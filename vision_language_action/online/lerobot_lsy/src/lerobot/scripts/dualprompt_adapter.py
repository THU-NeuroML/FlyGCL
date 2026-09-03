#!/usr/bin/env python

import copy
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.gcl_training import make_dataset_with_optional_gcl, make_train_dataloader, maybe_switch_gcl_stage
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir, get_step_identifier, load_training_state, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import format_big_number, get_safe_torch_device, has_method, init_logging
from lerobot.utils.wandb_utils import WandBLogger

from peft import get_peft_model, PeftConfig, PeftModel
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING


class PeftWrapperPolicy(torch.nn.Module):
    policy: PreTrainedPolicy

    def __init__(self, policy: PreTrainedPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, *args, **kwargs):
        return self.policy.forward(*args, **kwargs)


@dataclass
class DualPromptAdapterTrainPipelineConfig(TrainPipelineConfig):
    peft_cfg_path: Path | None = None
    peft_weight_path: Path | None = None
    max_episodes_rendered: int = 0

    def __post_init__(self):
        assert self.peft_cfg_path or self.peft_weight_path, "One from (peft_cfg_path, peft_weight_path) must be specified"


def build_multitask_eval_payload(
    cfg: DualPromptAdapterTrainPipelineConfig,
    step: int,
    step_id: str,
    eval_infos: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_task = {}
    per_task_eval_info = {}
    task_list = list(eval_infos.keys())
    for task, eval_info in eval_infos.items():
        per_task[task] = copy.deepcopy(eval_info.get("aggregated", {}))
        per_task_eval_info[task] = copy.deepcopy(eval_info)
    avg_success = sum(per_task[task]["pc_success"] for task in task_list) / len(task_list)
    avg_sum_reward = sum(per_task[task]["avg_sum_reward"] for task in task_list) / len(task_list)
    total_eval_s = sum(per_task[task]["eval_s"] for task in task_list)
    return {
        "step": step,
        "step_id": step_id,
        "seed": cfg.seed,
        "job_name": cfg.job_name,
        "output_dir": str(cfg.output_dir),
        "tasks": task_list,
        "per_task": per_task,
        "per_task_eval_info": per_task_eval_info,
        "avg_success": avg_success,
        "avg_sum_reward": avg_sum_reward,
        "eval_s": total_eval_s,
    }


def save_multitask_eval_payload(cfg: DualPromptAdapterTrainPipelineConfig, payload: dict[str, Any]) -> None:
    eval_dir = cfg.output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    latest_path = cfg.output_dir / "multitask_eval_info.json"
    step_path = eval_dir / f"multitask_eval_info_step_{payload['step_id']}.json"
    for path in (latest_path, step_path):
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
    logging.info(f"Saved structured multitask eval to {latest_path} and {step_path}")


def set_peft_module_train(peft_modules: list, train: bool = True):
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_modules[0].peft_config.peft_type]
    for peft_module in peft_modules:
        for name, module in peft_module.named_modules():
            if prefix in name or name == "":
                module.train(train)
            if "base_layer" in name:
                module.train(False)
    return peft_modules


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    peft_modules: list,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    peft_modules = set_peft_module_train(peft_modules)
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@torch.no_grad()
def estimate_dualprompt_prototypes(cfg, policy, peft_modules: list, dataset, device) -> None:
    prototype_batches = int(peft_modules[0].peft_config.dual_prototype_batches)
    if prototype_batches <= 0:
        logging.info("Skip DualPrompt prototype estimation because dual_prototype_batches <= 0")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)
    feature_sums: dict[int, torch.Tensor] = {}
    feature_counts: dict[int, int] = {}
    hooks = []

    def make_hook(module_id: int):
        def hook(module, inputs, output):
            features = module.collect_features(inputs[0]).detach().float().cpu()
            if module_id not in feature_sums:
                feature_sums[module_id] = features.sum(dim=0)
                feature_counts[module_id] = features.shape[0]
            else:
                feature_sums[module_id] += features.sum(dim=0)
                feature_counts[module_id] += features.shape[0]
        return hook

    for idx, peft_module in enumerate(peft_modules):
        if hasattr(peft_module, "dualprompt_router"):
            hooks.append(peft_module.register_forward_hook(make_hook(idx)))

    was_training = policy.training
    policy.train()
    try:
        for _ in range(prototype_batches):
            batch = next(dl_iter)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            policy.forward(batch)
    finally:
        for hook in hooks:
            hook.remove()
        policy.train(was_training)

    for idx, peft_module in enumerate(peft_modules):
        if idx not in feature_sums:
            continue
        prototype = feature_sums[idx] / max(feature_counts[idx], 1)
        task_id = peft_module._active_task
        expert_adapter_id = peft_module._forwarded_adapter_id
        peft_module.set_dualprompt_task_prototype(task_id, expert_adapter_id, prototype)
        logging.info(
            f"Layer {peft_module.layer_name}.{peft_module.layer_id}: saved DualPrompt prototype "
            f"for task {task_id} -> expert adapter {expert_adapter_id}"
        )


def setup_dualprompt_stage(peft_modules: list, peft_config) -> list[torch.nn.Parameter]:
    new_task_id = peft_config.num_learned_task
    train_params = []
    for peft_module in peft_modules:
        if not hasattr(peft_module, "dualprompt_router"):
            raise ValueError("dualprompt_adapter routing requires a dualprompt_router on every target module")
        key = f"{peft_module.layer_name}.{peft_module.layer_id}"
        for adapter_id in range(peft_module.num_adapters):
            peft_module.freeze_adapter(adapter_id)

        if new_task_id == 0 and peft_module.get_dualprompt_general_adapter_id() < 0:
            general_params = peft_module.add_adapter_only(-1)
            general_adapter_id = peft_module.num_adapters - 1
            peft_module.set_dualprompt_general_adapter_id(general_adapter_id)
            logging.info(f"Layer {key}: created general adapter {general_adapter_id}")
        else:
            general_adapter_id = peft_module.get_dualprompt_general_adapter_id()
            general_params = peft_module.unfreeze_adapter(general_adapter_id) if peft_config.dual_general_train else []
            logging.info(f"Layer {key}: reuse general adapter {general_adapter_id}")

        expert_params = peft_module.add_adapter_only(new_task_id)
        expert_adapter_id = peft_module.num_adapters - 1
        peft_module._forwarded_adapter_id = expert_adapter_id
        peft_module._forwarded_discriminator_id = -1
        peft_module._active_task = new_task_id
        train_params.extend(general_params)
        train_params.extend(expert_params)
        peft_config.structure[key] = [peft_module.num_adapters, peft_module.num_discriminators]
        logging.info(f"Layer {key}: created expert adapter {expert_adapter_id} for task {new_task_id}")
    peft_config.num_learned_task += 1
    return train_params


@parser.wrap()
def train(cfg: DualPromptAdapterTrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset, gcl_datasets = make_dataset_with_optional_gcl(cfg)

    eval_envs = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        task_list = [task.strip() for task in cfg.env.task.split(",") if task.strip()]
        if not task_list:
            raise ValueError("No valid tasks found in env.task")
        eval_envs = {}
        for task in task_list:
            env_cfg = copy.deepcopy(cfg.env)
            env_cfg.task = task
            eval_envs[task] = env_cfg

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    logging.info("Wrapping policy with DualPrompt adapter PEFT module")
    peft_wrapper_policy = PeftWrapperPolicy(policy=policy)
    if cfg.peft_weight_path:
        peft_policy = PeftModel.from_pretrained(
            peft_wrapper_policy,
            cfg.peft_weight_path,
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
        peft_config = peft_policy.peft_config["default"]
    else:
        peft_cfg = PeftConfig.from_pretrained(cfg.peft_cfg_path)
        peft_cfg.inference_mode = False
        peft_policy = get_peft_model(peft_wrapper_policy, peft_cfg)
        peft_config = peft_policy.peft_config["default"]

    if getattr(peft_config, "routing_mode", None) != "dualprompt_adapter":
        raise ValueError(f"dualprompt_adapter.py requires routing_mode='dualprompt_adapter', got {peft_config.routing_mode}")

    peft_modules = peft_policy.base_model.adapter_layers
    step = 0
    adapter_params = setup_dualprompt_stage(peft_modules, peft_config)

    logging.info("Creating optimizer and scheduler")
    optimizer = cfg.optimizer.build(adapter_params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler else None
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_adapter_params = sum(p.numel() for p in adapter_params)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_adapter_params=} ({format_big_number(num_adapter_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dataloader, dl_iter = make_train_dataloader(dataset, cfg, device)
    current_gcl_stage = 0
    peft_modules = set_peft_module_train(peft_modules)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step)

    logging.info("Start DualPrompt adapter training")
    for _ in range(step, cfg.steps):
        previous_gcl_stage = current_gcl_stage
        current_gcl_stage, dl_iter = maybe_switch_gcl_stage(
            gcl_datasets, cfg, device, step, current_gcl_stage, dl_iter
        )
        if gcl_datasets is not None and current_gcl_stage != previous_gcl_stage:
            logging.info(
                "GCL DualPrompt: finalizing stage %s and creating expert for stage %s",
                previous_gcl_stage,
                current_gcl_stage,
            )
            estimate_dualprompt_prototypes(cfg, policy, peft_modules, gcl_datasets[previous_gcl_stage], device)
            adapter_params = setup_dualprompt_stage(peft_modules, peft_config)
            optimizer = cfg.optimizer.build(adapter_params)
            remaining_steps = max(cfg.steps - step, 1)
            lr_scheduler = cfg.scheduler.build(optimizer, remaining_steps) if cfg.scheduler else None
            grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
            peft_modules = set_peft_module_train(peft_modules)

        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            peft_modules,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step, mode="train")
            train_tracker.reset_averages()

        if is_eval_step or is_saving_step:
            prototype_dataset = gcl_datasets[current_gcl_stage] if gcl_datasets is not None else dataset
            estimate_dualprompt_prototypes(cfg, policy, peft_modules, prototype_dataset, device)

        if cfg.env and is_eval_step:
            run_multitask_eval(cfg, policy, peft_modules, eval_envs, dataset, step, device, wandb_logger)

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            if hasattr(peft_policy.base_model, "finalize_router_state"):
                peft_policy.base_model.finalize_router_state()
            peft_policy.save_pretrained(str(checkpoint_dir / "adapter"))
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)

    logging.info("End of DualPrompt adapter training")
    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def run_multitask_eval(cfg, policy, peft_modules, eval_envs, dataset, step, device, wandb_logger) -> None:
    step_id = get_step_identifier(step, cfg.steps)
    logging.info(f"Eval policy at step {step}")
    to_train_module_list = []
    for peft_module in peft_modules:
        for name, parameter in peft_module.named_parameters():
            if parameter.requires_grad:
                to_train_module_list.append(name)
                parameter.requires_grad = False

    eval_infos = {}
    eval_metrics = {
        "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
        "pc_success": AverageMeter("success", ":.1f"),
        "eval_s": AverageMeter("eval_s", ":.3f"),
    }
    for task, eval_env_cfg in eval_envs.items():
        logging.info(f"Eval task {task}")
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            eval_info = eval_policy_with_env_init(
                eval_env_cfg,
                cfg.eval.batch_size,
                False,
                policy,
                cfg.eval.n_episodes,
                videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                max_episodes_rendered=cfg.max_episodes_rendered,
                start_seed=cfg.seed,
            )
            eval_infos[task] = eval_info
        eval_metrics[f"avg_sum_reward_{task}"] = AverageMeter(f"∑rwrd_{task}", ":.3f")
        eval_metrics[f"pc_success_{task}"] = AverageMeter(f"success_{task}", ":.1f")
        eval_metrics[f"eval_s_{task}"] = AverageMeter(f"eval_s_{task}", ":.3f")

    for peft_module in peft_modules:
        for name, parameter in peft_module.named_parameters():
            if name in to_train_module_list:
                parameter.requires_grad = True

    eval_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step)
    sum_avg_sum_reward = 0.0
    sum_pc_success = 0.0
    sum_eval_s = 0.0
    multitask_eval_payload = build_multitask_eval_payload(cfg, step, step_id, eval_infos)
    save_multitask_eval_payload(cfg, multitask_eval_payload)

    last_eval_info = None
    for task, eval_info in eval_infos.items():
        last_eval_info = eval_info
        avg_sum_reward = eval_info["aggregated"]["avg_sum_reward"]
        pc_success = eval_info["aggregated"]["pc_success"]
        eval_s = eval_info["aggregated"]["eval_s"]
        sum_avg_sum_reward += avg_sum_reward
        sum_pc_success += pc_success
        sum_eval_s += eval_s
        eval_tracker.__setattr__(f"avg_sum_reward_{task}", avg_sum_reward)
        eval_tracker.__setattr__(f"pc_success_{task}", pc_success)
        eval_tracker.__setattr__(f"eval_s_{task}", eval_s)

    eval_tracker.avg_sum_reward = sum_avg_sum_reward / len(eval_infos)
    eval_tracker.pc_success = sum_pc_success / len(eval_infos)
    eval_tracker.eval_s = sum_eval_s
    logging.info(eval_tracker)
    if wandb_logger and last_eval_info is not None:
        wandb_logger.log_dict({**eval_tracker.to_dict(), **last_eval_info}, step, mode="eval")
        if last_eval_info.get("video_paths"):
            wandb_logger.log_video(last_eval_info["video_paths"][-1], step, mode="eval")


if __name__ == "__main__":
    init_logging()
    train()
