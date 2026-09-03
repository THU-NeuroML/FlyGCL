#!/usr/bin/env python

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.gcl_training import make_dataset_with_optional_gcl, make_train_dataloader, maybe_switch_gcl_stage
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.scripts.multitask_eval_utils import build_multitask_eval_payload, save_multitask_eval_payload
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, get_safe_torch_device, has_method, init_logging
from lerobot.utils.wandb_utils import WandBLogger


@dataclass
class EWCTrainPipelineConfig(TrainPipelineConfig):
    ewc_lambda: float = 1000.0
    ewc_state_path: str | None = None
    ewc_fisher_batches: int = 200
    ewc_fisher_batch_size: int | None = None
    max_episodes_rendered: int = 0


EWC_STATE_NAME = "ewc_state.pt"


def trainable_named_parameters(policy: PreTrainedPolicy) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in policy.named_parameters() if param.requires_grad}


def load_ewc_state(path: str | Path | None, device: torch.device) -> dict[str, dict[str, torch.Tensor]] | None:
    if not path:
        return None
    state_path = Path(path)
    if not state_path.is_file():
        raise FileNotFoundError(f"EWC state not found: {state_path}")
    state = torch.load(state_path, map_location=device, weights_only=False)
    if "fisher" not in state or "theta_star" not in state:
        raise ValueError(f"Invalid EWC state at {state_path}; expected fisher and theta_star entries.")
    return {
        "fisher": {name: tensor.to(device) for name, tensor in state["fisher"].items()},
        "theta_star": {name: tensor.to(device) for name, tensor in state["theta_star"].items()},
    }


def compute_ewc_penalty(
    policy: PreTrainedPolicy,
    ewc_state: dict[str, dict[str, torch.Tensor]] | None,
    ewc_lambda: float,
) -> torch.Tensor:
    params = trainable_named_parameters(policy)
    device = get_device_from_parameters(policy)
    penalty = torch.zeros((), device=device)
    if ewc_state is None or ewc_lambda <= 0:
        return penalty

    fisher = ewc_state["fisher"]
    theta_star = ewc_state["theta_star"]
    missing = sorted(set(params) - set(fisher))
    if missing:
        raise KeyError(f"EWC state is missing {len(missing)} trainable parameters, first missing: {missing[:5]}")

    for name, param in params.items():
        previous_param = theta_star[name].to(device=device, dtype=param.dtype)
        importance = fisher[name].to(device=device, dtype=param.dtype)
        penalty = penalty + (importance * (param - previous_param).pow(2)).sum()
    return 0.5 * ewc_lambda * penalty


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    ewc_state: dict[str, dict[str, torch.Tensor]] | None,
    ewc_lambda: float,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        policy_loss, output_dict = policy.forward(batch)
        ewc_penalty = compute_ewc_penalty(policy, ewc_state, ewc_lambda)
        loss = policy_loss + ewc_penalty

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.policy_loss = policy_loss.item()
    train_metrics.ewc_penalty = ewc_penalty.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    output_dict = dict(output_dict or {})
    output_dict.update({"policy_loss": policy_loss.item(), "ewc_penalty": ewc_penalty.item()})
    return train_metrics, output_dict


def make_dataloader(dataset, cfg: EWCTrainPipelineConfig, device: torch.device, *, batch_size: int):
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def move_tensor_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
    return batch


def estimate_fisher(
    policy: PreTrainedPolicy,
    dataset,
    cfg: EWCTrainPipelineConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if cfg.ewc_fisher_batches <= 0:
        raise ValueError("ewc_fisher_batches must be positive to create the EWC state for future tasks.")

    batch_size = cfg.ewc_fisher_batch_size or cfg.batch_size
    old_batch_size = cfg.batch_size
    cfg.batch_size = batch_size
    try:
        _, dl_iter = make_train_dataloader(dataset, cfg, device, force_shuffle=True)
    finally:
        cfg.batch_size = old_batch_size
    params = trainable_named_parameters(policy)
    fisher = {name: torch.zeros_like(param, device=device) for name, param in params.items()}

    policy.train()
    for batch_idx in range(cfg.ewc_fisher_batches):
        batch = move_tensor_batch_to_device(next(dl_iter), device)
        policy.zero_grad(set_to_none=True)
        loss, _ = policy.forward(batch)
        loss.backward()
        for name, param in params.items():
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
        if (batch_idx + 1) % max(1, min(50, cfg.ewc_fisher_batches)) == 0:
            logging.info("Estimated EWC Fisher batches: %s/%s", batch_idx + 1, cfg.ewc_fisher_batches)

    for name in fisher:
        fisher[name] = (fisher[name] / cfg.ewc_fisher_batches).detach().cpu()
    policy.zero_grad(set_to_none=True)
    return fisher


def save_ewc_state(
    output_dir: Path,
    policy: PreTrainedPolicy,
    current_fisher: dict[str, torch.Tensor],
    previous_state: dict[str, dict[str, torch.Tensor]] | None,
) -> Path:
    params = trainable_named_parameters(policy)
    theta_star = {name: param.detach().cpu().clone() for name, param in params.items()}
    fisher = {name: tensor.detach().cpu().clone() for name, tensor in current_fisher.items()}

    if previous_state is not None:
        previous_fisher = previous_state["fisher"]
        for name in fisher:
            if name not in previous_fisher:
                raise KeyError(f"Previous EWC state is missing Fisher for parameter: {name}")
            fisher[name] = fisher[name] + previous_fisher[name].detach().cpu()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / EWC_STATE_NAME
    torch.save({"fisher": fisher, "theta_star": theta_star}, path)
    logging.info("Saved EWC state to %s", path)
    return path


@parser.wrap()
def train(cfg: EWCTrainPipelineConfig):
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

    ewc_state = load_ewc_state(cfg.ewc_state_path, device)
    if ewc_state is not None:
        logging.info("Loaded EWC state from %s", cfg.ewc_state_path)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"ewc_lambda={cfg.ewc_lambda}")
    logging.info(f"ewc_fisher_batches={cfg.ewc_fisher_batches}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dataloader, dl_iter = make_train_dataloader(dataset, cfg, device)
    current_gcl_stage = 0
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "policy_loss": AverageMeter("ploss", ":.3f"),
        "ewc_penalty": AverageMeter("ewc", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step)

    logging.info("Start EWC offline training")
    for _ in range(step, cfg.steps):
        previous_gcl_stage = current_gcl_stage
        current_gcl_stage, dl_iter = maybe_switch_gcl_stage(
            gcl_datasets, cfg, device, step, current_gcl_stage, dl_iter
        )
        if gcl_datasets is not None and current_gcl_stage != previous_gcl_stage:
            logging.info(
                "GCL EWC: estimating Fisher after stage %s before training stage %s",
                previous_gcl_stage,
                current_gcl_stage,
            )
            stage_fisher = estimate_fisher(policy, gcl_datasets[previous_gcl_stage], cfg, device)
            ewc_state_path = save_ewc_state(cfg.output_dir, policy, stage_fisher, ewc_state)
            ewc_state = load_ewc_state(str(ewc_state_path), device)
            policy.train()

        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        batch = move_tensor_batch_to_device(batch, device)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            ewc_state=ewc_state,
            ewc_lambda=cfg.ewc_lambda,
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
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            eval_infos = {}
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }

            for task in eval_envs.keys():
                eval_env_cfg = eval_envs[task]
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

            eval_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step)
            sum_avg_sum_reward = 0.0
            sum_pc_success = 0.0
            sum_eval_s = 0.0

            for task in eval_infos.keys():
                eval_info = eval_infos[task]
                avg_sum_reward = eval_info["aggregated"]["avg_sum_reward"]
                pc_success = eval_info["aggregated"]["pc_success"]
                eval_s = eval_info["aggregated"]["eval_s"]
                sum_avg_sum_reward += avg_sum_reward
                sum_pc_success += pc_success
                sum_eval_s += eval_s
                eval_tracker.__setattr__(f"avg_sum_reward_{task}", avg_sum_reward)
                eval_tracker.__setattr__(f"pc_success_{task}", pc_success)
                eval_tracker.__setattr__(f"eval_s_{task}", eval_s)

            eval_tracker.avg_sum_reward = sum_avg_sum_reward / len(eval_infos.keys())
            eval_tracker.pc_success = sum_pc_success / len(eval_infos.keys())
            eval_tracker.eval_s = sum_eval_s
            eval_payload = build_multitask_eval_payload(cfg, step, step_id, eval_infos)
            save_multitask_eval_payload(cfg, eval_payload)

            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    logging.info("Estimating EWC Fisher for future tasks")
    fisher = estimate_fisher(policy, dataset, cfg, device)
    save_ewc_state(cfg.output_dir, policy, fisher, ewc_state)
    logging.info("End of EWC training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
