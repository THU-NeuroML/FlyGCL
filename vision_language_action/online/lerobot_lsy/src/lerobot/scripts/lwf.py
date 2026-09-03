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
import torch.nn.functional as F
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
class LwFTrainPipelineConfig(TrainPipelineConfig):
    lwf_lambda: float = 0.1
    teacher_policy_path: str | None = None
    lwf_distill_old_tasks_only: bool = True
    max_episodes_rendered: int = 0


def make_dataloader(dataset, cfg: LwFTrainPipelineConfig, device: torch.device, *, batch_size: int):
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


def build_teacher_policy(cfg: LwFTrainPipelineConfig, dataset, device: torch.device) -> PreTrainedPolicy | None:
    if not cfg.teacher_policy_path:
        return None
    teacher_path = Path(cfg.teacher_policy_path)
    if not teacher_path.is_dir():
        raise FileNotFoundError(f"Teacher policy path not found: {teacher_path}")

    teacher_cfg = copy.deepcopy(cfg.policy)
    teacher_cfg.pretrained_path = str(teacher_path)
    teacher = make_policy(cfg=teacher_cfg, ds_meta=dataset.meta)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    logging.info("Loaded frozen LwF teacher from %s", teacher_path)
    return teacher


def prepare_ditflow_batch(policy: PreTrainedPolicy, batch: dict[str, Any]) -> dict[str, Any]:
    batch = policy.normalize_inputs(batch)
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
    batch = policy.normalize_targets(batch)
    return batch


def ditflow_velocity_prediction(
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if getattr(policy.config, "type", None) != "ditflow_mt" or not hasattr(policy, "dit_flow"):
        raise NotImplementedError("LwF velocity distillation currently supports policy.type='ditflow_mt' only.")
    prepared = prepare_ditflow_batch(policy, batch)
    global_cond = policy.dit_flow._prepare_global_conditioning(prepared)
    trajectory = prepared["action"]
    noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * trajectory
    return policy.dit_flow.velocity_net(noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond)


def filter_batch_by_mask(batch: dict[str, Any], mask: torch.Tensor) -> dict[str, Any] | None:
    mask = mask.bool().flatten()
    if mask.numel() == 0 or not torch.any(mask):
        return None
    filtered = {}
    batch_size = mask.shape[0]
    index = mask.detach().cpu().tolist()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            filtered[key] = value[mask]
        elif isinstance(value, list) and len(value) == batch_size:
            filtered[key] = [item for item, keep in zip(value, index) if keep]
        elif isinstance(value, tuple) and len(value) == batch_size:
            filtered[key] = tuple(item for item, keep in zip(value, index) if keep)
        else:
            filtered[key] = value
    return filtered


def maybe_filter_old_task_batch(
    batch: dict[str, Any],
    *,
    old_task_threshold: int | None,
    old_tasks_only: bool,
) -> dict[str, Any] | None:
    if not old_tasks_only or old_task_threshold is None:
        return batch
    gcl_task_id = batch.get("gcl_task_id")
    if gcl_task_id is None:
        # Non-GCL datasets do not expose the global source task id; keep the
        # original LwF behavior for those runs.
        return batch
    if gcl_task_id.ndim > 1:
        gcl_task_id = gcl_task_id.squeeze(-1)
    mask = gcl_task_id.to(device=batch["action"].device, dtype=torch.long) < int(old_task_threshold)
    return filter_batch_by_mask(batch, mask)


def compute_lwf_distillation_loss(
    student: PreTrainedPolicy,
    teacher: PreTrainedPolicy | None,
    batch: dict[str, Any],
    *,
    old_task_threshold: int | None = None,
    old_tasks_only: bool = True,
) -> torch.Tensor:
    device = get_device_from_parameters(student)
    if teacher is None:
        return torch.zeros((), device=device)
    if getattr(student.config, "type", None) != getattr(teacher.config, "type", None):
        raise ValueError(f"Student/teacher policy type mismatch: {student.config.type} vs {teacher.config.type}")
    if getattr(student.config, "type", None) != "ditflow_mt":
        raise NotImplementedError("LwF velocity distillation currently supports policy.type='ditflow_mt' only.")

    batch = maybe_filter_old_task_batch(
        batch,
        old_task_threshold=old_task_threshold,
        old_tasks_only=old_tasks_only,
    )
    if batch is None:
        return torch.zeros((), device=device)

    trajectory = batch["action"]
    noise = student.dit_flow.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)
    timesteps = student.dit_flow.noise_distribution.sample((trajectory.shape[0],)).to(trajectory.device)

    student_pred = ditflow_velocity_prediction(student, batch, noise, timesteps)
    with torch.no_grad():
        teacher_pred = ditflow_velocity_prediction(teacher, batch, noise, timesteps)
    return F.mse_loss(student_pred, teacher_pred)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    teacher_policy: PreTrainedPolicy | None,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lwf_lambda: float,
    old_task_threshold: int | None = None,
    old_tasks_only: bool = True,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    if teacher_policy is not None:
        teacher_policy.eval()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        policy_loss, output_dict = policy.forward(batch)
        distill_loss = compute_lwf_distillation_loss(
            policy,
            teacher_policy,
            batch,
            old_task_threshold=old_task_threshold,
            old_tasks_only=old_tasks_only,
        )
        loss = policy_loss + lwf_lambda * distill_loss

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
    train_metrics.lwf_distill_loss = distill_loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    output_dict = dict(output_dict or {})
    output_dict.update({"policy_loss": policy_loss.item(), "lwf_distill_loss": distill_loss.item()})
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: LwFTrainPipelineConfig):
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

    logging.info("Creating student policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    teacher_policy = build_teacher_policy(cfg, dataset, device)

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
    logging.info(f"lwf_lambda={cfg.lwf_lambda}")
    logging.info(f"lwf_distill_old_tasks_only={cfg.lwf_distill_old_tasks_only}")
    logging.info(f"teacher_policy_path={cfg.teacher_policy_path}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dataloader, dl_iter = make_train_dataloader(dataset, cfg, device)
    current_gcl_stage = 0
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "policy_loss": AverageMeter("ploss", ":.3f"),
        "lwf_distill_loss": AverageMeter("lwf", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step)

    logging.info("Start LwF offline training")
    for _ in range(step, cfg.steps):
        previous_gcl_stage = current_gcl_stage
        current_gcl_stage, dl_iter = maybe_switch_gcl_stage(
            gcl_datasets, cfg, device, step, current_gcl_stage, dl_iter
        )
        if gcl_datasets is not None and current_gcl_stage != previous_gcl_stage:
            logging.info(
                "GCL LwF: freezing teacher after stage %s for stage %s",
                previous_gcl_stage,
                current_gcl_stage,
            )
            teacher_policy = copy.deepcopy(policy)
            teacher_policy.to(device)
            teacher_policy.eval()
            for param in teacher_policy.parameters():
                param.requires_grad = False
            policy.train()

        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        batch = move_tensor_batch_to_device(batch, device)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            teacher_policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lwf_lambda=cfg.lwf_lambda,
            old_task_threshold=current_gcl_stage if gcl_datasets is not None else None,
            old_tasks_only=cfg.lwf_distill_old_tasks_only,
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

    logging.info("End of LwF training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
