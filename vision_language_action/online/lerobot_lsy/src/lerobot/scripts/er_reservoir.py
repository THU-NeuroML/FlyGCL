#!/usr/bin/env python

import copy
import json
import logging
import time
from collections import defaultdict
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
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import IMAGENET_STATS, make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.scripts.gcl_training import (
    make_dataset_with_optional_gcl,
    make_train_dataloader,
    maybe_switch_gcl_stage,
)
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
class ERReservoirTrainPipelineConfig(TrainPipelineConfig):
    replay_dataset: DatasetConfig | None = None
    replay_manifest_path: str | None = None
    replay_num_workers: int = 16
    replay_batch_size: int = 8

    max_episodes_rendered: int = 100


class ReservoirLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, episodes: list[int] | None = None, **kwargs):
        if episodes is None:
            raise ValueError("ReservoirLeRobotDataset requires a fixed list of episodes.")
        self._reservoir_episode_to_local = {episode: local_idx for local_idx, episode in enumerate(episodes)}
        super().__init__(*args, episodes=episodes, **kwargs)

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        original_ep_idx = item["episode_index"].item()
        try:
            local_ep_idx = self._reservoir_episode_to_local[original_ep_idx]
        except KeyError as exc:
            raise KeyError(
                f"Episode {original_ep_idx} is not part of this reservoir dataset episodes list."
            ) from exc

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, local_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, original_ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replay reservoir manifest not found: {manifest_path}")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("unit") != "episode" or manifest.get("strategy") != "reservoir":
        raise ValueError(f"Unsupported replay manifest format: {manifest_path}")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Replay manifest must contain a non-empty items list: {manifest_path}")
    return manifest


def _group_manifest_episodes(manifest: dict[str, Any]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for item in manifest["items"]:
        repo_id = item.get("repo_id")
        episode = item.get("episode")
        if not isinstance(repo_id, str) or not isinstance(episode, int):
            raise ValueError(f"Invalid reservoir item: {item}")
        key = (repo_id, episode)
        if key in seen:
            continue
        seen.add(key)
        grouped[repo_id].append(episode)
    return {repo_id: sorted(episodes) for repo_id, episodes in grouped.items()}


def _concat_lerobot_datasets(datasets: list[LeRobotDataset]) -> torch.utils.data.ConcatDataset:
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataset.meta = datasets[0].meta
    dataset.episode_data_index = datasets[0].episode_data_index.copy()
    for ds in datasets[1:]:
        offset = int(dataset.episode_data_index["to"][-1])
        dataset.episode_data_index["from"] = torch.cat(
            [dataset.episode_data_index["from"], ds.episode_data_index["from"] + offset]
        )
        dataset.episode_data_index["to"] = torch.cat(
            [dataset.episode_data_index["to"], ds.episode_data_index["to"] + offset]
        )
    return dataset


def make_replay_dataset(cfg: ERReservoirTrainPipelineConfig) -> LeRobotDataset | torch.utils.data.ConcatDataset:
    if cfg.replay_dataset is None:
        raise ValueError("replay_dataset must be provided for ER reservoir training.")
    if not cfg.replay_manifest_path:
        raise ValueError("replay_manifest_path must be provided for ER reservoir training.")

    manifest = _load_manifest(cfg.replay_manifest_path)
    grouped_episodes = _group_manifest_episodes(manifest)
    if not grouped_episodes:
        raise ValueError(f"Replay manifest has no usable episodes: {cfg.replay_manifest_path}")

    image_transforms = (
        ImageTransforms(cfg.replay_dataset.image_transforms) if cfg.replay_dataset.image_transforms.enable else None
    )

    first_repo_id = next(iter(grouped_episodes))
    meta = LeRobotDatasetMetadata(first_repo_id, root=cfg.replay_dataset.root, revision=cfg.replay_dataset.revision)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, meta)

    datasets = [
        ReservoirLeRobotDataset(
            repo_id,
            root=cfg.replay_dataset.root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.replay_dataset.revision,
            video_backend=cfg.replay_dataset.video_backend,
        )
        for repo_id, episodes in grouped_episodes.items()
    ]

    dataset = datasets[0] if len(datasets) == 1 else _concat_lerobot_datasets(datasets)
    logging.info(
        "Created reservoir replay dataset from %s repos and %s episodes: %s",
        len(grouped_episodes),
        sum(len(episodes) for episodes in grouped_episodes.values()),
        grouped_episodes,
    )

    if cfg.replay_dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()

    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock is not None else nullcontext():
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


@parser.wrap()
def train(cfg: ERReservoirTrainPipelineConfig):
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

    logging.info("Creating reservoir replay buffer dataset")
    replay_dataset = make_replay_dataset(cfg)

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
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

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
    logging.info(f"replay_dataset.num_frames={replay_dataset.num_frames if hasattr(replay_dataset, 'num_frames') else len(replay_dataset)}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        replay_sampler = EpisodeAwareSampler(
            replay_dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
        replay_sampler = None

    _, dl_iter = make_train_dataloader(dataset, cfg, device)
    current_gcl_stage = None

    replay_dataloader = torch.utils.data.DataLoader(
        replay_dataset,
        num_workers=cfg.replay_num_workers,
        batch_size=cfg.replay_batch_size,
        shuffle=shuffle,
        sampler=replay_sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    replay_dl_iter = cycle(replay_dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training with fixed reservoir replay")
    for _ in range(step, cfg.steps):
        current_gcl_stage, dl_iter = maybe_switch_gcl_stage(
            gcl_datasets, cfg, device, step, current_gcl_stage, dl_iter
        )
        start_time = time.perf_counter()
        batch = next(dl_iter)
        replay_batch = next(replay_dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in list(batch.keys()):
            if key not in replay_batch:
                # GCL-only metadata is absent from fixed-reservoir replay batches.
                batch.pop(key, None)
                continue
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
                replay_batch[key] = replay_batch[key].to(device, non_blocking=device.type == "cuda")
                batch[key] = torch.cat([batch[key], replay_batch[key]], dim=0)
            else:
                batch[key].extend(replay_batch[key])

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
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
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

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
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
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

            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )

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

            mean_avg_sum_reward = sum_avg_sum_reward / len(eval_infos.keys())
            mean_pc_success = sum_pc_success / len(eval_infos.keys())

            eval_tracker.avg_sum_reward = mean_avg_sum_reward
            eval_tracker.pc_success = mean_pc_success
            eval_tracker.eval_s = sum_eval_s

            eval_payload = build_multitask_eval_payload(cfg, step, step_id, eval_infos)
            save_multitask_eval_payload(cfg, eval_payload)

            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][-1], step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
