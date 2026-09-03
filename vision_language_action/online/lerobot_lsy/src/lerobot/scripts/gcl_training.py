import logging
from dataclasses import replace as dataclass_replace

import torch

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.gcl_dataset import build_gcl_datasets
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle


def make_dataset_with_optional_gcl(cfg):
    if not getattr(cfg.dataset, "use_gcl", False):
        return make_dataset(cfg), None

    repo_ids = [repo_id.strip() for repo_id in cfg.dataset.repo_id.split(",") if repo_id.strip()]
    if len(repo_ids) < 2:
        raise ValueError("dataset.use_gcl=true requires dataset.repo_id to contain multiple comma-separated repos.")

    source_datasets = []
    for repo_id in repo_ids:
        single_cfg = dataclass_replace(
            cfg,
            dataset=dataclass_replace(cfg.dataset, repo_id=repo_id, use_gcl=False),
        )
        source_datasets.append(make_dataset(single_cfg))

    gcl_datasets = build_gcl_datasets(
        source_datasets,
        n_percent=cfg.dataset.gcl_n_percent,
        m_percent=cfg.dataset.gcl_m_percent,
        seed=cfg.seed if cfg.seed is not None else 0,
    )
    if not gcl_datasets:
        raise ValueError("GCL dataset construction returned no stages.")

    logging.info(
        "GCL mode enabled: %d stages, n_percent=%s, m_percent=%s",
        len(gcl_datasets),
        cfg.dataset.gcl_n_percent,
        cfg.dataset.gcl_m_percent,
    )
    return gcl_datasets[0], gcl_datasets


def make_train_dataloader(dataset, cfg, device, *, force_shuffle=False):
    if force_shuffle or getattr(cfg.dataset, "use_gcl", False):
        shuffle = True
        sampler = None
    elif hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    return dataloader, cycle(dataloader)


def maybe_switch_gcl_stage(gcl_datasets, cfg, device, step, current_stage, dl_iter):
    if gcl_datasets is None:
        return current_stage, dl_iter

    steps_per_stage = max(cfg.steps // len(gcl_datasets), 1)
    stage_idx = min(step // steps_per_stage, len(gcl_datasets) - 1)
    if stage_idx == current_stage:
        return current_stage, dl_iter

    logging.info(
        "GCL: switching to stage %d/%d at step %d",
        stage_idx,
        len(gcl_datasets) - 1,
        step,
    )
    _, dl_iter = make_train_dataloader(
        gcl_datasets[stage_idx],
        cfg,
        device,
        force_shuffle=True,
    )
    return stage_idx, dl_iter
