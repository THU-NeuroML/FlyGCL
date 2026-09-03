import os
import random

import torch
from torch.utils.data import Dataset


class GCLDataset(Dataset):

    def __init__(self, source_datasets, sample_pairs):
        self.source_datasets = source_datasets
        self.sample_pairs = sample_pairs
        self.num_frames = sum(len(ds) for ds in source_datasets)
        self.num_episodes = sum(
            getattr(ds, 'num_episodes', 1) for ds in source_datasets
        )
        self.meta = source_datasets[0].meta
        self.episode_data_index = source_datasets[0].episode_data_index

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        src_task_id, src_idx = self.sample_pairs[idx]
        item = dict(self.source_datasets[src_task_id][src_idx])
        # Preserve the original source task id after GCL mixing. Individual
        # per-task datasets often carry local task_index=0, so LwF needs this
        # global id to distill only on tasks the teacher has actually seen.
        item["gcl_task_id"] = torch.tensor(src_task_id, dtype=torch.long)
        return item


def build_gcl_datasets(datasets, n_percent=50, m_percent=30, seed=0):
    num_tasks = len(datasets)
    if num_tasks == 0:
        return []

    rng = random.Random(int(seed))
    source_order = list(range(num_tasks))
    rng.shuffle(source_order)

    disjoint_num = int(round(num_tasks * (n_percent / 100.0)))
    disjoint_tasks = set(source_order[:disjoint_num])

    owners = list(range(num_tasks))
    rng.shuffle(owners)
    task_to_stage = {src: owners[i] for i, src in enumerate(source_order)}

    stage_pairs = [[] for _ in range(num_tasks)]
    blurry_pool = []

    for task_id in range(num_tasks):
        owner_stage = task_to_stage[task_id]
        indices = list(range(len(datasets[task_id])))
        rng.shuffle(indices)

        if task_id in disjoint_tasks:
            stage_pairs[owner_stage].extend((task_id, i) for i in indices)
        else:
            blur_cnt = int(len(indices) * (m_percent / 100.0))
            blurry_pool.extend((task_id, i) for i in indices[:blur_cnt])
            stage_pairs[owner_stage].extend(
                (task_id, i) for i in indices[blur_cnt:]
            )

    rng.shuffle(blurry_pool)
    for i, pair in enumerate(blurry_pool):
        stage_pairs[i % num_tasks].append(pair)

    boost_ids_raw = os.environ.get("GCL_BOOST_TASK_IDS", "").strip()
    boost_factor = int(os.environ.get("GCL_BOOST_FACTOR", "1") or "1")
    if boost_ids_raw and boost_factor > 1:
        boost_ids = {int(x) for x in boost_ids_raw.replace(";", ",").split(",") if x.strip()}
        for stage_idx in range(num_tasks):
            boosted_pairs = []
            for pair in stage_pairs[stage_idx]:
                if pair[0] in boost_ids:
                    boosted_pairs.extend([pair] * (boost_factor - 1))
            stage_pairs[stage_idx].extend(boosted_pairs)
        print(
            f"GCL target boost enabled: task_ids={sorted(boost_ids)} factor={boost_factor}",
            flush=True,
        )

    for i in range(num_tasks):
        rng.shuffle(stage_pairs[i])

    return [GCLDataset(datasets, stage_pairs[i]) for i in range(num_tasks)]
