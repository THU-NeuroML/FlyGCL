from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ACTION_GROUPS: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict(
    [("06", ("06",)), ("18", ("18",)), ("20", ("20",)), ("131415", ("13", "14", "15"))]
)
ACTION_TO_TASK = {
    action: task_id
    for task_id, actions in enumerate(ACTION_GROUPS.values())
    for action in actions
}


@dataclass(frozen=True)
class VideoToken:
    raw: str
    action: str

    @property
    def feature_name(self) -> str:
        return f"{self.raw}.npz"


@dataclass(frozen=True)
class SkillPair:
    better: VideoToken
    worse: VideoToken

    @property
    def sample_id(self) -> str:
        return f"{self.better.raw}||{self.worse.raw}"


def parse_token(raw: str) -> VideoToken:
    raw = raw.strip()
    parts = raw.split("_")
    if len(parts) < 4 or parts[0] not in ACTION_TO_TASK:
        raise ValueError(f"Invalid EgoExoLearn token: {raw!r}")
    return VideoToken(raw=raw, action=parts[0])


def read_pairs(path: str | Path) -> List[SkillPair]:
    pairs: List[SkillPair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.replace(",", " ").split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_no}: expected exactly two video tokens")
            better, worse = map(parse_token, fields)
            if ACTION_TO_TASK[better.action] != ACTION_TO_TASK[worse.action]:
                raise ValueError(f"{path}:{line_no}: pair crosses action groups")
            pairs.append(SkillPair(better=better, worse=worse))
    return pairs


def unique_videos(pairs: Iterable[SkillPair]) -> List[VideoToken]:
    videos: "OrderedDict[str, VideoToken]" = OrderedDict()
    for pair in pairs:
        videos.setdefault(pair.better.raw, pair.better)
        videos.setdefault(pair.worse.raw, pair.worse)
    return list(videos.values())


def fit_bradley_terry_scores(
    pairs: Sequence[SkillPair],
    iterations: int = 300,
    learning_rate: float = 0.08,
    l2: float = 1e-3,
) -> Dict[str, float]:
    """Fit deterministic task-local Bradley--Terry scores from training pairs only.

    Scores are centered and standardized independently per action task.  The
    implementation deliberately consumes no features and no validation pairs;
    it turns the transitive structure already present in the training comparison
    graph into an auxiliary target for the FlyGCL skill experts.
    """
    output: Dict[str, float] = {}
    for task_id in range(len(ACTION_GROUPS)):
        task_pairs = [
            pair for pair in pairs if ACTION_TO_TASK[pair.better.action] == task_id
        ]
        videos = sorted(
            {pair.better.raw for pair in task_pairs}
            | {pair.worse.raw for pair in task_pairs}
        )
        if not videos:
            continue
        index = {video: position for position, video in enumerate(videos)}
        better = np.asarray([index[pair.better.raw] for pair in task_pairs], dtype=np.int64)
        worse = np.asarray([index[pair.worse.raw] for pair in task_pairs], dtype=np.int64)
        scores = np.zeros(len(videos), dtype=np.float64)
        first_moment = np.zeros_like(scores)
        second_moment = np.zeros_like(scores)
        beta1, beta2 = 0.9, 0.999
        for step in range(1, max(int(iterations), 1) + 1):
            difference = np.clip(scores[better] - scores[worse], -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-difference))
            gradient = np.zeros_like(scores)
            np.add.at(gradient, better, probability - 1.0)
            np.add.at(gradient, worse, 1.0 - probability)
            gradient /= max(len(task_pairs), 1)
            gradient += float(l2) * scores / max(len(scores), 1)
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(gradient)
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            scores -= float(learning_rate) * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            scores -= scores.mean()
        scale = max(float(scores.std()), 1e-6)
        scores = np.clip(scores / scale, -4.0, 4.0)
        output.update({video: float(scores[position]) for video, position in index.items()})
    return output


def augment_pairs_from_scores(
    pairs: Sequence[SkillPair],
    scores: Mapping[str, float],
    task_id: int,
    ratio: float,
    seed: int,
    minimum_gap: float = 0.75,
) -> List[SkillPair]:
    """Add deterministic, high-confidence transitive comparisons within a task."""
    original = [pair for pair in pairs if ACTION_TO_TASK[pair.better.action] == int(task_id)]
    requested = int(round(len(original) * max(float(ratio), 0.0)))
    if requested <= 0:
        return original
    tokens = {token.raw: token for token in unique_videos(original)}
    candidates = [token for token in tokens.values() if token.raw in scores]
    known = {tuple(sorted((pair.better.raw, pair.worse.raw))) for pair in original}
    generated: List[SkillPair] = []
    rng = random.Random(int(seed) + int(task_id) * 7919)
    attempts = 0
    maximum_attempts = max(requested * 50, 1000)
    while len(generated) < requested and attempts < maximum_attempts and len(candidates) >= 2:
        attempts += 1
        first, second = rng.sample(candidates, 2)
        first_score, second_score = scores[first.raw], scores[second.raw]
        if abs(first_score - second_score) < float(minimum_gap):
            continue
        better, worse = (first, second) if first_score > second_score else (second, first)
        key = tuple(sorted((better.raw, worse.raw)))
        if key in known:
            continue
        known.add(key)
        generated.append(SkillPair(better=better, worse=worse))
    return original + generated


def load_feature(path: str | Path, expected_dim: int = 1024) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        if "arr_0" not in archive:
            raise KeyError(f"{path} does not contain arr_0")
        array = archive["arr_0"].astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != expected_dim:
        raise ValueError(f"Expected [T,{expected_dim}] in {path}, found {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array))


class ExoPools:
    """Deterministic 80/20 exo split and action-matched reference lookup."""

    def __init__(self, root: str | Path, seed: int = 42, val_ratio: float = 0.2):
        self.root = Path(root)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.pools: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}}
        for task_id, group_name in enumerate(ACTION_GROUPS):
            files = sorted((self.root / group_name).glob("*.npz"))
            if len(files) < 2:
                raise ValueError(f"Need at least two exo features for {group_name}, found {len(files)}")
            rng = random.Random(self.seed + task_id * 1009)
            rng.shuffle(files)
            n_val = min(len(files) - 1, max(1, int(round(len(files) * self.val_ratio))))
            self.pools["val"][group_name] = sorted(files[:n_val])
            self.pools["train"][group_name] = sorted(files[n_val:])

    def choose(self, task_id: int, split: str, sample_id: str) -> Path:
        group = list(ACTION_GROUPS.keys())[int(task_id)]
        candidates = self.pools[split][group]
        digest = hashlib.sha256(f"{self.seed}:{split}:{sample_id}".encode("utf-8")).digest()
        return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]

    def choose_many(self, task_id: int, split: str, sample_id: str, count: int) -> List[Path]:
        """Choose deterministic distinct references when the action pool permits."""
        group = list(ACTION_GROUPS.keys())[int(task_id)]
        candidates = self.pools[split][group]
        count = min(max(int(count), 1), len(candidates))
        digest = hashlib.sha256(f"{self.seed}:{split}:{sample_id}".encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "big") % len(candidates)
        return [candidates[(start + offset) % len(candidates)] for offset in range(count)]

    def counts(self) -> Dict[str, Dict[str, int]]:
        return {
            split: {group: len(files) for group, files in groups.items()}
            for split, groups in self.pools.items()
        }


class SkillDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[SkillPair],
        ego_root: str | Path,
        exo_pools: ExoPools,
        split: str,
        task_id: int,
        view: str,
        expected_dim: int = 1024,
        exo_references: int = 1,
        skill_scores: Optional[Mapping[str, float]] = None,
        synthetic_pair_ratio: float = 0.0,
        synthetic_pair_seed: int = 42,
        synthetic_minimum_gap: float = 0.75,
    ):
        self.original_pair_ids = {
            pair.sample_id
            for pair in pairs
            if ACTION_TO_TASK[pair.better.action] == int(task_id)
        }
        self.skill_scores = dict(skill_scores or {})
        self.pairs = augment_pairs_from_scores(
            pairs,
            self.skill_scores,
            task_id,
            synthetic_pair_ratio,
            synthetic_pair_seed,
            synthetic_minimum_gap,
        )
        self.ego_root = Path(ego_root)
        self.exo_pools = exo_pools
        self.split = split
        self.task_id = int(task_id)
        self.view = view
        self.expected_dim = int(expected_dim)
        self.exo_references = max(int(exo_references), 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        pair = self.pairs[index]
        result: Dict[str, object] = {
            "better": load_feature(self.ego_root / pair.better.feature_name, self.expected_dim),
            "worse": load_feature(self.ego_root / pair.worse.feature_name, self.expected_dim),
            "label": 1.0,
            "task_id": self.task_id,
            "sample_id": pair.sample_id,
            "pair_source": "observed" if pair.sample_id in self.original_pair_ids else "bt_transitive",
        }
        if self.skill_scores:
            better_score = float(self.skill_scores[pair.better.raw])
            worse_score = float(self.skill_scores[pair.worse.raw])
            result["better_skill_target"] = better_score
            result["worse_skill_target"] = worse_score
            result["skill_gap_target"] = max(better_score - worse_score, 0.05)
        if self.view == "ego_exo":
            exo_paths = self.exo_pools.choose_many(
                self.task_id, self.split, pair.sample_id, self.exo_references
            )
            exo_features = [load_feature(path, self.expected_dim) for path in exo_paths]
            result["exo"] = exo_features[0] if len(exo_features) == 1 else torch.stack(exo_features)
            result["exo_id"] = exo_paths[0].stem if len(exo_paths) == 1 else [path.stem for path in exo_paths]
        return result


def collate_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)  # type: ignore[arg-type]
        elif isinstance(values[0], (int, float)):
            dtype = torch.long if isinstance(values[0], int) else torch.float32
            output[key] = torch.tensor(values, dtype=dtype)
        else:
            output[key] = values
    return output


def build_manifest(
    train_pairs: Sequence[SkillPair],
    val_pairs: Sequence[SkillPair],
    ego_root: str | Path,
    exo_pools: ExoPools,
) -> Dict[str, object]:
    train_videos, val_videos = unique_videos(train_pairs), unique_videos(val_pairs)
    train_ids, val_ids = {v.raw for v in train_videos}, {v.raw for v in val_videos}
    ego_root = Path(ego_root)
    all_videos = {v.raw: v for v in train_videos + val_videos}
    missing = sorted(v.raw for v in all_videos.values() if not (ego_root / v.feature_name).is_file())

    def pair_counts(pairs: Sequence[SkillPair]) -> Dict[str, int]:
        result = {group: 0 for group in ACTION_GROUPS}
        for pair in pairs:
            result[list(ACTION_GROUPS.keys())[ACTION_TO_TASK[pair.better.action]]] += 1
        return result

    def video_counts(videos: Sequence[VideoToken]) -> Dict[str, int]:
        result = {group: 0 for group in ACTION_GROUPS}
        for video in videos:
            result[list(ACTION_GROUPS.keys())[ACTION_TO_TASK[video.action]]] += 1
        return result

    return {
        "schema": "flygcl_egoexolearn_manifest_v1",
        "dataset": "EgoExoLearn",
        "task_order": list(ACTION_GROUPS.keys()),
        "task_groups": {key: list(value) for key, value in ACTION_GROUPS.items()},
        "skill_pairs": {"train": pair_counts(train_pairs), "val": pair_counts(val_pairs)},
        "action_unique_videos": {"train": video_counts(train_videos), "val": video_counts(val_videos)},
        "action_train_val_overlap": {
            "count": len(train_ids & val_ids),
            "policy": "retained_by_user_request",
            "warning": "Action-classification validation contains videos also present in training.",
        },
        "exo": {
            "split_seed": exo_pools.seed,
            "val_ratio": exo_pools.val_ratio,
            "counts": exo_pools.counts(),
            "pairing": "deterministic action-matched reference; not synchronized",
        },
        "missing_ego_features": missing,
    }


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
