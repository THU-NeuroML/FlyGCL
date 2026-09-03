from __future__ import annotations

import ast
import csv
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AnticipationRecord:
    video_id: str
    start_sec: float
    end_sec: float
    labels: tuple[int, ...]
    view: str
    split: str
    home_session: int = -1
    primary_label: int = -1


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def load_valid_ids(path: Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_official_csv(
    path: Path, target: str, view: str, split: str, valid_ids: Sequence[int]
) -> List[AnticipationRecord]:
    mapping = {raw: index for index, raw in enumerate(valid_ids)}
    records: List[AnticipationRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_labels = ast.literal_eval(row[target])
            labels = tuple(sorted({mapping[int(label)] for label in raw_labels if int(label) in mapping}))
            if not labels or float(row["start_sec"]) < 3.0:
                continue
            records.append(
                AnticipationRecord(
                    video_id=row["video_id"],
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    labels=labels,
                    view=view,
                    split=split,
                )
            )
    return records


def assign_class_sessions(records: Iterable[AnticipationRecord], classes: int, sessions: int) -> List[int]:
    """Greedy frequency-balanced class partition, deterministic by class id."""
    counts = [0] * classes
    for record in records:
        for label in record.labels:
            counts[label] += 1
    loads = [0] * sessions
    assignment = [-1] * classes
    for label in sorted(range(classes), key=lambda item: (-counts[item], item)):
        session = min(range(sessions), key=lambda item: (loads[item], item))
        assignment[label] = session
        loads[session] += counts[label]
    return assignment


def attach_sessions(
    records: Sequence[AnticipationRecord], class_sessions: Sequence[int]
) -> List[AnticipationRecord]:
    output = []
    for record in records:
        # The rarest/earliest assigned label defines the partition unit while
        # retaining all valid multi-label targets for BCE and recall.
        home, primary = min((class_sessions[label], label) for label in record.labels)
        output.append(
            AnticipationRecord(
                record.video_id,
                record.start_sec,
                record.end_sec,
                record.labels,
                record.view,
                record.split,
                home,
                primary,
            )
        )
    return output


def choose_blurry_classes(classes: int, disjoint_ratio: float, seed: int) -> set[int]:
    order = list(range(int(classes)))
    random = __import__("random").Random(int(seed))
    random.shuffle(order)
    disjoint_count = min(
        len(order), max(0, int(round(len(order) * float(disjoint_ratio))))
    )
    return set(order[disjoint_count:])


def build_session_stream(
    records: Sequence[AnticipationRecord],
    session: int,
    seed: int,
    disjoint_ratio: float,
    blurry_ratio: float,
    blurry_classes: set[int] | None = None,
) -> List[AnticipationRecord]:
    """Build a deterministic Si-Blurry-like stream without split leakage.

    r_D of each home partition remains session-exclusive.  r_B is assigned to
    a deterministic non-home session, with the rest staying at home.  This
    changes stream membership only and never crosses train/val/test splits.
    """
    selected = []
    for record in records:
        key = f"{record.view}:{record.video_id}:{record.start_sec:.3f}:{record.labels}"
        draw = stable_unit_interval(key, seed)
        assigned = record.home_session
        primary = (
            record.primary_label if record.primary_label >= 0 else min(record.labels)
        )
        is_blurry = blurry_classes is None or primary in blurry_classes
        if is_blurry and draw < blurry_ratio:
            offset = 1 + int(stable_unit_interval(key, seed + 17) * 3)
            assigned = (record.home_session + offset) % 4
        if assigned == session:
            selected.append(record)
    return selected


def find_feature(feature_root: Path, video_id: str) -> Path:
    direct = feature_root / f"{video_id}.pt"
    if direct.is_file():
        return direct
    gazed = feature_root / f"{video_id}_50.pt"
    if gazed.is_file():
        return gazed
    return direct


def audit_feature_coverage(feature_root: Path, records: Sequence[AnticipationRecord]) -> Dict[str, int]:
    unique = {record.video_id for record in records}
    found = sum(find_feature(feature_root, video_id).is_file() for video_id in unique)
    return {"unique_videos": len(unique), "features_found": found, "features_missing": len(unique) - found}


def audit_split_overlap(split_records: Dict[str, Sequence[AnticipationRecord]]) -> Dict[str, Dict[str, int]]:
    """Distinguish allowed same-video reuse from forbidden exact sample reuse."""
    output: Dict[str, Dict[str, int]] = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if first not in split_records or second not in split_records:
            continue
        first_videos = {item.video_id for item in split_records[first]}
        second_videos = {item.video_id for item in split_records[second]}
        first_samples = {
            (item.video_id, item.start_sec, item.end_sec, item.labels)
            for item in split_records[first]
        }
        second_samples = {
            (item.video_id, item.start_sec, item.end_sec, item.labels)
            for item in split_records[second]
        }
        output[f"{first}_{second}"] = {
            "video_id_overlap": len(first_videos & second_videos),
            "exact_sample_overlap": len(first_samples & second_samples),
        }
    return output


def exact_sample_key(item: AnticipationRecord):
    return (item.video_id, item.start_sec, item.end_sec, item.labels)


def decontaminate_training_split(
    train: Sequence[AnticipationRecord],
    validation: Sequence[AnticipationRecord],
    test: Sequence[AnticipationRecord],
) -> tuple[List[AnticipationRecord], int]:
    """Remove projected-task duplicates from train, preserving official eval rows."""
    protected = {exact_sample_key(item) for item in validation}
    protected.update(exact_sample_key(item) for item in test)
    clean = [item for item in train if exact_sample_key(item) not in protected]
    return clean, len(train) - len(clean)


class AnticipationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[AnticipationRecord],
        feature_root: Path,
        classes: int,
        segments: int = 10,
        feature_fps: float = 5.0,
        anticipation_gap: float = 1.0,
        context_seconds: float = 2.0,
        cache_videos: int = 8,
    ) -> None:
        self.records = list(records)
        self.feature_root = feature_root
        self.classes = classes
        self.segments = segments
        self.feature_fps = feature_fps
        self.anticipation_gap = anticipation_gap
        self.context_seconds = context_seconds
        self.cache_videos = max(int(cache_videos), 0)
        self._feature_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = find_feature(self.feature_root, record.video_id)
        feature = self._feature_cache.pop(record.video_id, None)
        if feature is None:
            try:
                feature = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                feature = torch.load(path, map_location="cpu")
            if isinstance(feature, dict):
                feature = feature.get("features", feature.get("feature"))
            feature = torch.as_tensor(feature, dtype=torch.float32)
        if self.cache_videos > 0:
            self._feature_cache[record.video_id] = feature
            while len(self._feature_cache) > self.cache_videos:
                self._feature_cache.popitem(last=False)
        if feature.ndim != 2:
            raise ValueError(f"Expected [T,D] feature in {path}, got {tuple(feature.shape)}")
        end_sec = record.start_sec - self.anticipation_gap
        start_sec = end_sec - self.context_seconds
        start = max(0, int(start_sec * self.feature_fps))
        end = min(feature.shape[0], max(start + 1, int(end_sec * self.feature_fps) + 1))
        clip = feature[start:end]
        if clip.shape[0] != self.segments:
            clip = F.interpolate(
                clip.transpose(0, 1).unsqueeze(0),
                size=self.segments,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        labels = torch.zeros(self.classes, dtype=torch.float32)
        labels[list(record.labels)] = 1.0
        return {
            "feature": clip,
            "label": labels,
            "view": 0 if record.view == "ego" else 1,
            "sample_id": f"{record.view}:{record.video_id}:{record.start_sec:.3f}",
        }
