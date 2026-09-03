import csv
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

_WARNED_MESSAGES = set()


def ensure_dir(path: str) -> str:
    if path in ("", None):
        return ""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    warnings.warn(f"[SeqLoRAAnalysis] {message}", RuntimeWarning, stacklevel=2)


def to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return to_serializable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def save_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(to_serializable(payload), f, indent=2, sort_keys=True)


def load_json(path: str, default: Any = None) -> Any:
    if not path or not os.path.exists(path):
        warn_once(f"missing json file: {path}")
        return default
    with open(path, "r") as f:
        return json.load(f)


def safe_read_csv(path: str):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for post-hoc analysis") from exc

    if not path or not os.path.exists(path):
        warn_once(f"missing csv file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn_once(f"failed to read csv {path}: {exc}")
        return pd.DataFrame()


def safe_to_csv(df, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, na_rep="nan")


def save_tensor(path: str, tensor: torch.Tensor) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(tensor.detach().cpu(), path)


def append_csv(path: str, row: Dict[str, Any], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(row.keys())
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    normalized = {key: to_serializable(row.get(key, float("nan"))) for key in fieldnames}
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def append_csv_row(path: str, row_dict: Dict[str, Any], fieldnames: Optional[List[str]] = None) -> None:
    append_csv(path, row_dict, fieldnames=fieldnames)


def append_rows_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    for row in rows:
        append_csv(path, row, fieldnames=fieldnames)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def extract_batch(batch):
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images", batch.get("x")))
        labels = batch.get("label", batch.get("labels", batch.get("y")))
        indices = batch.get("index", batch.get("indices", batch.get("idx")))
        return images, labels, indices
    if isinstance(batch, (list, tuple)):
        images = batch[0] if len(batch) > 0 else None
        labels = batch[1] if len(batch) > 1 else None
        indices = batch[2] if len(batch) > 2 else None
        return images, labels, indices
    return None, None, None


def limit_batches(dataloader, max_batches: int):
    if dataloader is None:
        return
    for batch_idx, batch in enumerate(dataloader):
        if max_batches >= 0 and batch_idx >= max_batches:
            break
        yield batch_idx, batch
