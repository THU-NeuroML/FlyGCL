from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fieldnames})


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_md(path: Path, title: str, sections: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", ""])
        if isinstance(body, list):
            lines.extend(str(x) for x in body)
        else:
            lines.append(str(body))
        lines.append("")
    path.write_text("\n".join(lines))


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def to_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def mean(values: Iterable[object]) -> float:
    xs = [to_float(v) for v in values]
    xs = [x for x in xs if not math.isnan(x)]
    return float(sum(xs) / len(xs)) if xs else math.nan


def last_row(rows: Sequence[Mapping[str, object]], key: str = "session_id") -> Optional[Mapping[str, object]]:
    if not rows:
        return None
    return max(rows, key=lambda r: to_float(r.get(key, r.get("step", r.get("step_idx", 0))), 0.0))


def _csv_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value

