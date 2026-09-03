from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .io_utils import read_csv, read_json, to_float


def discover_analysis_dirs(run_dir: Path) -> List[Path]:
    candidates = []
    for pattern in ("analysis*", "*/analysis*", "posthoc*", "*/posthoc*"):
        candidates.extend(p for p in run_dir.glob(pattern) if p.is_dir())
    return sorted(set(candidates))


def first_existing(run_dir: Path, filenames: Iterable[str]) -> Optional[Path]:
    for base in [run_dir] + discover_analysis_dirs(run_dir):
        for name in filenames:
            path = base / name
            if path.exists():
                return path
    return None


def run_metadata(run_dir: Path, method: str = "", dataset: str = "", seed: str = "", config: str = "") -> Dict[str, object]:
    leaderboard = read_json(run_dir / "leaderboard_summary.json")
    cfg_path = Path(config) if config else (run_dir / "config.yaml")
    return {
        "run_dir": str(run_dir),
        "method": method or str(leaderboard.get("method", "")),
        "dataset": dataset or str(leaderboard.get("dataset", "")),
        "seed": seed or str(leaderboard.get("seed", "")),
        "config": str(cfg_path) if cfg_path.exists() else str(config or ""),
    }


def load_gcl_metric_rows(run_dir: Path) -> List[Dict[str, object]]:
    rows = []
    path = run_dir / "gcl_metrics.json"
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            import json

            payload = json.loads(line)
        except Exception:
            continue
        if payload.get("type") in {"session_end", "periodic_eval"}:
            rows.append(payload)
    return rows


def prefixed_group_rows(rows: List[Mapping[str, object]], groups: List[str], group_key: str = "group_name") -> Dict[str, Dict[str, object]]:
    out = {g: {} for g in groups}
    for row in rows:
        group = str(row.get(group_key, "all"))
        normalized = {"old_all": "old", "new_current": "new"}.get(group, group)
        if normalized in out:
            out[normalized] = dict(row)
    return out


def corr(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return math.nan
    xbar = sum(x for x, _ in pairs) / len(pairs)
    ybar = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    den_x = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return num / (den_x * den_y) if den_x > 0 and den_y > 0 else math.nan

