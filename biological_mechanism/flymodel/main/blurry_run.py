"""Run the fixed 80-task boundary-blur matrix."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import NamedTuple, Any
from .audit import atomic_json, canonical_sha256, file_sha256, runtime_identity
from .blurry_data import CELLS, build_blurry_subset
from .blurry_experiment import CONDITIONS, train_blurry
from .config import PROJECT_ROOT, SEEDS, Config

ETA = 1e-3
GAMMA = 10.0
DEFAULT_ROOT = PROJECT_ROOT / "results" / "boundary_blur"


class Task(NamedTuple):
    n: int
    m: int
    seed: int
    condition: str


def build_tasks() -> tuple[Task, ...]:
    tasks = tuple(Task(n, m, seed, condition) for n, m in CELLS for seed in SEEDS for condition in CONDITIONS)
    if len(tasks) != 80 or len(set(tasks)) != 80: raise RuntimeError("boundary-blur task construction failed")
    return tasks


TASKS = build_tasks()


def source_identity() -> dict[str, Any]:
    names = ("audit.py", "config.py", "data.py", "model.py", "experiment.py", "blurry_data.py", "blurry_metrics.py", "blurry_experiment.py", "blurry_run.py", "blurry_analysis.py")
    files = {f"flymodel/main/{name}": file_sha256(PROJECT_ROOT / "flymodel" / "main" / name) for name in names}
    return {"files": files, "sha256": canonical_sha256(files)}


def cell_key(task: Task) -> str: return f"n{task.n}m{task.m}"


def output_path(root: Path, task: Task) -> Path: return root / "runs" / cell_key(task) / f"seed{task.seed}_{task.condition}.json"


def expected_identity(index: int, task: Task) -> dict[str, Any]:
    rates = [1e-2, 1e-3, 1e-4] if task.condition.endswith("_el") else [1e-3]
    return {"task_index": index, "cell": [task.n, task.m], "seed": task.seed, "condition": task.condition, "eta": ETA, "gamma": GAMMA, "rates": rates, "train_per_stage": 10_000, "test_per_stage": 2_000}


def validate_existing(path: Path, index: int, task: Task) -> bool:
    if not path.is_file(): return False
    payload = json.loads(path.read_text(encoding="utf-8")); fingerprint = payload.get("fingerprint", {}); configuration = fingerprint.get("configuration", {})
    return payload.get("status") == "complete" and payload.get("identity") == expected_identity(index, task) and len(payload.get("evaluation", {}).get("records", [])) == 26 and fingerprint.get("sha256") == canonical_sha256(configuration) and configuration.get("source", {}).get("sha256") == source_identity()["sha256"]


def execute(index: int, device: str, root: Path) -> Path:
    task = TASKS[index]; output = output_path(root, task)
    if validate_existing(output, index, task): print(f"SKIP {index:03d} {output}", flush=True); return output
    if output.exists(): raise RuntimeError(f"conflicting existing result: {output}")
    cfg = Config(result_root=root); dataset = build_blurry_subset(task.seed, task.n, task.m, cfg); identity = expected_identity(index, task)
    configuration = {"identity": identity, "protocol": cfg.to_dict(), "class_groups": dataset["class_groups"].tolist(), "data_fingerprint": dataset["metadata"], "blur_stream": dataset["blur_metadata"], "source": source_identity(), "runtime": runtime_identity(device)}
    started = time.time(); payload = train_blurry(dataset, task.condition, ETA, GAMMA, task.seed, cfg, device)
    payload.update({"identity": identity, "class_groups": dataset["class_groups"].tolist(), "fingerprint": {"algorithm": "sha256-canonical-json", "sha256": canonical_sha256(configuration), "configuration": configuration}, "runtime": {**configuration["runtime"], "seconds": time.time() - started}})
    atomic_json(output, payload); print(f"DONE {index:03d} {output}", flush=True); return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--task-index", type=int, choices=range(len(TASKS))); parser.add_argument("--worker", type=int); parser.add_argument("--workers", type=int); parser.add_argument("--device", default="cuda"); parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT); args = parser.parse_args()
    if args.task_index is not None: indices = (args.task_index,)
    else:
        if args.worker is None or args.workers is None or not 0 <= args.worker < args.workers: parser.error("provide --task-index or valid worker partition")
        indices = range(args.worker, len(TASKS), args.workers)
    for index in indices: execute(index, args.device, args.result_root.resolve())


if __name__ == "__main__": main()
