#!/usr/bin/env python

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


RUN_DIR_RE = re.compile(
    r"^(?P<prefix>.+?)_seed_(?P<seed>\d+)_(?P<benchmark>.+?)_task_(?P<stage>\d+)(?P<suffix>.*)$"
)
TASK_NAME_RE = re.compile(r"Libero_[A-Za-z0-9]+_Task_(\d+)$")
WANDB_SUCCESS_RE = re.compile(r"^eval/pc_success_(Libero_[A-Za-z0-9]+_Task_(\d+))$")
WANDB_REWARD_RE = re.compile(r"^eval/avg_sum_reward_(Libero_[A-Za-z0-9]+_Task_(\d+))$")

SOURCE_PRIORITY = {
    "multitask_eval_info": 0,
    "eval_info_tree": 1,
    "wandb_summary": 2,
    "unknown": 99,
}


@dataclass(frozen=True)
class RunDescriptor:
    experiment_id: str
    seed: int
    stage: int
    benchmark: str
    run_dir: Path


@dataclass
class StageRecord:
    descriptor: RunDescriptor
    source: str
    success_by_task: dict[int, float] = field(default_factory=dict)
    reward_by_task: dict[int, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    source_path: Path | None = None
    mtime: float = 0.0


@dataclass
class SeedMetricSummary:
    seed: int
    auc: float | None
    fwt: float | None
    nbt: float | None
    faa: float | None
    issues: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize continual-learning metrics from eval JSONs or legacy wandb summaries."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Root directory to scan. Can be provided multiple times. Defaults to ./outputs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of seeds to include. Defaults to all discovered seeds.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks expected in the continual-learning benchmark.",
    )
    parser.add_argument(
        "--job-prefix",
        default="",
        help="Optional prefix filter for experiment ids. Leave empty to auto-discover all experiment groups.",
    )
    parser.add_argument(
        "--clamp-positive-nbt",
        action="store_true",
        help="Clamp negative forgetting to zero when computing NBT.",
    )
    parser.add_argument(
        "--csv-out",
        default="./outputs/cl_metric_summary.csv",
        help="CSV output path for the summary table. Use an empty string to disable CSV output.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def maybe_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_run_descriptor(run_dir: Path) -> RunDescriptor | None:
    match = RUN_DIR_RE.match(run_dir.name)
    if match is None:
        return None

    experiment_id = f"{match.group('prefix')}_{match.group('benchmark')}{match.group('suffix')}"
    return RunDescriptor(
        experiment_id=experiment_id,
        seed=int(match.group("seed")),
        stage=int(match.group("stage")),
        benchmark=match.group("benchmark"),
        run_dir=run_dir,
    )


def nearest_run_dir(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in [current, *current.parents]:
        if RUN_DIR_RE.match(candidate.name):
            return candidate
    return None


def choose_best_path(paths: list[Path], run_dir: Path) -> Path:
    return min(paths, key=lambda path: (len(path.relative_to(run_dir).parts), str(path)))


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def extract_from_multitask_json(path: Path, descriptor: RunDescriptor) -> StageRecord:
    data = load_json(path)
    results = data.get("per_task", {})
    record = StageRecord(
        descriptor=descriptor,
        source="multitask_eval_info",
        source_path=path,
        mtime=path.stat().st_mtime,
    )

    for task_name, task_metrics in results.items():
        task_match = TASK_NAME_RE.match(task_name)
        if task_match is None:
            continue
        task_idx = int(task_match.group(1))
        success = maybe_number(task_metrics.get("pc_success"))
        reward = maybe_number(task_metrics.get("avg_sum_reward"))
        if success is not None:
            record.success_by_task[task_idx] = success
        if reward is not None:
            record.reward_by_task[task_idx] = reward

    return record


def extract_from_eval_tree(paths: list[Path], descriptor: RunDescriptor) -> StageRecord:
    record = StageRecord(
        descriptor=descriptor,
        source="eval_info_tree",
        source_path=choose_best_path(paths, descriptor.run_dir),
        mtime=max(path.stat().st_mtime for path in paths),
    )

    for path in paths:
        task_match = TASK_NAME_RE.match(path.parent.name)
        if task_match is None:
            continue
        task_idx = int(task_match.group(1))
        data = load_json(path)
        aggregated = data.get("aggregated", {})
        success = maybe_number(aggregated.get("pc_success"))
        reward = maybe_number(aggregated.get("avg_sum_reward"))
        if success is not None:
            record.success_by_task[task_idx] = success
        if reward is not None:
            record.reward_by_task[task_idx] = reward

    return record


def extract_from_wandb_summary(paths: list[Path], descriptor: RunDescriptor) -> StageRecord:
    path = max(paths, key=lambda candidate: candidate.stat().st_mtime)
    data = load_json(path)
    record = StageRecord(
        descriptor=descriptor,
        source="wandb_summary",
        source_path=path,
        mtime=path.stat().st_mtime,
    )

    for key, value in data.items():
        success_match = WANDB_SUCCESS_RE.match(key)
        if success_match is not None:
            task_idx = int(success_match.group(2))
            parsed_value = maybe_number(value)
            if parsed_value is not None:
                record.success_by_task[task_idx] = parsed_value
            continue

        reward_match = WANDB_REWARD_RE.match(key)
        if reward_match is not None:
            task_idx = int(reward_match.group(2))
            parsed_value = maybe_number(value)
            if parsed_value is not None:
                record.reward_by_task[task_idx] = parsed_value

    return record


def collect_candidate_files(root: Path) -> dict[Path, dict[str, list[Path]]]:
    candidates: dict[Path, dict[str, list[Path]]] = {}
    patterns = ("multitask_eval_info.json", "eval_info.json", "wandb-summary.json")
    for pattern in patterns:
        for path in root.rglob(pattern):
            run_dir = nearest_run_dir(path)
            if run_dir is None:
                continue
            bucket = candidates.setdefault(
                run_dir,
                {"multitask": [], "eval": [], "wandb": []},
            )
            if pattern == "multitask_eval_info.json":
                bucket["multitask"].append(path)
            elif pattern == "eval_info.json":
                bucket["eval"].append(path)
            else:
                bucket["wandb"].append(path)
    return candidates


def stage_record_from_candidates(run_dir: Path, file_groups: dict[str, list[Path]]) -> StageRecord | None:
    descriptor = parse_run_descriptor(run_dir)
    if descriptor is None:
        return None

    if file_groups["multitask"]:
        return extract_from_multitask_json(choose_best_path(file_groups["multitask"], run_dir), descriptor)

    eval_tree_paths = [path for path in file_groups["eval"] if TASK_NAME_RE.match(path.parent.name)]
    if eval_tree_paths:
        return extract_from_eval_tree(eval_tree_paths, descriptor)

    if file_groups["wandb"]:
        return extract_from_wandb_summary(file_groups["wandb"], descriptor)

    return None


def add_quality_checks(record: StageRecord, num_tasks: int) -> None:
    expected_tasks = set(range(record.descriptor.stage + 1))
    success_tasks = set(record.success_by_task.keys())
    reward_tasks = set(record.reward_by_task.keys())

    missing_success = sorted(expected_tasks - success_tasks)
    missing_reward = sorted(expected_tasks - reward_tasks)
    if missing_success:
        record.issues.append(f"missing_success_tasks={missing_success}")
    if missing_reward:
        record.issues.append(f"missing_reward_tasks={missing_reward}")

    if record.success_by_task:
        success_values = [record.success_by_task.get(task_idx, 0.0) for task_idx in sorted(success_tasks)]
        reward_values = [record.reward_by_task.get(task_idx, 0.0) for task_idx in sorted(reward_tasks)]
        if all(value == 0.0 for value in success_values) and any(value > 0.0 for value in reward_values):
            record.issues.append("suspicious_all_zero_success_with_nonzero_reward")

    if record.descriptor.stage >= num_tasks:
        record.issues.append(f"stage_{record.descriptor.stage}_out_of_range")


def is_new_record_better(existing: StageRecord, new_record: StageRecord) -> bool:
    existing_priority = SOURCE_PRIORITY.get(existing.source, SOURCE_PRIORITY["unknown"])
    new_priority = SOURCE_PRIORITY.get(new_record.source, SOURCE_PRIORITY["unknown"])
    if new_priority != existing_priority:
        return new_priority < existing_priority
    return new_record.mtime > existing.mtime


def build_metric_matrix(
    stage_records: dict[int, StageRecord],
    metric_name: str,
    num_tasks: int,
) -> list[list[float | None]]:
    matrix: list[list[float | None]] = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]
    for stage, record in stage_records.items():
        values = record.success_by_task if metric_name == "success" else record.reward_by_task
        for task_idx, value in values.items():
            if 0 <= task_idx < num_tasks and 0 <= stage < num_tasks:
                matrix[task_idx][stage] = value
    return matrix


def compute_cl_metrics(
    matrix: list[list[float | None]],
    clamp_positive_nbt: bool,
) -> tuple[float | None, float | None, float | None, float | None, list[str]]:
    num_tasks = len(matrix)
    issues: list[str] = []

    final_stage = num_tasks - 1
    faa_missing_tasks = [task_idx for task_idx in range(num_tasks) if matrix[task_idx][final_stage] is None]
    faa = None
    if faa_missing_tasks:
        issues.append(f"faa_missing_tasks={faa_missing_tasks}")
    else:
        faa = mean(
            matrix[task_idx][final_stage]
            for task_idx in range(num_tasks)
            if matrix[task_idx][final_stage] is not None
        )

    for task_idx in range(num_tasks):
        suffix = matrix[task_idx][task_idx:]
        missing_positions = [task_idx + offset for offset, value in enumerate(suffix) if value is None]
        if missing_positions:
            issues.append(f"task_{task_idx}_missing_stages={missing_positions}")

    if issues:
        return None, None, None, faa, issues

    fwt = mean(matrix[idx][idx] for idx in range(num_tasks) if matrix[idx][idx] is not None)

    auc_values = []
    for task_idx in range(num_tasks):
        row = [value for value in matrix[task_idx][task_idx:] if value is not None]
        auc_values.append(mean(row))
    auc = mean(auc_values)

    nbt_values = []
    for task_idx in range(num_tasks - 1):
        diagonal = matrix[task_idx][task_idx]
        assert diagonal is not None
        row_diffs = []
        for stage in range(task_idx + 1, num_tasks):
            current = matrix[task_idx][stage]
            assert current is not None
            diff = diagonal - current
            if clamp_positive_nbt and diff < 0:
                diff = 0.0
            row_diffs.append(diff)
        if row_diffs:
            nbt_values.append(mean(row_diffs))
    nbt = mean(nbt_values) if nbt_values else 0.0

    return auc, fwt, nbt, faa, issues


def summarize_seed_metric(
    seed: int,
    stage_records: dict[int, StageRecord],
    metric_name: str,
    num_tasks: int,
    clamp_positive_nbt: bool,
) -> SeedMetricSummary:
    issues: list[str] = []
    missing_stages = [stage for stage in range(num_tasks) if stage not in stage_records]
    if missing_stages:
        issues.append(f"missing_stage_runs={missing_stages}")

    matrix = build_metric_matrix(stage_records, metric_name, num_tasks)
    auc, fwt, nbt, faa, metric_issues = compute_cl_metrics(matrix, clamp_positive_nbt)
    issues.extend(metric_issues)

    for stage in sorted(stage_records):
        for issue in stage_records[stage].issues:
            issues.append(f"stage_{stage}:{issue}")

    return SeedMetricSummary(seed=seed, auc=auc, fwt=fwt, nbt=nbt, faa=faa, issues=issues)


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "metric",
        "seed_count",
        "seed_list",
        "auc_mean",
        "auc_std",
        "fwt_mean",
        "fwt_std",
        "nbt_mean",
        "nbt_std",
        "faa_mean",
        "faa_std",
        "auc_values",
        "fwt_values",
        "nbt_values",
        "faa_values",
        "issues",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    roots = [Path(root) for root in (args.root or ["./outputs"])]
    roots = [root.resolve() for root in roots if root]

    stage_records_by_experiment: dict[str, dict[int, dict[int, StageRecord]]] = {}

    for root in roots:
        if not root.exists():
            continue
        candidate_files = collect_candidate_files(root)
        for run_dir, file_groups in candidate_files.items():
            descriptor = parse_run_descriptor(run_dir)
            if descriptor is None:
                continue
            if args.job_prefix and not descriptor.experiment_id.startswith(args.job_prefix):
                continue

            record = stage_record_from_candidates(run_dir, file_groups)
            if record is None:
                continue

            add_quality_checks(record, args.num_tasks)

            experiment_bucket = stage_records_by_experiment.setdefault(record.descriptor.experiment_id, {})
            seed_bucket = experiment_bucket.setdefault(record.descriptor.seed, {})
            existing = seed_bucket.get(record.descriptor.stage)
            if existing is None or is_new_record_better(existing, record):
                seed_bucket[record.descriptor.stage] = record

    if not stage_records_by_experiment:
        print("No matching continual-learning runs were found.")
        return

    csv_rows: list[dict[str, str]] = []

    for experiment_id in sorted(stage_records_by_experiment):
        seed_records = stage_records_by_experiment[experiment_id]
        discovered_seeds = sorted(seed_records)
        target_seeds = args.seeds if args.seeds else discovered_seeds

        print(f"\n## {experiment_id}")
        print(f"Discovered seeds: {', '.join(str(seed) for seed in discovered_seeds)}")

        for metric_name in ("success", "reward"):
            seed_summaries: list[SeedMetricSummary] = []
            experiment_issues: list[str] = []

            for seed in target_seeds:
                stage_records = seed_records.get(seed)
                if stage_records is None:
                    experiment_issues.append(f"missing_seed={seed}")
                    continue
                seed_summary = summarize_seed_metric(
                    seed=seed,
                    stage_records=stage_records,
                    metric_name=metric_name,
                    num_tasks=args.num_tasks,
                    clamp_positive_nbt=args.clamp_positive_nbt,
                )
                seed_summaries.append(seed_summary)

            valid_auc = [summary.auc for summary in seed_summaries if summary.auc is not None]
            valid_fwt = [summary.fwt for summary in seed_summaries if summary.fwt is not None]
            valid_nbt = [summary.nbt for summary in seed_summaries if summary.nbt is not None]
            valid_faa = [summary.faa for summary in seed_summaries if summary.faa is not None]

            print(
                f"{metric_name:>7}: "
                f"seed_count={len(seed_summaries)}/{len(target_seeds)}, "
                f"AUC={format_float(mean(valid_auc) if valid_auc else None)} +/- {format_float(safe_std(valid_auc) if valid_auc else None)}, "
                f"FWT={format_float(mean(valid_fwt) if valid_fwt else None)} +/- {format_float(safe_std(valid_fwt) if valid_fwt else None)}, "
                f"NBT={format_float(mean(valid_nbt) if valid_nbt else None)} +/- {format_float(safe_std(valid_nbt) if valid_nbt else None)}, "
                f"FAA={format_float(mean(valid_faa) if valid_faa else None)} +/- {format_float(safe_std(valid_faa) if valid_faa else None)}"
            )

            for summary in seed_summaries:
                issue_suffix = f" issues={summary.issues}" if summary.issues else ""
                print(
                    f"         seed={summary.seed} "
                    f"AUC={format_float(summary.auc)} "
                    f"FWT={format_float(summary.fwt)} "
                    f"NBT={format_float(summary.nbt)} "
                    f"FAA={format_float(summary.faa)}"
                    f"{issue_suffix}"
                )

            if experiment_issues:
                print(f"         aggregate_issues={experiment_issues}")

            csv_rows.append(
                {
                    "experiment": experiment_id,
                    "metric": metric_name,
                    "seed_count": str(len(seed_summaries)),
                    "seed_list": ",".join(str(summary.seed) for summary in seed_summaries),
                    "auc_mean": format_float(mean(valid_auc) if valid_auc else None),
                    "auc_std": format_float(safe_std(valid_auc) if valid_auc else None),
                    "fwt_mean": format_float(mean(valid_fwt) if valid_fwt else None),
                    "fwt_std": format_float(safe_std(valid_fwt) if valid_fwt else None),
                    "nbt_mean": format_float(mean(valid_nbt) if valid_nbt else None),
                    "nbt_std": format_float(safe_std(valid_nbt) if valid_nbt else None),
                    "faa_mean": format_float(mean(valid_faa) if valid_faa else None),
                    "faa_std": format_float(safe_std(valid_faa) if valid_faa else None),
                    "auc_values": ",".join(format_float(summary.auc) for summary in seed_summaries),
                    "fwt_values": ",".join(format_float(summary.fwt) for summary in seed_summaries),
                    "nbt_values": ",".join(format_float(summary.nbt) for summary in seed_summaries),
                    "faa_values": ",".join(format_float(summary.faa) for summary in seed_summaries),
                    "issues": " | ".join(
                        experiment_issues
                        + [f"seed_{summary.seed}:{';'.join(summary.issues)}" for summary in seed_summaries if summary.issues]
                    ),
                }
            )

    if args.csv_out:
        output_path = Path(args.csv_out)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        write_csv(csv_rows, output_path)
        print(f"\nCSV summary written to: {output_path}")


if __name__ == "__main__":
    main()
