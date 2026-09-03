#!/usr/bin/env python

import copy
import json
import logging
from pathlib import Path
from typing import Any


def build_multitask_eval_payload(cfg: Any, step: int, step_id: str, eval_infos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_task = {}
    per_task_eval_info = {}
    task_list = list(eval_infos.keys())

    for task, eval_info in eval_infos.items():
        per_task[task] = copy.deepcopy(eval_info.get("aggregated", {}))
        per_task_eval_info[task] = copy.deepcopy(eval_info)

    avg_success = sum(per_task[task]["pc_success"] for task in task_list) / len(task_list)
    avg_sum_reward = sum(per_task[task]["avg_sum_reward"] for task in task_list) / len(task_list)
    total_eval_s = sum(per_task[task]["eval_s"] for task in task_list)

    return {
        "step": step,
        "step_id": step_id,
        "seed": cfg.seed,
        "job_name": cfg.job_name,
        "output_dir": str(cfg.output_dir),
        "tasks": task_list,
        "per_task": per_task,
        "per_task_eval_info": per_task_eval_info,
        "avg_success": avg_success,
        "avg_sum_reward": avg_sum_reward,
        "eval_s": total_eval_s,
    }


def save_multitask_eval_payload(cfg: Any, payload: dict[str, Any]) -> None:
    eval_dir = Path(cfg.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    latest_path = Path(cfg.output_dir) / "multitask_eval_info.json"
    step_path = eval_dir / f"multitask_eval_info_step_{payload['step_id']}.json"

    for path in (latest_path, step_path):
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)

    logging.info(f"Saved structured multitask eval to {latest_path} and {step_path}")
