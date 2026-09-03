#!/usr/bin/env python

import copy
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from termcolor import colored

from peft import PeftModel

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy
from lerobot.scripts.clare import PeftWrapperPolicy, build_multitask_eval_payload, save_multitask_eval_payload
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class PEFTEvalMultiTaskConfig(EvalPipelineConfig):
    peft_weight_path: Path | None = None
    dataset: DatasetConfig | None = None
    step: int = 9
    step_id: str = "task_9"


def get_task_specific_cfg(base_cfg: PEFTEvalMultiTaskConfig, task: str) -> PEFTEvalMultiTaskConfig:
    task_cfg = copy.deepcopy(base_cfg)
    task_cfg.env.task = task
    if isinstance(task_cfg.env, LiberoEnv):
        task_idx = int(task.rsplit("_", maxsplit=1)[-1])
        task_cfg.env.task_id = f"task_{task_idx}"
    task_cfg.job_name = f"{base_cfg.job_name}_{task}"
    task_cfg.output_dir = Path(base_cfg.output_dir) / task
    return task_cfg


def parse_task_list(task_spec: str) -> list[str]:
    task_list = [task.strip() for task in task_spec.split(",") if task.strip()]
    if not task_list:
        raise ValueError(
            'env.task must be provided as a comma-separated string, e.g., "Libero_10_Task_0,Libero_10_Task_1"'
        )
    return task_list


def _to_namespace(value: Any):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def resolve_dataset_cfg(cfg: PEFTEvalMultiTaskConfig) -> DatasetConfig:
    if cfg.dataset is not None:
        dataset_cfg = cfg.dataset
    else:
        train_config_path = Path(cfg.policy.pretrained_path) / "train_config.json"
        if not train_config_path.is_file():
            raise ValueError(
                "dataset config must be provided, or train_config.json must exist under policy.pretrained_path"
            )
        with open(train_config_path) as handle:
            train_config = json.load(handle)
        dataset_cfg = _to_namespace(train_config["dataset"])

    if getattr(dataset_cfg, "root", None) is None:
        dataset_cfg.root = str(Path("./.cache/huggingface/lerobot") / dataset_cfg.repo_id)
    return dataset_cfg


PEFT_WRAPPED_POLICY_KEY_HINTS = (
    ".base_layer.",
    ".clare_func_adapters.",
    ".clare_residual_head_ema_adapters.",
    ".clare_residual_ensemble_gates.",
    ".clare_discriminators.",
    ".rp_head.",
    ".ws_head.",
)


def checkpoint_contains_wrapped_peft_policy(pretrained_dir: Path | str | None) -> bool:
    if pretrained_dir is None:
        return False
    model_file = Path(pretrained_dir) / "model.safetensors"
    if not model_file.is_file():
        return False

    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if any(token in key for token in PEFT_WRAPPED_POLICY_KEY_HINTS):
                return True
    return False


def make_policy_for_peft_eval(cfg, ds_meta) -> tuple[torch.nn.Module, bool]:
    restore_wrapped_state = checkpoint_contains_wrapped_peft_policy(getattr(cfg, "pretrained_path", None))
    policy_cfg = copy.deepcopy(cfg)
    if restore_wrapped_state:
        logging.warning(
            "Detected a PEFT-wrapped policy export under %s. "
            "This checkpoint is not a clean base policy; rebuilding the adapter structure first and then restoring "
            "the full wrapped policy state for offline evaluation.",
            cfg.pretrained_path,
        )
        policy_cfg.pretrained_path = None

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    return policy, restore_wrapped_state


def restore_wrapped_peft_policy_state(policy: torch.nn.Module, pretrained_dir: Path | str) -> None:
    model_file = Path(pretrained_dir) / "model.safetensors"
    wrapped_state = safe_load_file(str(model_file))
    load_result = policy.load_state_dict(wrapped_state, strict=False)

    missing_wrapped_keys = [
        key for key in load_result.missing_keys 
        if any(token in key for token in PEFT_WRAPPED_POLICY_KEY_HINTS)
        and "task_id" not in key  # task_id is a buffer, not saved in safetensors
    ]
    unexpected_wrapped_keys = [
        key for key in load_result.unexpected_keys if any(token in key for token in PEFT_WRAPPED_POLICY_KEY_HINTS)
    ]
    if missing_wrapped_keys or unexpected_wrapped_keys:
        raise ValueError(
            "Failed to restore wrapped PEFT policy state cleanly. "
            f"Missing wrapped keys: {missing_wrapped_keys}. "
            f"Unexpected wrapped keys: {unexpected_wrapped_keys}."
        )

    logging.info("Restored full wrapped PEFT policy state from %s", model_file)



def save_eval_info(output_dir: Path, info: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_info.json", "w") as handle:
        json.dump(info, handle, indent=2)


def load_eval_info(output_dir: Path) -> dict[str, Any] | None:
    eval_info_path = output_dir / "eval_info.json"
    if not eval_info_path.is_file():
        return None
    with open(eval_info_path) as handle:
        return json.load(handle)


def get_cached_eval_mismatch_reason(
    info: dict[str, Any],
    n_episodes: int,
    start_seed: int | None,
) -> str | None:
    per_episode = info.get("per_episode")
    aggregated = info.get("aggregated")
    if not isinstance(per_episode, list):
        return "missing per_episode list"
    if len(per_episode) != n_episodes:
        return f"episode count mismatch: expected {n_episodes}, found {len(per_episode)}"
    if not isinstance(aggregated, dict):
        return "missing aggregated metrics"

    actual_indices = [episode.get("episode_ix") for episode in per_episode if isinstance(episode, dict)]
    if actual_indices != list(range(n_episodes)):
        return "episode_ix sequence mismatch"

    expected_seeds = [None] * n_episodes if start_seed is None else list(range(start_seed, start_seed + n_episodes))
    actual_seeds = [episode.get("seed") for episode in per_episode if isinstance(episode, dict)]
    if actual_seeds != expected_seeds:
        return "seed sequence mismatch"

    required_metric_keys = {"avg_sum_reward", "avg_max_reward", "pc_success", "eval_s", "eval_ep_s"}
    missing_keys = sorted(required_metric_keys.difference(aggregated))
    if missing_keys:
        return f"missing aggregated keys: {missing_keys}"

    return None


@parser.wrap()
def eval_multitask_main(cfg: PEFTEvalMultiTaskConfig):
    if not cfg.env.task:
        raise ValueError(
            'env.task must be provided as a comma-separated string, e.g., "Libero_10_Task_0,Libero_10_Task_1"'
        )
    if cfg.peft_weight_path is None:
        raise ValueError("peft_weight_path must be provided")

    logging.info(pformat(asdict(cfg)))
    task_list = parse_task_list(cfg.env.task)
    dataset_cfg = resolve_dataset_cfg(cfg)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    ds_meta = LeRobotDatasetMetadata(
        dataset_cfg.repo_id, root=dataset_cfg.root, revision=dataset_cfg.revision
    )
    logging.info("Making policy.")
    policy, restore_wrapped_state = make_policy_for_peft_eval(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )

    logging.info("Wrapping policy with peft module")
    peft_wrapper_policy = PeftWrapperPolicy(policy=policy)
    peft_policy = PeftModel.from_pretrained(
        peft_wrapper_policy,
        cfg.peft_weight_path,
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    if restore_wrapped_state:
        restore_wrapped_peft_policy_state(policy, cfg.policy.pretrained_path)

    policy.eval()
    eval_infos: dict[str, dict[str, Any]] = {}

    autocast_ctx = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    with torch.no_grad(), autocast_ctx:
        for task in task_list:
            logging.info(f"\n========== Evaluating task: {task} ==========")
            task_cfg = get_task_specific_cfg(cfg, task)
            task_output_dir = Path(task_cfg.output_dir)
            cached_info = load_eval_info(task_output_dir)
            mismatch_reason = None
            if cached_info is not None:
                mismatch_reason = get_cached_eval_mismatch_reason(
                    cached_info,
                    n_episodes=task_cfg.eval.n_episodes,
                    start_seed=task_cfg.seed,
                )

            if cached_info is not None and mismatch_reason is None:
                logging.info("Reusing cached eval for %s from %s", task, task_output_dir / "eval_info.json")
                info = cached_info
            else:
                if mismatch_reason is not None:
                    logging.warning(
                        "Ignoring cached eval for %s from %s: %s",
                        task,
                        task_output_dir / "eval_info.json",
                        mismatch_reason,
                    )
                info = eval_policy_with_env_init(
                    env_cfg=task_cfg.env,
                    n_envs=task_cfg.eval.batch_size,
                    use_async_envs=task_cfg.eval.use_async_envs,
                    policy=policy,
                    n_episodes=task_cfg.eval.n_episodes,
                    max_episodes_rendered=0,  # disabled for headless server
                    videos_dir=task_output_dir / "videos",
                    start_seed=task_cfg.seed,
                )
                save_eval_info(task_output_dir, info)

            eval_infos[task] = info
            logging.info(
                "%s",
                {
                    "task": task,
                    **info["aggregated"],
                },
            )

    payload = build_multitask_eval_payload(cfg=cfg, step=cfg.step, step_id=cfg.step_id, eval_infos=eval_infos)
    save_multitask_eval_payload(cfg, payload)

    print("\n========== Multi-task Evaluation Summary ==========")
    for task in task_list:
        metric = payload["per_task"][task]
        print(f"{task}: success = {metric['pc_success']:.2f}%, avg_reward = {metric['avg_sum_reward']:.2f}")
    print(f"Average Success Rate across {len(task_list)} tasks: {payload['avg_success']:.2f}%")


if __name__ == "__main__":
    init_logging()
    eval_multitask_main()
