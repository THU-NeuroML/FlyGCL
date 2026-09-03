#!/usr/bin/env python
import copy
import json
import logging
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import torch
from termcolor import colored

from peft import PeftModel, get_peft_model

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy
from lerobot.scripts.clare import PeftWrapperPolicy
from lerobot.scripts.train_peft import make_merged_policy_for_eval_or_save
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.scripts.multitask_eval_utils import build_multitask_eval_payload, save_multitask_eval_payload
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


STAGE_ADAPTER_RE = re.compile(r"^stage_(\d+)$")


@dataclass
class AnyEvalMultiTaskConfig(EvalPipelineConfig):
    peft_weight_path: Path | None = None
    gcl_force_latest_adapter: bool = True
    gcl_forced_adapter_id: int | None = None
    dataset: DatasetConfig | None = None
    step: int = 0
    step_id: str = "quick"


def parse_task_list(task_spec: str) -> list[str]:
    task_list = [task.strip() for task in task_spec.split(",") if task.strip()]
    if not task_list:
        raise ValueError("env.task must be a comma-separated task list")
    return task_list


def _to_namespace(value: Any):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def resolve_dataset_cfg(cfg: AnyEvalMultiTaskConfig) -> DatasetConfig:
    if cfg.dataset is not None:
        dataset_cfg = cfg.dataset
    else:
        train_config_path = Path(cfg.policy.pretrained_path) / "train_config.json"
        if not train_config_path.is_file():
            raise ValueError("dataset config must be provided, or train_config.json must exist under policy.pretrained_path")
        with open(train_config_path) as handle:
            train_config = json.load(handle)
        dataset_cfg = _to_namespace(train_config["dataset"])
    if getattr(dataset_cfg, "root", None) is None:
        dataset_cfg.root = str(Path("./.cache/huggingface/lerobot") / dataset_cfg.repo_id)
    return dataset_cfg


def get_task_specific_cfg(base_cfg: AnyEvalMultiTaskConfig, task: str) -> AnyEvalMultiTaskConfig:
    task_cfg = copy.deepcopy(base_cfg)
    task_cfg.env.task = task
    task_idx = int(task.rsplit("_", maxsplit=1)[-1])
    if isinstance(task_cfg.env, LiberoEnv) or hasattr(task_cfg.env, "task_id"):
        task_cfg.env.task_id = f"task_{task_idx}"
    task_cfg.job_name = f"{base_cfg.job_name}_{task}"
    task_cfg.output_dir = Path(base_cfg.output_dir) / task
    return task_cfg


def save_eval_info(output_dir: Path, info: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_info.json", "w") as handle:
        json.dump(info, handle, indent=2)


def resolve_stage_adapter_dir(adapter_root: Path, cfg: AnyEvalMultiTaskConfig) -> Path:
    stage_dirs: list[tuple[int, Path]] = []
    for child in adapter_root.iterdir():
        if not child.is_dir():
            continue
        match = STAGE_ADAPTER_RE.match(child.name)
        if match is None:
            continue
        if (child / "adapter_config.json").is_file():
            stage_dirs.append((int(match.group(1)), child))
    if not stage_dirs:
        return adapter_root

    if cfg.gcl_forced_adapter_id is not None:
        if cfg.gcl_forced_adapter_id == 0 and (adapter_root / "adapter_config.json").is_file():
            logging.info("Using explicitly requested standard PEFT root adapter stage_0: %s", adapter_root)
            return adapter_root
        for stage_id, stage_dir in stage_dirs:
            if stage_id == cfg.gcl_forced_adapter_id:
                logging.info("Using explicitly requested standard PEFT stage adapter: %s", stage_dir)
                return stage_dir
        raise ValueError(f"gcl_forced_adapter_id={cfg.gcl_forced_adapter_id} not found under {adapter_root}")

    if cfg.gcl_force_latest_adapter:
        stage_id, stage_dir = max(stage_dirs, key=lambda item: item[0])
        logging.info("Using latest standard PEFT stage adapter stage_%s from %s", stage_id, adapter_root)
        return stage_dir

    return adapter_root


def resolve_peft_weight_path(cfg: AnyEvalMultiTaskConfig) -> Path | None:
    if cfg.peft_weight_path:
        peft_weight_path = str(cfg.peft_weight_path).strip().lower()
        if peft_weight_path in {"none", "null", "false", "off", "disable", "disabled"}:
            return None
        return Path(cfg.peft_weight_path)

    pretrained_path = Path(cfg.policy.pretrained_path)
    candidate = pretrained_path.parent / "adapter"
    if candidate.is_dir():
        return resolve_stage_adapter_dir(candidate, cfg)

    train_config_path = pretrained_path / "train_config.json"
    if train_config_path.is_file():
        with open(train_config_path) as handle:
            train_config = json.load(handle)
        peft_weight_path = train_config.get("peft_weight_path")
        if peft_weight_path and Path(peft_weight_path).is_dir():
            return Path(peft_weight_path)
    return None


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def _resolve_baseline_peft_cfg_path(policy_path: Path) -> Path | None:
    train_cfg = _load_json_if_exists(policy_path / "train_config.json")
    raw_path = train_cfg.get("peft_cfg_path")
    if not raw_path:
        return None
    cfg_path = Path(raw_path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    return cfg_path if cfg_path.exists() else None


def _try_load_baseline_full_checkpoint(policy, policy_path: Path, device: torch.device):
    peft_cfg_path = _resolve_baseline_peft_cfg_path(policy_path)
    model_path = policy_path / "model.safetensors"
    if peft_cfg_path is None or not model_path.exists():
        return policy, None, None

    from peft import PeftConfig
    from safetensors.torch import load_file as safe_load_file

    logging.info("Rebuilding baseline PEFT wrapper from %s and loading full checkpoint %s", peft_cfg_path, model_path)
    peft_config = PeftConfig.from_pretrained(str(peft_cfg_path))
    peft_policy = get_peft_model(PeftWrapperPolicy(policy=policy), peft_config, autocast_adapter_dtype=False)
    state = safe_load_file(str(model_path), device=str(device))
    missing, unexpected = peft_policy.load_state_dict(state, strict=False)
    logging.info("Baseline full checkpoint raw load_state_dict: missing=%s unexpected=%s", len(missing), len(unexpected))

    if len(unexpected) > 100 and any(not key.startswith("base_model.") for key in state):
        prefixed_state = {f"base_model.model.policy.{key}": value for key, value in state.items()}
        missing_prefixed, unexpected_prefixed = peft_policy.load_state_dict(prefixed_state, strict=False)
        logging.info(
            "Baseline full checkpoint prefixed load_state_dict: missing=%s unexpected=%s",
            len(missing_prefixed),
            len(unexpected_prefixed),
        )
        if len(unexpected_prefixed) < len(unexpected):
            missing, unexpected = missing_prefixed, unexpected_prefixed

    if missing:
        logging.info("First missing keys: %s", list(missing)[:20])
    if unexpected:
        logging.info("First unexpected keys: %s", list(unexpected)[:20])

    peft_modules = list(getattr(peft_policy.base_model, "adapter_layers", []))
    if peft_modules:
        latest_adapter_id = max(peft_module.num_adapters for peft_module in peft_modules) - 1
        for peft_module in peft_modules:
            peft_module._forwarded_adapter_id = min(latest_adapter_id, peft_module.num_adapters - 1)
            if hasattr(peft_module, "_forwarded_discriminator_id"):
                peft_module._forwarded_discriminator_id = -1
        logging.info("Baseline custom adapter layers detected; forcing latest adapter id %s", latest_adapter_id)
        return policy, peft_modules, latest_adapter_id

    merged_policy = make_merged_policy_for_eval_or_save(peft_policy)
    merged_policy.to(device)
    logging.info("Baseline standard PEFT checkpoint merged for evaluation")
    return merged_policy, None, None


def maybe_force_gcl_adapter(peft_modules, adapter_id: int | None):
    if peft_modules is None or adapter_id is None:
        return None
    original_adapter_ids = []
    for peft_module in peft_modules:
        if not hasattr(peft_module, "_forwarded_adapter_id") or not hasattr(peft_module, "num_adapters"):
            return None
        original_adapter_ids.append(peft_module._forwarded_adapter_id)
        peft_module._forwarded_adapter_id = min(adapter_id, peft_module.num_adapters - 1)
    return original_adapter_ids


def restore_gcl_adapter(peft_modules, original_adapter_ids) -> None:
    if peft_modules is None or original_adapter_ids is None:
        return
    for peft_module, adapter_id in zip(peft_modules, original_adapter_ids):
        peft_module._forwarded_adapter_id = adapter_id


@parser.wrap()
def eval_multitask_main(cfg: AnyEvalMultiTaskConfig):
    if not cfg.env.task:
        raise ValueError("env.task must be provided")
    logging.info(pformat(asdict(cfg)))
    task_list = parse_task_list(cfg.env.task)
    dataset_cfg = resolve_dataset_cfg(cfg)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    ds_meta = LeRobotDatasetMetadata(dataset_cfg.repo_id, root=dataset_cfg.root, revision=dataset_cfg.revision)

    logging.info("Making policy")
    if cfg.policy.pretrained_path:
        logging.info("Using checkpoint-saved normalization stats; env config only supplies feature shapes")
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    else:
        policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    peft_weight_path = resolve_peft_weight_path(cfg)
    peft_policy = None
    peft_modules = None
    latest_adapter_id = None
    policy_path_obj = Path(cfg.policy.pretrained_path) if cfg.policy.pretrained_path is not None else None
    if peft_weight_path is not None and peft_weight_path.is_dir():
        logging.info("Wrapping policy with PEFT adapter from %s", peft_weight_path)
        peft_policy = PeftModel.from_pretrained(
            PeftWrapperPolicy(policy=policy),
            peft_weight_path,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
        inner_policy = getattr(getattr(peft_policy, "base_model", None), "model", None)
        inner_policy = getattr(inner_policy, "policy", None)
        if inner_policy is None:
            raise ValueError("Could not locate inner PreTrainedPolicy inside PEFT wrapper")
        policy = inner_policy
        peft_modules = list(getattr(peft_policy.base_model, "adapter_layers", []))
        if peft_modules:
            latest_adapter_id = max(peft_module.num_adapters for peft_module in peft_modules) - 1
            forced_adapter_id = cfg.gcl_forced_adapter_id if cfg.gcl_forced_adapter_id is not None else latest_adapter_id
            if forced_adapter_id < 0 or forced_adapter_id > latest_adapter_id:
                raise ValueError(f"gcl_forced_adapter_id={forced_adapter_id} outside [0, {latest_adapter_id}]")
            logging.info(
                "Loaded PEFT adapter layers; latest adapter id %s; forced adapter id %s during GCL eval=%s",
                latest_adapter_id,
                forced_adapter_id,
                cfg.gcl_force_latest_adapter,
            )
        logging.info("Evaluating inner PEFT-wrapped policy object: %s", type(policy))
    else:
        logging.info("Evaluating plain policy checkpoint without PEFT adapter")

    policy.eval()
    eval_infos: dict[str, dict[str, Any]] = {}
    autocast_ctx = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    with torch.no_grad(), autocast_ctx:
        for task in task_list:
            logging.info("========== Evaluating task: %s ==========", task)
            task_cfg = get_task_specific_cfg(cfg, task)
            adapter_id_to_force = None
            if cfg.gcl_force_latest_adapter:
                adapter_id_to_force = cfg.gcl_forced_adapter_id if cfg.gcl_forced_adapter_id is not None else latest_adapter_id
            original_adapter_ids = maybe_force_gcl_adapter(peft_modules, adapter_id_to_force)
            if original_adapter_ids is not None:
                logging.info("Task %s: using forced GCL adapter_id=%s", task, adapter_id_to_force)
            try:
                info = eval_policy_with_env_init(
                    env_cfg=task_cfg.env,
                    n_envs=task_cfg.eval.batch_size,
                    use_async_envs=task_cfg.eval.use_async_envs,
                    policy=policy,
                    n_episodes=task_cfg.eval.n_episodes,
                    max_episodes_rendered=0,
                    videos_dir=Path(task_cfg.output_dir) / "videos",
                    start_seed=task_cfg.seed,
                )
            finally:
                restore_gcl_adapter(peft_modules, original_adapter_ids)
            save_eval_info(Path(task_cfg.output_dir), info)
            eval_infos[task] = info
            logging.info("%s", {"task": task, **info["aggregated"]})

    payload = build_multitask_eval_payload(cfg=cfg, step=cfg.step, step_id=cfg.step_id, eval_infos=eval_infos)
    save_multitask_eval_payload(cfg, payload)
    print("========== Multi-task Evaluation Summary ==========")
    for task in task_list:
        metric = payload["per_task"][task]
        print(f"{task}: success = {metric['pc_success']:.2f}%, avg_reward = {metric['avg_sum_reward']:.2f}")
    print(f"Average Success Rate across {len(task_list)} tasks: {payload['avg_success']:.2f}%")


if __name__ == "__main__":
    init_logging()
    eval_multitask_main()
