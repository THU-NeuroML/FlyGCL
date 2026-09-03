
import os
import json
import yaml

from omegaconf import DictConfig, OmegaConf
import pdb



def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]


def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_class_names(classes_names, class_ids_per_task):
    return [classes_names[class_id] for class_id in class_ids_per_task]


def get_dataset_class_names(workdir, dataset_name, long=False):
    dataset_key = str(dataset_name)
    # Keep backward compatibility for ImageNet-R naming variants.
    if dataset_key.lower() in {"imagenet_r", "imagenet-r"}:
        dataset_key = "imagenet_R"

    with open(os.path.join(workdir, "dataset_reqs", f"{dataset_key}_classes.txt"), "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    split_path = path.split("/")
    # Backward compatibility for legacy repo layouts.
    for marker in ["GCL_Baseline", "MindtheGap-GCL", "MindtheGap", "SVD_CL_CLIP", "CL_CLIP", "SVD_CL_CLIP_v2", "C_CLIP", "RAPF", "ZSCL"]:
        if marker in split_path:
            workdir_idx = split_path.index(marker)
            return "/".join(split_path[:workdir_idx + 1])

    # Robust fallback for Hydra-chdir runs: walk upward until we find
    # the seq_lora_gcl project root signature.
    cur = os.path.abspath(path)
    while True:
        has_main = os.path.isfile(os.path.join(cur, "main_gcl.py"))
        has_cfg = os.path.isdir(os.path.join(cur, "configs"))
        has_continual = os.path.isdir(os.path.join(cur, "continual_clip"))
        if has_main and has_cfg and has_continual:
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # For multi-seed launcher runs we may execute under a temporary seed dir
    # where Hydra changes cwd into ./experiment/**. Use the exported project
    # root as an explicit fallback when it has the expected repository shape.
    env_project_dir = os.environ.get("MULTI_SEED_PROJECT_DIR", "")
    if env_project_dir:
        env_project_dir = os.path.abspath(env_project_dir)
        has_main = os.path.isfile(os.path.join(env_project_dir, "main_gcl.py"))
        has_cfg = os.path.isdir(os.path.join(env_project_dir, "configs"))
        has_continual = os.path.isdir(os.path.join(env_project_dir, "continual_clip"))
        if has_main and has_cfg and has_continual:
            return env_project_dir

    raise ValueError(f"Cannot resolve project root from path: {path}")


