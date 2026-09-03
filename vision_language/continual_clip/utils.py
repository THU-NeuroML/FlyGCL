
import os
import yaml

from omegaconf import DictConfig, OmegaConf



def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]


def get_class_ids_per_task(args):
    total_classes = len(args.class_order)
    first_task_size = min(args.initial_increment, total_classes)

    if first_task_size <= 0:
        raise ValueError("initial_increment must be > 0")

    # Joint learning compatibility: task_num=1 or increment<=0 means single full task.
    yield args.class_order[:first_task_size]
    if getattr(args, "task_num", None) == 1 or args.increment <= 0 or first_task_size >= total_classes:
        return

    for i in range(first_task_size, total_classes, args.increment):
        yield args.class_order[i:i + args.increment]

def get_class_names(classes_names, class_ids_per_task):
    return [classes_names[class_id] for class_id in class_ids_per_task]


def get_dataset_class_names(workdir, dataset_name, long=False):
    dataset_req_name = {
        "imagenet_r": "imagenet_R",
        "imagenet-r": "imagenet_R",
    }.get(str(dataset_name).lower(), dataset_name)
    with open(os.path.join(workdir, "dataset_reqs", f"{dataset_req_name}_classes.txt"), "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    current = os.path.abspath(path)
    if not os.path.isdir(current):
        current = os.path.dirname(current)

    while True:
        if os.path.isdir(os.path.join(current, "class_orders")) and os.path.isdir(os.path.join(current, "configs")):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(path)
        current = parent

