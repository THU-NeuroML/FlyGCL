

import os
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50, CUB200, Food101,OxfordPet,Caltech101
)
from PIL import Image
from .utils import get_dataset_class_names, get_workdir


class GCLDatasetWrapper(Dataset):
    """Wrap a dataset into an OnlineSampler-compatible format for GCL."""

    def __init__(self, continuum_dataset, class_names, transform=None):
        self.continuum_dataset = continuum_dataset
        self.class_names = class_names
        self.transform = transform
        self._image_paths = None

        # Case 1: regular torch dataset with targets
        if isinstance(continuum_dataset, Dataset) and hasattr(continuum_dataset, 'targets') \
                and not hasattr(continuum_dataset, 'get_data'):
            self.targets = continuum_dataset.targets if isinstance(continuum_dataset.targets, list) \
                else list(continuum_dataset.targets)

        # Case 2: continuum wrappers over torchvision datasets
        elif hasattr(continuum_dataset, 'dataset') and hasattr(continuum_dataset.dataset, 'targets'):
            base = continuum_dataset.dataset
            self.targets = base.targets if isinstance(base.targets, list) else list(base.targets)

        # Case 3: ImageFolder-like / IMAGE_PATH continuum datasets (CUB200 returns a list)
        else:
            labels = None
            if not hasattr(continuum_dataset, '_y') or continuum_dataset._y is None:
                logging.info("Calling get_data() to populate continuum dataset targets for GCL...")
                loaded = continuum_dataset.get_data()
                if isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
                    labels = loaded[1]
                    xs = loaded[0]
                    if isinstance(xs, np.ndarray) and xs.size and isinstance(xs.reshape(-1)[0], (str, bytes, np.str_)):
                        self._image_paths = [str(item) for item in np.asarray(xs).reshape(-1)]

            if hasattr(continuum_dataset, '_y') and continuum_dataset._y is not None:
                labels = continuum_dataset._y
            elif labels is None and hasattr(continuum_dataset, 'targets'):
                labels = continuum_dataset.targets

            if labels is None:
                raise AttributeError("Unable to resolve dataset targets for GCLDatasetWrapper")

            self.targets = labels.tolist() if isinstance(labels, np.ndarray) else list(labels)

        self.classes = class_names

        logging.info(
            f"GCLDatasetWrapper: {len(self.targets)} samples, "
            f"{len(set(self.targets))} unique classes"
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self._image_paths is not None:
            x = Image.open(self._image_paths[index]).convert("RGB")
            y = self.targets[index]
        elif hasattr(self.continuum_dataset, 'dataset') and hasattr(self.continuum_dataset.dataset, 'targets'):
            x, y = self.continuum_dataset.dataset[index]
        else:
            x, y = self.continuum_dataset[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, int(y)


class ImageNet_C(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        self.data_path = self.data_path
        return super().get_data()


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(Dataset):
    """ImageNet-R list-based loader aligned with MindtheGap-GCL.

    Expected structure:
      <root>/train_list.txt
      <root>/val_list.txt
      <root>/imagenet-r/<class_name>/*.jpg
    """

    def __init__(self, data_path: str, train: bool = True, download: bool = False):
        self.train = train

        # Support either the manifest root or its nested image directory.
        raw_root = Path(data_path).expanduser().resolve()
        candidate_roots = [raw_root]
        if raw_root.name == "imagenet-r":
            candidate_roots.append(raw_root.parent)

        list_name = "train_list.txt" if train else "val_list.txt"
        root = None
        list_file = None
        for cand in candidate_roots:
            f = cand / list_name
            if f.exists():
                root = cand
                list_file = f
                break

        if root is None or list_file is None:
            raise FileNotFoundError(
                f"Cannot find {list_name} under '{raw_root}' or '{raw_root.parent}'. "
                "Please set dataset_root to the ImageNet-R root that contains list files."
            )

        self.data_path = str(root)
        self._x = []
        self._y = []

        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.split()
                self._x.append(str(root / rel_path))
                self._y.append(int(label))

        self._y = np.array(self._y, dtype=np.int64)
        self.targets = self._y.tolist()

        logging.info(f"ImageNet_R loaded: {len(self._x)} samples, train={train}")

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        from PIL import Image

        img = Image.open(self._x[index]).convert("RGB")
        label = int(self._y[index])
        return img, label

class VTAB(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    dataset_name = str(cfg.dataset).lower()

    if dataset_name == "cifar100":
        data_path = cfg.dataset_root
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes
    elif dataset_name in {"imagenet_r", "imagenet-r"}:
        data_path = cfg.dataset_root
        dataset = ImageNet_R(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif dataset_name == "cub200":
        data_path = cfg.dataset_root
        dataset = CUB200(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif dataset_name == "food101":
        data_path = cfg.dataset_root
        dataset = Food101(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif dataset_name == "oxford_pet":
        data_path = cfg.dataset_root
        dataset = OxfordPet(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif dataset_name == "caltech101":
        data_path = cfg.dataset_root
        dataset = Caltech101(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif dataset_name == "vtab":
        data_path = cfg.dataset_root
        dataset = VTAB(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif dataset_name == "imagenet_c":
        data_path = cfg.dataset_root
        dataset = ImageNet_C(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif dataset_name == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif dataset_name == "imagenet100":
        data_path = cfg.dataset_root
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join(get_workdir(os.getcwd()), "class_orders/train_100.txt" if is_train else "class_orders/val_100.txt")
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif dataset_name == "imagenet1000":
        data_path = cfg.dataset_root
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif dataset_name == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]
    
    else:
        raise ValueError(f"'{cfg.dataset}' is an invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        raise NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        raise ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                         "please choose from {'class', 'domain', 'task-agnostic'}.")

    return scenario, classes_names


def get_dataset_for_gcl(cfg, is_train, clip_transform):
    """
    Return a dataset wrapped for GCL OnlineSampler usage.

    Output dataset provides:
    - classes: class name list
    - targets: label list
    - __getitem__: (transformed_image, label)
    """
    continuum_dataset, class_names = get_dataset(cfg, is_train)
    gcl_dataset = GCLDatasetWrapper(continuum_dataset, class_names, transform=clip_transform)
    return gcl_dataset, class_names
