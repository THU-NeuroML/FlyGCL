# Continual vision-language learning

This package implements FlyGCL with CLIP ViT-B/16 and LoRA experts for continual image-text learning on CIFAR-100 and ImageNet-R. It includes training configurations, dataset adapters, ImageNet-R split manifests, launchers, evaluation utilities, and analysis code.

## Installation

```bash
conda create -n flygcl-clip python=3.10 -y
conda activate flygcl-clip
pip install -r requirements.txt
```

## Data

- CIFAR-100 is handled through the torchvision adapter and can be downloaded into a writable root.
- ImageNet-R uses the split manifests in `dataset_reqs/imagenet_r_split/`; place `train_list.txt` and `val_list.txt` in the dataset root expected by the loader.

## Run

CIFAR-100, one seed:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train/cifar100.sh \
  data/cifar100 outputs/cifar100 0
```

CIFAR-100, five seeds:

```bash
scripts/reproduce/cifar100_5seed.sh \
  data/cifar100 outputs/cifar100_5seed
```

ImageNet-R, one seed:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train/imagenet_r.sh \
  data/imagenet-r outputs/imagenet_r 2
```

Summarize completed runs:

```bash
python tools/summarize_results.py outputs/cifar100_5seed \
  --csv outputs/cifar100_5seed/summary.csv \
  --json outputs/cifar100_5seed/summary.json
```
