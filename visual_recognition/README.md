# Continual visual recognition

This package implements FlyGCL for online Si-Blurry class-incremental image recognition. The identifiers `flyprompt`, `flyadapter`, and `flylora` select prompt-, adapter-, and LoRA-based instantiations. A random-projection head routes instances to lightweight experts, and EMA heads provide complementary temporal predictions.

## Benchmarks

| Dataset | Sessions | Disjoint ratio | Blurry ratio | Backbone |
|---|---:|---:|---:|---|
| CIFAR-100 | 5 | 50% | 10% | ViT-B/16 |
| ImageNet-R | 5 | 50% | 10% | ViT-B/16 |
| CUB-200 | 5 | 50% | 10% | ViT-B/16 |

## Installation

```bash
conda create -n flygcl-vision python=3.10 -y
conda activate flygcl-vision
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
  --index-url https://download.pytorch.org/whl/cu117
pip install timm numpy pandas scikit-learn matplotlib
```

## Data

Pass the dataset root with `--data_dir`. The provided loaders support `cifar100`, `imagenet-r`, and `cub200`; inspect `datasets/` for their expected layouts.

## Run FlyGCL

From this directory, a CIFAR-100 invocation is:

```bash
python main.py \
  --method flyprompt \
  --dataset cifar100 \
  --data_dir data/cifar100 \
  --backbone vit_base_patch16_224 \
  --n_tasks 5 --n 50 --m 10 \
  --batchsize 64 --lr 0.005 --opt_name adam \
  --online_iter 3 --num_epochs 1 \
  --rp_dim 10000 --rp_ridge 10000 \
  --ema_ratio 0.9 0.99 \
  --use_amp --eval_period 1000 \
  --note flygcl_cifar100
```

Change `--dataset` and `--data_dir` for ImageNet-R or CUB-200. Adapter and LoRA variants are registered in the method/configuration code; use the corresponding prepared scripts after checking their arguments.

## Baselines and outputs

Baseline launchers are under `scripts/`, and outputs are written below `results/`. Paths and executables can be set through the environment variables documented in the launchers.

See `configuration/config.py` for all CLI flags.
