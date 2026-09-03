# Continual ego-exo video understanding

This package provides FlyGCL training and evaluation pipelines for EgoExoLearn skill assessment and action anticipation. FlyGCL uses video-specific adapter experts with routing and temporal integration over evolving first- and third-person experience.

## Coverage

| Task | Evaluation |
|---|---|
| Skill assessment | Ranking accuracy and continual-learning metrics |
| Action anticipation | Top-5 verb/noun recall and continual-learning metrics |

## Installation

```bash
conda create -n flygcl-egoexo python=3.10 -y
conda activate flygcl-egoexo
pip install -r requirements.txt
```

## Data

Download the datasets from their official project pages and follow the corresponding instructions to obtain the data and annotations:

- [EgoExoLearn](https://github.com/OpenGVLab/EgoExoLearn)
- [EgoExo-Fitness](https://github.com/iSEE-Laboratory/EgoExo-Fitness)

Set the downloaded dataset and feature paths through the relevant command-line arguments or configuration files.

## Reproduce

Run from this directory. Both commands use seeds 42, 43, and 44 by default.

```bash
python reproduce.py skill --device cuda --num-workers 4
```

```bash
python reproduce.py anticipation \
  --annotation-root data/balanced_full_annotation \
  --feature-root data/clip_features_5fps \
  --device cuda --num-workers 4
```

Results are written to `flygcl/runs/`. Skill assessment reports `A_last`, `F_T`, and `A_auc` for TL ego-exo, RN ego-exo, and ego-only variants. Anticipation reports ego/exo verb and noun metrics plus aggregate `A_last` and `A_auc` for ego-exo, ego-only, and exo-only inputs.

Implementation details are organized under `flygcl/`, with fixed experiment settings in `flygcl/configs/`.
