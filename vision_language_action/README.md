# Continual vision-language-action learning

This package contains FlyGCL implementations for continual language-conditioned robotic manipulation on LIBERO. A pretrained DiT flow-matching policy is adapted with routed lightweight experts and multi-timescale residual predictions.

The online and offline protocols use separate source trees because their training and evaluation flows differ:

```text
vision_language_action/
├── online/    # GCL / blurry continual stream
└── offline/   # Conventional task-wise continual learning
```

Each tree contains the local `lerobot_lsy/` and `peft_lsy/` runtime packages, experiment configurations, and launchers.

## Environment

Create separate environments if you plan to compare the two snapshots. From either `online/` or `offline/`, the local packages can be installed in editable mode:

```bash
conda create -n flygcl-vla python=3.10 -y
conda activate flygcl-vla
pip install -e ./peft_lsy
pip install -e ./lerobot_lsy
```

Install LIBERO and its simulator dependencies in the same environment before running the experiments.

## Data and pretrained policy

Dataset roots and the pretrained policy path are supplied through launcher arguments or environment variables. The launchers use repository-relative defaults under `data/`, `models/`, `outputs/`, and `logs/`; these local directories are ignored by Git.

## Code map

The main continual-adaptation components are located at:

```text
online|offline/peft_lsy/src/peft/tuners/clare/          # experts, routing, temporal heads
online|offline/lerobot_lsy/src/lerobot/scripts/clare.py # continual training loop
online/lerobot_lsy/src/lerobot/datasets/                 # online GCL stream support
```

Experiment launchers are under `online/bash/` and `offline/bash/`. Run them from the corresponding protocol directory. Weights & Biases logging is disabled unless explicitly enabled.
