# Biological mechanism experiments

This package contains the controlled olfactory continual-learning experiments used to study the biological mechanisms behind FlyGCL. The model follows the population scale of the *Drosophila* olfactory pathway:

```text
50-dimensional odor input
    -> 1,300 olfactory receptor neurons
    -> 50 projection neurons
    -> 2,000 Kenyon cells (top 5% retained)
    -> spatial expert
    -> one or three temporal heads
    -> 100-class prediction
```

The ORN–PN–KC encoder is fixed. Only bias-free linear readout heads downstream of the Kenyon cells are optimized.

## Experiments

The four model variants isolate spatial and temporal modularity:

| Variant | Spatial experts | Temporal heads |
|---|---:|---:|
| Baseline | 1 | 1 |
| EL | 1 | 3 |
| MoE | 5 | 1 |
| MoE+EL | 5 | 3 per expert |

The temporal heads use learning rates $10^{-2}$, $10^{-3}$, and $10^{-4}$. For routed models, each stage activates one expert during training. From the second stage onward, a new expert is initialized from the arithmetic mean of previously trained experts. At inference, a non-parametric router selects among active experts using cosine similarity between the input's KC representation and online stage prototypes. Multi-head predictions are integrated by averaging class probabilities.

Each seed contains 100 prototype-defined odor classes partitioned into five balanced regions, with 10,000 training samples and 2,000 test samples per region. Every stream contains 50,000 unique training samples over five stages:

- **Disjoint:** regions are presented sequentially without cross-stage redistribution.
- **Blurry (`n50m10`):** 50% of the classes are eligible for 10% cross-stage redistribution.
- **Joint (`n0m10`):** all classes are eligible for 10% cross-stage redistribution.

No replay or additional training samples are used.

## Package layout

```text
biological_mechanism/
├── flymodel/
│   ├── baseline/  # Audited data generation and the Disjoint single-head baseline
│   └── main/      # Inherited experts, temporal heads, routing, and blurry streams
└── requirements.txt
```

`flymodel/main/model.py` defines the fixed sensory encoder and expert bank. The Disjoint and boundary-blurred training loops are implemented in `flymodel/main/experiment.py` and `flymodel/main/blurry_experiment.py`, respectively.

## Environment

Create an isolated Python environment and install the package requirements:

```bash
cd biological_mechanism
python -m pip install -r requirements.txt
```

The runners accept any PyTorch device string through `--device`; use `cpu` for a CPU run or `cuda` for a CUDA device.

## Data assets

The experiment reads immutable per-seed assets from `data/olfactory/`. The data builder accepts the FlyWire tables `classification.csv`, `consolidated_cell_types.csv`, and `connections_princeton.csv`, together with a PN–KC statistics JSON file:

```bash
python -m flymodel.baseline.run \
  --raw-flywire-root /path/to/flywire_tables \
  --pn-kc-stats /path/to/flywire_stats.json \
  prepare-data
```

This generates the prototype dataset, fixed train/test samples, ORN–PN weights, PN–KC matrix, metadata, hashes, and audit records for the formal seeds.

## Reproduce the formal experiments

Run commands from `biological_mechanism/`. A single worker executes each complete matrix sequentially; use different `--worker` values with the same `--workers` count to distribute tasks across devices.

### Disjoint

The single-head Baseline is evaluated by the class-exclusive baseline protocol:

```bash
python -m flymodel.baseline.class_exclusive_run --worker 0 --workers 1 --device cuda
python -m flymodel.baseline.class_exclusive_analysis
```

EL, MoE, MoE+EL, and the temporal controls are evaluated by the main protocol:

```bash
python -m flymodel.main.run --worker 0 --workers 1 --device cuda
python -m flymodel.main.analysis
```

### Blurry and Joint

The boundary-blur matrix contains all four model variants. The manuscript settings are `n50m10` for Blurry and `n0m10` for Joint.

```bash
python -m flymodel.main.blurry_run --worker 0 --workers 1 --device cuda
python -m flymodel.main.blurry_analysis
```

By default, run records and aggregated analyses are written below `results/`. Each record includes the experiment identity, stream checks, source hashes, runtime information, evaluation checkpoints, and mechanism-specific audits.
