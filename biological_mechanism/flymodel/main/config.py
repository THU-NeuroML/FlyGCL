"""Fixed protocol for the hierarchical olfactory model."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ETA = 1e-3
GAMMAS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
SEEDS = tuple(range(5))
READOUTS = ("logits_mean", "softmax_mean", "softmax_max")
CONDITIONS = ("shared_el", "inherited_moe_el", "inherited_moe_mid", "inherited_moe_fast", "inherited_moe_slow")
METRICS = ("seen_anytime_auc", "final_accuracy", "current_adaptation", "old_retention", "average_forgetting", "worst_region_accuracy")

@dataclass(frozen=True)
class Config:
    data_root: Path = PROJECT_ROOT / "data" / "olfactory"
    result_root: Path = PROJECT_ROOT / "results" / "disjoint"
    n_classes: int = 100
    odor_dim: int = 50
    n_regions: int = 5
    n_orn: int = 1300
    orn_per_channel: int = 26
    n_pn: int = 50
    n_kc: int = 2000
    kc_topk: int = 100
    batch_size: int = 64
    evaluation_batch_size: int = 2048
    train_noise_sigma: float = 0.1
    evaluation_points_per_region: int = 5
    train_per_stage: int = 10_000
    test_per_stage: int = 2_000
    seed_namespace: tuple[int, ...] = (2026, 8, 23, 5)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["data_root"] = str(value["data_root"])
        value["result_root"] = str(value["result_root"])
        value["seed_namespace"] = list(value["seed_namespace"])
        return value
