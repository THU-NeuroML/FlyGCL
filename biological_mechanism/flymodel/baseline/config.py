"""Single source of truth for the fixed-encoder baseline protocol."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("shared", "unrouted_5", "random_routed_5", "online_routed_5", "oracle_routed_5")
FORMAL_SEEDS = tuple(range(5))
CALIBRATION_SEED = 5
LR_CANDIDATES = (3e-4, 1e-3)
LR_CONDITIONS = ("shared", "random_routed_5", "online_routed_5")

@dataclass(frozen=True)
class Config:
    data_root: Path = PROJECT_ROOT / "data" / "olfactory"
    result_root: Path = PROJECT_ROOT / "results" / "baseline_regions"
    raw_flywire_root: Path = PROJECT_ROOT / "data"
    pn_kc_stats_path: Path = PROJECT_ROOT / "data" / "flywire_stats.json"
    n_classes: int = 100
    odor_dim: int = 50
    n_train: int = 1_000_000
    n_test: int = 200_000
    n_regions: int = 5
    n_orn: int = 1300
    orn_per_channel: int = 26
    n_pn: int = 50
    n_kc: int = 2000
    kc_topk: int = 100
    batch_size: int = 64
    evaluation_batch_size: int = 2048
    train_noise_sigma: float = 0.1
    region_iterations: int = 12
    region_price_steps: int = 10
    region_price_rate: float = 0.35
    region_chunk_size: int = 16_384
    min_region_fraction: float = 0.15
    max_region_fraction: float = 0.25
    evaluation_points_per_region: int = 5
    lr_candidates: tuple[float, ...] = LR_CANDIDATES
    conditions: tuple[str, ...] = CONDITIONS
    formal_seeds: tuple[int, ...] = FORMAL_SEEDS
    calibration_seed: int = CALIBRATION_SEED
    seed_namespace: tuple[int, ...] = (2026, 8, 23, 4)
    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("data_root", "result_root", "raw_flywire_root", "pn_kc_stats_path"):
            value[key] = str(value[key])
        for key in ("lr_candidates", "conditions", "formal_seeds", "seed_namespace"):
            value[key] = list(value[key])
        return value
