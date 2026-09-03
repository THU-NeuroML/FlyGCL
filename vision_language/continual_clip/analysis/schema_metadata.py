"""Small compatibility metadata for analysis-stat outputs.

The training and analysis code writes many independent CSV/JSON artifacts.  This
module records which requested representation statistics are intentionally
available only as pending companion metadata, without changing existing metric
semantics.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Mapping, Optional


ANALYSIS_STATS_SCHEMA_VERSION = "2026-06-25.coverage.v1"

PENDING_STATS = {
    "sample_level_alignment.parquet": "sample_level_large_object_export_not_enabled",
    "sample_level_feature_drift.parquet": "sample_level_large_object_export_not_enabled",
    "confusion_matrix": "matrix_export_not_enabled",
    "per_class_acc_drop": "per_class_drop_export_not_enabled",
    "RDM_ref": "topology_rdm_export_not_enabled",
    "RDM_current": "topology_rdm_export_not_enabled",
    "RDM_drift": "topology_rdm_export_not_enabled",
    "prototype_pairwise_cos_ref": "topology_pairwise_export_not_enabled",
    "prototype_pairwise_cos_current": "topology_pairwise_export_not_enabled",
    "margin_projected_feature_drift": "requires_sample_delta_and_hard_negative_text_vectors",
    "value_ref": "value_vectors_not_recorded",
    "value_current": "value_vectors_not_recorded",
    "effective_context_drift": "value_vectors_not_recorded",
    "CLS_effective_context_drift": "value_vectors_not_recorded",
    "CLS_hidden_ref": "hidden_states_not_recorded",
    "CLS_hidden_current": "hidden_states_not_recorded",
    "layer_residual_CLS_ref": "hidden_states_not_recorded",
    "layer_residual_CLS_current": "hidden_states_not_recorded",
    "grad_norm": "training_dynamics_csv_not_enabled",
    "param_norm": "training_dynamics_csv_not_enabled",
    "update_norm": "training_dynamics_csv_not_enabled",
}


def analysis_stats_run_metadata() -> Dict[str, object]:
    """Fields safe to append to leaderboard/final summary JSON outputs."""
    return {
        "analysis_stats_schema_version": ANALYSIS_STATS_SCHEMA_VERSION,
        "analysis_stats_coverage_status": "partial_with_pending_companion_stats",
        "missing_stats_warning_count": 0,
        "pending_stats_warning_count": len(PENDING_STATS),
        "pending_stats_not_available_reason": dict(PENDING_STATS),
    }


def write_analysis_schema_metadata(
    output_dir: str,
    output_files: Optional[Mapping[str, Iterable[str]]] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> str:
    """Write a sidecar schema/availability file next to analysis CSV outputs."""
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, object] = analysis_stats_run_metadata()
    payload.update({
        "output_files": {
            name: list(fields)
            for name, fields in (output_files or {}).items()
        },
        "pending_stats": [
            {
                "field": field,
                "status": "missing_pending",
                "not_available_reason": reason,
            }
            for field, reason in sorted(PENDING_STATS.items())
        ],
    })
    if extra:
        payload.update(dict(extra))
    path = os.path.join(output_dir, "analysis_schema.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
