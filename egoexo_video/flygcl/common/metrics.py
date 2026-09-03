from __future__ import annotations

import math
from typing import Dict, List, Sequence


def summarize_accuracy_matrix(matrix: Sequence[Sequence[float]], counts: Sequence[int]) -> Dict[str, float]:
    if not matrix:
        return {}
    final = list(matrix[-1])
    macro = sum(final) / len(final)
    final_counts = list(counts[: len(final)])
    micro = sum(v * n for v, n in zip(final, final_counts)) / max(sum(final_counts), 1)
    forgetting_values: List[float] = []
    bwt_values: List[float] = []
    for task_id in range(len(final) - 1):
        history = [row[task_id] for row in matrix[task_id:] if len(row) > task_id]
        forgetting_values.append(max(history) - final[task_id])
        bwt_values.append(final[task_id] - matrix[task_id][task_id])
    session_macro = [sum(row) / len(row) for row in matrix if row]
    return {
        "final_macro_accuracy": macro,
        "final_micro_accuracy": micro,
        "average_forgetting": sum(forgetting_values) / max(len(forgetting_values), 1),
        "backward_transfer": sum(bwt_values) / max(len(bwt_values), 1),
        "cl_auc": sum(session_macro) / len(session_macro),
    }
