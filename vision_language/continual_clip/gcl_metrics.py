"""
GCL Metrics Module for General Continual Learning.
Implements evaluation metrics: A_auc, A_avg, A_last, F_last.
Adapted from Fly for CLIP-based continual learning.
"""

import numpy as np
from typing import Dict, List


class GCLMetrics:
    """
    Metrics for General Continual Learning (GCL).

    Metrics:
        - A_auc: Anytime accuracy (average of all periodic evaluations)
        - A_avg: Average accuracy across all sessions
        - A_last: Final accuracy on the last session
        - F_last: Final average forgetting
    """

    def __init__(self, num_classes: int, num_sessions: int):
        """
        Initialize GCL metrics tracker.

        Args:
            num_classes: Total number of classes in the dataset
            num_sessions: Number of sessions/tasks in the continual learning scenario
        """
        self.num_classes = num_classes
        self.num_sessions = num_sessions

        # Record accuracy at each periodic evaluation (for A_auc)
        self.eval_accs = []

        # Record accuracy at the end of each session (for A_avg, A_last)
        self.session_accs = []

        # Record per-class accuracy at each session (for F_last)
        self.cls_accs = np.zeros((num_sessions, num_classes))

    def add_eval_acc(self, acc: float):
        """
        Add accuracy from periodic evaluation during training.

        Args:
            acc: Accuracy value (0-100)
        """
        self.eval_accs.append(acc)

    def add_session_result(self, session_id: int, acc: float, cls_acc: np.ndarray):
        """
        Add results at the end of a session.

        Args:
            session_id: Current session ID (0-indexed)
            acc: Overall accuracy for this session
            cls_acc: Per-class accuracy array (shape: [num_classes])
        """
        self.session_accs.append(acc)
        if session_id < self.num_sessions:
            self.cls_accs[session_id] = cls_acc

    def compute_A_auc(self) -> float:
        """
        Compute A_auc: Average of all periodic evaluation accuracies.
        Reflects the model's anytime performance during training.

        Returns:
            A_auc value (0-100)
        """
        return np.mean(self.eval_accs) if len(self.eval_accs) > 0 else 0.0

    def compute_A_avg(self) -> float:
        """
        Compute A_avg: Average accuracy across all sessions.

        Returns:
            A_avg value (0-100)
        """
        return np.mean(self.session_accs) if len(self.session_accs) > 0 else 0.0

    def compute_A_last(self) -> float:
        """
        Compute A_last: Accuracy on the last session.

        Returns:
            A_last value (0-100)
        """
        return self.session_accs[-1] if len(self.session_accs) > 0 else 0.0

    def compute_F_last(self) -> float:
        """
        Compute F_last: Final average forgetting.

        For each class, compute: max(acc_before_last) - acc_last
        Then average across all classes that were seen before the last session.

        Returns:
            F_last value (0-100, higher means more forgetting)
        """
        if self.num_sessions <= 1:
            return 0.0

        acc_diff = []
        for j in range(self.num_classes):
            # Find maximum accuracy before the last session
            max_acc_before = np.max(self.cls_accs[:-1, j])
            if max_acc_before > 0:
                # Compute forgetting: peak accuracy - final accuracy
                forgetting = max_acc_before - self.cls_accs[-1, j]
                acc_diff.append(forgetting)

        return np.mean(acc_diff) if len(acc_diff) > 0 else 0.0

    def compute_BWT_last(self) -> float:
        """
        Compute BWT_last: Backward Transfer.

        For each class, compute: acc_last - acc_first_seen
        Then average across all classes.

        Returns:
            BWT_last value (-100 to 100, negative means forgetting)
        """
        if self.num_sessions <= 1:
            return 0.0

        bwt_vals = []
        for j in range(self.num_classes):
            per_cls_prev = self.cls_accs[:-1, j]
            seen_indices = np.where(per_cls_prev > 0)[0]
            if len(seen_indices) == 0:
                continue
            first_acc = per_cls_prev[seen_indices[0]]
            last_acc = self.cls_accs[-1, j]
            bwt_vals.append(last_acc - first_acc)

        return np.mean(bwt_vals) if len(bwt_vals) > 0 else 0.0

    def get_summary(self) -> Dict[str, float]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary with all computed metrics
        """
        return {
            'A_auc': self.compute_A_auc(),
            'A_avg': self.compute_A_avg(),
            'A_last': self.compute_A_last(),
            'F_last': self.compute_F_last(),
            'BWT_last': self.compute_BWT_last(),
        }

    def print_summary(self):
        """Print a formatted summary of all metrics."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("GCL Metrics Summary")
        print("="*50)
        print(f"A_auc (Anytime Accuracy):     {summary['A_auc']:.2f}%")
        print(f"A_avg (Average Accuracy):     {summary['A_avg']:.2f}%")
        print(f"A_last (Last Accuracy):       {summary['A_last']:.2f}%")
        print(f"F_last (Final Forgetting):    {summary['F_last']:.2f}%")
        print(f"BWT_last (Backward Transfer): {summary['BWT_last']:.2f}%")
        print("="*50 + "\n")
