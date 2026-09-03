from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class CLMethod(nn.Module, ABC):
    """Abstract base class for all continual learning methods."""

    @abstractmethod
    def adaptation(self, task_id: int, reset: bool = False) -> None:
        """Called at the start of each task to update internal state."""
        ...

    @abstractmethod
    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        """Returns logits over seen classes."""
        ...

    def after_task(self, train_loader=None) -> None:
        """Called after training each task. Override for post-task operations."""
        pass

    def auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Returns an auxiliary loss term to add to CE loss, or None."""
        return None
