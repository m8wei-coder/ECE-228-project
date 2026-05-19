from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base contract for Stage 2 backbone swaps.

    forward(x) must accept x with shape (B, T, F) and return shape (B, 1).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def validate_input(x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, F), got {tuple(x.shape)}")

    @staticmethod
    def validate_output(y: torch.Tensor) -> None:
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(f"Expected output shape (B, 1), got {tuple(y.shape)}")

