from __future__ import annotations

import numpy as np
import torch


def rmse_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[float, float]:
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    error = y_pred - y_true
    rmse = torch.sqrt(torch.mean(error**2)).item()
    score = torch.where(
        error < 0,
        torch.exp(-error / 13.0) - 1.0,
        torch.exp(error / 10.0) - 1.0,
    )
    return rmse, torch.sum(score).item()


def numpy_rmse_score(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    error = y_pred.reshape(-1) - y_true.reshape(-1)
    rmse = float(np.sqrt(np.mean(error**2)))
    score = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return rmse, float(np.sum(score))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

