from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from stage2_refactor.training.evaluator import rmse_score


@dataclass
class FitResult:
    best_metric: float
    best_epoch: int
    total_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_preds.append(y_pred.detach())
        all_labels.append(y_batch.detach())

    avg_loss = train_loss / len(train_loader)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    rmse, score = rmse_score(preds, labels)
    return avg_loss, rmse, score


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        total_loss += loss.item()
        all_preds.append(y_pred)
        all_labels.append(y_batch)

    avg_loss = total_loss / len(loader)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    rmse, score = rmse_score(preds, labels)
    return avg_loss, rmse, score


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int | None,
    best_model_path: str | Path,
    last_checkpoint_path: str | Path,
    best_checkpoint_path: str | Path,
    metadata: dict[str, Any],
    csv_logger: Any | None = None,
    wandb_logger: Any | None = None,
    resume_checkpoint_path: str | Path | None = None,
) -> FitResult:
    best_model_path = Path(best_model_path)
    last_checkpoint_path = Path(last_checkpoint_path)
    best_checkpoint_path = Path(best_checkpoint_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).exists():
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", best_metric)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement",
            epochs_without_improvement,
        )

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_loss, train_rmse, train_score = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = None
        val_rmse = None
        val_score = None
        if val_loader is not None:
            val_loss, val_rmse, val_score = evaluate_loader(model, val_loader, criterion, device)

        monitor_value = val_rmse if val_rmse is not None else train_loss
        improved = monitor_value < best_metric
        if improved:
            best_metric = float(monitor_value)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "metadata": metadata,
        }
        torch.save(checkpoint, last_checkpoint_path)
        if improved:
            torch.save(checkpoint, best_checkpoint_path)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_rmse": train_rmse,
            "train_score": train_score,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_score": val_score,
            "monitor_metric": monitor_value,
            "best_metric": best_metric,
            "epoch_seconds": time.time() - epoch_start,
            **metadata,
        }
        if csv_logger is not None:
            csv_logger.log(row)
        if wandb_logger is not None:
            wandb_logger.log(row)

        if val_loader is not None and patience is not None and epochs_without_improvement >= patience:
            break

    return FitResult(
        best_metric=best_metric,
        best_epoch=best_epoch + 1,
        total_seconds=time.time() - start_time,
    )

