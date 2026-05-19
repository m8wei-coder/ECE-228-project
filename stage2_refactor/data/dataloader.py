from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CMAPSSWindowDataset(Dataset):
    """Sliding-window C-MAPSS dataset.

    Each sample is X: (T, F), y: (1,). A DataLoader batch is therefore
    X: (B, T, F), y: (B, 1), matching the Stage 2 model contract.
    """

    def __init__(self, df: pd.DataFrame, sequence_length: int, features: list[str]):
        self.sequence_length = sequence_length
        self.features = features
        self.sequences, self.labels = self._generate_sequences(df)

    def _generate_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        seqs = []
        labels = []

        for engine_id in df["unit_number"].unique():
            engine_data = df[df["unit_number"] == engine_id].sort_values("time_cycles")
            x_values = engine_data[self.features].values
            y_values = engine_data["RUL_piecewise"].values

            for i in range(len(engine_data) - self.sequence_length + 1):
                seqs.append(x_values[i : i + self.sequence_length])
                labels.append(y_values[i + self.sequence_length - 1])

        return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_tensor = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return seq_tensor, label_tensor


@dataclass(frozen=True)
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame | None
    train_engine_ids: list[int]
    val_engine_ids: list[int]


def split_by_engine_id(
    df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> SplitResult:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    engine_ids = np.asarray(sorted(df["unit_number"].unique()))
    shuffled = engine_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_fraction)))

    val_ids = sorted(shuffled[:n_val].astype(int).tolist())
    train_ids = sorted(shuffled[n_val:].astype(int).tolist())

    train_df = df[df["unit_number"].isin(train_ids)].copy()
    val_df = df[df["unit_number"].isin(val_ids)].copy()
    return SplitResult(train_df=train_df, val_df=val_df, train_engine_ids=train_ids, val_engine_ids=val_ids)


def build_train_val_loaders(
    df: pd.DataFrame,
    features: list[str],
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    validation_enabled: bool,
    validation_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader | None, SplitResult]:
    if validation_enabled:
        split = split_by_engine_id(df, validation_fraction, seed)
        train_df = split.train_df
        val_df = split.val_df
    else:
        engine_ids = sorted(df["unit_number"].unique().astype(int).tolist())
        split = SplitResult(train_df=df.copy(), val_df=None, train_engine_ids=engine_ids, val_engine_ids=[])
        train_df = df
        val_df = None

    train_dataset = CMAPSSWindowDataset(train_df, sequence_length, features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    val_loader = None
    if val_df is not None:
        val_dataset = CMAPSSWindowDataset(val_df, sequence_length, features)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, val_loader, split


def build_final_test_windows(
    df_test: pd.DataFrame,
    features: list[str],
    sequence_length: int,
) -> np.ndarray:
    """Build one final window per test engine, matching Stage 1 main_test.py."""
    x_test = []
    for engine_id in df_test["unit_number"].unique():
        engine_data = df_test[df_test["unit_number"] == engine_id].sort_values("time_cycles")
        x_values = engine_data[features].values.astype(np.float32)

        if len(x_values) >= sequence_length:
            x_test.append(x_values[-sequence_length:])
        else:
            pad_len = sequence_length - len(x_values)
            pad = np.repeat(x_values[:1], pad_len, axis=0)
            x_test.append(np.vstack([pad, x_values]))

    return np.asarray(x_test, dtype=np.float32)

