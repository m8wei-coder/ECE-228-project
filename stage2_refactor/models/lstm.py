from __future__ import annotations

import torch
import torch.nn as nn

from stage2_refactor.models.base import BaseModel


class LSTMBaseline(BaseModel):
    """Stage 1 LSTM architecture, wrapped in the Stage 2 BaseModel contract."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        self.validate_output(out)
        return out

