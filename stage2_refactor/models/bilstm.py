from __future__ import annotations

import torch
import torch.nn as nn

from stage2_refactor.models.base import BaseModel


class BiLSTMBaseline(BaseModel):
    """Bidirectional LSTM variant with the Stage 1 regression head."""

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
            bidirectional=True,
        )
        head_input_size = hidden_size * 2
        self.fc1 = nn.Linear(head_input_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        _, (h_n, _) = self.lstm(x)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        self.validate_output(out)
        return out

