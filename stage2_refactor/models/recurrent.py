from __future__ import annotations

import torch
import torch.nn as nn

from stage2_refactor.models.base import BaseModel


class RecurrentBaseline(BaseModel):
    """Shared recurrent backbone with the Stage 1 regression head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        recurrent_type: str,
        bidirectional: bool = False,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        recurrent_classes = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        if recurrent_type not in recurrent_classes:
            raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")

        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.recurrent_name = module_name or recurrent_type
        recurrent = recurrent_classes[recurrent_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.add_module(self.recurrent_name, recurrent)

        head_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(head_input_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        encoded = self._encode(x)
        out = self.fc1(encoded)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        self.validate_output(out)
        return out

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        recurrent = getattr(self, self.recurrent_name)
        recurrent_out, hidden = recurrent(x)
        if not self.bidirectional:
            return recurrent_out[:, -1, :]

        h_n = hidden[0] if self.recurrent_type == "lstm" else hidden
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)
