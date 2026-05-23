"""AttnRecurrent: CBAM1D -> GRU/BiGRU -> regression head.

This is the Stage 4 model used for subsets whose Stage 3 finalist is the
pure recurrent backbone with no GNN branch (FD001 per
`stage3/docs/stage3_analysis.md`).

The recurrent block and regression head exactly mirror the Stage 2
finalists (GRUBaseline / BiGRUBaseline). The only addition is a CBAM1D
attention block applied to the (B, T, F) input window before it enters
the recurrent layer. When both attention sub-modules are disabled, the
model reduces to the corresponding Stage 2 finalist and serves as the
ablation control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_refactor.models.base import BaseModel
from stage4.models.attention import CBAM1D


class AttnRecurrent(BaseModel):
    """CBAM-augmented recurrent regressor.

    Args:
        input_size: F, retained sensors.
        recurrent_kind: "gru" or "bigru".
        hidden_size, num_layers, dropout: recurrent backbone hyperparams.
        use_channel_attn: enable the channel-attention sub-module of CBAM.
        use_temporal_attn: enable the temporal-attention sub-module of CBAM.
        attn_reduction: reduction ratio inside ChannelAttention.
        attn_kernel: 1-D conv kernel inside TemporalAttention.
    """

    def __init__(
        self,
        input_size: int,
        recurrent_kind: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_channel_attn: bool = True,
        use_temporal_attn: bool = True,
        attn_reduction: int = 4,
        attn_kernel: int = 7,
    ) -> None:
        super().__init__()
        if recurrent_kind not in {"gru", "bigru"}:
            raise ValueError(
                f"recurrent_kind must be 'gru' or 'bigru', got {recurrent_kind!r}"
            )
        self.input_size = input_size
        self.recurrent_kind = recurrent_kind
        self.use_channel_attn = use_channel_attn
        self.use_temporal_attn = use_temporal_attn

        self.attention = CBAM1D(
            num_channels=input_size,
            reduction=attn_reduction,
            kernel_size=attn_kernel,
            use_channel=use_channel_attn,
            use_temporal=use_temporal_attn,
        )

        bidirectional = recurrent_kind == "bigru"
        self.recurrent = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rec_dim = hidden_size * (2 if bidirectional else 1)

        # Stage 2 head: Linear -> ReLU -> Dropout -> Linear -> 1
        self.fc1 = nn.Linear(rec_dim, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"expected F={self.input_size}, got {x.shape[2]}"
            )

        x = self.attention(x)
        out, h_n = self.recurrent(x)
        if self.recurrent_kind == "gru":
            h = out[:, -1, :]
        else:
            # h_n shape: (num_layers * 2, B, H); concat last layer's fwd/bwd.
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        self.validate_output(h)
        return h
