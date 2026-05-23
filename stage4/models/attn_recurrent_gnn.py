"""AttnRecurrentGNN: CBAM1D applied to the input window of Stage 3 RGNN.

For subsets whose Stage 3 finalist is recurrent + GNN-physical (FD002,
FD003, FD004 per `stage3/docs/stage3_analysis.md`), Stage 4 wraps that
architecture with a CBAM1D attention block at the input.

Architecture
------------
    x : (B, T=30, F)
        |
        v
    CBAM1D(F)  -- channel-then-temporal gates over the input window
        |
        v
    (B, T, F)  -- attention-weighted features fed to BOTH branches
        |
        +-- Recurrent branch (GRU or BiGRU) -> (B, H or 2H)
        +-- GNN branch (DenseGCN over fixed adjacency) -> (B, gnn_hidden)
                                                |
                                            fuse v
                                          Linear -> ReLU -> Dropout -> Linear -> 1

Implementation note
-------------------
We delegate the entire recurrent + GNN + fusion block to
`stage3.RecurrentGNNFusion`, so the only Stage 4-local code is the CBAM
prefix and a thin wrapper around its forward(). This guarantees identical
backbone behaviour to Stage 3 when both CBAM sub-modules are disabled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_refactor.models.base import BaseModel
from stage3.models.recurrent_gnn import RecurrentGNNFusion
from stage4.models.attention import CBAM1D


class AttnRecurrentGNN(BaseModel):
    """CBAM1D + Stage 3 RecurrentGNNFusion.

    Args:
        input_size : F, retained sensors for the subset.
        recurrent_kind : "gru" or "bigru".
        hidden_size, num_layers, dropout : recurrent backbone hyperparams.
        sequence_length : T (default 30).
        adj_matrix : (F, F) fixed adjacency; required when use_gnn=True.
        gnn_hidden, gnn_layers, gnn_kind, gnn_dropout, gnn_pool : GNN
            branch hyperparams; ignored when use_gnn=False.
        use_gnn : whether to instantiate the GNN branch (Stage 3 ablation
            switch). If False the model degenerates to AttnRecurrent.
        use_channel_attn, use_temporal_attn : independent CBAM toggles.
        attn_reduction, attn_kernel : CBAM hyperparams.
    """

    def __init__(
        self,
        input_size: int,
        recurrent_kind: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        sequence_length: int = 30,
        adj_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        gnn_hidden: int = 32,
        gnn_layers: int = 2,
        gnn_kind: str = "gcn",
        gnn_dropout: float = 0.1,
        gnn_pool: str = "mean",
        use_gnn: bool = True,
        use_channel_attn: bool = True,
        use_temporal_attn: bool = True,
        attn_reduction: int = 4,
        attn_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.use_gnn = use_gnn
        self.use_channel_attn = use_channel_attn
        self.use_temporal_attn = use_temporal_attn

        self.attention = CBAM1D(
            num_channels=input_size,
            reduction=attn_reduction,
            kernel_size=attn_kernel,
            use_channel=use_channel_attn,
            use_temporal=use_temporal_attn,
        )

        # Delegate the recurrent + GNN + fusion head to the Stage 3 model.
        self.backbone = RecurrentGNNFusion(
            input_size=input_size,
            recurrent_kind=recurrent_kind,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            adj_matrix=adj_matrix,
            gnn_hidden=gnn_hidden,
            gnn_layers=gnn_layers,
            gnn_kind=gnn_kind,
            gnn_dropout=gnn_dropout,
            gnn_pool=gnn_pool,
            use_gnn=use_gnn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"expected F={self.input_size}, got {x.shape[2]}"
            )
        x = self.attention(x)
        out = self.backbone(x)
        self.validate_output(out)
        return out
