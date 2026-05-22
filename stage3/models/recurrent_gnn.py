"""RecurrentGNNFusion: temporal recurrent branch + spatial GNN branch + fusion head.

Inherits stage2_refactor.models.base.BaseModel so it plugs directly into
the teammate's trainer / evaluator: forward(x) takes (B, T, F) and returns
(B, 1) with the standard input/output validators.

Architecture
------------
Two-branch late fusion:

    x : (B, T=30, F)
        |
        +-- Recurrent branch ----------------------------+
        |   nn.GRU(input=F, hidden=H, num_layers=L,      |
        |          bidirectional={uni|bi}, ...)          |
        |   - uni : h_seq = out[:, -1, :]                |
        |   - bi  : h_seq = cat(h_n[-2], h_n[-1])        |
        |                          -> (B, H or 2H)       |
        |                                                |
        +-- GNN branch (optional, gated by use_gnn) -----+
            sensor-as-node features: x.transpose(1, 2)
                                       -> (B, F, T)
            GCN(node_feat_dim=T, ...) -> (B, gnn_hidden)
                                           |
                                    concat v
            fused = cat([h_seq, h_graph], dim=-1)
            head:   Linear -> ReLU -> Dropout -> Linear -> 1
                                                       |
                                                       v
                                                   (B, 1)

Ablation switch
---------------
`use_gnn=False` removes the GNN branch entirely (no parameters allocated,
no graph features computed). The head then operates on h_seq alone, so the
model degenerates to the same Stage 2 recurrent baseline used as the
ablation control.

Why a raw nn.GRU rather than stage2_refactor's GRUBaseline / BiGRUBaseline?
--------------------------------------------------------------------------
The Stage 2 backbones bake the regression head into their `forward`, so the
intermediate recurrent feature is never exposed. Re-using `nn.GRU` directly
here is a few lines and avoids touching teammate code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_refactor.models.base import BaseModel
from stage3.models.gnn_modules import GCN


class RecurrentGNNFusion(BaseModel):
    """Stage 3 fusion model. Honors BaseModel's (B, T, F) -> (B, 1) contract.

    Args:
        input_size : F, number of retained sensors (per subset: 14/21/16/21).
        recurrent_kind : "gru" | "bigru".
        hidden_size, num_layers, dropout : recurrent backbone hyperparams.
        sequence_length : T, used as the node feature dim when use_gnn=True.
                          Default 30 (Stage 1/2 convention).
        adj_matrix : (F, F) adjacency, numpy array or torch tensor. Required
                     when use_gnn=True; ignored otherwise.
        gnn_hidden, gnn_layers : GNN hyperparams; ignored when use_gnn=False.
        gnn_kind : "gcn" (default) or "gat" (placeholder; raises).
        gnn_dropout : dropout inside the GCN (between layers).
        gnn_pool : "mean" / "add" / "max" pooling over nodes.
        use_gnn : ablation switch; False disables the GNN branch.
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
    ) -> None:
        super().__init__()
        if recurrent_kind not in {"gru", "bigru"}:
            raise ValueError(
                f"recurrent_kind must be 'gru' or 'bigru', got {recurrent_kind!r}"
            )

        self.input_size = input_size
        self.recurrent_kind = recurrent_kind
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.use_gnn = use_gnn

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

        if use_gnn:
            if adj_matrix is None:
                raise ValueError("adj_matrix is required when use_gnn=True.")
            adj_t = (
                adj_matrix
                if isinstance(adj_matrix, torch.Tensor)
                else torch.as_tensor(adj_matrix, dtype=torch.float32)
            )
            if adj_t.shape != (input_size, input_size):
                raise ValueError(
                    f"adj_matrix shape {tuple(adj_t.shape)} != "
                    f"({input_size}, {input_size})"
                )

            if gnn_kind == "gcn":
                self.gnn = GCN(
                    n_nodes=input_size,
                    node_feat_dim=sequence_length,
                    gnn_hidden=gnn_hidden,
                    gnn_layers=gnn_layers,
                    dropout=gnn_dropout,
                    adjacency=adj_t,
                    pool=gnn_pool,
                )
            elif gnn_kind == "gat":
                raise NotImplementedError(
                    "gnn_kind='gat' is reserved for the ablation step."
                )
            else:
                raise ValueError(f"unknown gnn_kind: {gnn_kind!r}")
            fused_dim = rec_dim + gnn_hidden
        else:
            self.gnn = None
            fused_dim = rec_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def _recurrent_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, H) for "gru" or (B, 2H) for "bigru"."""
        out, h_n = self.recurrent(x)
        if self.recurrent_kind == "gru":
            return out[:, -1, :]
        # bidirectional: h_n shape (num_layers * 2, B, H); take the last
        # layer's forward (h_n[-2]) and backward (h_n[-1]) final states.
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)

    def _graph_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, gnn_hidden) by treating sensors as nodes.

        Node features = each sensor's full window of readings, i.e. transpose
        (B, T, F) -> (B, F, T). node_feat_dim therefore equals T.
        """
        assert self.gnn is not None
        node_feats = x.transpose(1, 2).contiguous()
        return self.gnn(node_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"expected F={self.input_size}, got {x.shape[2]}"
            )
        if self.use_gnn and x.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected T={self.sequence_length} for the GNN branch, "
                f"got T={x.shape[1]}"
            )

        h_seq = self._recurrent_features(x)
        if self.use_gnn:
            h_graph = self._graph_features(x)
            h = torch.cat([h_seq, h_graph], dim=-1)
        else:
            h = h_seq

        out = self.head(h)
        self.validate_output(out)
        return out
