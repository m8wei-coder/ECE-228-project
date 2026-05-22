"""GCN / GAT branches for the Stage 3 recurrent + GNN fusion model.

Both modules operate on a fixed-topology graph (one adjacency matrix per
subset, identical across all samples in a batch). Per-sample variation lives
entirely in the node features.

Input contract
--------------
- node_features : torch.Tensor of shape (B, N_nodes, node_feat_dim).
                  recurrent_gnn.py is responsible for collapsing each
                  (B, T=30, F) window into per-sensor descriptors here
                  (e.g. mean / std / last / slope across the time window).
- adjacency     : torch.Tensor of shape (N_nodes, N_nodes), passed once at
                  construction time and registered as a non-trainable buffer.

Output contract
---------------
- graph_embedding : torch.Tensor of shape (B, gnn_hidden), obtained by
                    global mean (or sum / max) pooling over the N node axis.

Design choice: dense + PyG DenseGCNConv
---------------------------------------
Every sample in a batch shares the same adjacency, so we keep a single dense
(N, N) tensor as a buffer and rely on torch_geometric.nn.dense.DenseGCNConv
to do the symmetric-normalized propagation D^-1/2 (A + I) D^-1/2 X W in one
batched matmul per layer. This is conceptually identical to Kipf & Welling
(2017) GCN. The alternative — PyG's sparse `GCNConv` with a `Batch` object
that disjoints B copies of the same graph — is heavier boilerplate without
any per-sample structural difference to exploit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv


class GCN(nn.Module):
    """Stacked GCN with global pooling.

    The adjacency is fixed at construction time and registered as a buffer,
    so .to(device) moves it along with parameters but no gradient flows
    through it.

    Self-loops
    ----------
    DenseGCNConv adds the identity matrix to the adjacency by default
    (`add_loop=True`). The adjacency tensors built by `stage3.build_graph`
    have a zero diagonal, so this default produces the conventional GCN
    propagation matrix `A + I` (Kipf & Welling 2017). We keep that default
    rather than baking self-loops into the saved .npy files.

    Args
    ----
    n_nodes : number of graph nodes (= retained sensors for the subset:
              14 / 16 / 21 in our setup).
    node_feat_dim : per-node input feature dim, supplied by recurrent_gnn.
    gnn_hidden : hidden + output channels per GCN layer; equals the dim of
                 the pooled graph embedding.
    gnn_layers : number of stacked DenseGCNConv layers (>= 1).
    dropout : dropout probability applied between (not after) GCN layers.
    adjacency : (n_nodes, n_nodes) dense adjacency matrix.
    pool : "mean" | "add" | "max" global pooling over nodes.
    """

    def __init__(
        self,
        n_nodes: int,
        node_feat_dim: int,
        gnn_hidden: int,
        gnn_layers: int,
        dropout: float,
        adjacency: torch.Tensor,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if gnn_layers < 1:
            raise ValueError("gnn_layers must be >= 1")
        if pool not in {"mean", "add", "max"}:
            raise ValueError(f"unknown pool: {pool!r}")
        adj = adjacency.to(torch.float32)
        if adj.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"adjacency shape {tuple(adj.shape)} != ({n_nodes}, {n_nodes})"
            )

        self.n_nodes = n_nodes
        self.gnn_hidden = gnn_hidden
        self.dropout = dropout
        self.pool = pool

        self.register_buffer("adj", adj)

        self.convs = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(gnn_layers):
            self.convs.append(DenseGCNConv(in_dim, gnn_hidden))
            in_dim = gnn_hidden

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError(
                f"expected node_features (B, N, F), got {tuple(node_features.shape)}"
            )
        if node_features.shape[1] != self.n_nodes:
            raise ValueError(
                f"expected N={self.n_nodes}, got {node_features.shape[1]}"
            )

        x = node_features
        last_idx = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            x = conv(x, self.adj)             # (B, N, gnn_hidden)
            x = F.relu(x)
            if i < last_idx:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "add":
            return x.sum(dim=1)
        return x.max(dim=1).values


class GAT(nn.Module):
    """Stacked GAT branch (placeholder).

    Reserved for the ablation step. Will mirror the GCN interface:
    `(B, N, F)` node features -> `(B, gnn_hidden)` graph embedding.

    Unlike GCN, GAT does not use edge weights; attention is learned over
    `edge_index` (so we will need to convert the dense adjacency to
    `edge_index` once at construction time). Plan: use
    `torch_geometric.nn.dense.DenseGATConv` with configurable heads and a
    last-layer head average (concat=False) to keep the output dim equal to
    gnn_hidden.

    Args (planned)
    --------------
    n_nodes, node_feat_dim, gnn_hidden, gnn_layers, dropout, adjacency, pool,
    heads (multi-head attention), attention_dropout.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # TODO(stage3-ablation): implement DenseGATConv stack with multi-head
        # attention. Until then, calling GAT raises.
        raise NotImplementedError("GAT branch is reserved for the ablation step.")

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
