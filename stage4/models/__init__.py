"""Stage 4 model module exports.

`AttnRecurrentGNN` is imported lazily so that environments without
`torch_geometric` (the local CPU sandbox used for shape checks) can still
exercise the non-GNN code path. Training in Colab pulls PyG explicitly.
"""

from stage4.models.attention import CBAM1D, ChannelAttention, TemporalAttention
from stage4.models.attn_recurrent import AttnRecurrent

__all__ = [
    "AttnRecurrent",
    "AttnRecurrentGNN",
    "CBAM1D",
    "ChannelAttention",
    "TemporalAttention",
]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "AttnRecurrentGNN":
        from stage4.models.attn_recurrent_gnn import AttnRecurrentGNN
        return AttnRecurrentGNN
    raise AttributeError(f"module 'stage4.models' has no attribute {name!r}")
