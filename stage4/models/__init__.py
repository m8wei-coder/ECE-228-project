from stage4.models.attention import CBAM1D, ChannelAttention, TemporalAttention
from stage4.models.attn_recurrent import AttnRecurrent

__all__ = [
    "AttnRecurrent",
    "AttnRecurrentGNN",
    "CBAM1D",
    "ChannelAttention",
    "TemporalAttention",
]


def __getattr__(name):
    if name == "AttnRecurrentGNN":
        from stage4.models.attn_recurrent_gnn import AttnRecurrentGNN
        return AttnRecurrentGNN
    raise AttributeError(f"module 'stage4.models' has no attribute {name!r}")
