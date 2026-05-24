"""CBAM-style 1D attention modules for C-MAPSS sensor-time tensors.

Designed to drop in front of the Stage 2 recurrent backbone / Stage 3
RecurrentGNNFusion. Input and output shapes are both `(B, T, F)` so the
downstream model code requires no changes.

Adapted from:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Adaptation notes
----------------
The original CBAM is defined for 2-D feature maps (B, C, H, W). For C-MAPSS
we have 1-D windows (B, T, F):
- "channels" are sensors (F);
- "spatial" is the time axis (T).

ChannelAttention -> "which sensors matter":
    pool along time (avg + max) -> shared MLP (F -> F/r -> F) ->
    add -> sigmoid -> gate of shape (B, 1, F) broadcast over T.

TemporalAttention -> "which time steps matter":
    pool along channels (avg + max) -> 1-D conv (kernel=7) ->
    sigmoid -> gate of shape (B, T, 1) broadcast over F.

CBAM1D applies channel-then-temporal in series, matching the CBAM paper's
ordering. Parameter count is dominated by the channel MLP (~2 * F * F / r)
and is well under 1k for F<=21, r=4 -- consistent with the project's
"lightweight augmentation" requirement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style gate over the sensor (channel) axis.

    Input  : (B, T, F)
    Output : (B, T, F)  -- same shape, multiplied element-wise by a per-sensor
                          scalar gate broadcast across the time axis.
    """

    def __init__(self, num_channels: int, reduction: int = 4) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        # Avoid a zero-width hidden layer when F is tiny.
        hidden = max(1, num_channels // reduction)
        self.num_channels = num_channels
        self.reduction = reduction
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, F), got {tuple(x.shape)}")
        if x.shape[2] != self.num_channels:
            raise ValueError(
                f"expected F={self.num_channels}, got {x.shape[2]}"
            )

        # Pool along time axis -> (B, F).
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

        gate = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))  # (B, F)
        gate = gate.unsqueeze(1)  # (B, 1, F) -> broadcast over T
        return x * gate


class TemporalAttention(nn.Module):
    """Spatial (time) attention via avg+max channel pooling and a 1-D conv.

    Input  : (B, T, F)
    Output : (B, T, F)  -- same shape, multiplied element-wise by a per-step
                          scalar gate broadcast across the channel axis.

    The 1-D conv operates on a 2-channel input (avg+max pooled along F),
    so the kernel size is the only meaningful hyperparameter.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        self.kernel_size = kernel_size
        # 2 input channels (avg + max) -> 1 output channel (the gate logits)
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, F), got {tuple(x.shape)}")

        # Pool along channel axis -> (B, T) twice, then stack as (B, 2, T).
        avg_pool = x.mean(dim=2)
        max_pool = x.max(dim=2).values
        pooled = torch.stack([avg_pool, max_pool], dim=1)  # (B, 2, T)

        gate = torch.sigmoid(self.conv(pooled))  # (B, 1, T)
        gate = gate.transpose(1, 2)              # (B, T, 1) -> broadcast over F
        return x * gate


class CBAM1D(nn.Module):
    """Channel-then-temporal attention block for (B, T, F) windows.

    Args:
        num_channels: F, number of retained sensors.
        reduction: ChannelAttention MLP reduction ratio (>=1).
        kernel_size: TemporalAttention 1-D conv kernel (positive odd int).
        use_channel: enable the channel-attention sub-module.
        use_temporal: enable the temporal-attention sub-module.
        attn_dropout: dropout probability applied to the post-gated features
            after each enabled sub-module. 0.0 disables (default), preserving
            backward compatibility with earlier Stage 4 runs.
        order: sub-module application order. "channel_first" (default) matches
            the CBAM paper; "temporal_first" swaps them so temporal attention
            runs before channel attention.

    When both `use_channel` and `use_temporal` are False the module is a
    no-op pass-through, which is the natural ablation control (Stage 3
    baseline).
    """

    _VALID_ORDERS = ("channel_first", "temporal_first")

    def __init__(
        self,
        num_channels: int,
        reduction: int = 4,
        kernel_size: int = 7,
        use_channel: bool = True,
        use_temporal: bool = True,
        attn_dropout: float = 0.0,
        order: str = "channel_first",
    ) -> None:
        super().__init__()
        if order not in self._VALID_ORDERS:
            raise ValueError(
                f"order must be one of {self._VALID_ORDERS}, got {order!r}"
            )
        if not 0.0 <= attn_dropout < 1.0:
            raise ValueError(
                f"attn_dropout must be in [0.0, 1.0), got {attn_dropout}"
            )
        self.use_channel = use_channel
        self.use_temporal = use_temporal
        self.attn_dropout_p = float(attn_dropout)
        self.order = order
        if use_channel:
            self.channel_attn = ChannelAttention(num_channels, reduction=reduction)
        else:
            self.channel_attn = None
        if use_temporal:
            self.temporal_attn = TemporalAttention(kernel_size=kernel_size)
        else:
            self.temporal_attn = None
        # Single shared Dropout layer is fine: Dropout has no learnable params
        # and re-applying it twice (after channel, after temporal) just gives
        # two independent dropout masks, which is exactly what we want.
        self.dropout = (
            nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.channel_attn if self.order == "channel_first" else self.temporal_attn
        second = self.temporal_attn if self.order == "channel_first" else self.channel_attn
        if first is not None:
            x = first(x)
            x = self.dropout(x)
        if second is not None:
            x = second(x)
            x = self.dropout(x)
        return x

    @staticmethod
    def parameter_count(num_channels: int, reduction: int = 4) -> int:
        """Theoretical parameter count for a full CBAM1D at given F, r.

        Useful for logging / sanity-checking the "lightweight" requirement.
        """
        hidden = max(1, num_channels // reduction)
        channel_params = 2 * num_channels * hidden  # two Linear layers, bias=False
        temporal_params = 2 * 1 * 7                 # 2-in-1-out conv, k=7, bias=False
        return channel_params + temporal_params
