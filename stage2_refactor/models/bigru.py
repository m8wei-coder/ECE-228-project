from __future__ import annotations

from stage2_refactor.models.recurrent import RecurrentBaseline


class BiGRUBaseline(RecurrentBaseline):
    """Bidirectional GRU variant with the Stage 1 regression head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            recurrent_type="gru",
            bidirectional=True,
            module_name="gru",
        )
