"""Model backbones that implement the BaseModel contract."""

from stage2_refactor.models.bigru import BiGRUBaseline
from stage2_refactor.models.bilstm import BiLSTMBaseline
from stage2_refactor.models.gru import GRUBaseline
from stage2_refactor.models.lstm import LSTMBaseline
from stage2_refactor.models.recurrent import RecurrentBaseline

MODEL_REGISTRY = {
    "lstm": LSTMBaseline,
    "gru": GRUBaseline,
    "bilstm": BiLSTMBaseline,
    "bigru": BiGRUBaseline,
}


def build_baseline_model(
    name: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> RecurrentBaseline:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name}. Choose from {sorted(MODEL_REGISTRY)}.")

    return MODEL_REGISTRY[name](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

__all__ = [
    "BiGRUBaseline",
    "BiLSTMBaseline",
    "GRUBaseline",
    "LSTMBaseline",
    "MODEL_REGISTRY",
    "RecurrentBaseline",
    "build_baseline_model",
]
