from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CMAPSS_COLUMNS = (
    ["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{idx}" for idx in range(1, 22)]
)


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if base_dir is None:
        return candidate
    return Path(base_dir).expanduser().resolve() / candidate


def read_cmapss_table(path: str | Path) -> pd.DataFrame:
    """Read either NASA C-MAPSS whitespace TXT files or CSV exports."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if set(CMAPSS_COLUMNS).issubset(df.columns):
            return df
        if df.shape[1] == len(CMAPSS_COLUMNS):
            df.columns = CMAPSS_COLUMNS
            return df
        raise ValueError(f"CSV {path} does not match the C-MAPSS schema.")

    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def read_rul(path: str | Path) -> pd.Series:
    """Read a C-MAPSS RUL file from whitespace TXT or one-column CSV."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.iloc[:, 0].astype("float32")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df.iloc[:, 0].astype("float32")


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

