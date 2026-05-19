from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SETTINGS_COLS = ["setting_1", "setting_2", "setting_3"]


@dataclass(frozen=True)
class PreprocessingParams:
    drop_cols: list[str]
    median_window: int = 5
    kmeans_clusters: int = 6
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10
    rul_window_size: int = 12
    rul_threshold: float = 0.2
    rul_patience: int = 1


def add_absolute_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rul = pd.DataFrame(df.groupby("unit_number")["time_cycles"].max()).reset_index()
    rul.columns = ["unit_number", "max_cycle"]
    df = df.merge(rul, on="unit_number", how="left")
    df["RUL_absolute"] = df["max_cycle"] - df["time_cycles"]
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def selected_sensor_columns(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    return [col for col in sensor_cols if col not in drop_cols]


def apply_median_filter(
    df: pd.DataFrame,
    features: list[str],
    median_window: int,
) -> pd.DataFrame:
    df = df.copy()
    for engine_id in df["unit_number"].unique():
        idx = df["unit_number"] == engine_id
        for col in features:
            df.loc[idx, col] = medfilt(df.loc[idx, col].values, kernel_size=median_window)
    return df


def fit_preprocessing(
    df: pd.DataFrame,
    params: PreprocessingParams,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit Stage 1 preprocessing objects and transform the training frame."""
    df = df.copy()
    features = selected_sensor_columns(df, params.drop_cols)

    df = apply_median_filter(df, features, params.median_window)
    df[features] = df[features].astype(float)

    kmeans = KMeans(
        n_clusters=params.kmeans_clusters,
        random_state=params.kmeans_random_state,
        n_init=params.kmeans_n_init,
    )
    df["condition"] = kmeans.fit_predict(df[SETTINGS_COLS])

    scalers: dict[int, StandardScaler] = {}
    for condition in range(params.kmeans_clusters):
        idx = df["condition"] == condition
        if idx.sum() == 0:
            continue
        scaler = StandardScaler()
        df.loc[idx, features] = scaler.fit_transform(df.loc[idx, features])
        scalers[condition] = scaler

    df.drop("condition", axis=1, inplace=True)

    artifact: dict[str, Any] = {
        "kmeans": kmeans,
        "scalers": scalers,
        "features": features,
        "drop_cols": params.drop_cols,
        "median_window": params.median_window,
        "kmeans_clusters": params.kmeans_clusters,
        "kmeans_random_state": params.kmeans_random_state,
        "kmeans_n_init": params.kmeans_n_init,
        "window_size": params.rul_window_size,
        "threshold": params.rul_threshold,
        "patience": params.rul_patience,
    }
    return df, artifact


def calculate_piecewise_rul(
    df: pd.DataFrame,
    features: list[str],
    w: int = 12,
    th: float = 0.2,
    patience: int = 1,
) -> tuple[pd.DataFrame, int]:
    """Replicate the Stage 1 knee-point-based initial RUL calculation."""
    df = df.copy()
    irul_list: list[float] = []

    for engine_id in df["unit_number"].unique():
        engine_data = df[df["unit_number"] == engine_id].sort_values("time_cycles")
        group_count = len(engine_data) // w

        if group_count < 2:
            irul_list.append(float(engine_data["RUL_absolute"].max()))
            continue

        x_values = engine_data[features].values
        centroids = np.array(
            [np.mean(x_values[i * w : (i + 1) * w], axis=0) for i in range(group_count)]
        )

        found_knee = False
        consecutive_count = 0
        for i in range(1, group_count):
            prev_mean = np.mean(centroids[:i], axis=0)
            dist = np.linalg.norm(prev_mean - centroids[i]) / (np.linalg.norm(prev_mean) + 1e-8)

            if dist > th:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= patience:
                knee_i = i - patience + 1
                knee_point_cycle = (knee_i + 1) * w
                max_cycle = engine_data["time_cycles"].max()
                irul_list.append(float(max_cycle - knee_point_cycle))
                found_knee = True
                break

        if not found_knee:
            irul_list.append(float(engine_data["RUL_absolute"].max()))

    global_initial_rul = int(np.min(irul_list))
    df["RUL_piecewise"] = df["RUL_absolute"].clip(upper=global_initial_rul)
    return df, global_initial_rul


def fit_transform_train(
    df_train: pd.DataFrame,
    params: PreprocessingParams,
    artifact_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df_train = add_absolute_rul(df_train)
    df_train, artifact = fit_preprocessing(df_train, params)
    df_train, initial_rul = calculate_piecewise_rul(
        df_train,
        artifact["features"],
        w=params.rul_window_size,
        th=params.rul_threshold,
        patience=params.rul_patience,
    )
    artifact["initial_rul"] = initial_rul

    if artifact_path is not None:
        save_preprocessing_artifact(artifact, artifact_path)

    return df_train, artifact


def transform_with_artifact(
    df: pd.DataFrame,
    artifact: dict[str, Any],
    apply_median_filter_to_data: bool = False,
) -> pd.DataFrame:
    """Apply saved Stage 1 preprocessing to train or test data."""
    df = df.copy()
    features = artifact["features"]

    if apply_median_filter_to_data:
        df = apply_median_filter(df, features, artifact.get("median_window", 5))

    df[features] = df[features].astype(float)
    df["condition"] = artifact["kmeans"].predict(df[SETTINGS_COLS])

    clusters = artifact.get("kmeans_clusters", 6)
    for condition in range(clusters):
        idx = df["condition"] == condition
        if idx.sum() == 0:
            continue
        if condition not in artifact["scalers"]:
            raise KeyError(f"Scaler for condition {condition} not found in artifact.")
        df.loc[idx, features] = artifact["scalers"][condition].transform(df.loc[idx, features])

    df.drop("condition", axis=1, inplace=True)
    return df


def save_preprocessing_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_preprocessing_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = ["kmeans", "scalers", "features", "initial_rul"]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"Missing keys in preprocessing artifact {path}: {missing}")
    return artifact

