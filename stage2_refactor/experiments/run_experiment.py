from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from stage2_refactor.data.dataloader import build_final_test_windows, build_train_val_loaders
from stage2_refactor.data.io import read_cmapss_table, read_rul
from stage2_refactor.data.preprocessing import (
    PreprocessingParams,
    fit_transform_train,
    load_preprocessing_artifact,
    transform_with_artifact,
)
from stage2_refactor.models.lstm import LSTMBaseline
from stage2_refactor.training.evaluator import count_parameters, rmse_score
from stage2_refactor.training.logging import CSVLogger, WandbLogger
from stage2_refactor.training.trainer import fit, set_seed


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def path_from_root(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def git_commit_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_model(input_size: int, model_cfg: dict[str, Any]) -> LSTMBaseline:
    if model_cfg.get("name", "lstm") != "lstm":
        raise ValueError("Stage 2.0 only supports the LSTM baseline.")
    return LSTMBaseline(
        input_size=input_size,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )


def merged_model_training_cfg(config: dict[str, Any], dataset_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_cfg = dict(config["model"])
    training_cfg = dict(config["training"])
    for key in ["hidden_size", "num_layers", "dropout"]:
        if key in dataset_cfg:
            model_cfg[key] = dataset_cfg[key]
    for key in ["batch_size", "learning_rate", "epochs"]:
        if key in dataset_cfg:
            training_cfg[key] = dataset_cfg[key]
    return model_cfg, training_cfg


@torch.no_grad()
def evaluate_test(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_true: np.ndarray,
    initial_rul: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true, dtype=torch.float32).reshape(-1, 1).to(device)
    y_pred = model(x_tensor)
    y_pred = torch.clamp(y_pred, min=0.0, max=float(initial_rul))
    y_tensor = torch.clamp(y_tensor, min=0.0, max=float(initial_rul))
    rmse, score = rmse_score(y_pred, y_tensor)
    return {"test_rmse": rmse, "test_score": score}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve() if args.project_root else repo_root_from_script()
    config = load_config(path_from_root(root, args.config))
    subset = args.subset.upper()
    dataset_cfg = dict(config["datasets"][subset])
    model_cfg, training_cfg = merged_model_training_cfg(config, dataset_cfg)

    if args.max_epochs is not None:
        training_cfg["epochs"] = args.max_epochs
    if args.seed is not None:
        training_cfg["seed"] = args.seed

    output_dir = path_from_root(root, args.output_dir or config["paths"]["output_dir"]) / subset
    checkpoint_dir = path_from_root(root, args.checkpoint_dir or config["paths"]["checkpoint_dir"]) / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = checkpoint_dir / f"{subset.lower()}_preprocessing.gz"
    external_artifact_path = (
        path_from_root(root, args.preprocessing_artifact_path)
        if args.preprocessing_artifact_path
        else None
    )
    best_model_path = checkpoint_dir / f"{subset.lower()}_lstm_best.pth"
    last_checkpoint_path = checkpoint_dir / f"{subset.lower()}_last_checkpoint.pt"
    best_checkpoint_path = checkpoint_dir / f"{subset.lower()}_best_checkpoint.pt"
    log_path = output_dir / "train_log.csv"
    summary_path = output_dir / "summary.json"

    seed = int(training_cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = path_from_root(root, dataset_cfg["train_file"])
    test_file = path_from_root(root, dataset_cfg["test_file"])
    rul_file = path_from_root(root, dataset_cfg["rul_file"])

    preprocessing_cfg = config["preprocessing"]
    params = PreprocessingParams(
        drop_cols=list(dataset_cfg["drop_cols"]),
        median_window=int(preprocessing_cfg["median_window"]),
        kmeans_clusters=int(preprocessing_cfg["kmeans_clusters"]),
        kmeans_random_state=int(preprocessing_cfg["kmeans_random_state"]),
        kmeans_n_init=int(preprocessing_cfg["kmeans_n_init"]),
        rul_window_size=int(dataset_cfg["rul_window_size"]),
        rul_threshold=float(dataset_cfg["rul_threshold"]),
        rul_patience=int(dataset_cfg["rul_patience"]),
    )

    if args.mode == "eval" and external_artifact_path is not None:
        artifact = load_preprocessing_artifact(external_artifact_path)
        df_train_raw = read_cmapss_table(train_file)
        df_train, _ = fit_transform_train(df_train_raw, params, artifact_path=None)
    else:
        df_train_raw = read_cmapss_table(train_file)
        df_train, artifact = fit_transform_train(df_train_raw, params, scaler_path)
    features = artifact["features"]
    initial_rul = int(artifact["initial_rul"])
    input_size = len(features)

    train_loader, val_loader, split = build_train_val_loaders(
        df=df_train,
        features=features,
        sequence_length=int(preprocessing_cfg["sequence_length"]),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=bool(training_cfg["shuffle"]),
        drop_last=bool(training_cfg["drop_last"]),
        validation_enabled=bool(config["validation"]["enabled"]),
        validation_fraction=float(config["validation"]["fraction"]),
        seed=seed,
    )

    model = build_model(input_size, model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = nn.MSELoss()
    metadata = {
        "subset": subset,
        "model": model_cfg["name"],
        "seed": seed,
        "git_commit": git_commit_hash(root),
        "input_size": input_size,
        "features": features,
        "initial_rul": initial_rul,
        "sequence_length": int(preprocessing_cfg["sequence_length"]),
        "validation_enabled": bool(config["validation"]["enabled"]),
        "parameter_count": count_parameters(model),
    }

    result: dict[str, Any] = {
        **metadata,
        "train_sequences": len(train_loader.dataset),
        "val_sequences": len(val_loader.dataset) if val_loader is not None else 0,
        "train_engine_ids": split.train_engine_ids,
        "val_engine_ids": split.val_engine_ids,
        "best_model_path": str(best_model_path),
        "preprocessing_artifact_path": str(scaler_path),
    }

    if args.mode in {"train", "train_eval"}:
        csv_logger = CSVLogger(log_path)
        wandb_logger = WandbLogger(
            enabled=bool(args.use_wandb),
            project=config["project_name"],
            run_name=f"{subset}-{model_cfg['name']}-seed{seed}",
            config={**config, "resolved_metadata": metadata},
        )
        patience = int(config["validation"]["patience"]) if val_loader is not None else None
        fit_result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=int(training_cfg["epochs"]),
            patience=patience,
            best_model_path=best_model_path,
            last_checkpoint_path=last_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            metadata=metadata,
            csv_logger=csv_logger,
            wandb_logger=wandb_logger,
            resume_checkpoint_path=last_checkpoint_path if args.resume else None,
        )
        wandb_logger.finish()
        result.update(
            {
                "best_metric": fit_result.best_metric,
                "best_epoch": fit_result.best_epoch,
                "total_train_seconds": fit_result.total_seconds,
            }
        )

    if args.mode in {"eval", "train_eval"}:
        artifact = load_preprocessing_artifact(external_artifact_path or scaler_path)
        df_test = read_cmapss_table(test_file)
        df_test = transform_with_artifact(
            df_test,
            artifact,
            apply_median_filter_to_data=bool(preprocessing_cfg["apply_median_filter_to_test"]),
        )
        x_test = build_final_test_windows(
            df_test,
            artifact["features"],
            int(preprocessing_cfg["sequence_length"]),
        )
        y_true = read_rul(rul_file).to_numpy(dtype=np.float32)

        if args.model_path:
            model.load_state_dict(torch.load(path_from_root(root, args.model_path), map_location=device))
        elif best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            raise FileNotFoundError(f"No model found at {best_model_path}.")

        result.update(evaluate_test(model, x_test, y_true, int(artifact["initial_rul"]), device))
        result["test_windows_shape"] = list(x_test.shape)

    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage2_refactor/configs/lstm_baseline.yaml")
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--mode", default="train_eval", choices=["train", "eval", "train_eval"])
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--preprocessing-artifact-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
