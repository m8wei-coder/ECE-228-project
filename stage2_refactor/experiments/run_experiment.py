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
from stage2_refactor.models.bigru import BiGRUBaseline
from stage2_refactor.models.bilstm import BiLSTMBaseline
from stage2_refactor.models.gru import GRUBaseline
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


def format_run_value(value: Any) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def build_run_id(model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> str:
    return "_".join(
        [
            f"seed{format_run_value(training_cfg['seed'])}",
            f"h{format_run_value(model_cfg['hidden_size'])}",
            f"l{format_run_value(model_cfg['num_layers'])}",
            f"d{format_run_value(model_cfg['dropout'])}",
            f"lr{format_run_value(training_cfg['learning_rate'])}",
            f"bs{format_run_value(training_cfg['batch_size'])}",
        ]
    )


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


def build_model(input_size: int, model_cfg: dict[str, Any]) -> torch.nn.Module:
    model_name = model_cfg.get("name", "lstm")
    model_classes = {
        "lstm": LSTMBaseline,
        "gru": GRUBaseline,
        "bilstm": BiLSTMBaseline,
        "bigru": BiGRUBaseline,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model {model_name}. Choose from {sorted(model_classes)}.")

    return model_classes[model_name](
        input_size=input_size,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )


def merged_model_training_cfg(
    config: dict[str, Any],
    dataset_cfg: dict[str, Any],
    selected_model_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "models" in config and selected_model_name in config["models"]:
        model_cfg = dict(config["models"][selected_model_name])
    else:
        model_cfg = dict(config["model"])
        model_cfg["name"] = selected_model_name

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
    if args.apply_median_filter_to_test is not None:
        config["preprocessing"]["apply_median_filter_to_test"] = args.apply_median_filter_to_test
    if args.validation_enabled is not None:
        config["validation"]["enabled"] = args.validation_enabled
    if args.validation_fraction is not None:
        config["validation"]["fraction"] = args.validation_fraction
    if args.validation_patience is not None:
        config["validation"]["patience"] = args.validation_patience
    subset = args.subset.upper()
    dataset_cfg = dict(config["datasets"][subset])
    selected_model_name = args.model or config.get("model", {}).get("name", "lstm")
    model_cfg, training_cfg = merged_model_training_cfg(config, dataset_cfg, selected_model_name)

    if args.max_epochs is not None:
        training_cfg["epochs"] = args.max_epochs
    if args.seed is not None:
        training_cfg["seed"] = args.seed
    if args.hidden_size is not None:
        model_cfg["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        model_cfg["num_layers"] = args.num_layers
    if args.dropout is not None:
        model_cfg["dropout"] = args.dropout
    if args.learning_rate is not None:
        training_cfg["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size

    model_name = model_cfg["name"]
    run_id = args.run_id or build_run_id(model_cfg, training_cfg)
    output_dir = path_from_root(root, args.output_dir or config["paths"]["output_dir"]) / model_name / subset / run_id
    checkpoint_dir = path_from_root(root, args.checkpoint_dir or config["paths"]["checkpoint_dir"]) / model_name / subset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = checkpoint_dir / f"{subset.lower()}_preprocessing.gz"
    external_artifact_path = (
        path_from_root(root, args.preprocessing_artifact_path)
        if args.preprocessing_artifact_path
        else None
    )
    best_model_path = checkpoint_dir / f"{subset.lower()}_{model_name}_best.pth"
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

    train_loader = None
    val_loader = None
    split = None

    if args.mode == "eval":
        artifact = load_preprocessing_artifact(external_artifact_path or scaler_path)
    else:
        df_train_raw = read_cmapss_table(train_file)
        df_train, artifact = fit_transform_train(df_train_raw, params, scaler_path)
        train_loader, val_loader, split = build_train_val_loaders(
            df=df_train,
            features=artifact["features"],
            sequence_length=int(preprocessing_cfg["sequence_length"]),
            batch_size=int(training_cfg["batch_size"]),
            shuffle=bool(training_cfg["shuffle"]),
            drop_last=bool(training_cfg["drop_last"]),
            validation_enabled=bool(config["validation"]["enabled"]),
            validation_fraction=float(config["validation"]["fraction"]),
            seed=seed,
        )
    features = artifact["features"]
    initial_rul = int(artifact["initial_rul"])
    input_size = len(features)

    model = build_model(input_size, model_cfg).to(device)
    criterion = nn.MSELoss()
    metadata = {
        "subset": subset,
        "model": model_cfg["name"],
        "run_id": run_id,
        "seed": seed,
        "git_commit": git_commit_hash(root),
        "input_size": input_size,
        "hidden_size": int(model_cfg["hidden_size"]),
        "num_layers": int(model_cfg["num_layers"]),
        "dropout": float(model_cfg["dropout"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "batch_size": int(training_cfg["batch_size"]),
        "features": features,
        "initial_rul": initial_rul,
        "sequence_length": int(preprocessing_cfg["sequence_length"]),
        "validation_enabled": bool(config["validation"]["enabled"]),
        "parameter_count": count_parameters(model),
    }

    result: dict[str, Any] = {
        **metadata,
        "train_sequences": len(train_loader.dataset) if train_loader is not None else None,
        "val_sequences": len(val_loader.dataset) if val_loader is not None else 0,
        "train_engine_ids": split.train_engine_ids if split is not None else [],
        "val_engine_ids": split.val_engine_ids if split is not None else [],
        "best_model_path": str(best_model_path),
        "preprocessing_artifact_path": str(external_artifact_path or scaler_path),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }

    if args.mode in {"train", "train_eval"}:
        if train_loader is None:
            raise RuntimeError("train_loader was not initialized for training mode.")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
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
    parser.add_argument("--model", choices=["lstm", "gru", "bilstm", "bigru"], default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--validation-enabled",
        dest="validation_enabled",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--validation-disabled",
        dest="validation_enabled",
        action="store_false",
    )
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--validation-patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--apply-median-filter-to-test",
        dest="apply_median_filter_to_test",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-apply-median-filter-to-test",
        dest="apply_median_filter_to_test",
        action="store_false",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
