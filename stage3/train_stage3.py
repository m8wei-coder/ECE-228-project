"""Stage 3 training entry.

Pure orchestration: every heavy lifting block — data pipeline, training loop,
test-time evaluation with `clamp(0, initial_rul)` — is reused verbatim from
stage2_refactor. We only assemble the RecurrentGNNFusion model and hand it
to the teammate's `fit()` function.

Protocol (aligned with Stage 2 finalists):
    sequence_length = 30
    validation_enabled = True (engine-level 15% split)
    apply_median_filter_to_test = False
    test eval: clamp y_pred & y_true to [0, initial_rul] then rmse_score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_refactor.data.io import read_cmapss_table, read_rul
from stage2_refactor.data.preprocessing import (
    PreprocessingParams,
    fit_transform_train,
    transform_with_artifact,
)
from stage2_refactor.data.dataloader import (
    build_final_test_windows,
    build_train_val_loaders,
)
from stage2_refactor.training.trainer import fit, set_seed
from stage2_refactor.training.evaluator import count_parameters
from stage2_refactor.training.logging import CSVLogger
from stage2_refactor.experiments.run_experiment import evaluate_test

from stage3.models.recurrent_gnn import RecurrentGNNFusion


SEQUENCE_LENGTH = 30
VALIDATION_ENABLED = True
VALIDATION_FRACTION = 0.15
VALIDATION_PATIENCE = 10
APPLY_MEDIAN_FILTER_TO_TEST = False
WEIGHT_DECAY = 1.0e-5


# Per-subset finalist config (mirrors stage3/configs/stage3.yaml).
SUBSET_CONFIGS: dict[str, dict[str, Any]] = {
    "FD001": {
        "train_file": "CMaps/train_FD001.txt",
        "test_file":  "CMaps/test_FD001.txt",
        "rul_file":   "CMaps/RUL_FD001.txt",
        "drop_cols":  ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                       "sensor_16", "sensor_18", "sensor_19"],
        "rul_window_size": 12, "rul_threshold": 0.2, "rul_patience": 1,
        "recurrent_kind": "gru",   "hidden_size": 90, "num_layers": 2,
        "dropout": 0.2, "learning_rate": 5.0e-4, "batch_size": 15, "epochs": 150,
    },
    "FD002": {
        "train_file": "CMaps/train_FD002.txt",
        "test_file":  "CMaps/test_FD002.txt",
        "rul_file":   "CMaps/RUL_FD002.txt",
        "drop_cols":  [],
        "rul_window_size": 12, "rul_threshold": 0.2, "rul_patience": 1,
        "recurrent_kind": "bigru", "hidden_size": 60, "num_layers": 2,
        "dropout": 0.1, "learning_rate": 5.0e-4, "batch_size": 15, "epochs": 150,
    },
    "FD003": {
        "train_file": "CMaps/train_FD003.txt",
        "test_file":  "CMaps/test_FD003.txt",
        "rul_file":   "CMaps/RUL_FD003.txt",
        "drop_cols":  ["sensor_1", "sensor_5",
                       "sensor_16", "sensor_18", "sensor_19"],
        "rul_window_size": 12, "rul_threshold": 0.2, "rul_patience": 2,
        "recurrent_kind": "gru",   "hidden_size": 90, "num_layers": 2,
        "dropout": 0.2, "learning_rate": 5.0e-4, "batch_size": 20, "epochs": 250,
    },
    "FD004": {
        "train_file": "CMaps/train_FD004.txt",
        "test_file":  "CMaps/test_FD004.txt",
        "rul_file":   "CMaps/RUL_FD004.txt",
        "drop_cols":  [],
        "rul_window_size": 12, "rul_threshold": 0.3, "rul_patience": 3,
        "recurrent_kind": "bigru", "hidden_size": 60, "num_layers": 2,
        "dropout": 0.1, "learning_rate": 5.0e-4, "batch_size": 10, "epochs": 150,
    },
}


def load_adjacency(subset: str, method: str, artifacts_dir: Path) -> torch.Tensor:
    path = artifacts_dir / f"adj_{subset}_{method}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency not found: {path}. "
            f"Run `python -m stage3.build_graph` first."
        )
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr)


def prepare_data(
    subset_cfg: dict, repo_root: Path, scaler_path: Path,
    seed: int, batch_size: int,
):
    """Run the Stage 2 data pipeline end-to-end for one subset.

    Returns:
        train_loader, val_loader, x_test, y_test, artifact
    """
    train_path = repo_root / subset_cfg["train_file"]
    test_path  = repo_root / subset_cfg["test_file"]
    rul_path   = repo_root / subset_cfg["rul_file"]

    params = PreprocessingParams(
        drop_cols=list(subset_cfg["drop_cols"]),
        rul_window_size=int(subset_cfg["rul_window_size"]),
        rul_threshold=float(subset_cfg["rul_threshold"]),
        rul_patience=int(subset_cfg["rul_patience"]),
    )
    df_train_raw = read_cmapss_table(train_path)
    df_train, artifact = fit_transform_train(df_train_raw, params, scaler_path)

    train_loader, val_loader, _split = build_train_val_loaders(
        df=df_train,
        features=artifact["features"],
        sequence_length=SEQUENCE_LENGTH,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        validation_enabled=VALIDATION_ENABLED,
        validation_fraction=VALIDATION_FRACTION,
        seed=seed,
    )

    df_test = read_cmapss_table(test_path)
    df_test = transform_with_artifact(
        df_test, artifact,
        apply_median_filter_to_data=APPLY_MEDIAN_FILTER_TO_TEST,
    )
    x_test = build_final_test_windows(
        df_test, artifact["features"], SEQUENCE_LENGTH,
    )
    y_test = read_rul(rul_path).to_numpy(dtype=np.float32)

    return train_loader, val_loader, x_test, y_test, artifact


def build_model(
    cfg: dict, input_size: int, adj: torch.Tensor | None,
    use_gnn: bool, gnn_kind: str, gnn_hidden: int, gnn_layers: int,
    gnn_dropout: float, gnn_pool: str,
) -> RecurrentGNNFusion:
    return RecurrentGNNFusion(
        input_size=input_size,
        recurrent_kind=cfg["recurrent_kind"],
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        sequence_length=SEQUENCE_LENGTH,
        adj_matrix=adj if use_gnn else None,
        gnn_hidden=gnn_hidden,
        gnn_layers=gnn_layers,
        gnn_kind=gnn_kind,
        gnn_dropout=gnn_dropout,
        gnn_pool=gnn_pool,
        use_gnn=use_gnn,
    )


def run(args: argparse.Namespace) -> dict:
    subset = args.subset.upper()
    if subset not in SUBSET_CONFIGS:
        raise ValueError(f"unknown subset: {subset}")

    cfg = dict(SUBSET_CONFIGS[subset])
    # CLI overrides
    for key, val in [
        ("recurrent_kind", args.recurrent_kind),
        ("hidden_size", args.hidden_size),
        ("num_layers", args.num_layers),
        ("dropout", args.dropout),
        ("learning_rate", args.learning_rate),
        ("batch_size", args.batch_size),
        ("epochs", args.epochs),
    ]:
        if val is not None:
            cfg[key] = val

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts_dir = (
        Path(args.artifacts_dir) if args.artifacts_dir
        else REPO_ROOT / "stage3" / "artifacts"
    )
    run_id = args.run_id or f"{subset.lower()}_seed{args.seed}"
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else artifacts_dir / "runs" / subset / run_id
    )
    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir
        else output_dir / "checkpoints"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler_path     = checkpoint_dir / f"{subset.lower()}_preprocessing.gz"
    best_model_path = checkpoint_dir / f"{subset.lower()}_rgnn_best.pth"
    last_ckpt_path  = checkpoint_dir / f"{subset.lower()}_last.pt"
    best_ckpt_path  = checkpoint_dir / f"{subset.lower()}_best.pt"
    log_path        = output_dir / "train_log.csv"
    summary_path    = output_dir / "summary.json"

    train_loader, val_loader, x_test, y_test, artifact = prepare_data(
        cfg, REPO_ROOT, scaler_path, args.seed, int(cfg["batch_size"]),
    )
    features = artifact["features"]
    input_size = len(features)
    initial_rul = int(artifact["initial_rul"])

    adj = None
    if args.use_gnn:
        adj = load_adjacency(subset, args.graph_method, artifacts_dir)
        if adj.shape != (input_size, input_size):
            raise ValueError(
                f"adjacency for {subset}/{args.graph_method} has shape "
                f"{tuple(adj.shape)} but data F={input_size}"
            )

    model = build_model(
        cfg, input_size, adj,
        use_gnn=args.use_gnn,
        gnn_kind=args.gnn_kind,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        gnn_dropout=args.gnn_dropout,
        gnn_pool=args.gnn_pool,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=WEIGHT_DECAY,
    )

    metadata = {
        "subset": subset,
        "run_id": run_id,
        "seed": args.seed,
        "use_gnn": bool(args.use_gnn),
        "graph_method": args.graph_method if args.use_gnn else None,
        "gnn_kind": args.gnn_kind if args.use_gnn else None,
        "gnn_hidden": args.gnn_hidden if args.use_gnn else None,
        "gnn_layers": args.gnn_layers if args.use_gnn else None,
        "gnn_dropout": args.gnn_dropout if args.use_gnn else None,
        "gnn_pool": args.gnn_pool if args.use_gnn else None,
        "recurrent_kind": cfg["recurrent_kind"],
        "hidden_size": int(cfg["hidden_size"]),
        "num_layers": int(cfg["num_layers"]),
        "dropout": float(cfg["dropout"]),
        "learning_rate": float(cfg["learning_rate"]),
        "batch_size": int(cfg["batch_size"]),
        "epochs": int(cfg["epochs"]),
        "weight_decay": WEIGHT_DECAY,
        "sequence_length": SEQUENCE_LENGTH,
        "validation_enabled": VALIDATION_ENABLED,
        "validation_fraction": VALIDATION_FRACTION,
        "apply_median_filter_to_test": APPLY_MEDIAN_FILTER_TO_TEST,
        "input_size": input_size,
        "features": list(features),
        "initial_rul": initial_rul,
        "parameter_count": count_parameters(model),
    }
    csv_logger = CSVLogger(log_path)

    patience = VALIDATION_PATIENCE if val_loader is not None else None
    fit_result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(cfg["epochs"]),
        patience=patience,
        best_model_path=best_model_path,
        last_checkpoint_path=last_ckpt_path,
        best_checkpoint_path=best_ckpt_path,
        metadata=metadata,
        csv_logger=csv_logger,
        wandb_logger=None,
        resume_checkpoint_path=last_ckpt_path if args.resume else None,
    )

    # Load best checkpoint for the test evaluation.
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_test(model, x_test, y_test, initial_rul, device)

    summary: dict[str, Any] = {
        **metadata,
        "best_metric": float(fit_result.best_metric),
        "best_epoch": int(fit_result.best_epoch),
        "total_train_seconds": float(fit_result.total_seconds),
        "test_rmse": float(test_metrics["test_rmse"]),
        "test_score": float(test_metrics["test_score"]),
        "test_engines": int(len(y_test)),
        "train_sequences": int(len(train_loader.dataset)),
        "val_sequences": int(len(val_loader.dataset)) if val_loader is not None else 0,
        "best_model_path": str(best_model_path),
        "preprocessing_artifact_path": str(scaler_path),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 RecurrentGNNFusion training entry.")
    p.add_argument("--subset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--graph-method", default="physical", choices=["physical", "pearson", "union"])
    p.add_argument("--use-gnn", dest="use_gnn", action="store_true", default=True)
    p.add_argument("--no-gnn", dest="use_gnn", action="store_false")
    p.add_argument("--recurrent-kind", default=None, choices=["gru", "bigru"])
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gnn-hidden", type=int, default=32)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--gnn-kind", default="gcn", choices=["gcn", "gat"])
    p.add_argument("--gnn-dropout", type=float, default=0.1)
    p.add_argument("--gnn-pool", default="mean", choices=["mean", "add", "max"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
