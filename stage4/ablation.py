"""Stage 4 ablation matrix.

For each subset, runs three attention configurations (channel-only,
temporal-only, full CBAM) on top of the Stage 3 best-architecture baseline,
across three seeds {7, 42, 123}. Then aggregates per-cell results into
stage4/artifacts/ablation/results.csv with mean ± std of RMSE and Score.

Baselines (no Stage 4 attention) are already in Stage 3 — we re-read them
from stage3/docs/stage3_analysis.md when writing the final report, so we
do not re-run them here.

Usage
-----
    python -m stage4.ablation \
        --subsets FD001,FD002,FD003,FD004 \
        --seeds   7,42,123
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# (cell_label, use_channel_attn, use_temporal_attn)
ABLATION_CELLS = [
    ("channel_only",  True,  False),
    ("temporal_only", False, True),
    ("cbam_full",     True,  True),
]


def run_one(subset: str, seed: int, cell: str,
            use_channel: bool, use_temporal: bool,
            graph_method: str,
            skip_existing: bool = True) -> dict[str, Any]:
    """Invoke stage4.train_stage4.run() in-process.

    If `skip_existing` is True and summary.json already exists for this run,
    load it from disk instead of retraining. This makes the matrix
    Ctrl+C-restart safe.
    """
    run_id = f"{subset.lower()}_{cell}_seed{seed}"
    summary_path = (REPO_ROOT / "stage4" / "artifacts"
                    / "runs" / subset / run_id / "summary.json")
    if skip_existing and summary_path.exists():
        print(f"[skip] {run_id}  (summary.json already exists)")
        return json.loads(summary_path.read_text())

    # Import inside the function so we don't require torch at module-import time.
    from stage4.train_stage4 import run as train_run

    args = argparse.Namespace(
        subset=subset,
        seed=seed,
        use_channel_attn=use_channel,
        use_temporal_attn=use_temporal,
        attn_reduction=4,
        attn_kernel=7,
        use_gnn=None,               # use subset default
        graph_method=graph_method,
        gnn_hidden=32,
        gnn_layers=2,
        gnn_kind="gcn",
        gnn_dropout=0.1,
        gnn_pool="mean",
        recurrent_kind=None,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        learning_rate=None,
        batch_size=None,
        epochs=None,
        run_id=run_id,
        artifacts_dir=None,
        stage3_artifacts_dir=None,
        output_dir=None,
        checkpoint_dir=None,
        resume=False,
    )
    return train_run(args)


def aggregate(results: list[dict[str, Any]], out_csv: Path) -> None:
    """Group by (subset, cell) and report mean ± std of test_rmse / test_score."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        key = (r["subset"], r["cell"])
        by_key.setdefault(key, []).append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset", "cell", "n_seeds",
            "rmse_mean", "rmse_std",
            "score_mean", "score_std",
        ])
        for (subset, cell), runs in sorted(by_key.items()):
            rmses = [float(r["test_rmse"])  for r in runs]
            scores = [float(r["test_score"]) for r in runs]
            rmse_std  = statistics.stdev(rmses)  if len(rmses)  > 1 else 0.0
            score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            writer.writerow([
                subset, cell, len(runs),
                f"{statistics.mean(rmses):.4f}",  f"{rmse_std:.4f}",
                f"{statistics.mean(scores):.4f}", f"{score_std:.4f}",
            ])
    print(f"Wrote aggregate to {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 4 ablation matrix runner.")
    p.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    p.add_argument("--seeds",   default="7,42,123")
    p.add_argument("--graph-method", default="physical",
                   choices=["physical", "pearson", "union"])
    p.add_argument("--cells", default=None,
                   help="Comma-separated subset of {channel_only,temporal_only,cbam_full}.")
    p.add_argument("--results-json", default=None)
    p.add_argument("--results-csv",  default=None)
    args = p.parse_args()

    subsets = [s.strip().upper() for s in args.subsets.split(",") if s.strip()]
    seeds   = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.cells is None:
        cells = ABLATION_CELLS
    else:
        wanted = {c.strip() for c in args.cells.split(",")}
        cells = [c for c in ABLATION_CELLS if c[0] in wanted]
        if not cells:
            raise SystemExit(f"no valid cells in {args.cells!r}")

    artifacts_dir = REPO_ROOT / "stage4" / "artifacts" / "ablation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_json = Path(args.results_json) if args.results_json else (artifacts_dir / "results.jsonl")
    results_csv  = Path(args.results_csv)  if args.results_csv  else (artifacts_dir / "results.csv")

    results: list[dict[str, Any]] = []
    with results_json.open("w") as jf:
        for subset in subsets:
            for cell_label, use_c, use_t in cells:
                for seed in seeds:
                    print(f"\n==== {subset} | {cell_label} | seed={seed} ====")
                    summary = run_one(subset, seed, cell_label, use_c, use_t,
                                      args.graph_method)
                    record = {"cell": cell_label, **summary}
                    results.append(record)
                    jf.write(json.dumps(record, default=str) + "\n")
                    jf.flush()

    aggregate(results, results_csv)


if __name__ == "__main__":
    main()
