"""Stage 4 Plan A tuning sweeps.

Targets the four CBAM-internal knobs identified in the post-batch analysis:

  Sweep 1 — attention dropout
            attn_dropout = 0.2  applied to cbam_full on FD002 and FD004
            (cbam_full is the universally-losing cell; dropout is the cleanest
            CBAM-only regularizer to try.)

  Sweep 2 — temporal kernel size
            attn_kernel ∈ {3, 11} applied to temporal_only on FD004
            (FD004 temporal_only is Stage 4's only clean baseline win.
             Sweep tests whether the locality of the temporal gate matters.)

  Sweep 3 — channel reduction ratio
            attn_reduction ∈ {2, 8} applied to channel_only on FD002
            (FD002 channel_only barely beats baseline on RMSE. Sweep tests
             whether more expressive/regularizing channel gate helps.)

  Sweep 4 — module order
            attn_order = "temporal_first" applied to cbam_full on FD002 and FD004
            (Original CBAM picks channel-first; this tests whether the
             temporal-then-channel order is more natural for sensor windows.)

Total = 6 + 6 + 6 + 6 = 24 runs across 3 seeds {7, 42, 123}.
All other hyperparameters match `stage4/ablation.py` (and therefore Stage 3
finalists) so deltas are attributable to the swept CBAM knob alone.

Outputs land under `stage4/artifacts/tune/runs/<SUBSET>/<run_id>/summary.json`
and are aggregated to `stage4/artifacts/tune/results.csv` with mean ± std.

Usage
-----
    python -m stage4.tune                       # run all 4 sweeps
    python -m stage4.tune --sweeps dropout      # only attention-dropout sweep
    python -m stage4.tune --seeds 7,42,123      # override seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Default ablation cell template — every value here matches stage4/ablation.py
# so that any divergence in results is attributable only to the swept knob.
DEFAULT_KNOBS = dict(
    attn_reduction=4,
    attn_kernel=7,
    attn_dropout=0.0,
    attn_order="channel_first",
    graph_method="physical",
    gnn_hidden=32,
    gnn_layers=2,
    gnn_kind="gcn",
    gnn_dropout=0.1,
    gnn_pool="mean",
)


def _make_sweeps() -> list[dict[str, Any]]:
    """Build the 24-run sweep list. Each entry describes one (subset, cell,
    knob-override) combination; the seed loop is added in `main()`."""
    sweeps: list[dict[str, Any]] = []

    # Sweep 1 — attention dropout on cbam_full
    for subset in ("FD002", "FD004"):
        sweeps.append({
            "tag": "dropout",
            "subset": subset,
            "cell": "cbam_full",
            "use_channel": True,
            "use_temporal": True,
            "overrides": {"attn_dropout": 0.2},
            "knob_label": "drop0.2",
        })

    # Sweep 2 — kernel size on FD004 temporal_only
    for k in (3, 11):
        sweeps.append({
            "tag": "kernel",
            "subset": "FD004",
            "cell": "temporal_only",
            "use_channel": False,
            "use_temporal": True,
            "overrides": {"attn_kernel": k},
            "knob_label": f"k{k}",
        })

    # Sweep 3 — reduction on FD002 channel_only
    for r in (2, 8):
        sweeps.append({
            "tag": "reduction",
            "subset": "FD002",
            "cell": "channel_only",
            "use_channel": True,
            "use_temporal": False,
            "overrides": {"attn_reduction": r},
            "knob_label": f"r{r}",
        })

    # Sweep 4 — order swap on cbam_full
    for subset in ("FD002", "FD004"):
        sweeps.append({
            "tag": "order",
            "subset": subset,
            "cell": "cbam_full",
            "use_channel": True,
            "use_temporal": True,
            "overrides": {"attn_order": "temporal_first"},
            "knob_label": "tempfirst",
        })

    return sweeps


def _run_id(sweep: dict[str, Any], seed: int) -> str:
    return f"{sweep['subset'].lower()}_{sweep['cell']}_{sweep['knob_label']}_seed{seed}"


def _run_one(sweep: dict[str, Any], seed: int,
             tune_dir: Path, skip_existing: bool = True) -> dict[str, Any]:
    """Invoke train_stage4.run() in-process for one sweep cell × seed."""
    run_id = _run_id(sweep, seed)
    out_dir = tune_dir / "runs" / sweep["subset"] / run_id
    summary_path = out_dir / "summary.json"

    if skip_existing and summary_path.exists():
        print(f"[skip] {run_id} (summary.json already exists)")
        return json.loads(summary_path.read_text())

    from stage4.train_stage4 import run as train_run

    knobs = {**DEFAULT_KNOBS, **sweep["overrides"]}
    args = argparse.Namespace(
        subset=sweep["subset"],
        seed=seed,
        use_channel_attn=sweep["use_channel"],
        use_temporal_attn=sweep["use_temporal"],
        attn_reduction=knobs["attn_reduction"],
        attn_kernel=knobs["attn_kernel"],
        attn_dropout=knobs["attn_dropout"],
        attn_order=knobs["attn_order"],
        use_gnn=None,                       # subset default (FD001 false; others true)
        graph_method=knobs["graph_method"],
        gnn_hidden=knobs["gnn_hidden"],
        gnn_layers=knobs["gnn_layers"],
        gnn_kind=knobs["gnn_kind"],
        gnn_dropout=knobs["gnn_dropout"],
        gnn_pool=knobs["gnn_pool"],
        recurrent_kind=None, hidden_size=None, num_layers=None, dropout=None,
        learning_rate=None, batch_size=None, epochs=None,
        run_id=run_id,
        artifacts_dir=str(tune_dir),        # tune-specific artifacts root
        stage3_artifacts_dir=None,
        output_dir=str(out_dir),
        checkpoint_dir=str(out_dir / "checkpoints"),
        resume=False,
    )
    return train_run(args)


def _aggregate(results: list[dict[str, Any]], out_csv: Path) -> None:
    """Group by (subset, sweep_tag, knob_label) and report mean ± std."""
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in results:
        key = (r["subset"], r["tag"], r["knob_label"])
        by_key.setdefault(key, []).append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset", "sweep", "knob", "cell", "n_seeds",
            "rmse_mean", "rmse_std",
            "score_mean", "score_std",
        ])
        for (subset, tag, knob), runs in sorted(by_key.items()):
            rmses  = [float(r["test_rmse"])  for r in runs]
            scores = [float(r["test_score"]) for r in runs]
            rmse_std  = statistics.stdev(rmses)  if len(rmses)  > 1 else 0.0
            score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            cell = runs[0]["cell"]
            writer.writerow([
                subset, tag, knob, cell, len(runs),
                f"{statistics.mean(rmses):.4f}",  f"{rmse_std:.4f}",
                f"{statistics.mean(scores):.4f}", f"{score_std:.4f}",
            ])
    print(f"\nWrote aggregate to {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 4 Plan A tuning sweeps.")
    p.add_argument("--seeds",  default="7,42,123")
    p.add_argument("--sweeps", default="dropout,kernel,reduction,order",
                   help="Comma-separated subset of {dropout, kernel, reduction, order}.")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false", default=True,
                   help="If set, re-train even when summary.json already exists.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    wanted_tags = {t.strip() for t in args.sweeps.split(",") if t.strip()}
    all_sweeps = _make_sweeps()
    sweeps = [s for s in all_sweeps if s["tag"] in wanted_tags]
    if not sweeps:
        raise SystemExit(
            f"No sweeps match --sweeps={args.sweeps!r}. "
            f"Choose from dropout, kernel, reduction, order."
        )

    tune_dir = REPO_ROOT / "stage4" / "artifacts" / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = tune_dir / "results.jsonl"
    results_csv   = tune_dir / "results.csv"

    print(f"Plan A: {len(sweeps)} sweep cells × {len(seeds)} seeds "
          f"= {len(sweeps) * len(seeds)} runs total")

    results: list[dict[str, Any]] = []
    with results_jsonl.open("w") as jf:
        for sweep in sweeps:
            for seed in seeds:
                print(f"\n==== [{sweep['tag']}] {sweep['subset']} | "
                      f"{sweep['cell']} | {sweep['knob_label']} | seed={seed} ====")
                summary = _run_one(sweep, seed, tune_dir,
                                   skip_existing=args.skip_existing)
                record = {
                    "tag": sweep["tag"],
                    "knob_label": sweep["knob_label"],
                    "cell": sweep["cell"],
                    **summary,
                }
                results.append(record)
                jf.write(json.dumps(record, default=str) + "\n")
                jf.flush()

    _aggregate(results, results_csv)


if __name__ == "__main__":
    main()
