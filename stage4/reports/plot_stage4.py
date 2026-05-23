"""Generate Stage 4 comparison figures from batch_summary.json.

Usage
-----
    python stage4/reports/plot_stage4.py \
        --batch-summary /content/drive/MyDrive/ece228_stage4/batch_summary.json \
        --out-dir stage4/reports/figures

Produces:
    figures/stage4_rmse_bars.png   -- 4-panel grid (one per subset), bar chart of
                                       Stage 3 baseline + 3 Stage 4 cells with std error bars.
    figures/stage4_score_bars.png  -- same, for NASA Score.

Each subset compares against its Stage 3 finalist baseline (hard-coded
from stage3/docs/stage3_analysis.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Stage 3 finalists, copied from stage3/docs/stage3_analysis.md.
STAGE3_BASELINE = {
    "FD001": {"rmse":  6.165, "rmse_std": 0.219, "score":   55.72, "score_std":   3.66,
              "label": "Stage 3\n(no-GNN)"},
    "FD002": {"rmse": 13.705, "rmse_std": 0.188, "score": 1010.90, "score_std": 333.90,
              "label": "Stage 3\n(GNN-physical)"},
    "FD003": {"rmse":  5.181, "rmse_std": 0.414, "score":   45.28, "score_std":   5.47,
              "label": "Stage 3\n(GNN-physical)"},
    "FD004": {"rmse": 15.900, "rmse_std": 0.585, "score": 1020.29, "score_std": 102.57,
              "label": "Stage 3\n(GNN-physical)"},
}

STAGE4_CELLS = [
    ("channel_only",  "channel\nonly"),
    ("temporal_only", "temporal\nonly"),
    ("cbam_full",     "full\nCBAM"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-summary", required=True,
                   help="Path to batch_summary.json from Cell 9.")
    p.add_argument("--out-dir", default="stage4/reports/figures")
    args = p.parse_args()

    import matplotlib.pyplot as plt  # imported lazily

    summary = json.loads(Path(args.batch_summary).read_text())
    agg = summary["aggregate"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _panel_for(ax, subset, metric):
        b = STAGE3_BASELINE[subset]
        labels = [b["label"]]
        means  = [b[metric]]
        stds   = [b[f"{metric}_std"]]
        for key, lbl in STAGE4_CELLS:
            try:
                row = next(x for x in agg
                           if x["subset"] == subset and x["cell"] == key)
            except StopIteration:
                means.append(float("nan"))
                stds.append(0.0)
                labels.append(lbl)
                continue
            means.append(row[f"{metric}_mean"])
            stds.append(row[f"{metric}_std"])
            labels.append(lbl)
        xs = range(len(labels))
        bars = ax.bar(xs, means, yerr=stds, capsize=4,
                      color=["#888"] + ["#3a86ff", "#ff006e", "#06d6a0"])
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(subset)
        ax.set_ylabel("RMSE" if metric == "rmse" else "NASA Score")
        # annotate values above bars
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=7)

    for metric, fname in [("rmse", "stage4_rmse_bars.png"),
                          ("score", "stage4_score_bars.png")]:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, subset in zip(axes, ["FD001", "FD002", "FD003", "FD004"]):
            _panel_for(ax, subset, metric)
        fig.suptitle(f"Stage 4 CBAM vs Stage 3 baseline — test {metric.upper()}")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
