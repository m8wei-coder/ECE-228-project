from __future__ import annotations

import csv
from pathlib import Path
from shutil import copyfile


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "stage2_refactor" / "reports"
FIG_DIR = REPORT_DIR / "figures"
RESULTS_DIR = ROOT / "ece228_stage2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def model_label(row: dict[str, str]) -> str:
    return (
        f"{row['model'].upper()} "
        f"h{row['hidden_size']} l{row['num_layers']} "
        f"d{row['dropout']} lr{row['learning_rate']}"
    )


def write_csv_copies() -> None:
    copyfile(RESULTS_DIR / "final_stage2_summary.csv", REPORT_DIR / "final_stage2_summary.csv")
    copyfile(RESULTS_DIR / "final_stage2_runs.csv", REPORT_DIR / "final_stage2_runs.csv")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '.title{font:700 20px Arial,sans-serif;fill:#111827}',
        '.label{font:12px Arial,sans-serif;fill:#374151}',
        '.tick{font:11px Arial,sans-serif;fill:#6b7280}',
        '.axis{stroke:#9ca3af;stroke-width:1}',
        '.grid{stroke:#e5e7eb;stroke-width:1}',
        '.err{stroke:#111827;stroke-width:1.5}',
        '.note{font:11px Arial,sans-serif;fill:#6b7280}',
        '</style>',
    ]


def write_bar_chart(
    rows: list[dict[str, str]],
    metric_mean: str,
    metric_std: str | None,
    title: str,
    out_path: Path,
    y_label: str,
) -> None:
    width, height = 860, 470
    left, right, top, bottom = 82, 36, 64, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [f(row, metric_mean) for row in rows]
    stds = [f(row, metric_std) if metric_std else 0.0 for row in rows]
    max_value = max(v + s for v, s in zip(values, stds)) * 1.15
    max_value = max(max_value, 1.0)
    bar_w = plot_w / len(rows) * 0.56
    colors = {"gru": "#0f766e", "bigru": "#b45309", "lstm": "#2563eb", "bilstm": "#7c3aed"}

    def y(value: float) -> float:
        return top + plot_h - (value / max_value) * plot_h

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="{left}" y="32">{title}</text>')
    lines.append(f'<text class="label" x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})">{y_label}</text>')

    for i in range(6):
        val = max_value * i / 5
        yy = y(val)
        lines.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end">{val:.1f}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>')

    for idx, row in enumerate(rows):
        cx = left + (idx + 0.5) * plot_w / len(rows)
        val = values[idx]
        std = stds[idx]
        bar_h = top + plot_h - y(val)
        x = cx - bar_w / 2
        color = colors.get(row["model"], "#4b5563")
        lines.append(f'<rect x="{x:.1f}" y="{y(val):.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        if metric_std:
            y1, y2 = y(val + std), y(max(val - std, 0))
            lines.append(f'<line class="err" x1="{cx:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{y2:.1f}"/>')
            lines.append(f'<line class="err" x1="{cx - 7:.1f}" y1="{y1:.1f}" x2="{cx + 7:.1f}" y2="{y1:.1f}"/>')
            lines.append(f'<line class="err" x1="{cx - 7:.1f}" y1="{y2:.1f}" x2="{cx + 7:.1f}" y2="{y2:.1f}"/>')
        lines.append(f'<text class="label" x="{cx:.1f}" y="{y(val) - 8:.1f}" text-anchor="middle">{val:.2f}</text>')
        lines.append(f'<text class="label" x="{cx:.1f}" y="{top + plot_h + 22}" text-anchor="middle">{row["subset"]}</text>')
        lines.append(f'<text class="tick" x="{cx:.1f}" y="{top + plot_h + 40}" text-anchor="middle">{row["model"].upper()}</text>')

    lines.append(f'<text class="note" x="{left}" y="{height - 20}">Error bars show one standard deviation across 3 seeds.</text>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


def write_scatter_runs(runs: list[dict[str, str]], out_path: Path) -> None:
    width, height = 900, 470
    left, right, top, bottom = 82, 36, 64, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    values = [f(row, "test_rmse") for row in runs]
    max_value = max(values) * 1.15
    colors = {"gru": "#0f766e", "bigru": "#b45309"}
    seed_offsets = {"7": -18, "42": 0, "123": 18}

    def y(value: float) -> float:
        return top + plot_h - (value / max_value) * plot_h

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="{left}" y="32">Per-Seed RMSE for Final Stage 2 Choices</text>')

    for i in range(6):
        val = max_value * i / 5
        yy = y(val)
        lines.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end">{val:.1f}</text>')

    for idx, subset in enumerate(subsets):
        cx = left + (idx + 0.5) * plot_w / len(subsets)
        lines.append(f'<text class="label" x="{cx:.1f}" y="{top + plot_h + 28}" text-anchor="middle">{subset}</text>')
        subset_rows = [row for row in runs if row["subset"] == subset]
        for row in subset_rows:
            x = cx + seed_offsets.get(row["seed"], 0)
            yy = y(f(row, "test_rmse"))
            color = colors.get(row["model"], "#4b5563")
            lines.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="6" fill="{color}"/>')
            lines.append(f'<text class="tick" x="{x:.1f}" y="{yy - 10:.1f}" text-anchor="middle">{row["seed"]}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>')
    lines.append(f'<text class="label" x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})">RMSE</text>')
    lines.append(f'<rect x="{width - 220}" y="24" width="12" height="12" fill="#0f766e"/><text class="label" x="{width - 202}" y="35">GRU</text>')
    lines.append(f'<rect x="{width - 150}" y="24" width="12" height="12" fill="#b45309"/><text class="label" x="{width - 132}" y="35">BiGRU</text>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


def markdown_table(summary: list[dict[str, str]]) -> str:
    lines = [
        "| Subset | Final model | Config | RMSE mean +/- std | Score mean +/- std | Params |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary:
        config = f"h{row['hidden_size']} l{row['num_layers']} d{row['dropout']} lr{row['learning_rate']}"
        lines.append(
            f"| {row['subset']} | {row['model'].upper()} | {config} | "
            f"{fmt_mean_std(f(row, 'rmse_mean'), f(row, 'rmse_std'))} | "
            f"{fmt_mean_std(f(row, 'score_mean'), f(row, 'score_std'))} | "
            f"{int(float(row['params'])):,} |"
        )
    return "\n".join(lines)


def write_analysis(summary: list[dict[str, str]], runs: list[dict[str, str]]) -> None:
    table = markdown_table(summary)
    fd001 = next(row for row in summary if row["subset"] == "FD001")
    fd003 = next(row for row in summary if row["subset"] == "FD003")
    fd002 = next(row for row in summary if row["subset"] == "FD002")
    fd004 = next(row for row in summary if row["subset"] == "FD004")

    text = f"""# Stage 2 Backbone Analysis

## Final Results

{table}

![Final RMSE](figures/final_rmse.svg)

![Final Score](figures/final_score.svg)

![Per-seed RMSE](figures/per_seed_rmse.svg)

![Parameter Count](figures/parameter_count.svg)

## Main Findings

The final Stage 2 experiments select different recurrent backbones depending on operating-condition complexity. FD001 and FD003, the simpler one-condition subsets, are best served by the tuned GRU configuration (`h90`, `2` layers, dropout `0.2`, learning rate `5e-4`). FD002 and FD004, which contain six operating conditions, are better served by the tuned BiGRU configuration (`h60`, `2` layers, dropout `0.1`, learning rate `5e-4`).

GRU is the strongest choice on simple subsets:

- FD001: RMSE {f(fd001, 'rmse_mean'):.3f} +/- {f(fd001, 'rmse_std'):.3f}, Score {f(fd001, 'score_mean'):.2f} +/- {f(fd001, 'score_std'):.2f}.
- FD003: RMSE {f(fd003, 'rmse_mean'):.3f} +/- {f(fd003, 'rmse_std'):.3f}, Score {f(fd003, 'score_mean'):.2f} +/- {f(fd003, 'score_std'):.2f}.

BiGRU is the stronger choice on the complex operating-condition subsets:

- FD002: RMSE {f(fd002, 'rmse_mean'):.3f} +/- {f(fd002, 'rmse_std'):.3f}, Score {f(fd002, 'score_mean'):.2f} +/- {f(fd002, 'score_std'):.2f}.
- FD004: RMSE {f(fd004, 'rmse_mean'):.3f} +/- {f(fd004, 'rmse_std'):.3f}, Score {f(fd004, 'score_mean'):.2f} +/- {f(fd004, 'score_std'):.2f}.

## Ablation Answers

1. **Which backbone is best on which sub-dataset?** GRU wins on FD001 and FD003; BiGRU wins on FD002 and FD004. There is not a single universal winner across all four subsets.

2. **Does bidirectionality help consistently?** No. Bidirectionality helps most clearly on the more complex FD002/FD004 settings. On FD001/FD003, the simpler GRU is more parameter-efficient and achieves better mean RMSE.

3. **Is the architecture gap larger on simple or complex subsets?** The evidence points to a larger practical gap on complex subsets. FD004 in particular strongly favored BiGRU over the transferred GRU seed-42 comparison, while FD003 showed only a small GRU/BiGRU difference.

4. **How does parameter count trade off against accuracy?** The best GRU setting is compact, with roughly 82k trainable parameters on FD001/FD003. The chosen BiGRU setting is still modest at roughly 99k parameters on FD002/FD004, and the added bidirectional capacity appears worthwhile for complex operating conditions.

## Recommendation for Stage 3

If Stage 3 can use subset-specific backbones, carry forward GRU for FD001/FD003 and BiGRU for FD002/FD004. If Stage 3 requires one global recurrent backbone, BiGRU is the more conservative choice because it handles the complex six-condition subsets better, although GRU remains the stronger and lighter option for the simple subsets.

## Notes

- All final results use three seeds: 7, 42, and 123.
- Model selection used engine-level validation (`validation_enabled=True`) to avoid sample-window leakage between train and validation.
- The checked-in Stage 1 FD001 checkpoint evaluated at RMSE 8.72 with the original `reproduce/main_test.py`; the refactored FD001 LSTM retraining run produced RMSE 8.26. The tuned Stage 2 GRU improves FD001 to RMSE 6.16 +/- 0.22.
"""
    (REPORT_DIR / "stage2_analysis.md").write_text(text)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_csv(RESULTS_DIR / "final_stage2_summary.csv")
    runs = read_csv(RESULTS_DIR / "final_stage2_runs.csv")
    write_csv_copies()
    write_bar_chart(summary, "rmse_mean", "rmse_std", "Final Stage 2 RMSE by Subset", FIG_DIR / "final_rmse.svg", "RMSE")
    write_bar_chart(summary, "score_mean", "score_std", "Final Stage 2 NASA Score by Subset", FIG_DIR / "final_score.svg", "Score")
    write_bar_chart(summary, "params", None, "Final Stage 2 Parameter Count by Subset", FIG_DIR / "parameter_count.svg", "Trainable parameters")
    write_scatter_runs(runs, FIG_DIR / "per_seed_rmse.svg")
    write_analysis(summary, runs)


if __name__ == "__main__":
    main()
