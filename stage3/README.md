# Stage 3: Recurrent + GNN Fusion for C-MAPSS RUL

## Goal

Add a graph-neural-network branch alongside the Stage 2 best recurrent backbone, to explicitly model the spatial relationship between engine sensors. The recurrent branch encodes per-window temporal dynamics; the GNN branch encodes inter-sensor correlations (Pearson and/or physical topology); the two are fused and decoded to a single RUL value per window.

## Design Principles

1. **No edits to teammate code.** Stage 3 lives entirely in this `stage3/` directory and imports `stage2_refactor` as a library.
2. **Reuse stage 2 data / training / evaluation paths verbatim.** Same `read_cmapss_table`, same `fit_transform_train`, same `build_train_val_loaders`, same `rmse_score`, same `clamp(0, initial_rul)` test protocol.
3. **Per-subset backbone selection.** FD001/FD003 use GRU; FD002/FD004 use BiGRU (per `stage2_refactor/reports/stage2_analysis.md` finalist decision).

## Backbone Configuration (per subset)

| Subset | Recurrent backbone | Hidden | Layers | Dropout | LR | Stage 2 finalist RMSE | Stage 2 finalist Score |
|---|---|---:|---:|---:|---:|---:|---:|
| FD001 | GRU   | 90 | 2 | 0.2 | 5e-4 |  6.17 |   55.7 |
| FD002 | BiGRU | 60 | 2 | 0.1 | 5e-4 | 14.50 | 1879.5 |
| FD003 | GRU   | 90 | 2 | 0.2 | 5e-4 |  5.37 |   46.0 |
| FD004 | BiGRU | 60 | 2 | 0.1 | 5e-4 | 16.77 | 1194.2 |

These are the numbers Stage 3 must beat (or at least match) per subset.

## Protocol (aligned with Stage 2 finalists)

- `validation_enabled = True` (engine-level split, 15% by `unit_number`)
- `apply_median_filter_to_test = false` (Stage 1 / Stage 2 default; do not toggle it on, even though our earlier Stage 1 baseline numbers were medfilt-on)
- Seeds: `{7, 42, 123}` (same as Stage 2 finalists), report mean ± std
- Test-time evaluation: clamp both `y_pred` and `y_true` to `[0, initial_rul]` before `rmse_score`

## Layout

```
stage3/
├── README.md               # this file
├── docs/
│   ├── recon_report.md     # Stage 1 reproduce/ recon
│   ├── baseline_metrics.md # our Stage 1 baseline numbers
│   └── stage2_recon.md     # Stage 2 framework recon
├── build_graph.py          # adjacency builders: Pearson + physical topology
├── models/
│   ├── __init__.py
│   ├── gnn_modules.py      # GCN / GAT layers (torch_geometric)
│   └── recurrent_gnn.py    # RecurrentGNNFusion(BaseModel)
├── train_stage3.py         # training entry; reuses stage2_refactor fit/evaluate
├── ablation.py             # ablation matrix runner
├── configs/
│   └── stage3.yaml         # per-subset backbone + GNN hyperparams + graph params
└── artifacts/              # adjacency .npy, checkpoints, predictions (gitignored)
```

## Adjacency Sources

- **Pearson correlation graph (default)**: `reproduce/exports/FD00X/corr_matrix.npy` — already produced by `reproduce/export_artifacts.py`. The matrix is `(F+1, F+1)` because the last row/column is `RUL_absolute`; **slice it down to `(F, F)`** (sensor-only) before use.
- **Physical topology graph (alternative)**: hand-defined edges based on engine schematic (fan / LPC / HPC / combustor / HPT / LPT / nozzle stages → which sensors observe which stage). Built in `build_graph.py` per the engine diagram in `reproduce/CMaps/Damage Propagation Modeling.pdf`.

## Running

Set up the env once (already done if you followed Stage 2 setup):

```bash
conda activate ece228
pip install torch-geometric  # the only extra dep beyond Stage 2's requirements
```

Train one subset:

```bash
python -m stage3.train_stage3 \
  --subset FD001 \
  --graph pearson \
  --run-id rgnn_fd001_pearson_seed42 \
  --seed 42
```

Run full ablation matrix:

```bash
python -m stage3.ablation --subsets FD001,FD002,FD003,FD004 \
                          --graphs pearson,physical \
                          --seeds 7,42,123
```

## Environment

`ece228` conda env (see `stage3/docs/baseline_metrics.md` for the pinned package list). Extra: `torch-geometric` (and its scatter/sparse deps if not already pulled in).
