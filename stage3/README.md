# Stage 3: Recurrent + GNN Fusion for C-MAPSS RUL

This directory contains the Stage 3 contribution to the project: a GNN
branch that runs in parallel to Stage 2's recurrent backbone and is fused
through a shared regression head. Each sensor is a graph node; the
adjacency comes from one of three constructions (physical / pearson /
union). The recurrent half is unchanged from Stage 2 and the
`stage2_refactor/` package is imported as a library — no teammate code
is modified.

## Layout

```text
stage3/
├── build_graph.py             # adjacency matrix builder (pearson | physical | union)
├── models/
│   ├── gnn_modules.py         # DenseGCNConv-based GCN branch (+ GAT placeholder)
│   └── recurrent_gnn.py       # RecurrentGNNFusion: inherits BaseModel, fuses recurrent + GNN
├── train_stage3.py            # training entry; reuses stage2_refactor data / fit / evaluate
├── configs/
│   └── stage3.yaml            # per-subset finalist config + GNN / graph hyperparams
├── notebooks/
│   └── stage3_colab.ipynb     # 10-cell Colab notebook, executed with outputs
├── artifacts/
│   ├── adj_{FD001..FD004}_{pearson,physical,union}.npy   # 12 precomputed graphs (~4 KB each, in repo)
│   └── runs/ ...              # per-run summary.json / train_log.csv / checkpoints (gitignored)
├── docs/
│   ├── recon_report.md        # Stage 1 reproduce/ recon
│   ├── baseline_metrics.md    # our Stage 1 baseline numbers (RMSE/Score, initial_rul)
│   ├── stage2_recon.md        # Stage 2 framework recon
│   ├── fd001_results.md       # initial FD001 sanity check (3 seeds, local CPU)
│   └── stage3_analysis.md     # FINAL Stage 3 ablation analysis + recommendations
├── requirements.txt           # third-party deps (union of Stage 2 + Stage 3)
└── README.md                  # this file
```

## Goal

Stage 1 / Stage 2 treat each window as F independent univariate time
series stacked into a tensor and let a recurrent backbone discover any
cross-sensor coupling on its own. Stage 3 adds an explicit spatial prior:
the C-MAPSS sensors are physically arranged along the engine gas path
(Fan → LPC → HPC → Combustor → HPT → LPT → Nozzle, with the core shaft
linking HPC and HPT), and a GCN over that topology should help when the
degradation signature is distributed across components — i.e. when the
operating regime or fault count is non-trivial (FD002, FD003, FD004).

The model is `RecurrentGNNFusion`. It inherits `stage2_refactor.models.base.BaseModel`
so the Stage 2 trainer / evaluator run it without changes:

```text
x : (B, 30, F)
    ├── recurrent branch:  nn.GRU (uni or bidirectional) -> (B, H)  or (B, 2H)
    └── GNN branch:        x.transpose(1,2) -> (B, F, 30) -> GCN -> (B, gnn_hidden)
                                                                      |
                  fused = cat([h_rec, h_gnn], -1) -> head -> (B, 1)
```

A `use_gnn=False` switch on the model disables the GNN branch entirely.
In that mode the parameter count is identical to the Stage 2 finalists
(81,901 for FD001/FD003 GRU; 99,061 for FD002/FD004 BiGRU), giving us a
fair ablation control.

## Per-subset backbone (locked from Stage 2 finalists)

| Subset | Backbone | hidden | layers | dropout | lr |
|---|---|---:|---:|---:|---:|
| FD001 / FD003 | GRU   | 90 | 2 | 0.2 | 5e-4 |
| FD002 / FD004 | BiGRU | 60 | 2 | 0.1 | 5e-4 |

## Dependencies and environment

See `requirements.txt`. Two equivalent environments are used:

- **Local** (smoke tests, FD001 fast turnaround): conda env `ece228` with
  Python 3.11, scikit-learn 1.4.2, torch 2.x, torch_geometric 2.7.0.
- **Colab T4 GPU** (full FD001–FD004 × 4 variants × 3 seeds = 48 runs):
  `pip install -q torch_geometric pyyaml joblib`; everything else ships
  with Colab.

The repo is the source of truth for raw data: it expects
`CMaps/{train,test,RUL}_FD00X.txt` at the repo root (committed by the
Stage 2 owner).

## Run a single training

From the repo root, in the `ece228` env:

```bash
python -m stage3.train_stage3 \
  --subset FD001 \
  --graph-method physical \
  --use-gnn \
  --seed 42 \
  --run-id fd001_gnn_seed42
```

Common CLI flags (all optional unless noted):

| Flag | Purpose | Default |
|---|---|---|
| `--subset` | FD001 / FD002 / FD003 / FD004 | **required** |
| `--graph-method` | physical / pearson / union | physical |
| `--use-gnn` / `--no-gnn` | enable / disable GNN branch | --use-gnn |
| `--recurrent-kind` | gru / bigru | per-subset default |
| `--hidden-size`, `--num-layers`, `--dropout` | backbone hyperparams | per-subset default |
| `--learning-rate`, `--batch-size`, `--epochs` | training hyperparams | per-subset default |
| `--seed` | RNG seed | 42 |
| `--gnn-hidden`, `--gnn-layers`, `--gnn-kind`, `--gnn-pool` | GNN hyperparams | 32 / 2 / gcn / mean |
| `--run-id`, `--output-dir`, `--checkpoint-dir` | output paths | auto from subset + seed |
| `--artifacts-dir` | parent for the auto path | `stage3/artifacts/` |
| `--resume` | resume from `last_checkpoint.pt` | False |

Each run writes `summary.json` + `train_log.csv` + 3 checkpoint files to
the output / checkpoint dirs.

## Run the full ablation on Colab

Open `notebooks/stage3_colab.ipynb` in Colab (the version checked in
already carries the executed cell outputs from the run that produced
the final paper numbers). Run cells in order:

- **Cell 1** — `git clone` / `git pull` this repo
- **Cell 2** — `pip install torch_geometric pyyaml joblib`
- **Cell 3** — mount Drive at `/content/drive/MyDrive/ece228_stage3/`
  for resumable run outputs
- **Cell 4** — env self-check (versions + GPU)
- **Cell 5** — `python -m stage3.build_graph` (regenerates the 12 `.npy` files)
- **Cell 6** — hyperparameter knobs (only cell to edit for one-off runs)
- **Cell 7** — train one configuration
- **Cell 8** — read back `summary.json` and print a tidy report
- **Cell 9** — *batch*: 4 subsets × {GNN-physical, no-GNN} × 3 seeds = **24 runs**, resumable
- **Cell 10** — *batch*: 2 graphs (pearson + union) × 4 subsets × 3 seeds = **24 runs**, resumable

Cell 9 + Cell 10 together produce the full ablation table in
`docs/stage3_analysis.md`. Both cells skip runs whose `summary.json`
already exists in Drive, so a Colab disconnect mid-batch is recovered
by simply re-running the cell.

## Result summary

The headline finding (full table in `docs/stage3_analysis.md`): the
**physical** graph beats both pearson and union on every subset, and
GNN-physical wins (on either RMSE, Score, or both) on **FD003 and
FD004** while remaining tied with the pure-recurrent baseline on
**FD002 RMSE** and giving a large Score improvement on FD002 (−168
absolute, −14%) and FD004 (−108 absolute, −10%). On **FD001** — the
simplest subset (one condition, one fault) — adding GNN hurts; the
recommendation there is the pure GRU baseline.

| Subset | Recommended for Stage 3 |
|---|---|
| FD001 | pure GRU (no GNN) |
| FD002 | BiGRU + GNN (physical) |
| FD003 | GRU   + GNN (physical) |
| FD004 | BiGRU + GNN (physical) |

## Relationship to Stage 1 / Stage 2

- `stage2_refactor/data/{io,preprocessing,dataloader}.py` — Stage 3
  imports these directly for raw-data reading, KMeans-by-condition
  scaling, piecewise-RUL clipping, and engine-level train/val
  splitting. The whole Stage 2 data contract (sequence length 30,
  apply_median_filter_to_test = False, validation_enabled = True) is
  inherited.
- `stage2_refactor/training/trainer.py:fit` — Stage 3 calls this for
  the actual training loop, so checkpointing, early stopping (`patience=10`),
  and CSV logging are exactly the same as Stage 2 finalists.
- `stage2_refactor/training/evaluator.py:rmse_score` + the
  `clamp(0, initial_rul)` test protocol from
  `stage2_refactor.experiments.run_experiment.evaluate_test` —
  Stage 3 uses these so the final test numbers are computed with the
  same code as Stage 2.
- No file under `stage2_refactor/` is modified by Stage 3.
- Raw data lives at the repo root in `CMaps/` (added by the Stage 2 owner).
