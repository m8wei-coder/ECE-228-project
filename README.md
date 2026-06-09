# ECE 228 Project: C-MAPSS Remaining Useful Life Prediction

This repository contains our ECE 228 project on remaining useful life (RUL)
prediction for turbofan engines using the NASA C-MAPSS benchmark. The work
starts from a reproduced LSTM baseline, then builds a shared experiment
framework for recurrent backbones, graph fusion, and lightweight attention.

The main metric is test RMSE. We also report the NASA scoring function, which
penalizes late predictions more heavily and is useful for judging safety risk.

## Repository Layout

```text
.
├── CMaps/                 # NASA C-MAPSS raw txt files and RUL labels
├── reproduce/             # original Stage 1 reproduction code
├── stage2_refactor/       # shared data pipeline, trainer, evaluator, recurrent baselines
├── stage3/                # recurrent + GNN fusion models and graph ablations
├── stage4/                # CBAM-style channel / temporal attention experiments
└── README.md
```

The later stages reuse the earlier code instead of copying it. In particular,
Stage 3 and Stage 4 import the `stage2_refactor` data, training, and evaluation
modules so the comparisons use the same preprocessing and metric code.

## Data

The project expects the C-MAPSS files under `CMaps/`:

```text
train_FD001.txt ... train_FD004.txt
test_FD001.txt  ... test_FD004.txt
RUL_FD001.txt   ... RUL_FD004.txt
```

The four subsets differ by operating conditions and fault modes:

| Subset | Operating conditions | Fault modes |
|---|---:|---:|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

## Environment

Each stage has its own requirements file. For local runs, create one Python
environment and install the stage you want to reproduce:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r stage2_refactor/requirements.txt
```

For Stage 3 or Stage 4:

```bash
pip install -r stage3/requirements.txt
pip install -r stage4/requirements.txt
```

Most full experiment sweeps were run in Colab with a T4 GPU. The checked-in
notebooks in `stage3/notebooks/` and `stage4/notebooks/` include the executed
outputs used for the reported tables.

## How to Reproduce

Run a Stage 2 recurrent baseline:

```bash
python -m stage2_refactor.experiments.run_experiment \
  --subset FD001 \
  --model gru \
  --mode train_eval \
  --run-id fd001_gru_seed42
```

Run a Stage 3 recurrent + GNN model:

```bash
python -m stage3.train_stage3 \
  --subset FD004 \
  --graph-method physical \
  --use-gnn \
  --seed 42 \
  --run-id fd004_gnn_seed42
```

Run a Stage 4 attention model:

```bash
python -m stage4.train_stage4 \
  --subset FD004 \
  --use-temporal-attn \
  --seed 42 \
  --run-id fd004_temporal_seed42
```

The main batch scripts are:

```bash
python -m stage4.ablation --subsets FD001,FD002,FD003,FD004 --seeds 7,42,123
python -m stage4.tune
```

See the stage-specific READMEs for full command-line options:

- `stage2_refactor/README.md`
- `stage3/README.md`
- `stage4/README.md`

## Method Summary

Stage 1 reproduced the original LSTM pipeline and established the data contract.
Stage 2 refactored the pipeline and compared recurrent backbones. The best
subset-specific recurrent choices were GRU for FD001/FD003 and BiGRU for
FD002/FD004.

Stage 3 added a graph branch in parallel with the recurrent backbone. Each
sensor is treated as a graph node, and the node feature is the 30-step sensor
window. Three graph constructions were tested: physical topology, Pearson
correlation, and their union.

Stage 4 added a small CBAM-style attention block at the input. Channel attention
weights sensors, temporal attention weights time steps, and the full CBAM model
uses both gates in series.

## Final Results

The table below lists the recommended configuration after all stages.

| Subset | Recommended model | Test RMSE | NASA Score |
|---|---|---:|---:|
| FD001 | GRU | 6.165 +/- 0.219 | 55.72 +/- 3.66 |
| FD002 | BiGRU + physical GNN | 13.705 +/- 0.188 | 1010.90 +/- 333.90 |
| FD003 | GRU + physical GNN | 5.181 +/- 0.414 | 45.28 +/- 5.47 |
| FD004 | BiGRU + physical GNN + temporal attention | 15.470 +/- 0.792 | 920.47 +/- 128.00 |

Main observations:

- GRU is enough for the single-condition subsets, while BiGRU helps on the
  six-condition subsets.
- The physical graph is the most reliable graph construction. Pearson and union
  graphs are too dense for this setup.
- GNN fusion improves the harder subsets, especially FD002 and FD004 in NASA
  Score.
- Full CBAM is not consistently helpful. The clean Stage 4 gain is FD004 with
  temporal-only attention.

Detailed tables and discussion are in:

- `stage2_refactor/reports/stage2_analysis.md`
- `stage3/docs/stage3_analysis.md`
- `stage4/docs/stage4_analysis.md`

## Quick Checks

These checks are useful before pushing changes:

```bash
python3 stage2_refactor/tools/smoke_read_data.py --data-dir CMaps
PYTHONPYCACHEPREFIX=/tmp/ece228_pycache python3 -m compileall -q reproduce stage2_refactor stage3 stage4
```

`outputs/`, checkpoints, and run artifacts are intentionally left out of git.
