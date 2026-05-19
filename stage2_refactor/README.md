# Stage 2 Refactor

This directory contains the Stage 2.0 modular refactor of the Stage 1 C-MAPSS
LSTM baseline. The original `reproduce/` directory is left untouched.

## Layout

```text
stage2_refactor/
├── data/
├── models/
├── training/
├── configs/
├── experiments/
├── notebooks/
└── docs/
```

## Baseline Defaults

The default config preserves Stage 1 behavior:

- LSTM only.
- `seq_length = 30`.
- Train preprocessing applies the median filter.
- Test preprocessing does not apply the median filter.
- Validation is disabled.
- Best model is selected by training loss.

See `docs/data_contract.md` for the full audited contract.

## Run on Colab

Open:

```text
stage2_refactor/notebooks/stage2_lstm_baseline_colab.ipynb
```

The notebook mounts Google Drive, installs dependencies, and runs FD001 with
checkpoints/logs written to Drive.

## Run from a Python Environment

From the repo root:

```bash
pip install -r stage2_refactor/requirements.txt
python -m stage2_refactor.experiments.run_experiment \
  --subset FD001 \
  --model lstm \
  --run-id baseline_seed42 \
  --mode train_eval
```

Use a Drive-backed checkpoint directory in Colab:

```bash
python -m stage2_refactor.experiments.run_experiment \
  --subset FD001 \
  --model lstm \
  --run-id baseline_seed42 \
  --mode train_eval \
  --checkpoint-dir /content/drive/MyDrive/ece228_stage2/checkpoints \
  --output-dir /content/drive/MyDrive/ece228_stage2/outputs \
  --resume
```

Outputs are written under model-specific directories, for example:

```text
/content/drive/MyDrive/ece228_stage2/outputs/lstm/FD001/baseline_seed42/summary.json
```

If `--run-id` is omitted, the CLI creates one from the main hyperparameters:

```text
seed42_h60_l4_d0p1_lr0p002_bs15
```

## Stage 2.1 Backbones

The core Stage 2.1 recurrent backbones are implemented:

- `lstm`
- `gru`
- `bilstm`
- `bigru`

Run one model by changing only `--model`:

```bash
python -m stage2_refactor.experiments.run_experiment \
  --subset FD001 \
  --model gru \
  --run-id gru_seed42_h60_l4 \
  --mode train_eval \
  --checkpoint-dir /content/drive/MyDrive/ece228_stage2/checkpoints \
  --output-dir /content/drive/MyDrive/ece228_stage2/outputs \
  --resume
```

The preprocessing, data interface, metrics, checkpointing, and training loop
stay shared across backbones.

For grid search or multi-seed runs, use a unique `--run-id` for each
configuration so results and checkpoints are not overwritten.

## Local Smoke Checks

This local machine may not have the full PyTorch/scientific stack installed, so
the included smoke check avoids third-party imports:

```bash
python3 stage2_refactor/tools/smoke_read_data.py --data-dir CMaps
python3 -m py_compile $(find stage2_refactor -name '*.py')
```
