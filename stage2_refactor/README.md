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
  --mode train_eval
```

Use a Drive-backed checkpoint directory in Colab:

```bash
python -m stage2_refactor.experiments.run_experiment \
  --subset FD001 \
  --mode train_eval \
  --checkpoint-dir /content/drive/MyDrive/ece228_stage2/checkpoints \
  --output-dir /content/drive/MyDrive/ece228_stage2/outputs \
  --resume
```

## Local Smoke Checks

This local machine may not have the full PyTorch/scientific stack installed, so
the included smoke check avoids third-party imports:

```bash
python3 stage2_refactor/tools/smoke_read_data.py --data-dir CMaps
python3 -m py_compile $(find stage2_refactor -name '*.py')
```

