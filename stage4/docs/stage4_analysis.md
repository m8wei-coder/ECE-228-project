# Stage 4 CBAM Attention Analysis

> Note: this report has the structure ready; the numbers in the **Final
> Results** and **Per-cell Contribution** sections will be filled in once
> the Colab batch (`stage4/notebooks/stage4_colab.ipynb`, Cell 9) finishes.
> Baselines below come directly from `stage3/docs/stage3_analysis.md`.

## Goal and Method

Stage 4 adds a CBAM (Convolutional Block Attention Module) style
attention block on top of the Stage 3 best-architecture-per-subset
finalist. The motivation is to test whether lightweight, inductive-bias-
preserving attention modules can squeeze additional RUL prediction
accuracy out of the recurrent (+ optional GNN) backbones we already
have, without resorting to a Transformer architecture.

**CBAM adaptation to (B, T=30, F).** The original CBAM is defined over
2-D feature maps; we adapt it to our 1-D sensor-time windows:

- **ChannelAttention** ("which sensors matter"): pool along time (avg +
  max), feed through a shared MLP `F -> F/r -> F` with reduction ratio
  `r=4`, add the two branches and sigmoid. Gate has shape `(B, 1, F)`
  and is broadcast across the time axis.
- **TemporalAttention** ("which time steps matter"): pool along the
  channel axis (avg + max), feed the resulting `(B, 2, T)` tensor
  through a 1-D conv with kernel 7, sigmoid. Gate has shape
  `(B, T, 1)` and is broadcast across the channel axis.
- **CBAM1D** applies channel-then-temporal in series.

**Backbone per subset (locked by Stage 3 finalists).**

| Subset | Backbone selected                            |
|---|---|
| FD001 | GRU (no GNN; Stage 3 said pure GRU wins here) |
| FD002 | BiGRU + GNN-physical                         |
| FD003 | GRU   + GNN-physical                         |
| FD004 | BiGRU + GNN-physical                         |

**Parameter overhead.** With `r=4`, the CBAM1D parameter count is

| F | CBAM1D params | Backbone params | Overhead |
|---:|---:|---:|---:|
| 14 (FD001) |  98 | 81 901  | +0.12% |
| 16 (FD003) | 142 | 82 441  | +0.17% |
| 21 (FD002/FD004) | 224 | 99 061 | +0.23% |

This is well within the "lightweight augmentation" specification from
the project methodology (Stage 4 paragraph), and is the empirical
guarantee that any observed RMSE / Score change is driven by the
inductive bias of the attention gates rather than additional capacity.

**Protocol (aligned with Stage 3 finalists for direct comparability).**

- 3 seeds `{7, 42, 123}`
- engine-level validation (15%), `patience=10` early stopping
- `apply_median_filter_to_test = False`
- clamp `y_pred` and `y_true` to `[0, initial_rul]` before computing
  RMSE / NASA Score
- 4 subsets × 3 attention cells × 3 seeds = **36 runs** on Colab T4 GPU

**Ablation cells** (each compared against its Stage 3 baseline):

1. `channel_only`  — channel attention only, no temporal gate
2. `temporal_only` — temporal attention only, no channel gate
3. `cbam_full`     — full CBAM (channel then temporal)

## Final Results

> Filled in by `stage4/notebooks/stage4_colab.ipynb` Cell 9.
> Copy the printed table here once the 36 runs complete.

| Subset | Variant | Test RMSE (mean ± std) | Test Score (mean ± std) |
|---|---|---:|---:|
| FD001 | stage3 baseline (no-gnn)      |  6.165 ± 0.219 |    55.72 ±   3.66 |
| FD001 | stage4 channel_only           |  _TBD_ |    _TBD_ |
| FD001 | stage4 temporal_only          |  _TBD_ |    _TBD_ |
| FD001 | stage4 cbam_full              |  _TBD_ |    _TBD_ |
| FD002 | stage3 baseline (gnn-physical) | 13.705 ± 0.188 |  1010.90 ± 333.90 |
| FD002 | stage4 channel_only           |  _TBD_ |    _TBD_ |
| FD002 | stage4 temporal_only          |  _TBD_ |    _TBD_ |
| FD002 | stage4 cbam_full              |  _TBD_ |    _TBD_ |
| FD003 | stage3 baseline (gnn-physical) |  5.181 ± 0.414 |    45.28 ±   5.47 |
| FD003 | stage4 channel_only           |  _TBD_ |    _TBD_ |
| FD003 | stage4 temporal_only          |  _TBD_ |    _TBD_ |
| FD003 | stage4 cbam_full              |  _TBD_ |    _TBD_ |
| FD004 | stage3 baseline (gnn-physical) | 15.900 ± 0.585 |  1020.29 ± 102.57 |
| FD004 | stage4 channel_only           |  _TBD_ |    _TBD_ |
| FD004 | stage4 temporal_only          |  _TBD_ |    _TBD_ |
| FD004 | stage4 cbam_full              |  _TBD_ |    _TBD_ |

## Per-cell Contribution (filled in after Cell 9)

The headline question Stage 4 has to answer is **which of the two
attention sub-modules matters, and on which subsets**. The methodology
predicted channel attention should help most on FD002 / FD004 (multi-
operating-condition, multi-fault) where inter-sensor weighting is
most informative.

We will tabulate, for each subset:

| Subset | Δ RMSE (channel) | Δ RMSE (temporal) | Δ RMSE (full CBAM) |
|---|---:|---:|---:|
| FD001 | _TBD_ | _TBD_ | _TBD_ |
| FD002 | _TBD_ | _TBD_ | _TBD_ |
| FD003 | _TBD_ | _TBD_ | _TBD_ |
| FD004 | _TBD_ | _TBD_ | _TBD_ |

and the analogous Score deltas. Negative numbers = improvement over the
Stage 3 baseline.

Figures are produced by `stage4/reports/plot_stage4.py` (4-panel RMSE
bar chart + 4-panel Score bar chart with std error bars across the
three seeds).

## Notes

- All Stage 4 results above were trained on Colab T4 GPU using the
  same `stage2_refactor` data / training / evaluation modules and the
  same Stage 3 GNN modules — the only Stage-4-local code is
  `stage4/models/attention.py`, `stage4/models/attn_recurrent.py`,
  `stage4/models/attn_recurrent_gnn.py`, and `stage4/train_stage4.py`.
- Teammate's Stage 2 / Stage 3 code is not modified.
- 3 seeds: 7, 42, 123 (matching Stage 2 / Stage 3 finalists).
- `validation_enabled=True`, engine-level 15% split, `patience=10`.
- `apply_median_filter_to_test = False`.
- Graph: `physical` (Stage 3 winner) for FD002/FD003/FD004.
- Raw run summaries are stored under
  `DRIVE_ROOT/runs/{SUBSET}/{run_id}/summary.json`;
  aggregate in `DRIVE_ROOT/batch_summary.json`.
- The fully-executed Colab notebook will be committed at
  `stage4/notebooks/stage4_colab.ipynb` and serves as the reproducibility
  artifact for these numbers.
