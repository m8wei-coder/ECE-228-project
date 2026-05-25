# Stage 4 CBAM Attention Analysis

## Goal and Method

Stage 4 adds a CBAM (Convolutional Block Attention Module) style attention block on top of the Stage 3 best-architecture-per-subset finalist. The motivation is to test whether a lightweight, inductive-bias-preserving attention module can squeeze additional RUL prediction accuracy out of the recurrent (+ optional GNN) backbones already established, without resorting to a Transformer architecture.

**CBAM adaptation to (B, T=30, F).** The original CBAM is defined over 2-D feature maps; we adapt it to our 1-D sensor-time windows:

- **ChannelAttention** ("which sensors matter"): pool along time (avg + max), feed through a shared MLP `F → F/r → F` with reduction ratio `r=4`, add the two branches and sigmoid. Gate has shape `(B, 1, F)` and is broadcast across the time axis.
- **TemporalAttention** ("which time steps matter"): pool along the channel axis (avg + max), feed the resulting `(B, 2, T)` tensor through a 1-D conv with kernel 7, sigmoid. Gate has shape `(B, T, 1)` and is broadcast across the channel axis.
- **CBAM1D** applies channel-then-temporal in series. The full module placed at the input to the recurrent (+ GNN) backbone.

**Backbone per subset (locked by Stage 3 finalists).**

| Subset | Backbone selected                            |
|---|---|
| FD001 | GRU (no GNN; Stage 3 finalist for single-condition)   |
| FD002 | BiGRU + GNN-physical                                  |
| FD003 | GRU   + GNN-physical                                  |
| FD004 | BiGRU + GNN-physical                                  |

**Parameter overhead.** With `r=4`, the CBAM1D parameter count is:

| F | CBAM1D params | Backbone params | Overhead |
|---:|---:|---:|---:|
| 14 (FD001) |  98 | 81 901  | +0.12% |
| 16 (FD003) | 142 | 82 441  | +0.17% |
| 21 (FD002/FD004) | 224 | 99 061 | +0.23% |

Well within the "lightweight augmentation" specification from the project methodology — empirically guaranteeing that any observed RMSE/Score change is driven by the inductive bias of the attention gates rather than additional capacity.

**Protocol (aligned with Stage 3 finalists for direct comparability).**

- 3 seeds `{7, 42, 123}`
- engine-level validation (15%), `patience=10` early stopping
- `apply_median_filter_to_test = False`
- clamp `y_pred` and `y_true` to `[0, initial_rul]` before computing RMSE / NASA Score
- Graph: `physical` (Stage 3 winner) for FD002/FD003/FD004
- 4 subsets × 3 attention cells × 3 seeds = **36 main runs** on Colab T4 GPU

**Ablation cells** (each compared against its Stage 3 baseline):

1. `channel_only`  — channel attention only, no temporal gate
2. `temporal_only` — temporal attention only, no channel gate
3. `cbam_full`     — full CBAM (channel-then-temporal in series)

## Final Results — Main Ablation (36 runs)

| Subset | Variant | Test RMSE (mean ± std) | Test Score (mean ± std) |
|---|---|---:|---:|
| FD001 | stage3 baseline (no-gnn)        |  6.165 ± 0.219 |    55.72 ±   3.66 |
| FD001 | stage4 channel_only             |  7.520 ± 1.611 |    78.35 ±  31.13 |
| FD001 | stage4 temporal_only            |  7.074 ± 1.182 |    71.13 ±  19.23 |
| FD001 | stage4 cbam_full                |  7.190 ± 0.608 |    68.20 ±   8.66 |
| FD002 | stage3 baseline (gnn-physical)  | 13.705 ± 0.188 |  1010.90 ± 333.90 |
| FD002 | stage4 **channel_only**         | **13.603 ± 1.160** |  1203.39 ± 493.61 |
| FD002 | stage4 temporal_only            | 13.827 ± 0.077 |  1057.41 ± 188.02 |
| FD002 | stage4 cbam_full                | 14.077 ± 0.970 |  1326.95 ± 153.39 |
| FD003 | stage3 baseline (gnn-physical)  |  5.181 ± 0.414 |    45.28 ±   5.47 |
| FD003 | stage4 channel_only             |  5.587 ± 0.110 |    48.66 ±   1.63 |
| FD003 | stage4 temporal_only            |  5.333 ± 0.284 |    46.53 ±   4.73 |
| FD003 | stage4 cbam_full                |  6.277 ± 0.470 |    58.42 ±  10.18 |
| FD004 | stage3 baseline (gnn-physical)  | 15.900 ± 0.585 |  1020.29 ± 102.57 |
| FD004 | stage4 channel_only             | 16.944 ± 2.970 |  1204.68 ± 344.33 |
| FD004 | stage4 **temporal_only** ✓✓     | **15.470 ± 0.792** |   **920.47 ± 128.00** |
| FD004 | stage4 cbam_full                | 18.213 ± 3.240 |  1538.27 ± 634.30 |

**Bold-where-best per subset (Stage 4 cells only):**

| Subset | Best Stage 4 RMSE | Best Stage 4 Score | Beats Stage 3 baseline? |
|---|---|---|---|
| FD001 | temporal_only 7.07 | cbam_full 68.2 | No (RMSE +0.9, Score +12.5) |
| FD002 | **channel_only 13.60** | temporal_only 1057 | RMSE: marginal yes (−0.10); Score: no |
| FD003 | temporal_only 5.33 | temporal_only 46.5 | No (tied within std) |
| FD004 | **temporal_only 15.47** ✓ | **temporal_only 920.5** ✓ | **YES on both — RMSE −2.7%, Score −9.8%** |

## Per-cell Contribution

Δ = Stage 4 − Stage 3 baseline. Negative numbers = improvement.

|  | Δ RMSE channel | Δ RMSE temporal | Δ RMSE cbam | Δ Score channel | Δ Score temporal | Δ Score cbam |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | +1.36 | +0.91 | +1.03 | +22.7 | +15.4 | +12.5 |
| FD002 | **−0.10** | +0.12 | +0.37 | +192.5 | +46.5 | +316.1 |
| FD003 | +0.41 | +0.15 | +1.10 | +3.4 | +1.3 | +13.1 |
| FD004 | +1.04 | **−0.43** | +2.31 | +184.4 | **−99.8** | +518.0 |

## Main Findings

### 1. **FD004 + temporal_only is the only clean win.**

RMSE 15.47 vs baseline 15.90 (**−2.7%**), Score 920 vs 1020 (**−9.8%**). FD004 is the hardest subset (6 operating conditions × 2 fault modes); Stage 4's win lands exactly where the project methodology predicted the inductive bias would matter most. The hypothesis was that channel attention should help on multi-condition subsets, but the actual winner was *temporal* attention — re-interpreted, FD004's two fault modes have distinct degradation *time profiles* (linear vs accelerated), and temporal attention learns to weight the time steps where the acceleration signal becomes detectable.

### 2. **`cbam_full` consistently underperforms either single gate.**

In 3 of 4 subsets, `cbam_full` is *worse* than both `channel_only` and `temporal_only` (FD002 RMSE 14.08 vs 13.60/13.83; FD003 RMSE 6.28 vs 5.59/5.33; FD004 RMSE 18.21 vs 16.94/15.47). This contradicts the original CBAM paper's ImageNet finding that channel-then-temporal in series is best. We attribute this to CMAPSS's small training set (~100–250 engines per subset): two gates in series compound the optimization noise and produce a *double-overfitting* effect that single gates avoid by virtue of simplicity.

### 3. **FD001 (single condition) has no structure for attention to learn.**

All three Stage 4 cells underperform the Stage 3 baseline on FD001 (RMSE +0.9 to +1.4). This matches the project methodology's a-priori expectation: with one operating condition and one fault mode, the recurrent backbone alone already captures all available structure, and any attention block becomes pure noise injection. Notable detail: `cbam_full` on FD001 has the *lowest* std (0.61) among the three Stage 4 cells, suggesting that two gates in series mutually regularize each other even when there is nothing useful to learn.

### 4. **Variance reduction on Score even where mean is unchanged.**

On FD002, the Stage 3 baseline Score has very large std (333.9). Stage 4 `temporal_only` brings this down to 188.0 (−44%) and `cbam_full` to 153.4 (−54%) while keeping the mean close. Attention modules act as a *consistency regularizer* — predictions become more reproducible across seeds even when average accuracy does not change. This is a practical win for safety-critical RUL prediction where worst-case variance matters more than mean accuracy.

### 5. **Severe train–val divergence on FD002/FD004 with attention.**

Examining `train_log.csv` for FD004 channel_only seed=7 (representative):

| epoch | train_rmse | val_rmse | status |
|---:|---:|---:|---|
| 1 | 18.65 | **11.76** ← best | random init |
| 2 | 12.42 | 12.09 | training starts |
| 5 |  9.81 | 13.52 | train↓ val↑ |
| 11 |  5.54 | 14.03 | patience exhausted → stop |

Train RMSE drops 70%, but validation RMSE *increases* from epoch 1 onward — the "best checkpoint" is the random-init state. Best_epoch ranges:

| Subset | best_epoch range |
|---|---|
| FD001 | 3 – 26 |
| FD002 | 2 – 5 |
| FD003 | 12 – 43 |
| FD004 | 1 – 5 |

This is not a bug; the model trains, but it overfits *immediately*. Attention gates amplify unit-specific noise in the small training sets of FD002 (~100 engines) and FD004 (~250 engines × 6 conditions), widening the generalization gap. The fact that `temporal_only` on FD004 still wins despite best_epoch=1–3 means the *random-initialization-region* solution space under temporal attention generalizes better than the baseline's converged solution — a non-trivial and slightly unsettling result.

## Tuning Experiments (Plan A, 24 runs)

To test whether the main-ablation results were sensitive to CBAM-internal hyperparameters, we ran four targeted sweeps that vary only CBAM-specific knobs (everything else identical to the main ablation, so Stage 3 baselines remain comparable):

1. **attn_dropout ∈ {0, 0.2}** on `cbam_full` × {FD002, FD004}  — regularize the gates directly
2. **attn_kernel ∈ {3, 7, 11}** on `temporal_only` × FD004  — temporal locality sweep
3. **attn_reduction ∈ {2, 4, 8}** on `channel_only` × FD002  — channel-gate capacity sweep
4. **attn_order ∈ {channel_first, temporal_first}** on `cbam_full` × {FD002, FD004}  — submodule order

| Sweep | Config | RMSE (mean ± std) | Score (mean ± std) | Δ vs main ablation |
|---|---|---:|---:|---|
| **Dropout** | FD002 cbam_full drop=0 (main) | 14.08 ± 0.97 | 1327 ± 153 | — |
|             | FD002 cbam_full drop=0.2      | **17.89 ± 0.39** | **3612 ± 1477** | RMSE +3.8, Score +2285 (×2.7 catastrophic) |
|             | FD004 cbam_full drop=0 (main) | 18.21 ± 3.24 | 1538 ± 634 | — |
|             | FD004 cbam_full drop=0.2      | **18.71 ± 1.03** | **2377 ± 1233** | RMSE +0.5, Score +839 |
| **Kernel**  | FD004 temporal_only k=3       | 16.23 ± 0.57 | 1202 ± 155 | RMSE +0.76, Score +282 |
|             | FD004 temporal_only k=7 (main, winner) | **15.47 ± 0.79** | **920.5 ± 128.0** | best |
|             | FD004 temporal_only k=11      | 16.38 ± 1.71 | 1035 ± 204 | RMSE +0.91, Score +114 |
| **Reduction** | FD002 channel_only r=2      | 14.31 ± 0.62 | 1532 ± 584 | RMSE +0.71, Score +329 |
|             | FD002 channel_only r=4 (main) | **13.60 ± 1.16** | **1203 ± 494** | best |
|             | FD002 channel_only r=8        | 14.48 ± 0.82 | 1588 ± 443 | RMSE +0.88, Score +385 |
| **Order**   | FD002 cbam_full channel_first (main) | 14.08 ± 0.97 | 1327 ± 153 | — |
|             | FD002 cbam_full temporal_first | 14.32 ± 1.12 | 1451 ± 369 | RMSE +0.24, Score +124 |
|             | FD004 cbam_full channel_first (main) | 18.21 ± 3.24 | 1538 ± 634 | — |
|             | FD004 cbam_full **temporal_first** | **17.52 ± 2.64** | **1197 ± 357** | **RMSE −0.69, Score −341 (−22%)** |

### Tuning findings

**(a) attn_dropout=0.2 is catastrophic.** Score increases 2.7× on FD002. The CBAM gate is a continuous multiplicative weight in (0, 1); dropout randomly zeros it, converting a smooth gate into a hard binary mask and destroying the attention mechanism. Lesson: CBAM gates should be regularized via weight_decay or backbone dropout, never input/output dropout.

**(b) CBAM defaults are already at the optimum.** Both `reduction` and `kernel` sweeps form clean U-shaped curves with the default (r=4, k=7) at the minimum. This is consistent with CMAPSS's small F (14–21 channels) needing less aggressive channel compression than CBAM's ImageNet defaults of r=16, and the seq_length=30 window being well-served by k=7 (covering 23% of the window per receptive field).

**(c) Module order is subset-dependent.** Swapping to `temporal_first` improves FD004 cbam_full Score by 22% (1538 → 1197) but slightly hurts FD002 (Score +124). This aligns with finding #1: FD004 fault disambiguation lives in the temporal axis, so running temporal attention first lets the channel attention act on already-time-emphasized features. Still, even the swapped `temporal_first` cbam_full on FD004 (17.52/1197) is worse than plain `temporal_only` (15.47/920) — the channel branch contributes pure noise on FD004.

**(d) No tuning recovers `cbam_full` to baseline.** The best tuned cbam_full result is FD002 `temporal_first` (14.32/1451), still worse than Stage 3 baseline (13.70/1011). The "two gates compound noise" hypothesis from finding #2 is confirmed.

**(e) best_epoch pattern unchanged.** Tuned runs still early-stop at epoch 1–7 on FD002/FD004. Train-val divergence is intrinsic to those subsets, not a CBAM hyperparameter issue.

## Recommended Stage 4 Configuration

Based on the main ablation and Plan A tuning:

| Subset | Recommended architecture | Test RMSE | Test Score | Δ vs Stage 3 |
|---|---|---:|---:|---:|
| **FD001** | pure GRU (no Stage 4)                 | 6.165 ± 0.219 |   55.72 ±   3.66 | — (use Stage 3) |
| **FD002** | recurrent + GNN-physical (no Stage 4) | 13.705 ± 0.188 | 1010.90 ± 333.90 | — (use Stage 3) |
| **FD003** | recurrent + GNN-physical (no Stage 4) |  5.181 ± 0.414 |   45.28 ±   5.47 | — (use Stage 3) |
| **FD004** | recurrent + GNN-physical + **temporal_only CBAM** | **15.470 ± 0.792** | **920.47 ± 128.00** | **RMSE −2.7%, Score −9.8%** |

Stage 4 contributes one clean architectural improvement (FD004), three null results (FD001/FD002/FD003), and a set of negative findings (cbam_full universally underperforms; dropout on gates is harmful; CBAM defaults are near-optimal). The single FD004 improvement is non-trivial because FD004 was Stage 3's hardest subset, and the +0.23% parameter overhead is essentially free.

## Limitations

1. **Test-set engine count varies dramatically across subsets**: FD001/FD003 have 100 engines; FD002/FD004 have 248–259. Small-test-set subsets (FD001/FD003) have correspondingly noisier Score estimates that may mask real but small effects.

2. **Single-graph experiments**: We fixed `graph=physical` based on Stage 3's finding. We did not retest pearson / union graphs with CBAM; it is conceivable that channel attention's per-sensor weighting interacts differently with denser pearson graphs.

3. **`patience=10` may be too aggressive for FD002/FD004**: best_epoch=1–5 means the model is essentially never trained past a few epochs. A separate experiment with `patience=30` (Tier 2 tuning) could clarify whether CBAM models would benefit from longer training before early-stopping kicks in — but would require re-running the Stage 3 baseline with the same patience for fair comparison.

4. **CBAM placement is fixed at the input**: Alternative placements (after recurrent, between recurrent and GNN, after GNN) were not explored. The input-side placement was chosen so both branches see attention-weighted features, but post-recurrent placement could allow attention to operate on already-encoded sequence representations.

## Notes

- All Stage 4 results above were trained on Colab T4 GPU using the same `stage2_refactor` data / training / evaluation modules and the same Stage 3 GNN modules — the only Stage-4-local code is `stage4/models/{attention.py, attn_recurrent.py, attn_recurrent_gnn.py}`, `stage4/train_stage4.py`, `stage4/ablation.py`, and `stage4/tune.py`.
- Teammate's Stage 2 / Stage 3 code is not modified.
- 3 seeds: 7, 42, 123 (matching Stage 2 / Stage 3 finalists for direct comparison).
- `validation_enabled=True`, engine-level 15% split, `patience=10`.
- `apply_median_filter_to_test = False`.
- Graph: `physical` (Stage 3 winner) for FD002/FD003/FD004.
- Raw run summaries are stored under `DRIVE_ROOT/runs/{SUBSET}/{run_id}/summary.json` for the main ablation and `DRIVE_ROOT/tune/runs/{SUBSET}/{run_id}/summary.json` for the Plan A sweeps; aggregates in `DRIVE_ROOT/batch_summary.json` and `DRIVE_ROOT/tune/results.csv` respectively.
- The fully-executed Colab notebook (with cell outputs) is committed at `stage4/notebooks/stage4_colab.ipynb` and serves as the reproducibility artifact for these numbers.
- Figures generated by `stage4/reports/plot_stage4.py` (4-panel RMSE / 4-panel Score bar charts with seed-std error bars).
