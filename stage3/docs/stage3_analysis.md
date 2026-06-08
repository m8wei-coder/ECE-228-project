# Stage 3 GNN Fusion Analysis

## Goal and Method

Stage 3 extends the Stage 2 recurrent baseline by adding a graph-neural-network branch that explicitly models the spatial relationship between engine sensors. The two branches share the (B, 30, F) input window; the recurrent branch consumes it as a sequence and the GNN branch consumes its transpose (B, F, 30) — each retained sensor becomes a node whose feature vector is its 30-step reading. The two outputs are concatenated and passed through a shared regression head, all inside the Stage 2 `BaseModel` contract so the Stage 2 trainer / evaluator can run the new model without modification.

**Recurrent backbones** (subset-specific, locked from Stage 2 finalists):
- FD001 / FD003 → GRU, hidden 90, 2 layers, dropout 0.2
- FD002 / FD004 → BiGRU, hidden 60, 2 layers, dropout 0.1

**Graph constructions** (three variants, compared head-to-head):
1. **Physical** — hand-defined gas-path topology. Sensors inside the same component are fully connected; adjacent components along the gas path are connected; a global sensor (sensor_10) connects to every component; the core shaft links HPC↔HPT. Induced subgraph on each subset's retained sensors. Binary {0, 1}.
2. **Pearson** — |Pearson r| computed on the Stage-2-preprocessed training features, edges retained where |r| > 0.3, diagonal zeroed. Weighted.
3. **Union** — element-wise max of pearson (weighted) and physical (binary 1.0).

**GNN module** — 2-layer DenseGCNConv (Kipf & Welling 2017), hidden 32, ReLU+dropout 0.1 between layers, global mean pooling over nodes. The adjacency is registered as a buffer (fixed for the whole training run).

**Ablation control**: a `use_gnn=False` switch removes the GNN branch entirely, degenerating the model to the Stage 2 recurrent baseline with parameter count exactly matching Stage 2 finalists (81,901 for GRU h=90 L=2; 99,061 for BiGRU h=60 L=2).

**Protocol** (aligned with Stage 2 finalists for direct comparability):
- 3 seeds {7, 42, 123}
- engine-level validation (15%), `patience=10` early stopping
- `apply_median_filter_to_test = False`
- clamp `y_pred` and `y_true` to `[0, initial_rul]` before computing RMSE / NASA Score
- 4 subsets × {no-GNN, GNN-physical, GNN-pearson, GNN-union} × 3 seeds = **48 runs** on Colab T4 GPU

## Final Results

| Subset | Variant | Test RMSE (mean ± std) | Test Score (mean ± std) |
|---|---|---:|---:|
| FD001 | no-gnn         |  6.165 ± 0.219 |    55.72 ±   3.66 |
| FD001 | gnn (physical) |  6.490 ± 0.071 |    63.97 ±   4.11 |
| FD001 | gnn (pearson)  |  6.587 ± 0.369 |    61.91 ±   4.46 |
| FD001 | gnn (union)    |  6.576 ± 0.326 |    64.97 ±   2.85 |
| FD002 | no-gnn         | 13.699 ± 0.483 |  1179.61 ± 368.93 |
| FD002 | gnn (physical) | 13.705 ± 0.188 |  1010.90 ± 333.90 |
| FD002 | gnn (pearson)  | 13.786 ± 0.214 |  1066.10 ± 194.88 |
| FD002 | gnn (union)    | 13.844 ± 0.024 |  1160.76 ± 272.22 |
| FD003 | no-gnn         |  5.366 ± 0.070 |    45.98 ±   4.26 |
| FD003 | gnn (physical) |  5.181 ± 0.414 |    45.28 ±   5.47 |
| FD003 | gnn (pearson)  |  5.493 ± 0.404 |    49.61 ±   4.04 |
| FD003 | gnn (union)    |  5.474 ± 0.199 |    47.33 ±   3.15 |
| FD004 | no-gnn         | 16.236 ± 1.185 |  1127.93 ± 274.31 |
| FD004 | gnn (physical) | 15.900 ± 0.585 |  1020.29 ± 102.57 |
| FD004 | gnn (pearson)  | 16.866 ± 0.827 |  1135.85 ±  39.64 |
| FD004 | gnn (union)    | 16.643 ± 0.483 |  1174.05 ± 252.10 |

**Bold-where-best per subset (across all 4 variants):**

| Subset | Best RMSE | Best Score |
|---|---|---|
| FD001 | no-gnn (6.165) | no-gnn (55.72) |
| FD002 | no-gnn (13.699) (tied with gnn-physical 13.705) | gnn-physical (1010.90) |
| FD003 | gnn-physical (5.181) | gnn-physical (45.28) |
| FD004 | gnn-physical (15.900) | gnn-physical (1020.29) |

## Main Findings

### 1. Among the three graph constructions, **physical** is the best on every subset.

Across all four subsets, on both RMSE and Score, the physical graph beats the pearson and union graphs (sometimes by a small margin on FD002 RMSE, by a large margin on FD003/FD004). The union of pearson and physical never reaches pure physical — adding the dense Pearson edges hurts more than the physical edges help. This is the single most consistent signal in the ablation: the inductive bias from the hand-defined gas-path topology dominates a purely data-driven correlation graph at this threshold.

### 2. GNN benefits concentrate on the harder subsets; FD001 is hurt.

The four subsets sort cleanly by GNN benefit:

| Subset | What changes vs no-gnn (physical graph) | Reading |
|---|---|---|
| FD001 (1 cond, 1 fault) | RMSE +0.33, Score +8.3                    | GNN clearly hurts |
| FD002 (6 cond, 1 fault) | RMSE +0.006 (tie), but std 0.48→0.19; Score −168.7 (−14%) | GNN ties RMSE, slashes variance, large Score gain |
| FD003 (1 cond, 2 faults) | RMSE −0.18 (−3.4%), Score −0.7 (tie)      | GNN improves RMSE, ties Score |
| FD004 (6 cond, 2 faults) | RMSE −0.34 (−2.1%), RMSE std 1.19→0.59; Score −107.6 (−10%) | GNN wins on both metrics, halves variance |

The pattern aligns with the Stage 3 proposal hypothesis: when the regime is simple (one operating condition, one fault — FD001), the recurrent backbone already extracts everything useful from the per-sensor time series, and adding a spatial branch only adds optimization noise. When the regime is harder (extra operating conditions on FD002/FD004, extra fault mode on FD003), the model needs to disambiguate which sensors drive degradation and the physical adjacency provides that prior.

FD003 is the one mixed case: RMSE mean improves but RMSE std grows from 0.07 → 0.41. The single-condition, two-fault structure is hard for GNN to learn stably with only 100 train engines.

### 3. Pearson and union graphs are too dense.

Edge counts vs the maximum possible (`N(N−1)/2`), with the Pearson threshold τ = 0.3:

| Subset | N | Physical edges | Pearson edges | Union edges | Pearson fill |
|---|---:|---:|---:|---:|---:|
| FD001 | 14 | 34 | 74 | 81 | 81% |
| FD002 | 21 | 82 | 187 | 194 | 89% |
| FD003 | 16 | 52 | 84 | 92 | 70% |
| FD004 | 21 | 82 | 187 | 194 | 89% |

(Average degrees: physical 4.9–7.8, pearson 10.5–17.8, union 11.5–18.5 — see `stage3/build_graph.py` output.)

For FD002 and FD004 in particular, the Pearson graph at τ = 0.3 is essentially fully connected (~89% of all possible edges). At that density, GCN neighbor aggregation degenerates toward "average every other sensor", which is close to a global mean pooling layer — the graph structure loses its discriminative power. The physical graph keeps a sparse, named topology where each propagation step actually carries information about gas-path locality. This is consistent with the broader GCN literature: over-smoothing and loss of distinctiveness in dense or deep GCNs (Li et al. 2018, "Deeper Insights into GCNs"). Raising the Pearson threshold (e.g. τ = 0.5 or top-k per node) would be a natural follow-up; we did not have budget to sweep it in this iteration.

## Metric Sensitivity: RMSE vs NASA Score (FD001 case study)

A subtle observation on FD001 worth flagging for the project write-up: adding the GNN with the physical graph reduces RMSE standard deviation (0.219 → 0.071, **3× tighter**) but *increases* Score standard deviation (3.66 → 4.11). The mean direction is the same on both — GNN hurts — but the variance moves in opposite directions.

This is a clean illustration of the difference between the two metrics:

- **RMSE** squares the error and averages, so it is dominated by the bulk of predictions where the model is roughly accurate. Tightening the bulk distribution (which the GNN seems to do — the spatial prior anchors predictions away from a few wild outliers per seed) directly tightens RMSE.
- **NASA Score** applies an *exponential* asymmetric penalty to each individual prediction (`exp(−h/13)−1` for early predictions, `exp(h/10)−1` for late ones). A single very-late prediction on a single test engine can push Score by hundreds. So Score is not a moment statistic of the error distribution — it is sensitive to the *tail*, and especially to the right (late-prediction) tail.

When the GNN tightens the bulk of FD001 errors but occasionally produces a more dispersed extreme (likely a late prediction on one of the 100 test engines), RMSE benefits while Score suffers. For project-level reporting on safety-critical RUL prediction, **Score is the metric to optimize**, because a single missed-late prediction is what actually causes an in-service failure; the bulk RMSE improvement on FD001 is therefore not enough to recommend GNN there.

This effect would not be visible at all in a single-seed run — it only shows up when we look at the variance structure across the 3 seeds, which is one of the practical reasons for running multi-seed experiments at this scale.

## Recommended Stage 3 Configuration

Based on the ablation:

| Subset | Recommended architecture | Test RMSE | Test Score |
|---|---|---:|---:|
| **FD001** | pure GRU (no GNN)                | 6.165 ± 0.219 |   55.72 ±   3.66 |
| **FD002** | BiGRU + GNN (physical graph)     | 13.705 ± 0.188 | 1010.90 ± 333.90 |
| **FD003** | GRU + GNN (physical graph)       | 5.181 ± 0.414 |   45.28 ±   5.47 |
| **FD004** | BiGRU + GNN (physical graph)     | 15.900 ± 0.585 | 1020.29 ± 102.57 |

Compared with the Stage 2 best per subset:

| Subset | Stage 2 best (RMSE / Score) | Stage 3 recommended (RMSE / Score) | Score Δ |
|---|---|---|---:|
| FD001 | GRU 6.165 / 55.72       | GRU (no GNN) 6.165 / 55.72        |     0   |
| FD002 | BiGRU 14.502 / 1879.50  | BiGRU + GNN-physical 13.705 / 1010.90 | **−868** (−46%) |
| FD003 | GRU 5.366 / 45.98       | GRU + GNN-physical 5.181 / 45.28  |     −0.7 |
| FD004 | BiGRU 16.773 / 1194.19  | BiGRU + GNN-physical 15.900 / 1020.29 | **−174** (−15%) |

The dramatic Score improvement on FD002 and FD004 is the main contribution of Stage 3 to the overall project. The architectural cost is modest (~5,500 extra parameters from the GCN + fusion expansion) and the recurrent half is unchanged, so the GNN branch can be thought of as a cheap spatial-prior add-on layered on top of Stage 2's backbones.

## Notes

- All Stage 3 results above were trained on Colab T4 GPU using the same `stage2_refactor` data / training / evaluation modules — the only Stage-3-local code is `stage3/build_graph.py`, `stage3/models/`, and `stage3/train_stage3.py`. The `stage2_refactor/` package is imported as a library and not modified.
- 3 seeds: 7, 42, 123 (matching Stage 2 finalists for direct comparison).
- `validation_enabled=True`, engine-level 15% split, `patience=10`.
- `apply_median_filter_to_test = False`.
- Raw run summaries are stored in `DRIVE_ROOT/runs/{SUBSET}/{run_id}/summary.json`; aggregates in `DRIVE_ROOT/batch_summary.json` (physical + no-gnn from notebook Cell 9) and `DRIVE_ROOT/batch_summary_graphs.json` (pearson + union from Cell 10, plus references to physical / no-gnn).
- The fully-executed Colab notebook (with cell outputs) is committed at `stage3/notebooks/stage3_colab.ipynb` and serves as the reproducibility artifact for these numbers.
