# Stage 4: CBAM-style Attention Augmentation

## Goal

Add lightweight CBAM (Convolutional Block Attention Module) style attention on top of the Stage 2 / Stage 3 best architecture per subset, and quantify the contribution of each attention sub-module (channel-only, temporal-only, full CBAM) to RUL prediction quality.

The attention is deliberately small — a few hundred parameters — so it acts as an inductive-bias-preserving augmentation rather than a Transformer replacement.

## CBAM Adapted to (B, T=30, F)

We treat the input window as a 1-D sequence of `F` sensor channels and `T=30` time steps.

- **ChannelAttention** ("which sensors matter"): pool along time (avg + max) → shared MLP (`F → F/r → F`, `r=4`) → sigmoid gate broadcast over time. Shape: `(B, T, F) -> (B, 1, F) -> elementwise multiply -> (B, T, F)`.
- **TemporalAttention** ("which time steps matter"): pool along channels (avg + max) → 1-D conv `k=7` → sigmoid gate broadcast over channels. Shape: `(B, T, F) -> (B, T, 1) -> elementwise multiply -> (B, T, F)`.
- **CBAM1D**: channel then temporal in series. Output shape equals input shape — drops in directly before the recurrent / GNN branches.

## Layout

```
stage4/
├── README.md
├── models/
│   ├── __init__.py
│   ├── attention.py            # ChannelAttention / TemporalAttention / CBAM1D
│   ├── attn_recurrent.py       # AttnRecurrent  (CBAM + GRU/BiGRU)
│   └── attn_recurrent_gnn.py   # AttnRecurrentGNN (CBAM + stage3.RecurrentGNNFusion)
├── train_stage4.py             # training entry; reuses stage2_refactor fit/evaluate
├── ablation.py                 # ablation matrix runner
├── configs/stage4.yaml         # per-subset backbone + CBAM hyperparams
├── docs/
│   └── stage4_analysis.md      # written after the ablation completes
└── artifacts/                  # runs/, ablation/ (gitignored)
```

## Protocol

- `sequence_length = 30`
- `validation_enabled = True`, engine-level 15% split, `patience = 10`
- `apply_median_filter_to_test = False`
- Seeds: `{7, 42, 123}`
- Test-time evaluation: clamp `y_pred` and `y_true` to `[0, initial_rul]` before `rmse_score`
- Graph: `physical` (Stage 3 winner) for FD002/FD003/FD004
- Optimizer: Adam, weight_decay = 1e-5, lr = 5e-4

## Running

```bash
# Stage 4 with full CBAM, FD002, seed 42
python -m stage4.train_stage4 \
    --subset FD002 \
    --use-channel-attn --use-temporal-attn \
    --seed 42 \
    --run-id fd002_cbam_seed42

# Full ablation matrix
python -m stage4.ablation \
    --subsets FD001,FD002,FD003,FD004 \
    --seeds 7,42,123
```

## Ablation Matrix

3 attention configurations × 4 subsets × 3 seeds = **36 runs**. Each cell is compared with its Stage 3 baseline.

| Cell | Baseline (no Stage 4 attention) | Stage 4 variant |
|---|---|---|
| channel-only | Stage 3 finalist | + ChannelAttention   |
| temporal-only | Stage 3 finalist | + TemporalAttention |
| full CBAM    | Stage 3 finalist | + ChannelAttention then TemporalAttention |

## Expected Outcome

Per the project methodology hypothesis, channel attention should help most on FD002 / FD004 (multi-condition, multi-fault) where inter-sensor weighting is genuinely informative. FD001 / FD003 (single condition) should see at most a small gain.
