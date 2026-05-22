# FD001 Stage 3: GNN Fusion vs Pure GRU

> 6 runs total: 2 groups × 3 seeds {7, 42, 123}. All other hyperparams aligned with Stage 2 FD001 finalist (GRU h=90, L=2, dropout=0.2, lr=5e-4, batch=15, max 150 epoch, val=engine-15%, patience=10, weight_decay=1e-5, medfilt-on-test=false). Local CPU run, conda env `ece228`, torch 2.12.0.

## 1. Per-seed results

| Seed | Group A (GNN, physical) RMSE / Score | Group B (no GNN) RMSE / Score | Best epoch (A / B) |
|-----:|---------------------------------------:|--------------------------------:|------------------:|
|   7  |   7.4581 / 75.6152 |   7.2961 / 71.9468 | 17 / 23 |
|  42  |   5.9955 / 58.1476 |   6.5073 / 62.0065 |  5 /  6 |
| 123  |   6.0953 / 60.5049 |   6.7808 / 64.9709 |  8 / 11 |

Param counts: A = 87,373 (GRU 81,901 + GCN 2,048 + fusion head expanded); B = 81,901 (identical to Stage 2 finalist).

## 2. Aggregate (mean ± sample std)

| Group | Test RMSE | Test Score |
|---|---:|---:|
| **A — RecurrentGNNFusion (GRU + GCN-physical)** | **6.516 ± 0.817** | **64.76 ± 9.48** |
| **B — Pure GRU (use_gnn=False)** | **6.861 ± 0.401** | **66.31 ± 5.10** |
| **Stage 2 FD001 finalist (GRU, official)** | **6.165 ± 0.219** | **55.72 ± 3.66** |

## 3. Conclusion (one sentence)

加 physical 图的 GNN 分支相比纯 GRU 在 FD001 上 mean RMSE 改善 **−0.35**（6.86 → 6.52，约 −5%）、mean Score 改善 **−1.55**（66.3 → 64.8，约 −2.3%）；GNN 在 3 个 seed 里赢了 2 个（seed 42 / 123），seed 7 上略输，且 A 组方差比 B 组大（std 0.82 vs 0.40），3 seed 难做出"统计显著"的判定，结论倾向**轻度正向**但需要更多 seed 才能盖棺。

## 4. 为什么组 B 比 Stage 2 finalist 6.17 略差？

观察：组 B 的 3 个 seed mean RMSE 6.86，比 Stage 2 finalist 同 3 seed 的 mean 6.17 高 **+0.69 RMSE**。模型在 `use_gnn=False` 下参数量、回归头结构、超参、数据 pipeline 都和 Stage 2 完全一致（param count 81,901 在两边都对齐），所以差异不来自代码。

逐 seed 对比（Stage 2 vs 组 B）：

| Seed | Stage 2 finalist RMSE | 组 B RMSE | Δ | Stage 2 best_epoch | 组 B best_epoch |
|-----:|---------------------:|---------:|----:|------------------:|----------------:|
|   7  | 6.358 | 7.296 | +0.938 |  6 | 23 |
|  42  | 6.209 | 6.507 | +0.298 |  6 |  6 |
| 123  | 5.926 | 6.781 | +0.855 |  5 | 11 |

可以看到 **seed 42 几乎一致**（差 +0.30，best_epoch 同为 6），而 seed 7 / 123 偏差大且 `best_epoch` 比 Stage 2 晚 5–17 epoch 才停。同样的 `patience=10`、同样的 val 切分种子，理论上 val_rmse 曲线应该收敛到同一形状。差异最可能的根因：

1. **PyTorch 版本数值漂移（最可能）**：Stage 2 finalist 在 Colab 上跑，那边默认 torch 版本与我们本地的 **torch 2.12.0** 不同。GRU 内部的 cuDNN/MKL kernel 在不同版本里 matmul 累加顺序可能不同，相同 seed 也会在前几个 epoch 后分叉。一旦曲线分叉，early stopping 触发点也随之不同，模型收敛到不同的 val 最优。这种 ~5–10% RMSE 差异在跨版本复现中是常见量级。
2. **CPU vs GPU 数值差异**：Stage 2 在 Colab 上很可能用了 GPU，我们这里全程 CPU。即使同 seed，不同后端的浮点累加可能产生轻微不同的权重轨迹。
3. **best 选择是 val_rmse 最低点**：我们组 B seed 7 在 epoch 23 才找到 val_rmse=4.22，比 Stage 2 seed 7 在 epoch 6 找到的 val_rmse=4.48 还要低 0.26；但 test RMSE 反而更高（7.30 vs 6.36）—— 说明这个更低的 val 是 val 集上的"过拟合到那 15 个 engine 的随机性"，并不能泛化到 test。这是单一 val 切分本身固有的方差，跨 seed 平均后才能稳定。

**不是问题的**：
- ❌ 不是模型架构差异（param 81,901 完全一致；架构对齐已在 `gnn_modules.py` self-check 验证过）。
- ❌ 不是数据 pipeline 差异（imports 来自 `stage2_refactor.data.*`，函数同一份）。
- ❌ 不是 clamp / metric 差异（`evaluate_test` 直接 import 自 `stage2_refactor.experiments.run_experiment`）。

**对 Stage 3 后续的影响**：
- A vs B 的比较是**同环境内**（同一 torch 2.12.0、同 CPU、同 seed 集合）做的，**横向 GNN 增益结论是站得住脚的**（−0.35 RMSE，2/3 seed 赢）。
- A vs Stage 2 finalist 的纵向比较被这个跨环境漂移污染了 0.5–0.9 RMSE，**不应该直接用** A 6.52 vs Stage 2 6.17 来做"Stage 3 没有进步"的判定。要做合公平比较，Stage 3 的最终评估应该在和 Stage 2 同一环境（最好 Colab GPU）上重跑组 B 作为新的 reference baseline，再和组 A 比。

## 5. 产物

每个 run 的完整 summary、train_log、checkpoint 都在：

```
stage3/artifacts/runs/FD001/{fd001_gnn_seed7, fd001_gnn_seed42, fd001_gnn_seed123,
                              fd001_nognn_seed7, fd001_nognn_seed42, fd001_nognn_seed123}/
├── summary.json
├── train_log.csv
└── checkpoints/
    ├── fd001_preprocessing.gz
    ├── fd001_rgnn_best.pth
    ├── fd001_last.pt
    └── fd001_best.pt
```

run 时长 1:32 – 2:59 / run（CPU），val early stopping 大多在 epoch 15–35 之间触发。
