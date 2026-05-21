# Stage 1 Baseline LSTM 指标汇总

> 跑法：在独立 conda 环境 `ece228`（python 3.11、scikit-learn 1.4.2、threadpoolctl 3.6.0、torch 2.12.0）里，对 4 个子集分别执行
> `python reproduce/export_artifacts.py --dataset FD00X`。
> 该脚本只做：加载已保存的 `.gz` 预处理工件 → 用 `apply_saved_preprocessing` 处理 train/test → 加载 `.pth` checkpoint → `model.eval()` + `no_grad` 推理 → 落盘。**无任何训练步**。

## 1. 测试集 RMSE / Score（vs 论文 Table 9）

NASA 不对称 Score 与论文一致：`h<0 → exp(-h/13)-1`，`h≥0 → exp(h/10)-1`。预测和真值在评估前都先 `clamp(0, initial_rul)`。

| 子集 | 我们 RMSE | 论文 RMSE | RMSE Δ | 我们 Score | 论文 Score | Score Δ |
|------|---------:|---------:|------:|----------:|---------:|--------:|
| FD001 |   **8.28** |   7.78 |  +0.50 |   **110.56** |    100 |    +10.56 |
| FD002 |  **18.62** |  17.64 |  +0.98 |  **5407.2** |   1440 |  +3967.2 |
| FD003 |   **7.04** |   8.03 |  −0.99 |    **79.98** |    104 |   −24.02 |
| FD004 |  **17.43** |  17.63 |  −0.20 |  **2036.2** |   2390 |   −353.8 |

观察：
- **FD001 / FD003 / FD004 都在合理范围内**（RMSE 偏离 ≤ 1.0；FD003、FD004 的 Score 实际还优于论文）。
- **FD002 Score 偏差很大**（5407 vs 1440）。RMSE 只多 ~1 但 Score 多 ~4×，说明在少量大误差样本上 `exp` 项被放大；和论文使用的具体超参 / 训练步数差异有关（FD002 工况复杂，训练 150 epoch 可能不够）。需要 Stage 3 报告时如实写出来。

## 2. 实际使用的 initial_rul（验证修复）

| 子集 | initial_rul（从 `.gz` 读出） | 论文 Table 7 | 之前的错误值 |
|------|---------------------------:|------------:|------------:|
| FD001 |  **81** |  78 | 116 |
| FD002 | **104** | 103 | 116 |
| FD003 |  **77** |  79 | 116 |
| FD004 |  **88** |  87 | （未受影响） |

**修复确认**：四个值都各自不同，且都贴近论文 Table 7 的 78/103/79/87（偏差 ±2 以内）。原先「三个子集都被算成 116」的 bug 已被修掉。

读取来源（每个子集）：`reproduce/logs/cmapss_condition_scalers_fd00X.gz["initial_rul"]`，同时也存在 `reproduce/exports/FD00X/metadata.pkl["initial_rul"]`。

## 3. Stage 3 需要的产物 — 实际路径

每个子集目录 `reproduce/exports/FD00X/` 下，**12 个文件**齐备（已通过 `ls` 验证存在）：

| 文件 | 用途 | FD001 shape / 备注 |
|---|---|---|
| `X_train.npy` | 训练滑窗张量 | (17731, 30, 14) |
| `y_train.npy` | 训练 label（clip 后） | (17731,) |
| `X_test.npy` | 测试每 engine 最后 30 步 | (100, 30, 14) |
| `y_test.npy` | 测试真值（来自 `RUL_FD00X.csv`） | (100,) |
| `corr_matrix.npy` | Pearson 相关矩阵（含 `RUL_absolute`） | (15, 15) ⇒ 14 sensors + RUL |
| `corr_matrix.csv` | 同上 CSV 版（带行/列名） | 同上 |
| `retained_sensors.pkl` | 保留传感器名 list | 14 个名字 |
| `metadata.pkl` | features / initial_rul / 超参 | dict |
| `metadata.csv` | 同上 CSV 版 | |
| `y_pred_lstm.npy` | baseline 测试预测 | (100,) |
| `baseline_lstm_metrics.pkl` | RMSE / Score / clamp 上界 | dict |
| `baseline_lstm_metrics.csv` | 同上 CSV 版 | |

各子集 shape 概览：

| 子集 | X_train | X_test | 保留传感器数 | corr_matrix |
|---|---|---|---:|---|
| FD001 | (17731, 30, 14) | (100, 30, 14) | 14 | (15, 15) |
| FD002 | (46219, 30, 21) | (259, 30, 21) | 21 | (22, 22) |
| FD003 | (21820, 30, 16) | (100, 30, 16) | 16 | (17, 17) |
| FD004 | (54028, 30, 21) | (248, 30, 21) | 21 | (22, 22) |

提醒：`corr_matrix` 末尾一行/列是 `RUL_absolute`，Stage 3 做节点图时按需切片成 `(N_sensors, N_sensors)`。同时它是在**原始未滤波、未归一化的 train CSV** 上算的——若 Stage 3 要"与模型输入一致的相关性"，得另起一条基于 `apply_saved_preprocessing` 后 df 的算法（见 `recon_report.md` E.②）。

## 4. 运行环境记录

```
conda env: ece228
python   : 3.11
sklearn  : 1.4.2     (与训练时保存 .gz 的版本一致)
threadpoolctl: 3.6.0  (修掉 macOS 上 get_config 返回 None 的崩溃)
torch    : 2.12.0
numpy    : 2.4.6
pandas   : 3.0.3
scipy    : 1.17.1
joblib   : 1.5.3
```

之后 Stage 3 的所有代码都在 `ece228` 环境里运行。
