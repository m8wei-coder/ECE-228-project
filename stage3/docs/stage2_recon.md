# Stage 2 框架摸底报告

> 范围：`stage2_refactor/` 全部 Python + 配置 + 报告，外加 `docs/data_contract.md`、`reports/*.csv`、`reports/stage2_analysis.md`。仅阅读，零改动。

## A. Stage 2 做了什么

### A.1 循环结构：四个 backbone，统一接口

实现在 `stage2_refactor/models/`：

| 模块 | 类 | 关键层 | head 输入维 |
|---|---|---|---:|
| `lstm.py` | `LSTMBaseline` | `nn.LSTM` | `hidden_size` |
| `gru.py` | `GRUBaseline` | `nn.GRU` | `hidden_size` |
| `bilstm.py` | `BiLSTMBaseline` | `nn.LSTM(bidirectional=True)` | `hidden_size*2`（concat `h_n[-2]`+`h_n[-1]`） |
| `bigru.py` | `BiGRUBaseline` | `nn.GRU(bidirectional=True)` | `hidden_size*2`（同上） |

**4 个模型共享同一个回归 head**：`Linear(H_in → H/2) → ReLU → Dropout → Linear(H/2 → 1)`。

单向版本（LSTM/GRU）走 `out[:, -1, :]` 取最后时间步；双向版本走 `h_n[-2]`、`h_n[-1]`（最后一层正/反向）concat。

配置入口在 `configs/lstm_baseline.yaml`，4 个 backbone 默认起点都是 `hidden_size=60 / num_layers=4 / dropout=0.1`（和 Stage 1 LSTM 一致），但每个子集另有 dataset-level 覆盖（详见 yaml 的 `datasets.FD00X`）。最终选用的 finalist 配置见 B 节。

### A.2 预处理：100% 复刻 Stage 1，外加一个可选的验证集

`data/preprocessing.py` 完全沿用 reproduce 那套逻辑（功能一一对应）：

| 步骤 | 函数 | 与 reproduce 的差异 |
|---|---|---|
| 中值滤波（kernel=5，逐 engine 逐 sensor） | `apply_median_filter` | 一致 |
| 删除 `drop_cols`（从 yaml 读，不是从 .gz） | `selected_sensor_columns` | 一致 |
| KMeans(6 cluster)（仅在 3 个 setting 上） | `fit_preprocessing` | 一致（参数 `random_state=42`, `n_init=10`） |
| 按 condition 分组 `StandardScaler` | `fit_preprocessing` | 一致 |
| Piecewise RUL knee point + clip | `calculate_piecewise_rul` | 算法逐行一致；得到的 `initial_rul` 也是 81/104/77/88（写在 `docs/data_contract.md` 表里） |
| 保存预处理工件（kmeans + scalers + features + initial_rul + 超参） | `save_preprocessing_artifact` → joblib.dump 到 `<subset>_preprocessing.gz` | 路径变了，键一致 |
| 滑窗 dataset | `data/dataloader.py:CMAPSSWindowDataset` | label 形状从 Stage 1 的标量改成 `(1,)`，batch 后是 `(B,1)`，配合 `BaseModel` 输出 `(B,1)` |
| 测试窗口（每 engine 最后 30 步，前向 pad） | `build_final_test_windows` | 一致 |

**两点 Stage 1 没有、Stage 2 加上的能力**：

1. **engine-level 验证集**（`split_by_engine_id` in `dataloader.py`）：按 `unit_number` 划分（不是按窗口），避免同一台引擎的窗口同时出现在 train/val。默认 `validation.enabled=false`（baseline 与 Stage 1 一致），**但 finalist 实验全部 `validation_enabled=True`**（见 `final_stage2_runs.csv`）。
2. **测试时中值滤波可配**：`apply_median_filter_to_test`（默认 `false`，与 `reproduce/main_test.py` 行为一致；我们之前跑出 RMSE 8.28 是因为走的 `export_artifacts.py` 路径，那里测试侧 medfilt 是开的——data_contract.md 第 137-140 行也写了这件事，FD001 LSTM checkpoint 在 medfilt off/on 下 RMSE 是 8.72/8.28）。

### A.3 数据契约

- **原始数据**：仓库根 `CMaps/` 下的 15 个 NASA C-MAPSS 原始 .txt（队友 commit `09ed2b9` 入库的）。
- **`read_cmapss_table`**（`data/io.py`）兼容两种输入：`.csv`（带表头或纯 26 列无表头）和 `.txt`（whitespace 分隔，无表头）。所以我们之前写的 `reproduce/csv/*.csv` 也能被它直接读。
- **26 列硬契约**：`unit_number, time_cycles, setting_1..3, sensor_1..21`。
- **保留传感器**：FD001 = 14（drop 1/5/6/10/16/18/19）、FD002 = 21、FD003 = 16（drop 1/5/16/18/19）、FD004 = 21。与我们 baseline_metrics.md 完全一致。
- **`initial_rul`**：FD001=81、FD002=104、FD003=77、FD004=88（与我们一致）。data_contract.md 明确写"these differ from the Stage 2 plan's expected 78/103/79/87... documents and preserves the Stage 1 actual values rather than forcing the plan values"。
- **dtype**：`float32`，window 数组也是 `np.float32`。
- **label shape**：每个样本 `(1,)`，batch 后 `(B, 1)`。

## B. Stage 2 结果

来源：`reports/final_stage2_runs.csv`（13 行 = 1 header + 12 finalist 单次 run）和 `reports/final_stage2_summary.csv`（4 行汇总）。Finalist 用 3 个 seed = {7, 42, 123}，`validation_enabled=True`。

### B.1 Final（每子集选定的"最佳循环主干"）

| Subset | Best | Config | RMSE mean ± std | Score mean ± std | Params |
|---|---|---|---:|---:|---:|
| FD001 | **GRU** | h=90, L=2, d=0.2, lr=5e-4 | **6.165 ± 0.219** | **55.72 ± 3.66** | 81,901 |
| FD002 | **BiGRU** | h=60, L=2, d=0.1, lr=5e-4 | **14.502 ± 1.226** | **1879.50 ± 688.11** | 99,061 |
| FD003 | **GRU** | h=90, L=2, d=0.2, lr=5e-4 | **5.366 ± 0.070** | **45.98 ± 4.26** | 82,441 |
| FD004 | **BiGRU** | h=60, L=2, d=0.1, lr=5e-4 | **16.773 ± 1.070** | **1194.19 ± 39.51** | 99,061 |

**所选 backbone**：**没有单一全局最优**。GRU 赢简单子集（FD001、FD003，单工况）；BiGRU 赢复杂子集（FD002、FD004，6 工况）。这两个配置都比 Stage 1 baseline 论文的"4-layer LSTM h=60"小、且效果更好。

`stage2_analysis.md` 在 §Recommendation 明确写："If Stage 3 requires one global recurrent backbone, **BiGRU** is the more conservative choice"。

### B.2 vs. 我们之前的 baseline (FD001-4 RMSE 8.28/18.62/7.04/17.43)

| 子集 | Stage 1 baseline (我们) | Stage 2 best | 改进 |
|---|---:|---:|---|
| FD001 |  8.28 |  6.17 | −2.11（GRU 大幅好于 LSTM） |
| FD002 | 18.62 | 14.50 | −4.12（BiGRU 帮助大） |
| FD003 |  7.04 |  5.37 | −1.67 |
| FD004 | 17.43 | 16.77 | −0.66（FD004 训练 epochs 仍少） |

**全部 4 个子集 Stage 2 都更好**。但有 2 个口径差异要注意：

1. **训练数据 vs. 测试数据**：Stage 2 finalist 启用了 validation（engine-level 切 15%），所以训练用的窗口比我们 baseline 少了 ~15%；尽管如此还是更好——说明 GRU/BiGRU + early stopping 比 LSTM 单纯靠 epoch 上限更有效。
2. **我们的 baseline 数 8.28 / 18.62 / 7.04 / 17.43 是用 medfilt-on 的 test 路径算的**（走 `export_artifacts.py`），data_contract.md 同口径下报 FD001=8.2575。所以"8.28 → 6.17"是同口径下的真实提升。

## C. 模型接口（Stage 3 关键）

### C.1 `BaseModel` 契约（`models/base.py`）

```python
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def validate_input(x):  # 强制 ndim == 3，即 (B, T, F)
    @staticmethod
    def validate_output(y): # 强制 shape == (B, 1)
```

- **输入**：`x` 必须是 `(B, T, F) = (B, 30, F_subset)` 的 float32 张量
- **输出**：必须是 `(B, 1)`，即每条样本一个 RUL 标量（包了一层）

每个具体模型的 `forward` 都显式调用 `validate_input(x)` + 末尾 `validate_output(out)`，所以**接口非常硬**。

### C.2 配置驱动？换模型容易吗？

**是配置驱动**。`experiments/run_experiment.py:build_model`（行 85-101）是个 dispatch dict：

```python
model_classes = {"lstm": LSTMBaseline, "gru": GRUBaseline,
                 "bilstm": BiLSTMBaseline, "bigru": BiGRUBaseline}
```

加新模型只要：① 在 `models/` 下加一个继承 `BaseModel` 的类；② 在 `model_classes` 字典里注册一个 key；③ 在 `argparse` 的 `--model choices=[...]` 里加一项；④ 在 yaml 的 `models.<name>` 加一个 block。**改 4 个地方，无需碰已有 backbone 代码**。

### C.3 有没有"只取 backbone 特征、不过回归头"的接口？

**没有**。所有 4 个 backbone 的 `forward` 都把 backbone 计算（取最后时间步 / concat 双向 h_n）和 fc1/fc2 写在一起，**没有暴露 `forward_features` 这种中间表征**。

对 Stage 3 GNN 分支融合的影响：要么 (1) 新写一个继承 `BaseModel` 的融合模型，**内部自己复用 `nn.LSTM`/`nn.GRU`**，不沿用已有的 4 个 backbone 类；要么 (2) 给每个 backbone 加 4-6 行的 `forward_features`，并把 `forward` 改成 `self.head(self.forward_features(x))`——这是侵入式改造。

## D. 训练 / 评估

### D.1 `training/trainer.py:fit`（训练循环）

- 每个 epoch 跑 `train_one_epoch`（标准前向/反向/optimizer step）
- 如果有 val_loader：跑 `evaluate_loader`，按 `val_rmse` 选 best；否则按 `train_loss` 选 best（Stage 1 行为）
- **checkpoint 三件套**：
  - `best_model_path`（`*_best.pth`）：纯 state_dict
  - `last_checkpoint_path`（`*_last_checkpoint.pt`）：含 model+optimizer+epoch+best_metric → 支持 `--resume`
  - `best_checkpoint_path`（`*_best_checkpoint.pt`）：best 时刻的完整快照
- **early stopping**：仅在启用 val 时生效（`patience` 来自 `config["validation"]["patience"]`，默认 10）
- **CSV logger** 写 `train_log.csv`，每 epoch 一行（train/val loss、rmse、score、epoch_seconds 等）。可选 wandb（`--use-wandb`）

### D.2 `training/evaluator.py:rmse_score`

```python
error = y_pred - y_true
rmse  = sqrt(mean(error**2))
score = sum( where(error<0, exp(-error/13)-1, exp(error/10)-1) )
```

完全是 NASA Score 标准式。**这个函数自己不做 clamp**。

### D.3 clamp 在哪？

clamp 在 `run_experiment.py:evaluate_test`（行 125-140）：

```python
y_pred = torch.clamp(y_pred, 0.0, float(initial_rul))
y_tensor = torch.clamp(y_tensor, 0.0, float(initial_rul))
rmse, score = rmse_score(y_pred, y_tensor)
```

**与 Stage 1 `main_test.py` 完全一致**：评估测试集前，y_pred 和 y_true 都先 clamp 到 `[0, initial_rul]`。但是注意 **训练阶段不 clamp**（`train_one_epoch` / `evaluate_loader` 都没 clamp），只在 test summary 那一步 clamp。这意味着早期 epoch 的 train_rmse 数字可能很大（被未 clamp 的预测拉爆），不能直接拿来当 final score 比对。

### D.4 怎么跑一次实验？

```bash
python -m stage2_refactor.experiments.run_experiment \
  --subset FD001 \
  --model gru \
  --mode train_eval \
  --run-id my_first_run \
  [--checkpoint-dir <dir>] [--output-dir <dir>] [--resume]
```

`--mode` 三选一：`train` / `eval` / `train_eval`。`--config` 默认指向 `stage2_refactor/configs/lstm_baseline.yaml`。

输出：
- `<output_dir>/<model>/<subset>/<run_id>/summary.json` —— 整轮结果（含 test_rmse、test_score、metadata、features list、initial_rul、parameter_count 等）
- `<output_dir>/.../train_log.csv` —— 每 epoch 日志
- `<checkpoint_dir>/.../<subset>_<model>_best.pth` —— 最佳模型权重
- `<checkpoint_dir>/.../<subset>_preprocessing.gz` —— 预处理工件（kmeans + scalers + features + initial_rul + 超参，与 Stage 1 `.gz` 结构一致）

run_id 不指定时自动生成 `seed42_h60_l4_d0p1_lr0p002_bs15` 这种。

## E. 给 Stage 3 的接入建议

### E.1 (a) 用 stage2_refactor / (b) 在 reproduce/ 上接？

**强烈推荐 (a) 复用 `stage2_refactor/`**。理由：

| 维度 | (a) stage2_refactor | (b) reproduce |
|---|---|---|
| 模型接口 | `BaseModel(B,T,F)→(B,1)` 已硬契约 | network.py 把 LSTM+head 写死，没有抽象 |
| 数据 pipeline | `fit_transform_train` + `build_train_val_loaders` 一行调到位 | 要自己手工组 `CMAPSSDataset` + `update_preprocessing_metadata` |
| 训练/checkpoint/log | 有 `fit()`、CSV/WandB logger、resume 支持 | 只有 `main.py` 的 inline 循环，无 resume |
| 验证集 | engine-level split 已实现 | 没有 |
| 评估口径 | `evaluate_test` 已含 clamp，和 Stage 1 同口径 | 散在 `main_test.py` / `export_artifacts.py` 两个版本 |
| Baseline 数 | finalist GRU/BiGRU 数已发布，可直接对照 | 只有 8.28/18.62/7.04/17.43 一个版本 |
| 配置驱动 | yaml，加新模型改 4 个文件 | 没有 |
| 已花的工程量 | 队友已封好 | 我得自己造一遍 |

唯一劣势：stage2_refactor 假设数据从 `CMaps/` 读，不依赖 `reproduce/exports/*.npy`。但这不算问题——它会动态构造滑窗，性能开销很小，且不会和 Stage 2 finalist 的数对不上。

### E.2 GNN 分支最干净的接入点

**最干净的方式：写一个继承 `BaseModel` 的新类 `RecurrentGNNFusion`，把循环主干和 GNN 都装进去，head 留给融合后的输出**。

伪代码草图（不要现在写）：

```python
# stage2_refactor/models/recurrent_gnn.py
class RecurrentGNNFusion(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 adj_matrix, gnn_hidden, gnn_layers, recurrent_kind="bigru"):
        super().__init__()
        # 1) 复用 stage 2 已有的循环模块（裸 nn.GRU/nn.LSTM，不是 GRUBaseline 整个）
        self.recurrent = nn.GRU(input_size, hidden_size, num_layers,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers>1 else 0)
        # 2) GNN over sensor nodes，adj 来自 reproduce/exports/FD00X/corr_matrix.npy（切掉 RUL 行/列）
        self.gnn = GCN(input_dim=..., hidden=gnn_hidden, layers=gnn_layers,
                       adj=adj_matrix)
        # 3) 融合 head: [h_recurrent, h_gnn] → fc → 1
        fused_dim = hidden_size*2 + gnn_hidden  # if bidirectional
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fused_dim//2, 1))

    def forward(self, x):
        self.validate_input(x)
        # 循环分支
        _, h_n = self.recurrent(x)
        h_seq = torch.cat([h_n[-2], h_n[-1]], dim=-1)        # (B, 2H)
        # GNN 分支
        node_feats = self._build_node_features(x)            # (B, N_sensors, d_node)
        h_graph = self.gnn(node_feats)                       # (B, gnn_hidden)
        # 融合
        out = self.head(torch.cat([h_seq, h_graph], dim=-1)) # (B, 1)
        self.validate_output(out)
        return out
```

注册：在 `experiments/run_experiment.py:build_model` 字典加 `"rgnn": RecurrentGNNFusion`；yaml 加一段 `models.rgnn`；argparse `--model choices` 加 `"rgnn"`。

**为什么不直接复用 `GRUBaseline` / `BiGRUBaseline`？** 因为它们的 `forward` 已经把 backbone 计算 + fc1/fc2 缝死了，没法只拿到 `(B, 2H)` 的 backbone 特征。复用裸 `nn.GRU` 是开销最低、最对称、不动现有代码的做法。

### E.3 数据格式与我 stage3 脚手架的兼容性

我之前在 `stage3/docs/baseline_metrics.md` 里假设产物都在 `reproduce/exports/FD00X/*.npy`。**stage2_refactor 不用这套产物**，它直接读 `CMaps/` 然后在内存里构造滑窗。差异：

| 我之前的假设 | Stage 2 框架的真实行为 | 处理 |
|---|---|---|
| 从 `reproduce/exports/FD00X/X_train.npy` 等读 | 调 `fit_transform_train` + `build_train_val_loaders` 动态产生 | 我 Stage 3 直接走 Stage 2 数据 pipeline，不再依赖 `.npy` |
| Pearson 邻接来自 `reproduce/exports/FD00X/corr_matrix.npy` | Stage 2 自己不算 corr | **保留 `corr_matrix.npy` 这条产线**（在 `reproduce/exports/` 已落盘），GNN 邻接从那里读。或者在 stage3/ 里加个一次性脚本，用 `transform_with_artifact` 输出的 preprocessed df 重新算 corr（这更符合"和模型输入同分布"原则）。 |
| 评估 clamp + initial_rul 来自 reproduce 的 `metadata.pkl` | Stage 2 的 `summary.json` 也含 `initial_rul`；preprocessing artifact `.gz` 里也有 | 用 Stage 2 的 `evaluate_test` 函数即可，不需要再读 `metadata.pkl` |

**结论**：我 Stage 3 代码应该 import `stage2_refactor`，不重复造数据 pipeline，邻接矩阵单独维护一份在 `stage3/artifacts/` 或继续用 `reproduce/exports/`。`reproduce/exports/*.npy` 现在变成"备用快照"，不是必需。

## F. 一些 Stage 3 要注意的"坑"

1. **Stage 2 finalist 的最佳 GRU 是 `h=90, L=2, d=0.2`**，**不是** yaml 默认的 `h=60, L=4, d=0.1`。yaml 里 `datasets.FD003` 段也是默认的 60/4/0.1。所以 yaml 默认配置不会直接复现 finalist 数字——要按 `final_stage2_runs.csv` 显式传 `--hidden-size 90 --num-layers 2 --dropout 0.2 --learning-rate 5e-4 --validation-enabled`。
2. **Stage 2 finalist 全部 `validation_enabled=True`**（engine-level 切 15%）。Stage 3 如果想公平对比 Stage 2，也要开 val，否则训练数据量不同。
3. **`apply_median_filter_to_test`**：yaml 默认 false（Stage 1 行为）。我们之前 export 用的是 true。**Stage 3 选哪种要先定**——影响约 ±0.5 RMSE。建议跟 Stage 2 同口径（false），因为 Stage 2 finalist 数字是 false 下产出的。
4. **训练时不 clamp**：Stage 2 trainer 早期 epoch 的 `train_rmse` 数字会很大（log 里看着吓人），只有 test 时才 clamp。Stage 3 监控曲线时心里有数。
5. **`stage2_refactor/__init__.py` 是空文件**（已确认），但 `models/__init__.py` 注册了 4 个 backbone。Stage 3 import 时用 `from stage2_refactor.models.base import BaseModel` 这种全限定路径，跟 `experiments/run_experiment.py` 风格保持一致。
6. **wandb 是 requirements.txt 的硬依赖**，但 `--use-wandb` 默认 false。如果不打算用，本地装 wandb 也无害；不装的话只要不传 `--use-wandb` 就不会 import。

## G. 找不到 / 未确认的项

- **未找到**：`stage2_refactor/notebooks/stage2_full_pipeline_colab.ipynb` 我没读（README.md 还提到一个 `stage2_lstm_baseline_colab.ipynb` 但实际文件名不同）。如果 Stage 3 要在 Colab 跑，需要再核查 notebook 内容。
- **未直接验证**：yaml 中 4 个子集 dataset-level 的 `hidden_size/num_layers/dropout` 默认 = Stage 1 reproduce 的同名超参。但 finalist 实际用的是 `h=90, L=2, d=0.2`（GRU）和 `h=60, L=2, d=0.1`（BiGRU），**没写进 yaml**，仅记录在 `final_stage2_runs.csv` 的 `run_id` 和列里。Stage 3 接入前最好和队友确认这套 finalist 配置是否打算合入 yaml。
