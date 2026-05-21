# Stage 1 复现代码摸底报告

> 范围：`reproduce/` 下 7 个源文件 + `logs.tar` 归档清单。仅阅读，无任何文件改动。

## A. 数据流

### A.1 原始数据从哪里读？路径写死吗？
- 读取入口：`load_and_filter_data` (`data_preprocessing.py:16`) 用 `pd.read_csv(file_path)`。
- 路径全部在 `config.py` 里**硬编码为相对路径**：
  - `./csv/train/train_FD00X.csv`
  - `./csv/test/test_FD00X.csv`
  - `./csv/rul/RUL_FD00X.csv`
- 注意：仓库里**没有 `./csv/` 目录，也没有 CSV 文件**。原始 C-MAPSS 数据在 `../ECE 228/CMaps/` 下，是无表头的 `.txt`。代码假设 CSV 有表头 `unit_number, time_cycles, setting_1..3, sensor_1..21`（见 `data_preprocessing.py:20-31`）。所以**跑 main.py 之前必须先 .txt → .csv 加表头并放到 `reproduce/csv/...` 下**。README 第 24 行也提到了这点，但没给转换脚本。

### A.2 预处理流水线步骤顺序
全部集中在 `load_and_filter_data` + `calculate_piecewise_rul`，顺序如下（这与论文里的「相关性筛选 → 滤波 → 归一化 → initial RUL → 滑窗」**不完全一致**）：

| 步骤 | 实现位置 | 说明 |
|---|---|---|
| 1. 计算 `RUL_absolute = max_cycle − time_cycles` | `data_preprocessing.py:20-24` | 每个 engine 内做 groupby |
| 2. "相关性筛选" → 实际是**硬编码 drop_cols** | `data_preprocessing.py:33-36` + `config.py` 每个子集的 `drop_cols` | **没有真的算 Pearson 再筛**，是直接把 `sensor_1/5/6/10/16/18/19` 这种列表写死在 config 里 |
| 3. 中值滤波 `medfilt(kernel=5)` | `data_preprocessing.py:38-46` | 逐 engine、逐 sensor 做 |
| 4. KMeans(n_clusters=6) 在 3 个 setting 上聚类，给出 `condition` | `data_preprocessing.py:50-52` | **所有子集都用 6 类**，FD001/FD003 实际只有 1 个工况，仍用 6 类 |
| 5. 按 condition 分组分别 `StandardScaler.fit_transform` | `data_preprocessing.py:55-66` | 每个 condition 一个独立 scaler |
| 6. 保存 `{kmeans, scalers, features, drop_cols}` 到 `.gz` | `data_preprocessing.py:69-77` | |
| 7. Piecewise initial RUL：逐 engine 滑窗均值 + 拐点检测 | `calculate_piecewise_rul` (`data_preprocessing.py:83-154`) | 取所有 engine knee RUL 的**最小值**作为 global `initial_rul`，再 `clip(upper=initial_rul)` |
| 8. 把 `initial_rul, window_size, threshold, patience` 追加进 `.gz` | `main.py:31-37` 调 `update_preprocessing_metadata` | |
| 9. 滑窗成 `(seq_length, n_features)` 张量 | `CMAPSSDataset._generate_sequences` (`data_preprocessing.py:162-175`) | 步长 1，label 取窗口最后一步的 `RUL_piecewise` |

测试侧 (`main_test.py:11-60` 或 `export_artifacts.py:99-150`) 走的是「加载 .gz → `kmeans.predict` → 逐 condition `scaler.transform` → 每个 engine 取最后 `seq_length` 步」，所以**仅 export_artifacts.py 的测试路径有 medfilt**（`apply_saved_preprocessing` 加了 medfilt），**`main_test.py:11-60` 的测试路径没有 medfilt**。这是一处不一致，Stage 3 要复用时建议统一走 export_artifacts.py 里的版本。

### A.3 每个子集最终留下哪些传感器
由 `drop_cols` 决定（21 个原始传感器减去 drop_cols）：

| 子集 | 删除的 sensor | 保留数 |
|---|---|---|
| FD001 | 1, 5, 6, 10, 16, 18, 19 | **14** |
| FD002 | （无） | **21** |
| FD003 | 1, 5, 16, 18, 19 | **16** |
| FD004 | （无） | **21** |

### A.4 滑窗后张量 shape
- 训练侧 (`CMAPSSDataset`)：`X_train.shape = (N_windows, 30, N_features)`，`y_train.shape = (N_windows,)`，每个 engine 贡献 `len(engine) − 30 + 1` 个窗口，全 engine 拼接。`N_features` 见 A.3。
- 测试侧 (`prepare_test_data` / `prepare_test_data_from_saved_artifact`)：`X_test.shape = (N_engines, 30, N_features)`，每个 engine **只取最后 30 步**（长度不足则前向 pad 第一行）。`y_test.shape = (N_engines,)` 来自 `RUL_FD00X.csv`。

---

## B. 关键超参（来自 `config.py`）

| 项 | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| `window_size` (w，用于 piecewise RUL) | 12 | 12 | 12 | 12 |
| `threshold` (Th) | 0.2 | 0.2 | 0.2 | **0.3** |
| `patience` | 1 | 1 | 2 | 3 |
| `seq_length` (LSTM 输入窗) | 30 | 30 | 30 | 30 |
| `batch_size` | 15 | 15 | 20 | 10 |
| `learning_rate` | 2e-3 | 2e-3 | 1e-3 | 1e-3 |
| `epochs` | 150 | 150 | 250 | **20** |
| `hidden_size` | 60 | 60 | 90 | 30 |
| `num_layers` | 4 | 4 | 6 | 2 |
| `dropout` | 0.1 | 0.1 | 0.1 | 0.1 |
| optimizer | Adam, weight_decay=1e-5（`main.py:71-75`） | 同 | 同 | 同 |
| 归一化方式 | 按 6 个 KMeans condition 分组的 StandardScaler |  |  |  |
| `initial_rul` | **运行时计算**，不在 config | 同 | 同 | 同 |

### 与论文 Table 7（w=12, Th=0.2, initial RUL = 78/103/79/87）的差异
- **w=12**：四个子集都一致 ✅
- **Th=0.2**：FD001/FD002/FD003 一致 ✅，**FD004 用了 0.3** ❌
- **initial RUL**：代码里**完全由 `calculate_piecewise_rul` 在线计算**（取所有 engine knee RUL 的最小值），**没有硬编码 78/103/79/87**。能不能复现这四个数，要真跑一遍才知道——**未直接验证**。
- README 第 4 行也明确写了「some of the hyperparameters used are different from the article configuration」。
- 另外 `patience` 这个超参（连续 patience 个窗口都越过阈值才确认 knee）论文里没有提到，是代码作者自己加的。

---

## C. 模型结构

`network.py:4-38` 定义 `CMAPSS_LSTM`：

```
Input: (B, 30, N_features)
  └─ nn.LSTM(input_size=N_features, hidden_size=H, num_layers=L,
             batch_first=True, dropout=0.1 if L>1 else 0)
  └─ 取最后时间步 lstm_out[:, -1, :]               → (B, H)
  └─ Linear(H → H//2) → ReLU → Dropout(0.1)        → (B, H//2)
  └─ Linear(H//2 → 1)                              → (B, 1)
```

- `hidden_size` 和 `num_layers` 每个子集不同（见 B 表），所以**每个子集训练一个独立的模型，权重独立保存**到 `./logs/cmapss_lstm_fd00X.pth`。
- 没有共享 backbone，没有多任务头。
- Loss = `nn.MSELoss`（`main.py:69`）。
- 评估指标 `calculate_metrics` 在 `network.py:40-60` 和 `export_artifacts.py:15-32` 各有一份，**逻辑等价**（NASA 不对称指数：h<0 用 exp(-h/13)-1，h≥0 用 exp(h/10)-1）。

---

## D. 已有产物

### D.1 训练落盘的东西
| 产物 | 路径 | 由谁写 | 是否有 |
|---|---|---|---|
| LSTM checkpoint | `./logs/cmapss_lstm_fd00X.pth` | `main.py:122`（每次 train loss 创新低就覆盖） | ✅ |
| 归一化对象（kmeans + 6 个 scaler + features + drop_cols + initial_rul + 元数据） | `./logs/cmapss_condition_scalers_fd00X.gz` | `data_preprocessing.py:69-77` + `main.py:31-37` | ✅（mean/std 隐含在 scaler 对象里，**不是单独 .npy**） |
| 训练曲线 / 每 epoch RMSE&Score | — | 只 `print`，不落盘 | ❌ |
| 测试集预测 `y_pred` | `./exports/{DS}/y_pred_lstm.npy` | `export_artifacts.evaluate_baseline_lstm:322-327` | ✅（**需要先跑 `python export_artifacts.py --dataset FD00X`** 才会有） |
| Pearson 相关矩阵 | `./exports/{DS}/corr_matrix.npy` + `.csv` | `export_artifacts.compute_corr_matrix:225-248` | ✅（同上，要先跑 export） |
| 保留传感器名单 | `./exports/{DS}/retained_sensors.pkl` | `export_artifacts.export_train_arrays:261-262` | ✅（同上） |
| `X_train/y_train/X_test/y_test` `.npy` | `./exports/{DS}/` | `export_artifacts.export_train_arrays / export_test_arrays` | ✅（同上） |
| 元数据（features, initial_rul, 超参） | `./exports/{DS}/metadata.{pkl,csv}` | `export_artifacts.export_metadata:347-381` | ✅（同上） |
| 测试 RMSE/Score | `./exports/{DS}/baseline_lstm_metrics.{pkl,csv}` | `export_artifacts.evaluate_baseline_lstm:336-342` | ✅（同上） |

### D.2 `logs.tar` 解压后里面是什么
（`tar -tvf` 看到的清单，未解压）

```
./logs/cmapss_lstm_fd001.pth                          435 558 B
./logs/cmapss_lstm_fd002.pth                          442 278 B
./logs/cmapss_lstm_fd003.pth                        1 486 886 B
./logs/cmapss_lstm_fd004.pth                           60 198 B
./logs/cmapss_condition_scalers_fd001.gz               17 009 B
./logs/cmapss_condition_scalers_fd002.gz               38 365 B
./logs/cmapss_condition_scalers_fd003.gz               19 845 B
./logs/cmapss_condition_scalers_fd004.gz               43 089 B
./logs/.ipynb_checkpoints/
```

**只含权重和 scaler `.gz`，不含任何 RMSE/Score 数字文件**。最终的每子集 RMSE / Score —— **未找到**（既没有 metrics JSON，也没有训练日志 txt）。要拿到这四个数，只能：
1. 解开 `logs.tar` → 把 4 个 `.pth` 和 4 个 `.gz` 放到 `reproduce/logs/`
2. 准备好 csv 数据
3. 跑 `python main_test.py --dataset FD00X`（或 `python export_artifacts.py --dataset FD00X`，后者还会顺便落盘 `y_pred_lstm.npy` 和 metrics csv）

---

## E. Stage 3 缺口（最关键）

| Stage 3 需要 | 现状 | 直接可用 / 需要补 |
|---|---|---|
| ① 已预处理好的滑窗张量 | `export_artifacts.export_train_arrays / export_test_arrays` 已经会写 `X_train.npy / y_train.npy / X_test.npy / y_test.npy` 到 `./exports/{DS}/` | **代码可用**；但产物文件还**没生成**，需要先跑过一遍 export_artifacts.py（依赖 CSV 数据 + `logs/*.gz`） |
| ② 每个子集的 Pearson 相关矩阵 | `compute_corr_matrix` 已实现，落盘到 `./exports/{DS}/corr_matrix.{npy,csv}` | **代码可用**；同样需要先跑 export。**两点要注意**：(a) 它是在**原始未滤波、未归一化的 train CSV** 上算的，和模型实际输入的特征分布不一致；(b) 矩阵末尾还包含 `RUL_absolute` 行/列，做图的邻接矩阵时要切掉。如果 Stage 3 想用「与模型输入一致的相关性」，需要补：在 `compute_corr_matrix` 里加一个选项，让它在 `apply_saved_preprocessing` 之后的 df 上算，**而不是 raw CSV**。建议改 `export_artifacts.py:225-248`，加 1 个 `use_preprocessed=True` 分支约 10 行。 |
| ③ 保留传感器的索引/名字 | `./exports/{DS}/retained_sensors.pkl`（即 `feature_cols`），也存在 `metadata.pkl["features"]` 里 | **完全可用**，无需补 |
| ④ 可复用的循环主干代码 | `network.CMAPSS_LSTM` 把 LSTM + FC head 写死在一起，`forward` 直接返回 RUL 标量 | **部分可用，但耦合**。Stage 3 要把 LSTM 输出作为「时序分支特征」再和 GNN 分支融合，目前没有「只取 backbone 特征」的接口。**需要补**：在 `network.py` 里加一个 `forward_features(x)` 方法（4–6 行），返回 `lstm_out[:, -1, :]`（形状 `(B, H)`），原 `forward` 改成调用它再过 FC head。或者新写 `network_stage3.py`，把 LSTM backbone 单独抽出来 + 留一个组合点位给 GNN 分支。**不要现在改**。 |
| ⑤ 评估函数（RMSE + NASA Score） | `network.calculate_metrics` 和 `export_artifacts.calculate_metrics` 各一份，逻辑等价 | **可用**。但**注意 clamp 不在指标函数内**：`main_test.py:90-91` 和 `export_artifacts.py:312-318` 在调用前先 `clamp(y_pred, y_true, max=initial_rul)`，Stage 3 复用时要保持同样的 clamp 顺序，否则数字会和 baseline 不可比。建议 Stage 3 写一个薄包装 `eval_rul(y_pred, y_true, initial_rul)`，把 clamp + metric 封一起，但**先不动现有文件**。 |

### 额外要提醒的几个坑（不属于五项缺口，但会影响 Stage 3）
1. **数据 CSV 还没生成**。原始 `.txt` 在 `../ECE 228/CMaps/`，需要先转 CSV（加表头：`unit_number,time_cycles,setting_1,setting_2,setting_3,sensor_1,...,sensor_21`）并放到 `reproduce/csv/{train,test,rul}/`。这是跑 Stage 1 复现和拿 D.2 那四个 RMSE 数字的前置。
2. **`main_test.py` 与 `export_artifacts.py` 的测试预处理不一致**：前者没做 medfilt，后者做了。论文里 medfilt 应当 train/test 一致。Stage 3 建议**统一走 `export_artifacts.py` 的 `apply_saved_preprocessing`** 作为唯一入口，不要复用 `main_test.py:11-60`。
3. **`drop_cols` 是手写不是算出来的**。如果 Stage 3 想用「Pearson 相关性筛传感器」做图节点选择，要明确这是和 Stage 1 不一样的筛法，否则跑出来的图节点数和 baseline 输入维度不一致，融合会出问题。
4. **没有验证集**。`main.py` 整个训练只看 train loss 选 best checkpoint（`main.py:120-122`），Stage 3 若要公平比较，可能需要补一个 val split——但这是改进而非缺口，**先不动**。
5. **FD004 epochs 只有 20**（其它是 150/150/250），明显欠拟合。如果对照 baseline 时 FD004 数字很难看，先看这个。
