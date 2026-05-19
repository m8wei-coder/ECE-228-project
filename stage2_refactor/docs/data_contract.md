# Stage 2.0 Data Contract

This document records the actual Stage 1 behavior found in `reproduce/`.
The refactored `stage2_refactor/` code keeps these defaults so the LSTM
baseline can be re-verified before Stage 2.1 model swaps.

## Source Data

- Current repo data source: NASA C-MAPSS whitespace-delimited TXT files in `CMaps/`.
- Supported compatibility path: CSV exports with the same 26 C-MAPSS columns.
- Raw columns:
  - `unit_number`
  - `time_cycles`
  - `setting_1`, `setting_2`, `setting_3`
  - `sensor_1` through `sensor_21`

## Preprocessing Contract

- Training data gets `RUL_absolute = max_cycle_for_engine - time_cycles`.
- Selected sensor columns are median-filtered per engine during training.
- Median kernel size: `5`.
- KMeans condition clustering:
  - Uses only `setting_1`, `setting_2`, `setting_3`.
  - `n_clusters = 6`.
  - `random_state = 42`.
  - `n_init = 10`.
- Scaling:
  - Condition-wise `StandardScaler`.
  - Fit on training data only.
  - Applied only to retained sensor features.
  - Test data uses the saved training KMeans and scalers.
- Stage 1 test-time behavior:
  - `main_test.py` does not apply the median filter to test sensor values.
  - The refactor preserves this with `apply_median_filter_to_test: false`.

## RUL Contract

- Piecewise RUL is computed from the Stage 1 knee-point heuristic.
- RUL heuristic parameters:
  - `window_size = 12`
  - `threshold = 0.2` for FD001, FD002, FD003
  - `threshold = 0.3` for FD004
  - `patience = 1` for FD001, FD002
  - `patience = 2` for FD003
  - `patience = 3` for FD004
- Stage 1 saved artifact actual initial RUL values:

| Subset | Initial RUL |
|---|---:|
| FD001 | 81 |
| FD002 | 104 |
| FD003 | 77 |
| FD004 | 88 |

These differ from the Stage 2 plan's expected `78/103/79/87`. The refactor
documents and preserves the Stage 1 actual values rather than forcing the plan
values, because changing them would change the baseline.

## Feature Contract

Stage 1 uses retained sensor features only. Operational settings are used for
condition clustering, not as model input features.

| Subset | Dropped sensors | Retained features | n_features |
|---|---|---|---:|
| FD001 | 1, 5, 6, 10, 16, 18, 19 | sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21 | 14 |
| FD002 | none | sensors 1-21 | 21 |
| FD003 | 1, 5, 16, 18, 19 | sensors 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21 | 16 |
| FD004 | none | sensors 1-21 | 21 |

The Stage 2 plan suggested FD002/FD004 may have 24 features. The Stage 1 code
and saved artifacts use only the 21 sensor columns as model features.

## Windowing Contract

- Sequence length: `30`.
- Training input `X` shape per batch: `(B, 30, F)`.
- Training target `y` shape per batch in the refactor: `(B, 1)`.
  - Stage 1 returned scalar labels per sample and then squeezed model output.
  - The refactor keeps the same scalar target value but batches it as `(B, 1)`
    to match the model output contract.
- Training windows:
  - Sliding windows for every engine.
  - One label per window.
  - Label equals the `RUL_piecewise` value at the final timestep of the window.
- Test windows:
  - One final window per test engine.
  - If an engine has fewer than 30 cycles, repeat the first row at the front.
  - No sliding test windows are used for baseline evaluation.

## Dtype Contract

- Dataset tensors are `torch.float32`.
- Numpy arrays used for exported windows are `np.float32`.
- Model input is normalized post-preprocessing sensor data.
- Model output is a single RUL prediction per sequence.

## Model Interface Contract

Every Stage 2 model must inherit from `BaseModel` and implement:

```python
forward(x)
```

with:

- input shape: `(B, T, F)`
- output shape: `(B, 1)`

The current LSTM baseline validates these shapes at runtime.

## Validation Contract

Stage 1 did not use a validation split or early stopping. It saved the best
model by training loss. The refactor therefore defaults to:

```json
"validation": {
  "enabled": false,
  "fraction": 0.15,
  "patience": 10
}
```

An engine-level validation split is implemented but disabled by default. When
enabled later, engines are split by `unit_number`, not by sample window.

## Baseline Verification Note

The prompt listed the Stage 1 FD001 target as RMSE approximately `7.78`.
During Stage 2.0 verification, the checked-in Stage 1 artifacts did not
reproduce that number:

- Original `reproduce/logs.tar` FD001 checkpoint evaluated with original
  `reproduce/main_test.py`: `Test RMSE = 8.72`.
- Same original checkpoint and preprocessing artifact evaluated through the
  refactored eval path:
  - test median filter disabled: `test_rmse = 8.7202`.
  - test median filter enabled: `test_rmse = 8.2788`.
- Refactored FD001 training run in Colab from scratch:
  - `test_rmse = 8.2575`.
  - `test_score = 105.8406`.
  - `initial_rul = 81`.
  - `test_windows_shape = [100, 30, 14]`.

Conclusion: the current repository artifacts and original Stage 1 evaluation
script verify an FD001 baseline around `8.72` for the saved checkpoint, not
`7.78`. The refactored pipeline is consistent with the checked-in Stage 1
behavior and produced a better FD001 run (`8.26`) when retrained in Colab.
