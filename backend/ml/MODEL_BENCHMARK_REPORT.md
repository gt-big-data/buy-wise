# MODEL_BENCHMARK_REPORT

Generated: 2026-04-13 23:23:17
Models: xgb_reg_7d.joblib, xgb_cls_7d.joblib, xgb_reg_14d.joblib, xgb_cls_14d.joblib

---

## Success Metric Gates

| Metric                     |   Target | Result 7d                | Result 14d               |
|----------------------------|----------|--------------------------|--------------------------|
| R² ≥ 0.75                  |     0.75 | 0.9413 — ✅ PASS          | 0.9441 — ✅ PASS          |
| MAE ≤ $2.00                |      2.0 | 12.8547 — ❌ FAIL         | 14.5999 — ❌ FAIL         |
| MAPE ≤ 5%                  |      5.0 | 7.83 — ❌ FAIL            | 9.36 — ❌ FAIL            |
| Latency p99 ≤200ms         |    200.0 | 1.483 — ✅ PASS           | 3.996 — ✅ PASS           |
| Classifier F1 ≥ 0.50       |      0.5 | N/A                      | 0.4723 — ❌ FAIL          |
| Classifier Prec ≥ 0.60     |      0.6 | N/A                      | 0.4723 — ❌ FAIL          |

---

## 1. Regression Metrics — Hold-out Test Set

| Horizon | MAE ($) | RMSE ($) | R²     | MAPE (%) | R² Gate            |
|---------|---------|----------|--------|----------|--------------------|
| 7d      | 12.8547  | 50.7657   | 0.9413 | 7.83%    | 0.9413 — ✅ PASS |
| 14d      | 14.5999  | 49.3687   | 0.9441 | 9.36%    | 0.9441 — ✅ PASS |

**Metric definitions**

- **MAE** — average |predicted − actual| in dollars.
- **RMSE** — √(mean squared error); penalises large errors more than MAE.
- **R²** — proportion of price variance explained (1.0 = perfect).
- **MAPE** — mean |error / actual| × 100; scale-independent.

---

## 2. Cross-Validation Results (TimeSeriesSplit, 5 folds)

### 2a. 7-Day Horizon

| Fold     | MAE ($) | RMSE ($) | R²     | MAPE (%) |
|----------|---------|----------|--------|----------|
| 1        | 6.0592  | 13.9806   | 0.9867 | 6.69%    |
| 2        | 7.2092  | 21.1867   | 0.9713 | 6.54%    |
| 3        | 8.0294  | 17.1393   | 0.9812 | 9.93%    |
| 4        | 9.2280  | 41.3408   | 0.9203 | 5.97%    |
| 5        | 10.7590  | 52.5659   | 0.9147 | 6.61%    |
| **Mean** | **8.2570** | **29.2427** | **0.9548** | **7.15%** |
| **Std**  | 1.8154  | 16.8423   | 0.0346 | 1.58%    |

### 2b. 14-Day Horizon

| Fold     | MAE ($) | RMSE ($) | R²     | MAPE (%) |
|----------|---------|----------|--------|----------|
| 1        | 7.6577  | 16.9941   | 0.9802 | 7.60%    |
| 2        | 8.2623  | 23.7274   | 0.9637 | 7.48%    |
| 3        | 9.1554  | 18.5426   | 0.9778 | 11.67%    |
| 4        | 10.1347  | 38.6379   | 0.9282 | 7.29%    |
| 5        | 12.0820  | 53.0653   | 0.9100 | 8.15%    |
| **Mean** | **9.4584** | **30.1935** | **0.9520** | **8.44%** |
| **Std**  | 1.7397  | 15.3802   | 0.0314 | 1.84%    |

---

## 3. BUY / WAIT Classifier

> **WAIT threshold:** price drop ≥ **8%** → WAIT.
>
> **Decision threshold:** auto-tuned per horizon to maximise F1 on hold-out set.
>
> **Method:** A dedicated `XGBClassifier` is trained on the binary label
> directly. `scale_pos_weight` compensates for class imbalance.
> `max_delta_step=1` stabilises gradient updates for sparse positives.
> `predict_proba[:,1]` gives P(WAIT) used as the confidence score.

### 3a. 7-Day Horizon — Summary Metrics

_Classifier results not available — re-run dataset.py to generate labels._

### 3b. 14-Day Horizon — Summary Metrics

| Metric                    | Value    |
|---------------------------|----------|
| WAIT threshold            | 0.715    |
| WAIT prevalence (test)    | 14.6%    |
| scale_pos_weight          | 6.08     |
| Accuracy                  | 0.8459   |
| Precision (WAIT)          | 0.4723   |
| Recall (WAIT)             | 0.4724   |
| F1 Score                  | 0.4723   |

### 3c. 14-Day Horizon — Confusion Matrix

Rows = actual label; Columns = predicted label.

|                     | **Predicted BUY**       | **Predicted WAIT**      |
|---------------------|-------------------------|-------------------------|
| **Actual BUY**      | TN = 73206  (77.7%)   | FP = 7260   (7.7%)   |
| **Actual WAIT**     | FN = 7256   (7.7%)   | TP = 6497   (6.9%)   |

**Interpretation**

- **TP = 6497** — Correct WAIT: price dropped ≥71% and model said WAIT.
- **TN = 73206** — Correct BUY: price did not drop and model said BUY.
- **FP = 7260** — False WAIT: model said WAIT but price did not drop (unnecessary wait).
- **FN = 7256** — Missed drop: model said BUY but price dropped ≥71% (missed saving).

### 3d. 14-Day Horizon — Metric Definitions

| Metric    | Formula                      | Value    | What it means |
|-----------|------------------------------|----------|---------------|
| Accuracy  | (TP + TN) / Total            | 0.8459   | Overall correct predictions |
| Precision | TP / (TP + FP)               | 0.4723   | Of all WAIT calls, how many were right |
| Recall    | TP / (TP + FN)               | 0.4724   | Of all actual drops, how many were caught |
| F1 Score  | 2·(P·R)/(P+R)               | 0.4723   | Harmonic mean; primary metric for imbalanced data |

_Class balance: 14.6% WAIT / 85.4% BUY. F1 is the primary metric due to imbalance._

### 3e. 14-Day Horizon — Precision-Recall Curve (sampled)

Shows the precision/recall trade-off at different decision thresholds.
The chosen threshold is marked with ◀.

| Threshold | Precision | Recall   |    |
|-----------|-----------|----------|----|
| 0.001     | 0.1460    | 1.0000   | |
| 0.004     | 0.1872    | 0.9983   | |
| 0.065     | 0.2114    | 0.9800   | |
| 0.132     | 0.2382    | 0.9439   | |
| 0.220     | 0.2678    | 0.8840   | |
| 0.328     | 0.3062    | 0.8034   | |
| 0.472     | 0.3600    | 0.7061   | |
| 0.642     | 0.4334    | 0.5633   | |
| 0.786     | 0.5546    | 0.3565   | |
| 0.972     | 1.0000    | 0.0001   | |

### 3f. 14-Day Horizon — Confidence Band Analysis

P(WAIT) confidence buckets for predictions above the 0.5 mark.
A well-calibrated model shows rising accuracy as confidence rises.

| Confidence Band | N Predictions | Band Accuracy | % True WAIT |
|-----------------|---------------|---------------|-------------|
| 50–60%           |          5250 |        22.6% |       22.6% |
| 60–70%           |          5614 |        25.6% |       25.6% |
| 70–80%           |          6578 |        34.0% |       34.0% |
| 80–90%           |          5654 |        50.0% |       50.0% |
| 90–100%          |          2421 |        73.0% |       73.0% |

---

## 4. Latency Benchmarks (reg + cls, batch_size=1)

Each measurement covers one regressor + one classifier call. Measured over 200 runs after 10 warm-up calls.

### 7-Day Horizon

| Metric | Value     | Target    | Result    |
|--------|-----------|-----------|-----------|
| Mean   | 1.049ms  | —         | —         |
| Median | 1.019ms  | —         | —         |
| p95    | 1.368ms  | —         | —         |
| p99    | 1.483ms  | <200ms     | 1.483 — ✅ PASS |

### 14-Day Horizon

| Metric | Value     | Target    | Result    |
|--------|-----------|-----------|-----------|
| Mean   | 2.376ms  | —         | —         |
| Median | 2.295ms  | —         | —         |
| p95    | 3.128ms  | —         | —         |
| p99    | 3.996ms  | <200ms     | 3.996 — ✅ PASS |

---

## 5. Feature Importance (Top 15)

### 7-Day Horizon — Regressor

| Rank | Feature                                   | Importance |
|------|-------------------------------------------|------------|
| 1    | amazon_ma_7d                              | 0.4328     |
| 2    | price                                     | 0.3833     |
| 3    | amazon_ma_14d                             | 0.0681     |
| 4    | list_price                                | 0.0265     |
| 5    | log_mean_price                            | 0.0208     |
| 6    | global_min                                | 0.0139     |
| 7    | new_price                                 | 0.0100     |
| 8    | global_max                                | 0.0076     |
| 9    | roll7_mean                                | 0.0067     |
| 10   | count_new                                 | 0.0061     |
| 11   | price_lag1                                | 0.0057     |
| 12   | global_mean                               | 0.0046     |
| 13   | price_vs_global_max                       | 0.0012     |
| 14   | used_price                                | 0.0010     |
| 15   | global_norm_sd                            | 0.0010     |

### 7-Day Horizon — Classifier

| Rank | Feature                                   | Importance |
|------|-------------------------------------------|------------|
| 1    | dist_from_30d_low                         | 0.2087     |
| 2    | price_minmax_asin                         | 0.0879     |
| 3    | price_z_asin                              | 0.0803     |
| 4    | rsi_14d                                   | 0.0405     |
| 5    | pct_change_14d                            | 0.0405     |
| 6    | price_vs_global_min                       | 0.0262     |
| 7    | global_mean                               | 0.0234     |
| 8    | price_vs_global_mean                      | 0.0224     |
| 9    | month                                     | 0.0215     |
| 10   | log_mean_price                            | 0.0208     |
| 11   | global_norm_sd                            | 0.0197     |
| 12   | count_used                                | 0.0191     |
| 13   | amazon_pct_change_7d                      | 0.0182     |
| 14   | price_vs_global_max                       | 0.0181     |
| 15   | roll7_max                                 | 0.0175     |

### 14-Day Horizon — Regressor

| Rank | Feature                                   | Importance |
|------|-------------------------------------------|------------|
| 1    | amazon_ma_14d                             | 0.4164     |
| 2    | amazon_ma_7d                              | 0.3435     |
| 3    | price                                     | 0.0871     |
| 4    | log_mean_price                            | 0.0446     |
| 5    | global_mean                               | 0.0433     |
| 6    | list_price                                | 0.0229     |
| 7    | global_min                                | 0.0096     |
| 8    | price_lag1                                | 0.0072     |
| 9    | global_max                                | 0.0064     |
| 10   | count_new                                 | 0.0039     |
| 11   | roll7_mean                                | 0.0019     |
| 12   | new_price                                 | 0.0013     |
| 13   | price_vs_global_max                       | 0.0009     |
| 14   | price_vs_global_min                       | 0.0009     |
| 15   | global_norm_sd                            | 0.0008     |

### 14-Day Horizon — Classifier

| Rank | Feature                                   | Importance |
|------|-------------------------------------------|------------|
| 1    | dist_from_30d_low                         | 0.2239     |
| 2    | price_minmax_asin                         | 0.0953     |
| 3    | price_z_asin                              | 0.0859     |
| 4    | pct_change_14d                            | 0.0375     |
| 5    | rsi_14d                                   | 0.0327     |
| 6    | price_vs_global_min                       | 0.0288     |
| 7    | month                                     | 0.0225     |
| 8    | global_norm_sd                            | 0.0210     |
| 9    | log_mean_price                            | 0.0208     |
| 10   | global_mean                               | 0.0208     |
| 11   | price_vs_global_mean                      | 0.0207     |
| 12   | global_max                                | 0.0196     |
| 13   | list_price                                | 0.0192     |
| 14   | price_vs_global_max                       | 0.0188     |
| 15   | global_min                                | 0.0185     |

---

## 6. Temporal Error Analysis

### 6a. Monthly Error Trend — 7d

| Month       |    N |  MAE ($) | MAPE (%) |  RMSE ($) |
|-------------|------|----------|----------|-----------|
| 2025-09     | 11739 |  12.2758 |    6.95% |   44.4745 |
| 2025-10     | 13832 |  12.2859 |    6.95% |   48.5680 |
| 2025-11     | 13510 |  12.1836 |    6.98% |   45.1968 |
| 2025-12     | 14311 |  12.9610 |    7.13% |   51.5568 |
| 2026-01     | 14467 |  12.1230 |    6.96% |   45.8625 |
| 2026-02     | 13292 |  12.1061 |    7.05% |   43.6580 |
| 2026-03     | 14973 |  12.3698 |    6.98% |   47.7772 |
| 2026-04     | 1985 |  11.6205 |    6.96% |   41.3809 |

### 6b. Price-Tier Error — 7d

| Tier              |    N |  MAE ($) | MAPE (%) |
|-------------------|------|----------|----------|
| Q1 (cheap)        | 24922 |   0.8084 |    5.45% |
| Q2                | 24258 |   2.5908 |    6.30% |
| Q3                | 24554 |   8.1863 |    7.53% |
| Q4 (expensive)    | 24375 |  37.9366 |    8.75% |

### 6c. Early vs Late Test Period — 7d

| Period            |  MAE ($) | MAPE (%) |
|-------------------|----------|----------|
| Early test period |  12.8835 |    7.85% |
| Late test period  |  12.8257 |    7.82% |

### 6d. Monthly Error Trend — 14d

| Month       |    N |  MAE ($) | MAPE (%) |  RMSE ($) |
|-------------|------|----------|----------|-----------|
| 2025-09     | 11690 |  14.3711 |    8.98% |   46.9471 |
| 2025-10     | 13782 |  14.2299 |    8.99% |   44.8522 |
| 2025-11     | 13420 |  13.6529 |    8.81% |   41.4413 |
| 2025-12     | 14288 |  14.5364 |    9.00% |   50.2694 |
| 2026-01     | 14446 |  14.0491 |    8.87% |   44.2147 |
| 2026-02     | 13265 |  13.8408 |    8.91% |   44.9305 |
| 2026-03     | 13437 |  14.1041 |    8.92% |   44.4370 |

### 6e. Price-Tier Error — 14d

| Tier              |    N |  MAE ($) | MAPE (%) |
|-------------------|------|----------|----------|
| Q1 (cheap)        | 23964 |   1.0920 |    7.35% |
| Q2                | 23373 |   3.5028 |    8.52% |
| Q3                | 23443 |  10.4357 |    9.58% |
| Q4 (expensive)    | 23548 |  41.5500 |   10.27% |

### 6f. Early vs Late Test Period — 14d

| Period            |  MAE ($) | MAPE (%) |
|-------------------|----------|----------|
| Early test period |  14.7938 |    9.39% |
| Late test period  |  14.4024 |    9.34% |

---

## 7. Per-ASIN Accuracy (Best 5 by RMSE)

### 7a. 7-Day Horizon

| ASIN       | N obs |  MAE ($) | RMSE ($) |     R² | MAPE (%) |
|------------|-------|----------|----------|--------|----------|
| B0DQSJ4QHB |    10 |   2.4170 |   4.3688 | 0.9986 |    1.45% |
| B0DM5D9VPH |     2 |   5.4015 |   6.1075 | 0.9996 |    4.65% |
| B09B27J11J |    20 |   3.6818 |   6.1359 | 0.9914 |    4.81% |
| B08QMG7Z4R |    24 |   5.6889 |   8.3566 | 0.9966 |    5.35% |
| B0GQVBT32P |    20 |   5.7191 |   8.4819 | 0.9973 |    7.01% |

### 7b. 14-Day Horizon

| ASIN       | N obs |  MAE ($) | RMSE ($) |     R² | MAPE (%) |
|------------|-------|----------|----------|--------|----------|
| B0DQSJ4QHB |     3 |   4.7180 |   5.0121 | 0.9983 |    6.53% |
| B0FV8MD221 |     9 |   4.0182 |   6.6032 | 0.9832 |    3.87% |
| B0G1PJLWLZ |    23 |   4.2746 |   8.5723 | 0.9973 |    5.57% |
| B0CLNBSHBY |    17 |   7.3157 |  10.1920 | 0.9933 |    5.31% |
| B0GD21577Z |    72 |   5.5452 |  11.8130 | 0.9942 |    7.94% |

---

## 8. Production Recommendation

**Recommendation: ❌ FURTHER TUNING REQUIRED**

The following gates were not met:
  - MAE ≤ $2.00 (7d) = 12.8547$
  - MAE ≤ $2.00 (14d) = 14.5999$
  - MAPE ≤ 5% (7d) = 7.83%
  - MAPE ≤ 5% (14d) = 9.36%
  - Classifier F1 ≥ 0.50 (14d) = 0.4723
  - Classifier Prec ≥ 0.60 (14d) = 0.4723

Suggested next steps:
  - Regression: tune `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`
  - Classifier precision low (too many false WAITs): raise `CLASSIFY_THRESHOLD` in xgboost_price_model.py
  - Classifier recall low (missing too many drops): lower `CLASSIFY_THRESHOLD`
  - Check Section 3 PR curve to find a better threshold manually
  - Review Section 6 for temporal distribution shift
  - If latency gate fails: reduce `n_estimators` or `max_depth`

---

## 9. Model Configuration

| Parameter              | Value                               |
|------------------------|-------------------------------------|
| Algorithm              | XGBoost (gradient boosted trees)    |
| Regression objective   | reg:squarederror (MSE)              |
| Classifier objective   | binary:logistic                     |
| Classifier eval metric | aucpr (PR-AUC)                      |
| WAIT label threshold   | 8% price drop                  |
| Decision threshold     | auto (max F1)                       |
| Horizons               | 7-day, 14-day                           |

**Base hyperparameters (shared)**

| Parameter          | Value                               |
|--------------------|-------------------------------------|
| n_estimators       | 1200                                |
| learning_rate      | 0.015                               |
| max_depth          | 8                                   |
| min_child_weight   | 4                                   |
| subsample          | 0.75                                |
| colsample_bytree   | 0.75                                |
| reg_alpha          | 0.8                                 |
| reg_lambda         | 1.5                                 |
| random_state       | 42                                  |
| n_jobs             | -1                                  |

**Classifier-only hyperparameters**

| Parameter          | Value                               |
|--------------------|-------------------------------------|
| max_delta_step     | 1                                   |
| gamma              | 1.0                                 |
| tree_method        | hist                                |
| max_bin            | 512                                 |
| grow_policy        | lossguide                           |

| CV strategy        | TimeSeriesSplit (5 folds)           |
| Early stopping     | 50 rounds                           |
| Train/test split   | Time-based (no data leakage)        |
