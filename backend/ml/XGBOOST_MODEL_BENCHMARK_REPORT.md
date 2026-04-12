# XGBOOST_MODEL_BENCHMARK_REPORT

Generated: 2026-04-12 16:54:50
Model: XGBoost (xgb_reg_7d.joblib, xgb_reg_14d.joblib)

---

## Success Metric Gates

| Metric               |   Target | Result 7d              | Result 14d             |
|----------------------|----------|------------------------|------------------------|
| R² > 0.75            |     0.75 | 0.9346 — PASS        | 0.9404 — PASS        |
| MAE < $2.00          |      2.0 | 12.5883 — FAIL       | 14.4762 — FAIL       |
| MAPE < 5%            |      5.0 | 6.98 — FAIL          | 8.71 — FAIL          |
| Latency p99 <200ms   |    200.0 | 0.680 — PASS         | 0.916 — PASS         |

---

## 1. Regression Metrics — Hold-out Test Set

| Horizon | MAE ($) | RMSE ($) | R²     | MAPE (%) | R² Gate           |
|---------|---------|----------|--------|----------|-------------------|
| 7d      | 12.5883  | 54.2130   | 0.9346 | 6.98%    | 0.9346 — PASS |
| 14d      | 14.4762  | 51.4299   | 0.9404 | 8.71%    | 0.9404 — PASS |

**Metric definitions**

- **MAE** (Mean Absolute Error) — average |predicted − actual| in dollars.  
  Less sensitive to outliers; treats every error with equal weight.
- **RMSE** (Root Mean Squared Error) — √(mean squared error) in dollars.  
  Penalises large errors more strongly than MAE while keeping the result
  in the original price unit, improving interpretability.
- **R²** (Coefficient of Determination) — proportion of price variance explained.  
  1.0 = perfect prediction; 0 = no better than predicting the global mean.
  Provides insight into how well the model captures underlying data patterns.
- **MAPE** (Mean Absolute Percentage Error) — average |error / actual| × 100.  
  Scale-independent; useful for comparing accuracy across different price ranges.

---

## 2. Cross-Validation Results (TimeSeriesSplit, 5 folds)

### 2a. 7-Day Horizon

| Fold     | MAE ($) | RMSE ($) | R²     | MAPE (%) |
|----------|---------|----------|--------|----------|
| 1        | 6.2159  | 13.9818   | 0.9868 | 8.03%    |
| 2        | 6.6264  | 20.5749   | 0.9729 | 5.92%    |
| 3        | 7.9994  | 16.7361   | 0.9820 | 11.23%    |
| 4        | 8.9266  | 38.7597   | 0.9307 | 5.95%    |
| 5        | 10.9580  | 57.8299   | 0.9002 | 6.70%    |
| **Mean** | **8.1453** | **29.5765** | **0.9545** | **7.57%** |
| **Std**  | 1.9087  | 18.5181   | 0.0376 | 2.22%    |

### 2b. 14-Day Horizon

| Fold     | MAE ($) | RMSE ($) | R²     | MAPE (%) |
|----------|---------|----------|--------|----------|
| 1        | 7.6361  | 16.7851   | 0.9809 | 8.17%    |
| 2        | 7.9177  | 23.0623   | 0.9658 | 7.29%    |
| 3        | 9.2853  | 18.2468   | 0.9785 | 12.79%    |
| 4        | 10.1041  | 36.8213   | 0.9356 | 7.55%    |
| 5        | 12.1468  | 55.2432   | 0.9050 | 8.17%    |
| **Mean** | **9.4180** | **30.0317** | **0.9532** | **8.79%** |
| **Std**  | 1.8276  | 16.1605   | 0.0324 | 2.27%    |

---

## 3. Latency Benchmarks (batch_size=1)

Measured on 200 predictions after 10 warm-up calls.

| Metric   | 7d | 14d | Target | Result |
|----------|----------|----------|--------|--------|
| Mean     | 0.311ms | 0.337ms | —      | —      |
| Median   | 0.284ms | 0.319ms | —      | —      |
| p95      | 0.478ms | 0.432ms | —      | —      |
| p99      | 0.680ms | 0.916ms | <200ms   | 0.916 — PASS |

---

## 4. Feature Importance (Top 15)

### 4a. 7-Day Horizon

| Rank | Feature                                   | Importance |
|------|-------------------------------------------|------------|
| 1    | price                                     | 0.4466     |
| 2    | amazon_ma_7d                              | 0.3272     |
| 3    | amazon_ma_14d                             | 0.0557     |
| 4    | list_price                                | 0.0398     |
| 5    | global_min                                | 0.0241     |
| 6    | global_mean                               | 0.0238     |
| 7    | log_mean_price                            | 0.0180     |
| 8    | new_price                                 | 0.0175     |
| 9    | global_max                                | 0.0140     |
| 10   | count_new                                 | 0.0077     |
| 11   | price_lag1                                | 0.0036     |
| 12   | price_vs_list                             | 0.0019     |
| 13   | roll30_mean                               | 0.0017     |
| 14   | price_vs_global_min                       | 0.0016     |
| 15   | roll14_mean                               | 0.0014     |

### 4b. 14-Day Horizon

| Rank | Feature                                   | Importance |
|------|-------------------------------------------|------------|
| 1    | amazon_ma_14d                             | 0.4445     |
| 2    | amazon_ma_7d                              | 0.2593     |
| 3    | price                                     | 0.1457     |
| 4    | log_mean_price                            | 0.0459     |
| 5    | global_mean                               | 0.0442     |
| 6    | list_price                                | 0.0221     |
| 7    | global_min                                | 0.0101     |
| 8    | roll14_mean                               | 0.0034     |
| 9    | new_price                                 | 0.0033     |
| 10   | global_max                                | 0.0029     |
| 11   | count_new                                 | 0.0029     |
| 12   | roll30_mean                               | 0.0026     |
| 13   | price_vs_list                             | 0.0020     |
| 14   | price_lag1                                | 0.0009     |
| 15   | week_of_year                              | 0.0007     |

---

## 5. Temporal Error Analysis

### 5a. Monthly Error Trend — 7d

A well-behaved model should show gradually increasing error over time,
not random spikes — spikes indicate distribution shift.

| Month       |    N |  MAE ($) | MAPE (%) |  RMSE ($) |
|-------------|------|----------|----------|-----------|
| 2025-09     | 11876 |  12.5324 |    6.93% |   55.7886 |
| 2025-10     | 13973 |  13.1717 |    7.06% |   61.4941 |
| 2025-11     | 13682 |  12.4375 |    6.97% |   50.3304 |
| 2025-12     | 14386 |  11.9584 |    6.92% |   46.4151 |
| 2026-01     | 14612 |  12.6740 |    6.96% |   53.0701 |
| 2026-02     | 13364 |  12.7116 |    6.96% |   53.8070 |
| 2026-03     | 15112 |  12.7483 |    7.03% |   57.7862 |
| 2026-04     | 1990 |  11.7412 |    7.15% |   51.0722 |

### 5b. Price-Tier Error — 7d

Shows whether prediction accuracy degrades for cheap vs expensive products.

| Tier              |    N |  MAE ($) | MAPE (%) |
|-------------------|------|----------|----------|
| Q1 (cheap)        | 25085 |   0.8399 |    5.85% |
| Q2                | 24425 |   2.5040 |    6.13% |
| Q3                | 24798 |   8.1090 |    7.42% |
| Q4 (expensive)    | 24687 |  39.0027 |    8.53% |

### 5c. Early vs Late Test Period — 7d

Splits the test set at the median date. Growing error in the late period
suggests the model struggles as it predicts further from its training window.

| Period            |  MAE ($) | MAPE (%) |
|-------------------|----------|----------|
| Early test period |  12.6118 |    6.98% |
| Late test period  |  12.5645 |    6.98% |

### 5d. Monthly Error Trend — 14d

A well-behaved model should show gradually increasing error over time,
not random spikes — spikes indicate distribution shift.

| Month       |    N |  MAE ($) | MAPE (%) |  RMSE ($) |
|-------------|------|----------|----------|-----------|
| 2025-09     | 11876 |  14.5400 |    8.83% |   50.0765 |
| 2025-10     | 13973 |  13.8998 |    8.63% |   46.7922 |
| 2025-11     | 13682 |  14.9535 |    8.69% |   54.9816 |
| 2025-12     | 14386 |  14.1255 |    8.73% |   49.2150 |
| 2026-01     | 14612 |  14.7784 |    8.57% |   55.0285 |
| 2026-02     | 13364 |  13.9513 |    8.67% |   46.2435 |
| 2026-03     | 13620 |  15.0932 |    8.87% |   56.3384 |

### 5e. Price-Tier Error — 14d

Shows whether prediction accuracy degrades for cheap vs expensive products.

| Tier              |    N |  MAE ($) | MAPE (%) |
|-------------------|------|----------|----------|
| Q1 (cheap)        | 24189 |   1.0353 |    6.80% |
| Q2                | 23600 |   3.4312 |    8.38% |
| Q3                | 23960 |  10.2302 |    9.36% |
| Q4 (expensive)    | 23764 |  43.4071 |   10.31% |

### 5f. Early vs Late Test Period — 14d

Splits the test set at the median date. Growing error in the late period
suggests the model struggles as it predicts further from its training window.

| Period            |  MAE ($) | MAPE (%) |
|-------------------|----------|----------|
| Early test period |  14.4409 |    8.71% |
| Late test period  |  14.5116 |    8.71% |

---

## 6. Per-ASIN Accuracy (Best 5 by RMSE)

### 6a. 7-Day Horizon

| ASIN       | N obs |  MAE ($) | RMSE ($) |     R² | MAPE (%) |
|------------|-------|----------|----------|--------|----------|
| B0DM5D9VPH |     2 |   0.9394 |   1.0912 | 0.9973 |    1.99% |
| B0DQSJ4QHB |    10 |   1.1610 |   1.6184 | 0.9962 |    4.63% |
| B0B1J7N2F2 |     9 |   2.5126 |   3.8186 | 0.9973 |    2.63% |
| B0GQVF4SV7 |    18 |   3.6986 |   5.0903 | 0.9947 |    4.45% |
| B081HH5X61 |    12 |   5.6119 |   7.8093 | 0.9930 |    5.16% |

### 6b. 14-Day Horizon

| ASIN       | N obs |  MAE ($) | RMSE ($) |     R² | MAPE (%) |
|------------|-------|----------|----------|--------|----------|
| B071DGMNMX |     5 |   4.1808 |   6.3847 | 0.9718 |    8.90% |
| B09L951YB4 |     3 |   5.4233 |   7.2656 | 0.9378 |    7.86% |
| B0G1PJLWLZ |    23 |   3.2684 |   7.4386 | 0.9941 |    3.20% |
| B0B1J7N2F2 |     2 |   6.4986 |   8.7888 | 0.9847 |    4.69% |
| B08QMG7Z4R |    17 |   9.3820 |  14.3165 | 0.9927 |    6.84% |

---

## 7. Production Recommendation

**Recommendation: FURTHER TUNING REQUIRED ❌**

The following gates were not met:
  - ❌ MAE < $2.00 (7d) = 12.5883$
  - ❌ MAE < $2.00 (14d) = 14.4762$
  - ❌ MAPE < 5% (7d) = 6.98%
  - ❌ MAPE < 5% (14d) = 8.71%

Suggested next steps:
  - Increase training data (more ASINs or longer history)
  - Tune `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight` in BASE_PARAMS
  - Add missing features (promotions, competitor pricing, release date)
  - Review Section 5 for distribution shift across the test period
  - If latency gate fails: reduce `n_estimators`, lower `max_depth`, or limit features

---

## 8. Model Configuration

| Parameter          | Value                               |
|--------------------|-------------------------------------|
| Algorithm          | XGBoost (gradient boosted trees)    |
| Objective          | reg:squarederror (MSE)              |
| Horizons           | 7-day, 14-day                       |
| n_estimators       | 1000                                |
| learning_rate      | 0.05                                |
| max_depth          | 6                                   |
| min_child_weight   | 3                                   |
| subsample          | 0.8                                 |
| colsample_bytree   | 0.8                                 |
| reg_alpha          | 0.1                                 |
| reg_lambda         | 1.0                                 |
| random_state       | 42                                  |
| n_jobs             | -1                                  |
| CV strategy        | TimeSeriesSplit (5 folds)           |
| Early stopping     | 50 rounds on val RMSE               |
| Latency n_runs     | 200                                   |
| Latency n_warmup   | 10                                   |
| Train/test split   | Time-based (no data leakage)        |
