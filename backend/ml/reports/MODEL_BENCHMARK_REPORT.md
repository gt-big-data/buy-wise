# MODEL_BENCHMARK_REPORT

Generated: 2026-04-12 12:54:54
Model: LightGBM (lgbm_7d.lgb, lgbm_14d.lgb)

---

## Success Metric Gates

| Metric | Target | Result |
|--------|--------|--------|
| R² > 0.75          | 0.75   | 0.9250 — ✅ PASS |
| F1 > 0.70          | 0.70   | 0.1590 — ❌ FAIL |
| Latency p99 <200ms | 200ms  | 0.8ms — ✅ PASS |

---

## 1. Regression Metrics (Normalized Space)

Predictions are evaluated in normalized label space (z-score per ASIN).
Dollar-space metrics are shown in Section 1b.

| Horizon | MAE    | RMSE   | R²     | R² Gate |
|---------|--------|--------|--------|---------|
| 7d      | 0.0583 | 0.1369 | 0.9250 | ✅ PASS |
| 14d      | 0.1144 | 0.1966 | 0.8453 | ✅ PASS |

### 1b. Regression Metrics (Dollar Space)

| Horizon | MAE ($) | RMSE ($) | Mean Actual Price ($) |
|---------|---------|----------|----------------------|
| 7d      | $0.04  | $0.13   | $140.12               |
| 14d      | $0.08  | $0.19   | $139.95               |

---

## 2. Classification Metrics (BUY / WAIT)

Predictions are piped through `PriceRecommendationClassifier` (threshold=0.05).
Ground truth: actual price drop >= 5% at forecast horizon.

| Horizon | Precision | Recall | F1     | Accuracy | F1 Gate |
|---------|-----------|--------|--------|----------|---------|
| 7d      | 0.100     | 0.372  | 0.158  | 0.906    | ❌ FAIL |
| 14d      | 0.101     | 0.368  | 0.159  | 0.908    | ❌ FAIL |

### Confusion Matrices

**7d horizon:**

|              | Predicted BUY | Predicted WAIT |
|--------------|---------------|----------------|
| Actual BUY   | TN=88844          | FP=7813             |
| Actual WAIT  | FN=1468          | TP=871             |

**14d horizon:**

|              | Predicted BUY | Predicted WAIT |
|--------------|---------------|----------------|
| Actual BUY   | TN=85925          | FP=7343             |
| Actual WAIT  | FN=1419          | TP=827             |

### Confidence Calibration

Mean confidence score for correct vs incorrect WAIT predictions.
Well-calibrated model: correct WAITs should have higher confidence.

| Horizon | Mean Conf (Correct WAIT) | Mean Conf (Incorrect WAIT) |
|---------|--------------------------|----------------------------|
| 7d      | 0.995                    | 1.000                      |
| 14d      | 0.995                    | 1.000                      |

---

## 3. Latency Benchmarks (batch_size=1)

Measured on 200 predictions after 10 warm-up calls.

| Metric     | Value    | Target   | Result |
|------------|----------|----------|--------|
| Mean       | 0.795ms    | —        | —      |
| Median     | 0.796ms    | —        | —      |
| p95        | 0.819ms    | —        | —      |
| p99        | 0.844ms    | <200ms   | ✅ PASS |

---

## 4. Production Recommendation

**Recommendation: FURTHER TUNING REQUIRED**

The following gates were not met:
- ❌ F1 = 0.1590 (target > 0.70)

Suggested next steps:
- Increase training data (more ASINs or longer history)
- Tune `num_leaves`, `learning_rate`, `min_child_samples` in train.py
- Add missing features (promotions, competitor pricing, release date)
- Run `threshold_optimizer.py` to find optimal WAIT threshold

---

## 5. Model Configuration

| Parameter          | Value   |
|--------------------|---------|
| Algorithm          | LightGBM (gradient boosted trees) |
| Objective          | regression (MSE) |
| Horizons           | 7-day, 14-day   |
| Features           | See features.py FEATURE_COLS |
| Train/test split   | Time-based (no data leakage) |
| Normalization      | Per-ASIN z-score (global_mean, global_norm_sd) |
| 30d proxy          | predicted_price_14d used as predicted_price_30d |
| Early stopping     | 50 rounds on val RMSE |
| Random seed        | 42 |
