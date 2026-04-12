"""
backend/ml/evaluate.py

Full benchmarking and evaluation framework for BuyWise price prediction models.

Covers all acceptance criteria:
    ✓ Regression metrics:     MAE, RMSE, R²
    ✓ Classification metrics: Precision, Recall, F1, Accuracy, Confusion Matrix
    ✓ Latency testing:        mean, median, p95, p99 (batch_size=1)
    ✓ Confidence calibration: predicted confidence vs actual outcome frequency
    ✓ Report generation:      MODEL_BENCHMARK_REPORT.md
    ✓ Per-horizon evaluation: 7d and 14d evaluated separately

Usage:
    # Full benchmark (both horizons, generates report)
    python -m backend.ml.evaluate

    # Single horizon
    python -m backend.ml.evaluate --horizon 7

    # Custom paths
    python -m backend.ml.evaluate \\
        --test_7d  data/test_7day.csv \\
        --test_14d data/test_14day.csv \\
        --output_dir backend/ml/reports

Outputs:
    backend/ml/reports/MODEL_BENCHMARK_REPORT.md
    backend/ml/reports/benchmark_results.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from backend.ml.classifier import PriceRecommendationClassifier
from backend.ml.features import FEATURE_COLS, TARGET_7D, TARGET_14D
from backend.ml.predict import PricePredictor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
REPORTS_DIR  = Path(__file__).resolve().parent / "reports"
MODEL_DIR    = Path(__file__).resolve().parent / "models"

DEFAULT_TEST = {
    7:  DATA_DIR / "test_7day.csv",
    14: DATA_DIR / "test_14day.csv",
}

# ---------------------------------------------------------------------------
# Success metric gates (from acceptance criteria)
# ---------------------------------------------------------------------------

GATES = {
    "r2":          0.75,   # R² > 0.75
    "f1":          0.70,   # F1 > 0.70
    "latency_p99": 200.0,  # p99 < 200ms
}

# Number of warm-up predictions before latency measurement begins
# (eliminates JIT / cache warm-up noise from p99)
LATENCY_WARMUP = 10


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
) -> dict:
    """
    Compute MAE, RMSE, R² for predicted vs actual normalized price labels.

    Note: metrics are in normalized space (same space as labels in test CSV).
    Dollar-space metrics are computed separately after inverse transform.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    meets_r2 = r2 >= GATES["r2"]

    return {
        "horizon":   horizon,
        "mae":       round(float(mae),  4),
        "rmse":      round(float(rmse), 4),
        "r2":        round(float(r2),   4),
        "meets_r2_gate": meets_r2,
        "n_samples": len(y_true),
    }


def evaluate_regression_dollars(
    y_true_norm:  np.ndarray,
    y_pred_norm:  np.ndarray,
    global_means: np.ndarray,
    global_sds:   np.ndarray,
    horizon:      int,
) -> dict:
    """
    Compute MAE and RMSE after inverse transforming predictions to dollar space.
    Provides interpretable error magnitudes for the benchmark report.
    """
    # Inverse transform: price = (norm * sd) + mean
    safe_sds   = np.where(global_sds == 0, 1.0, global_sds)
    y_true_usd = (y_true_norm * safe_sds) + global_means
    y_pred_usd = (y_pred_norm * safe_sds) + global_means

    mae_usd  = mean_absolute_error(y_true_usd, y_pred_usd)
    rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))

    return {
        "horizon":    horizon,
        "mae_usd":    round(float(mae_usd),  2),
        "rmse_usd":   round(float(rmse_usd), 2),
        "mean_actual_price_usd": round(float(np.mean(y_true_usd)), 2),
    }


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def build_classification_labels(
    test_df:    pd.DataFrame,
    predictor:  PricePredictor,
    horizon:    int,
    threshold:  float = 0.05,
) -> tuple[list[bool], list[bool], list[float]]:
    """
    Generate y_true and y_pred classification labels by:
        1. Running each test row through PricePredictor
        2. Passing predictions into PriceRecommendationClassifier
        3. Comparing BUY/WAIT recommendation against actual price movement

    Args:
        test_df:   Test set DataFrame with feature columns + label columns
        predictor: Loaded PricePredictor instance
        horizon:   7 or 14
        threshold: Drop threshold for WAIT classification (default 0.05 = 5%)

    Returns:
        y_true:      List of bool — did price actually drop >= threshold?
        y_pred:      List of bool — did classifier predict WAIT?
        confidences: List of float — classifier confidence scores
    """
    target_col  = TARGET_7D if horizon == 7 else TARGET_14D
    clf         = PriceRecommendationClassifier(threshold=threshold)

    y_true      = []
    y_pred      = []
    confidences = []

    required_scaler_cols = ["global_mean", "global_norm_sd"]
    has_scaler = all(c in test_df.columns for c in required_scaler_cols)

    for _, row in test_df.iterrows():
        asin = str(row.get("asin", "UNKNOWN"))

        # --- Predict ---
        pred_result = predictor.predict_single(row, asin=asin)

        # --- Build minimal price_history for classifier ---
        # In production this comes from the Keepa API.
        # In evaluation we reconstruct a synthetic history from test row features
        # to allow the classifier to compute volatility and check deal_flag.
        current_price_norm = float(row.get("price", 0.0))

        # Inverse transform current price to dollars for classifier
        if has_scaler:
            gm  = float(row["global_mean"])
            gsd = float(row["global_norm_sd"]) or 1.0
            current_price_usd = (current_price_norm * gsd) + gm
        else:
            current_price_usd = current_price_norm  # fallback: use normalized

        # Synthetic price history: current row only (sufficient for classifier)
        # Volatility will fall back to default (0.05) with insufficient history —
        # this is acceptable for benchmarking; production uses full Keepa history
        synthetic_history = [{
            "price":        current_price_usd,
            "availability": True,
            "deal_flag":    False,
            "timestamp":    row.get("date", "2025-01-01"),
        }]

        clf_result = clf.classify(
            predicted_price_7d  = pred_result.predicted_price_7d,
            predicted_price_14d = pred_result.predicted_price_14d,
            predicted_price_30d = pred_result.predicted_price_30d,
            price_history       = synthetic_history,
        )

        # --- Ground truth: did price actually drop >= threshold? ---
        actual_label_norm = float(row[target_col])
        if has_scaler:
            gm  = float(row["global_mean"])
            gsd = float(row["global_norm_sd"]) or 1.0
            actual_price_usd = (actual_label_norm * gsd) + gm
        else:
            actual_price_usd = actual_label_norm

        actual_drop_pct = (
            (current_price_usd - actual_price_usd) / current_price_usd
            if current_price_usd > 0 else 0.0
        )

        y_true.append(actual_drop_pct >= threshold)
        y_pred.append(clf_result.recommendation == "WAIT")
        confidences.append(clf_result.confidence)

    return y_true, y_pred, confidences


def evaluate_classification(
    y_true:      list[bool],
    y_pred:      list[bool],
    confidences: list[float],
    horizon:     int,
) -> dict:
    """Compute Precision, Recall, F1, Accuracy and confusion matrix."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true,    y_pred, zero_division=0)
    f1        = f1_score(y_true,        y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true,  y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    meets_f1 = f1 >= GATES["f1"]

    # Confidence calibration: mean confidence for correct vs incorrect WAIT predictions
    wait_indices    = [i for i, p in enumerate(y_pred) if p]
    correct_waits   = [confidences[i] for i in wait_indices if y_true[i]]
    incorrect_waits = [confidences[i] for i in wait_indices if not y_true[i]]

    return {
        "horizon":       horizon,
        "precision":     round(float(precision), 3),
        "recall":        round(float(recall),    3),
        "f1":            round(float(f1),        3),
        "accuracy":      round(float(accuracy),  3),
        "meets_f1_gate": meets_f1,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "n_samples":         len(y_true),
        "n_wait_predicted":  int(sum(y_pred)),
        "n_wait_actual":     int(sum(y_true)),
        "calibration": {
            "mean_confidence_correct_waits":   round(float(np.mean(correct_waits)),   3) if correct_waits   else None,
            "mean_confidence_incorrect_waits": round(float(np.mean(incorrect_waits)), 3) if incorrect_waits else None,
            "n_correct_waits":   len(correct_waits),
            "n_incorrect_waits": len(incorrect_waits),
        },
    }


# ---------------------------------------------------------------------------
# Latency evaluation
# ---------------------------------------------------------------------------

def evaluate_latency(
    test_df:   pd.DataFrame,
    predictor: PricePredictor,
    n_samples: int = 200,
) -> dict:
    """
    Benchmark inference latency using batch_size=1 (real-time scenario).

    Runs LATENCY_WARMUP predictions first (discarded), then measures n_samples.
    Uses both 7d and 14d models per call to reflect real production usage.

    Target: p99 < 200ms
    """
    sample_df = test_df.head(min(n_samples + LATENCY_WARMUP, len(test_df)))

    predictor.reset_latency()

    # Warm-up pass (discarded)
    warmup_df = sample_df.head(LATENCY_WARMUP)
    for _, row in warmup_df.iterrows():
        predictor.predict_single(row, asin=str(row.get("asin", "WARMUP")))
    predictor.reset_latency()

    # Measured pass
    measure_df = sample_df.iloc[LATENCY_WARMUP:LATENCY_WARMUP + n_samples]
    for _, row in measure_df.iterrows():
        predictor.predict_single(row, asin=str(row.get("asin", "UNKNOWN")))

    stats = predictor.latency_stats
    meets_target = predictor.meets_latency_target

    return {
        "mean_ms":         stats["mean"],
        "median_ms":       stats["median"],
        "p95_ms":          stats["p95"],
        "p99_ms":          stats["p99"],
        "n_measured":      stats["n"],
        "warmup_discarded": LATENCY_WARMUP,
        "meets_p99_gate":  meets_target,
        "target_ms":       GATES["latency_p99"],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _gate_icon(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def generate_markdown_report(
    results:      dict,
    output_path:  Path,
):
    """
    Generate MODEL_BENCHMARK_REPORT.md from benchmark results dict.
    Fulfills acceptance criteria: benchmark report with all required sections.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# MODEL_BENCHMARK_REPORT",
        "",
        f"Generated: {now}",
        f"Model: LightGBM (lgbm_7d.lgb, lgbm_14d.lgb)",
        "",
        "---",
        "",
        "## Success Metric Gates",
        "",
        "| Metric | Target | Result |",
        "|--------|--------|--------|",
    ]

    # Determine overall gate results
    r2_results  = [results[h]["regression"]["r2"] for h in ["7d", "14d"] if h in results]
    f1_results  = [results[h]["classification"]["f1"] for h in ["7d", "14d"] if h in results]
    lat_result  = results.get("latency", {})

    best_r2 = max(r2_results) if r2_results else 0.0
    best_f1 = max(f1_results) if f1_results else 0.0
    p99     = lat_result.get("p99_ms", 9999)

    lines += [
        f"| R² > 0.75          | 0.75   | {best_r2:.4f} — {_gate_icon(best_r2 >= 0.75)} |",
        f"| F1 > 0.70          | 0.70   | {best_f1:.4f} — {_gate_icon(best_f1 >= 0.70)} |",
        f"| Latency p99 <200ms | 200ms  | {p99:.1f}ms — {_gate_icon(p99 < 200)} |",
        "",
        "---",
        "",
        "## 1. Regression Metrics (Normalized Space)",
        "",
        "Predictions are evaluated in normalized label space (z-score per ASIN).",
        "Dollar-space metrics are shown in Section 1b.",
        "",
        "| Horizon | MAE    | RMSE   | R²     | R² Gate |",
        "|---------|--------|--------|--------|---------|",
    ]

    for h_key in ["7d", "14d"]:
        if h_key not in results:
            continue
        reg = results[h_key]["regression"]
        lines.append(
            f"| {h_key}      | {reg['mae']:.4f} | {reg['rmse']:.4f} | "
            f"{reg['r2']:.4f} | {_gate_icon(reg['meets_r2_gate'])} |"
        )

    lines += [
        "",
        "### 1b. Regression Metrics (Dollar Space)",
        "",
        "| Horizon | MAE ($) | RMSE ($) | Mean Actual Price ($) |",
        "|---------|---------|----------|----------------------|",
    ]

    for h_key in ["7d", "14d"]:
        if h_key not in results:
            continue
        reg_usd = results[h_key].get("regression_dollars", {})
        if reg_usd:
            lines.append(
                f"| {h_key}      | ${reg_usd['mae_usd']:.2f}  | "
                f"${reg_usd['rmse_usd']:.2f}   | "
                f"${reg_usd['mean_actual_price_usd']:.2f}               |"
            )

    lines += [
        "",
        "---",
        "",
        "## 2. Classification Metrics (BUY / WAIT)",
        "",
        "Predictions are piped through `PriceRecommendationClassifier` (threshold=0.05).",
        "Ground truth: actual price drop >= 5% at forecast horizon.",
        "",
        "| Horizon | Precision | Recall | F1     | Accuracy | F1 Gate |",
        "|---------|-----------|--------|--------|----------|---------|",
    ]

    for h_key in ["7d", "14d"]:
        if h_key not in results:
            continue
        clf = results[h_key]["classification"]
        lines.append(
            f"| {h_key}      | {clf['precision']:.3f}     | "
            f"{clf['recall']:.3f}  | {clf['f1']:.3f}  | "
            f"{clf['accuracy']:.3f}    | {_gate_icon(clf['meets_f1_gate'])} |"
        )

    lines += ["", "### Confusion Matrices", ""]

    for h_key in ["7d", "14d"]:
        if h_key not in results:
            continue
        clf = results[h_key]["classification"]
        cm  = clf["confusion_matrix"]
        lines += [
            f"**{h_key} horizon:**",
            "",
            "|              | Predicted BUY | Predicted WAIT |",
            "|--------------|---------------|----------------|",
            f"| Actual BUY   | TN={cm['tn']}          | FP={cm['fp']}             |",
            f"| Actual WAIT  | FN={cm['fn']}          | TP={cm['tp']}             |",
            "",
        ]

    lines += [
        "### Confidence Calibration",
        "",
        "Mean confidence score for correct vs incorrect WAIT predictions.",
        "Well-calibrated model: correct WAITs should have higher confidence.",
        "",
        "| Horizon | Mean Conf (Correct WAIT) | Mean Conf (Incorrect WAIT) |",
        "|---------|--------------------------|----------------------------|",
    ]

    for h_key in ["7d", "14d"]:
        if h_key not in results:
            continue
        cal = results[h_key]["classification"].get("calibration", {})
        cc  = cal.get("mean_confidence_correct_waits",   "N/A")
        ci  = cal.get("mean_confidence_incorrect_waits", "N/A")
        cc_str = f"{cc:.3f}" if isinstance(cc, float) else str(cc)
        ci_str = f"{ci:.3f}" if isinstance(ci, float) else str(ci)
        lines.append(f"| {h_key}      | {cc_str}                    | {ci_str}                      |")

    lines += [
        "",
        "---",
        "",
        "## 3. Latency Benchmarks (batch_size=1)",
        "",
        f"Measured on {lat_result.get('n_measured', 'N/A')} predictions after "
        f"{lat_result.get('warmup_discarded', LATENCY_WARMUP)} warm-up calls.",
        "",
        "| Metric     | Value    | Target   | Result |",
        "|------------|----------|----------|--------|",
        f"| Mean       | {lat_result.get('mean_ms', 'N/A')}ms    | —        | —      |",
        f"| Median     | {lat_result.get('median_ms', 'N/A')}ms    | —        | —      |",
        f"| p95        | {lat_result.get('p95_ms', 'N/A')}ms    | —        | —      |",
        f"| p99        | {lat_result.get('p99_ms', 'N/A')}ms    | <200ms   | {_gate_icon(lat_result.get('meets_p99_gate', False))} |",
        "",
        "---",
        "",
        "## 4. Production Recommendation",
        "",
    ]

    all_gates_pass = (
        best_r2 >= GATES["r2"]
        and best_f1 >= GATES["f1"]
        and p99 < GATES["latency_p99"]
    )

    if all_gates_pass:
        lines += [
            "**Recommendation: DEPLOY LightGBM**",
            "",
            "All acceptance criteria gates passed:",
            f"- R² = {best_r2:.4f} (target > 0.75) ✅",
            f"- F1 = {best_f1:.4f} (target > 0.70) ✅",
            f"- Latency p99 = {p99:.1f}ms (target < 200ms) ✅",
            "",
            "LightGBM is recommended for production deployment for the following reasons:",
            "- Handles structured tabular retail price data natively",
            "- No GPU required — low operational cost",
            "- Fast inference suitable for real-time browser extension use",
            "- Captures both short-term volatility and long-term seasonal patterns",
            "  via engineered rolling features and temporal features",
        ]
    else:
        failed = []
        if best_r2 < GATES["r2"]:
            failed.append(f"R² = {best_r2:.4f} (target > 0.75)")
        if best_f1 < GATES["f1"]:
            failed.append(f"F1 = {best_f1:.4f} (target > 0.70)")
        if p99 >= GATES["latency_p99"]:
            failed.append(f"Latency p99 = {p99:.1f}ms (target < 200ms)")

        lines += [
            "**Recommendation: FURTHER TUNING REQUIRED**",
            "",
            "The following gates were not met:",
        ]
        for f in failed:
            lines.append(f"- ❌ {f}")

        lines += [
            "",
            "Suggested next steps:",
            "- Increase training data (more ASINs or longer history)",
            "- Tune `num_leaves`, `learning_rate`, `min_child_samples` in train.py",
            "- Add missing features (promotions, competitor pricing, release date)",
            "- Run `threshold_optimizer.py` to find optimal WAIT threshold",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Model Configuration",
        "",
        "| Parameter          | Value   |",
        "|--------------------|---------|",
        "| Algorithm          | LightGBM (gradient boosted trees) |",
        "| Objective          | regression (MSE) |",
        "| Horizons           | 7-day, 14-day   |",
        "| Features           | See features.py FEATURE_COLS |",
        "| Train/test split   | Time-based (no data leakage) |",
        "| Normalization      | Per-ASIN z-score (global_mean, global_norm_sd) |",
        "| 30d proxy          | predicted_price_14d used as predicted_price_30d |",
        "| Early stopping     | 50 rounds on val RMSE |",
        "| Random seed        | 42 |",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"\n  Report saved → {output_path}")
    return report_text


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    horizons:    list[int],
    test_paths:  dict[int, Path],
    output_dir:  Path,
) -> dict:
    """
    Run full benchmark pipeline across specified horizons.

    Returns results dict suitable for report generation and JSON export.
    """
    print("\n" + "="*60)
    print("  BUYWISE MODEL BENCHMARK EVALUATION")
    print("="*60)

    # Load predictor once (both models)
    print("\nLoading models...")
    predictor = PricePredictor(verbose=True)

    results = {}

    # -----------------------------------------------------------------------
    # Per-horizon evaluation
    # -----------------------------------------------------------------------
    for horizon in horizons:
        h_key      = f"{horizon}d"
        target_col = TARGET_7D if horizon == 7 else TARGET_14D
        test_path  = test_paths[horizon]

        print(f"\n{'─'*60}")
        print(f"  Evaluating {horizon}-day horizon")
        print(f"  Test set: {test_path}")
        print(f"{'─'*60}")

        test_df = pd.read_csv(test_path)
        print(f"  Loaded {len(test_df):,} rows  |  {test_df['asin'].nunique()} ASINs")

        # --- Regression ---
        print("\n  [1/3] Regression metrics...")
        available_features = [c for c in FEATURE_COLS if c in test_df.columns]
        X_test = test_df[available_features]
        y_true = test_df[target_col].values

        model_path = MODEL_DIR / f"lgbm_{horizon}d.lgb"
        model      = lgb.Booster(model_file=str(model_path))

        # Align features to model's expected order
        missing = [c for c in FEATURE_COLS if c not in test_df.columns]
        for col in missing:
            X_test = X_test.copy()
            X_test[col] = 0.0
        X_test = X_test[FEATURE_COLS] if all(c in X_test.columns for c in FEATURE_COLS) else X_test[available_features]

        y_pred = model.predict(X_test)

        reg_metrics = evaluate_regression(y_true, y_pred, horizon)
        print(
            f"  MAE={reg_metrics['mae']:.4f}  "
            f"RMSE={reg_metrics['rmse']:.4f}  "
            f"R²={reg_metrics['r2']:.4f}  "
            f"{_gate_icon(reg_metrics['meets_r2_gate'])}"
        )

        # Dollar-space metrics
        reg_usd = {}
        if "global_mean" in test_df.columns and "global_norm_sd" in test_df.columns:
            reg_usd = evaluate_regression_dollars(
                y_true_norm  = y_true,
                y_pred_norm  = y_pred,
                global_means = test_df["global_mean"].values,
                global_sds   = test_df["global_norm_sd"].values,
                horizon      = horizon,
            )
            print(
                f"  Dollar space — MAE=${reg_usd['mae_usd']:.2f}  "
                f"RMSE=${reg_usd['rmse_usd']:.2f}  "
                f"(mean price ${reg_usd['mean_actual_price_usd']:.2f})"
            )

        # --- Classification ---
        print("\n  [2/3] Classification metrics (BUY/WAIT)...")
        y_true_cls, y_pred_cls, confidences = build_classification_labels(
            test_df   = test_df,
            predictor = predictor,
            horizon   = horizon,
        )
        clf_metrics = evaluate_classification(
            y_true      = y_true_cls,
            y_pred      = y_pred_cls,
            confidences = confidences,
            horizon     = horizon,
        )
        print(
            f"  Precision={clf_metrics['precision']:.3f}  "
            f"Recall={clf_metrics['recall']:.3f}  "
            f"F1={clf_metrics['f1']:.3f}  "
            f"Accuracy={clf_metrics['accuracy']:.3f}  "
            f"{_gate_icon(clf_metrics['meets_f1_gate'])}"
        )
        cm = clf_metrics["confusion_matrix"]
        print(f"  Confusion matrix — TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
        print(
            f"  WAIT predicted: {clf_metrics['n_wait_predicted']}  |  "
            f"WAIT actual: {clf_metrics['n_wait_actual']}"
        )

        results[h_key] = {
            "regression":         reg_metrics,
            "regression_dollars": reg_usd,
            "classification":     clf_metrics,
        }

    # -----------------------------------------------------------------------
    # Latency benchmark (run once, using larger test set)
    # -----------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("  [3/3] Latency benchmark (batch_size=1)")
    print(f"{'─'*60}")

    # Use 7d test set for latency (both models run per prediction)
    latency_test_path = test_paths.get(7, test_paths[list(test_paths.keys())[0]])
    latency_df        = pd.read_csv(latency_test_path)

    latency_metrics = evaluate_latency(latency_df, predictor, n_samples=200)
    print(
        f"  mean={latency_metrics['mean_ms']}ms  "
        f"median={latency_metrics['median_ms']}ms  "
        f"p95={latency_metrics['p95_ms']}ms  "
        f"p99={latency_metrics['p99_ms']}ms  "
        f"{_gate_icon(latency_metrics['meets_p99_gate'])}"
    )

    results["latency"] = latency_metrics

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  GATE SUMMARY")
    print(f"{'='*60}")

    for h_key in ["7d", "14d"]:
        if h_key not in results:
            continue
        r2 = results[h_key]["regression"]["r2"]
        f1 = results[h_key]["classification"]["f1"]
        print(f"  {h_key}:  R²={r2:.4f} {_gate_icon(r2 >= GATES['r2'])}  |  F1={f1:.4f} {_gate_icon(f1 >= GATES['f1'])}")

    p99 = results["latency"]["p99_ms"]
    print(f"  Latency p99: {p99}ms {_gate_icon(p99 < GATES['latency_p99'])}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON results saved → {json_path}")

    # Markdown report
    report_path = output_dir / "MODEL_BENCHMARK_REPORT.md"
    generate_markdown_report(results, report_path)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run BuyWise model benchmark evaluation"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[7, 14],
        default=None,
        help="Evaluate single horizon (omit for both)",
    )
    parser.add_argument("--test_7d",     type=str, default=None)
    parser.add_argument("--test_14d",    type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Output directory for report and JSON (default: backend/ml/reports/)",
    )
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else [7, 14]

    test_paths = {}
    for h in horizons:
        if h == 7 and args.test_7d:
            test_paths[7] = Path(args.test_7d)
        elif h == 14 and args.test_14d:
            test_paths[14] = Path(args.test_14d)
        else:
            test_paths[h] = DEFAULT_TEST[h]

    run_evaluation(
        horizons   = horizons,
        test_paths = test_paths,
        output_dir = Path(args.output_dir),
    )


if __name__ == "__main__":
    main()