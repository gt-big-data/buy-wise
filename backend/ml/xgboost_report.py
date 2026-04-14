"""
xgboost_report.py
=================
Generates MODEL_BENCHMARK_REPORT.md from a RunResult produced by
xgboost_price_model.main().

Not meant to be run directly — called automatically at end of main():

    from xgboost_report import generate_report
    generate_report(run_result)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from xgboost_price_model import RunResult

_HERE = Path(__file__).parent

# ── Metric gates ──────────────────────────────────────────────────────────────
LATENCY_P99_TARGET_MS: float = 200.0

GATES: dict[str, dict] = {
    "r2":          {"target": 0.75,                  "op": ">=", "label": "R² ≥ 0.75",             "fmt": ".4f", "unit": ""},
    "mae":         {"target": 2.00,                  "op": "<=", "label": "MAE ≤ $2.00",            "fmt": ".4f", "unit": "$"},
    "mape":        {"target": 5.00,                  "op": "<=", "label": "MAPE ≤ 5%",              "fmt": ".2f", "unit": "%"},
    "latency_p99": {"target": LATENCY_P99_TARGET_MS, "op": "<=", "label": f"Latency p99 ≤{LATENCY_P99_TARGET_MS:.0f}ms", "fmt": ".3f", "unit": "ms"},
    "cls_f1":      {"target": 0.50,                  "op": ">=", "label": "Classifier F1 ≥ 0.50",   "fmt": ".4f", "unit": ""},
    "cls_prec":    {"target": 0.60,                  "op": ">=", "label": "Classifier Prec ≥ 0.60", "fmt": ".4f", "unit": ""},
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_gate(metric: str, value: float) -> tuple[str, bool]:
    cfg    = GATES[metric]
    passed = (value >= cfg["target"]) if cfg["op"] == ">=" else (value <= cfg["target"])
    label  = "✅ PASS" if passed else "❌ FAIL"
    return f"{value:{cfg['fmt']}} — {label}", passed


def _early_late_split(
    test_df: pd.DataFrame,
    y_pred:  np.ndarray,
    y_true:  np.ndarray,
) -> list[tuple[str, float, float]]:
    df = test_df[["date"]].copy().reset_index(drop=True)
    df["abs_err"] = np.abs(y_true - y_pred)
    df["pct_err"] = (
        np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)) * 100
    )
    mid  = df["date"].median()
    rows = []
    for label, mask in [
        ("Early test period", df["date"] <= mid),
        ("Late test period",  df["date"] > mid),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append((label, float(sub["abs_err"].mean()), float(sub["pct_err"].mean())))
    return rows


def _confidence_bands(
    confidence: np.ndarray,
    y_true_cls: np.ndarray,
) -> list[tuple]:
    """Bucket WAIT predictions by confidence and compute accuracy per band."""
    bands  = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    labels = ["50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
    rows   = []
    for (lo, hi), lbl in zip(bands, labels):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            rows.append((lbl, 0, float("nan"), float("nan")))
            continue
        sub_conf = confidence[mask]
        sub_true = y_true_cls[mask]
        pred     = (sub_conf > 0.5).astype(int)
        acc      = float((pred == sub_true).mean())
        pct_wait = float(sub_true.mean() * 100)
        rows.append((lbl, int(mask.sum()), acc, pct_wait))
    return rows


def _pr_curve_table(
    precisions: np.ndarray,
    recalls:    np.ndarray,
    thresholds: np.ndarray,
    n_points:   int = 10,
) -> list[tuple[float, float, float]]:
    """
    Sample n_points evenly spaced along the PR curve.
    Returns list of (threshold, precision, recall).
    """
    indices = np.linspace(0, len(thresholds) - 1, n_points, dtype=int)
    return [
        (float(thresholds[i]), float(precisions[i]), float(recalls[i]))
        for i in indices
    ]


# ── Public entry point ────────────────────────────────────────────────────────
def generate_report(run_result: "RunResult") -> None:
    from xgboost_price_model import (
        BASE_PARAMS, CLS_EXTRA_PARAMS, N_CV_FOLDS, EARLY_STOPPING,
        WAIT_THRESHOLD, CLASSIFY_THRESHOLD,
    )

    now         = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    horizons    = sorted(run_result.horizons)
    model_files = ", ".join(
        f"xgb_reg_{h}d.joblib, xgb_cls_{h}d.joblib" for h in horizons
    )

    lines: list[str] = []
    def ln(s: str = "") -> None:
        lines.append(s)

    # ═══════════════════════════════════════════════════════════════════════
    # Header
    # ═══════════════════════════════════════════════════════════════════════
    ln("# MODEL_BENCHMARK_REPORT")
    ln()
    ln(f"Generated: {now}")
    ln(f"Models: {model_files}")
    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 0 — Success Metric Gates
    # ═══════════════════════════════════════════════════════════════════════
    ln("## Success Metric Gates")
    ln()
    h_headers = " | ".join(f"{'Result ' + str(h) + 'd':<24}" for h in horizons)
    ln(f"| {'Metric':<26} | {'Target':>8} | {h_headers} |")
    ln(f"|{'-'*28}|{'-'*10}|" + "|".join(["-" * 26] * len(horizons)) + "|")

    all_passed    = True
    failed_gates: list[str] = []

    for metric, cfg in GATES.items():
        row = [f"| {cfg['label']:<26} | {cfg['target']:>8} |"]
        for h in horizons:
            hr  = run_result.horizons[h]
            cls = hr.classifier
            if metric == "latency_p99":
                val = hr.latency["p99"]
            elif metric == "cls_f1":
                val = cls.f1 if cls is not None else float("nan")
            elif metric == "cls_prec":
                val = cls.precision if cls is not None else float("nan")
            else:
                val = hr.holdout[metric]

            if np.isnan(val):
                row.append(f" {'N/A':<24} |")
            else:
                res_str, passed = _check_gate(metric, val)
                if not passed:
                    all_passed = False
                    failed_gates.append(
                        f"{cfg['label']} ({h}d) = {val:{cfg['fmt']}}{cfg['unit']}"
                    )
                row.append(f" {res_str:<24} |")
        ln("".join(row))

    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1 — Regression Metrics
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 1. Regression Metrics — Hold-out Test Set")
    ln()
    ln("| Horizon | MAE ($) | RMSE ($) | R²     | MAPE (%) | R² Gate            |")
    ln("|---------|---------|----------|--------|----------|--------------------|")
    for h in horizons:
        ho        = run_result.horizons[h].holdout
        r2_str, _ = _check_gate("r2", ho["r2"])
        ln(f"| {h}d      | {ho['mae']:.4f}  | {ho['rmse']:.4f}   | "
           f"{ho['r2']:.4f} | {ho['mape']:.2f}%    | {r2_str} |")
    ln()
    ln("**Metric definitions**")
    ln()
    ln("- **MAE** — average |predicted − actual| in dollars.")
    ln("- **RMSE** — √(mean squared error); penalises large errors more than MAE.")
    ln("- **R²** — proportion of price variance explained (1.0 = perfect).")
    ln("- **MAPE** — mean |error / actual| × 100; scale-independent.")
    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2 — Cross-Validation
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 2. Cross-Validation Results (TimeSeriesSplit, 5 folds)")
    ln()
    sub_labels = iter("abcdefgh")
    for h in horizons:
        label   = next(sub_labels)
        cv_list = run_result.horizons[h].cv_results
        cv_df   = pd.DataFrame(cv_list)
        ln(f"### 2{label}. {h}-Day Horizon")
        ln()
        ln("| Fold     | MAE ($) | RMSE ($) | R²     | MAPE (%) |")
        ln("|----------|---------|----------|--------|----------|")
        for row in cv_list:
            ln(f"| {row['fold']:<8} | {row['mae']:.4f}  | {row['rmse']:.4f}   | "
               f"{row['r2']:.4f} | {row['mape']:.2f}%    |")
        ln(f"| **Mean** | **{cv_df['mae'].mean():.4f}** | "
           f"**{cv_df['rmse'].mean():.4f}** | "
           f"**{cv_df['r2'].mean():.4f}** | "
           f"**{cv_df['mape'].mean():.2f}%** |")
        ln(f"| **Std**  | {cv_df['mae'].std():.4f}  | {cv_df['rmse'].std():.4f}   | "
           f"{cv_df['r2'].std():.4f} | {cv_df['mape'].std():.2f}%    |")
        ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 3 — BUY / WAIT Classifier
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 3. BUY / WAIT Classifier")
    ln()
    threshold_note = (
        f"fixed at {CLASSIFY_THRESHOLD:.3f} (CLASSIFY_THRESHOLD override)"
        if CLASSIFY_THRESHOLD is not None
        else "auto-tuned per horizon to maximise F1 on hold-out set"
    )
    ln(f"> **WAIT threshold:** price drop ≥ **{WAIT_THRESHOLD:.0%}** → WAIT.")
    ln(f">")
    ln(f"> **Decision threshold:** {threshold_note}.")
    ln(f">")
    ln("> **Method:** A dedicated `XGBClassifier` is trained on the binary label")
    ln("> directly. `scale_pos_weight` compensates for class imbalance.")
    ln("> `max_delta_step=1` stabilises gradient updates for sparse positives.")
    ln("> `predict_proba[:,1]` gives P(WAIT) used as the confidence score.")
    ln()

    sub_labels = iter("abcdefghijklmnop")
    for h in horizons:
        hr  = run_result.horizons[h]
        cls = hr.classifier

        ln(f"### 3{next(sub_labels)}. {h}-Day Horizon — Summary Metrics")
        ln()
        if cls is None:
            ln("_Classifier results not available — re-run dataset.py to generate labels._")
            ln()
            continue

        ln("| Metric                    | Value    |")
        ln("|---------------------------|----------|")
        ln(f"| WAIT threshold            | {cls.threshold:.3f}    |")
        ln(f"| WAIT prevalence (test)    | {cls.wait_pct:.1f}%    |")
        ln(f"| scale_pos_weight          | {cls.scale_pos_weight:.2f}     |")
        ln(f"| Accuracy                  | {cls.accuracy:.4f}   |")
        ln(f"| Precision (WAIT)          | {cls.precision:.4f}   |")
        ln(f"| Recall (WAIT)             | {cls.recall:.4f}   |")
        ln(f"| F1 Score                  | {cls.f1:.4f}   |")
        ln()

        # ── Confusion matrix ─────────────────────────────────────────────
        ln(f"### 3{next(sub_labels)}. {h}-Day Horizon — Confusion Matrix")
        ln()
        ln("Rows = actual label; Columns = predicted label.")
        ln()
        tn, fp, fn, tp = cls.confusion.ravel()
        total = tn + fp + fn + tp
        ln("|                     | **Predicted BUY**       | **Predicted WAIT**      |")
        ln("|---------------------|-------------------------|-------------------------|")
        ln(f"| **Actual BUY**      | TN = {tn:<6} ({tn/total:.1%})   | FP = {fp:<6} ({fp/total:.1%})   |")
        ln(f"| **Actual WAIT**     | FN = {fn:<6} ({fn/total:.1%})   | TP = {tp:<6} ({tp/total:.1%})   |")
        ln()
        ln("**Interpretation**")
        ln()
        ln(f"- **TP = {tp}** — Correct WAIT: price dropped ≥{cls.threshold:.0%} and model said WAIT.")
        ln(f"- **TN = {tn}** — Correct BUY: price did not drop and model said BUY.")
        ln(f"- **FP = {fp}** — False WAIT: model said WAIT but price did not drop (unnecessary wait).")
        ln(f"- **FN = {fn}** — Missed drop: model said BUY but price dropped ≥{cls.threshold:.0%} (missed saving).")
        ln()

        # ── Metric definitions ───────────────────────────────────────────
        ln(f"### 3{next(sub_labels)}. {h}-Day Horizon — Metric Definitions")
        ln()
        ln("| Metric    | Formula                      | Value    | What it means |")
        ln("|-----------|------------------------------|----------|---------------|")
        ln(f"| Accuracy  | (TP + TN) / Total            | {cls.accuracy:.4f}   | Overall correct predictions |")
        ln(f"| Precision | TP / (TP + FP)               | {cls.precision:.4f}   | Of all WAIT calls, how many were right |")
        ln(f"| Recall    | TP / (TP + FN)               | {cls.recall:.4f}   | Of all actual drops, how many were caught |")
        ln(f"| F1 Score  | 2·(P·R)/(P+R)               | {cls.f1:.4f}   | Harmonic mean; primary metric for imbalanced data |")
        ln()
        ln(f"_Class balance: {cls.wait_pct:.1f}% WAIT / {100-cls.wait_pct:.1f}% BUY. "
           f"F1 is the primary metric due to imbalance._")
        ln()

        # ── Precision-Recall curve (sampled) ────────────────────────────
        ln(f"### 3{next(sub_labels)}. {h}-Day Horizon — Precision-Recall Curve (sampled)")
        ln()
        ln("Shows the precision/recall trade-off at different decision thresholds.")
        ln("The chosen threshold is marked with ◀.")
        ln()
        ln("| Threshold | Precision | Recall   |    |")
        ln("|-----------|-----------|----------|----|")
        if len(cls.pr_thresholds) > 0:
            for thr, prec, rec in _pr_curve_table(
                cls.pr_precisions, cls.pr_recalls, cls.pr_thresholds
            ):
                marker = " ◀ chosen" if abs(thr - cls.threshold) < 0.02 else ""
                ln(f"| {thr:.3f}     | {prec:.4f}    | {rec:.4f}   |{marker} |")
        else:
            ln("_PR curve data not available._")
        ln()

        # ── Confidence bands ─────────────────────────────────────────────
        ln(f"### 3{next(sub_labels)}. {h}-Day Horizon — Confidence Band Analysis")
        ln()
        ln(f"P(WAIT) confidence buckets for predictions above the 0.5 mark.")
        ln("A well-calibrated model shows rising accuracy as confidence rises.")
        ln()
        ln("| Confidence Band | N Predictions | Band Accuracy | % True WAIT |")
        ln("|-----------------|---------------|---------------|-------------|")
        for band_lbl, n, acc, pct_wait in _confidence_bands(cls.confidence, cls.y_true_cls):
            if n == 0:
                ln(f"| {band_lbl:<15}  | {'—':>13} | {'—':>13} | {'—':>11} |")
            else:
                ln(f"| {band_lbl:<15}  | {n:>13} | {acc:>12.1%} | {pct_wait:>10.1f}% |")
        ln()

    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 4 — Latency
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 4. Latency Benchmarks (reg + cls, batch_size=1)")
    ln()
    _lat0 = run_result.horizons[horizons[0]].latency
    ln(f"Each measurement covers one regressor + one classifier call. "
       f"Measured over {_lat0.get('n_runs', 200)} runs after "
       f"{_lat0.get('n_warmup', 10)} warm-up calls.")
    ln()
    latency_rows = [
        ("Mean",   "mean",   None),
        ("Median", "median", None),
        ("p95",    "p95",    None),
        ("p99",    "p99",    LATENCY_P99_TARGET_MS),
    ]
    for h in horizons:
        lat = run_result.horizons[h].latency
        ln(f"### {h}-Day Horizon")
        ln()
        ln("| Metric | Value     | Target    | Result    |")
        ln("|--------|-----------|-----------|-----------|")
        for row_label, key, target in latency_rows:
            val = lat.get(key, float("nan"))
            if target is not None:
                gate_str, _ = _check_gate("latency_p99", val)
                ln(f"| {row_label:<6} | {val:.3f}ms  | <{target:.0f}ms     | {gate_str} |")
            else:
                ln(f"| {row_label:<6} | {val:.3f}ms  | —         | —         |")
        ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 5 — Feature Importance
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 5. Feature Importance (Top 15)")
    ln()
    for h in horizons:
        for suffix, label in [("_reg", "Regressor"), ("_cls", "Classifier")]:
            imp_path = _HERE / f"feature_importance_{h}d{suffix}.csv"
            ln(f"### {h}-Day Horizon — {label}")
            ln()
            if imp_path.exists():
                imp   = pd.read_csv(imp_path, index_col=0).squeeze()
                top15 = imp.nlargest(15).reset_index()
                top15.columns = ["Feature", "Importance"]
                ln(f"| {'Rank':<4} | {'Feature':<41} | Importance |")
                ln(f"|{'-'*6}|{'-'*43}|{'-'*12}|")
                for rank, (_, row) in enumerate(top15.iterrows(), 1):
                    ln(f"| {rank:<4} | {row['Feature']:<41} | {row['Importance']:.4f}     |")
            else:
                ln("_Feature importance CSV not found._")
            ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 6 — Temporal Error Analysis
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 6. Temporal Error Analysis")
    ln()
    sub_labels = iter("abcdefghijklmnop")
    for h in horizons:
        hr = run_result.horizons[h]

        monthly_path = _HERE / f"temporal_error_{h}d.csv"
        ln(f"### 6{next(sub_labels)}. Monthly Error Trend — {h}d")
        ln()
        if monthly_path.exists():
            m = pd.read_csv(monthly_path)
            ln("| Month       |    N |  MAE ($) | MAPE (%) |  RMSE ($) |")
            ln("|-------------|------|----------|----------|-----------|")
            for _, row in m.iterrows():
                ln(f"| {str(row['month']):<11} | {int(row['n']):>4} | "
                   f"{row['mae']:>8.4f} | {row['mape']:>7.2f}% | {row['rmse']:>9.4f} |")
        else:
            ln("_temporal_error CSV not found._")
        ln()

        tier_path = _HERE / f"price_tier_error_{h}d.csv"
        ln(f"### 6{next(sub_labels)}. Price-Tier Error — {h}d")
        ln()
        if tier_path.exists():
            t = pd.read_csv(tier_path)
            ln("| Tier              |    N |  MAE ($) | MAPE (%) |")
            ln("|-------------------|------|----------|----------|")
            for _, row in t.iterrows():
                ln(f"| {str(row['price_tier']):<17} | {int(row['n']):>4} | "
                   f"{row['mae']:>8.4f} | {row['mape']:>7.2f}% |")
        else:
            ln("_price_tier_error CSV not found._")
        ln()

        ln(f"### 6{next(sub_labels)}. Early vs Late Test Period — {h}d")
        ln()
        ln("| Period            |  MAE ($) | MAPE (%) |")
        ln("|-------------------|----------|----------|")
        for label, mae, mape in _early_late_split(hr.test_df, hr.y_pred, hr.y_true):
            ln(f"| {label:<17} | {mae:>8.4f} | {mape:>7.2f}% |")
        ln()

    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 7 — Per-ASIN Accuracy
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 7. Per-ASIN Accuracy (Best 5 by RMSE)")
    ln()
    sub_labels = iter("abcdefgh")
    for h in horizons:
        label     = next(sub_labels)
        asin_path = _HERE / f"per_asin_accuracy_{h}d.csv"
        ln(f"### 7{label}. {h}-Day Horizon")
        ln()
        if asin_path.exists():
            a = pd.read_csv(asin_path).head(5)
            ln("| ASIN       | N obs |  MAE ($) | RMSE ($) |     R² | MAPE (%) |")
            ln("|------------|-------|----------|----------|--------|----------|")
            for _, row in a.iterrows():
                ln(f"| {row['asin']:<10} | {int(row['n_obs']):>5} | "
                   f"{row['mae']:>8.4f} | {row['rmse']:>8.4f} | "
                   f"{row['r2']:>6.4f} | {row['mape']:>7.2f}% |")
        else:
            ln("_per_asin_accuracy CSV not found._")
        ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 8 — Production Recommendation
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 8. Production Recommendation")
    ln()
    if all_passed:
        ln("**Recommendation: ✅ READY FOR DEPLOYMENT**")
        ln()
        ln("All metric gates passed:")
        for h in horizons:
            ho  = run_result.horizons[h].holdout
            lat = run_result.horizons[h].latency
            cls = run_result.horizons[h].classifier
            ln(f"  - R²   ({h}d) = {ho['r2']:.4f}  (target ≥ {GATES['r2']['target']})")
            ln(f"  - MAE  ({h}d) = ${ho['mae']:.4f}  (target ≤ ${GATES['mae']['target']:.2f})")
            ln(f"  - MAPE ({h}d) = {ho['mape']:.2f}%  (target ≤ {GATES['mape']['target']:.0f}%)")
            ln(f"  - Latency p99 ({h}d) = {lat.get('p99',0):.3f}ms  (target < {LATENCY_P99_TARGET_MS:.0f}ms)")
            if cls is not None:
                ln(f"  - Classifier F1   ({h}d) = {cls.f1:.4f}  (target ≥ {GATES['cls_f1']['target']})")
                ln(f"  - Classifier Prec ({h}d) = {cls.precision:.4f}  (target ≥ {GATES['cls_prec']['target']})")
    else:
        ln("**Recommendation: ❌ FURTHER TUNING REQUIRED**")
        ln()
        ln("The following gates were not met:")
        for fg in failed_gates:
            ln(f"  - {fg}")
        ln()
        ln("Suggested next steps:")
        ln("  - Regression: tune `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`")
        ln("  - Classifier precision low (too many false WAITs): raise `CLASSIFY_THRESHOLD` in xgboost_price_model.py")
        ln("  - Classifier recall low (missing too many drops): lower `CLASSIFY_THRESHOLD`")
        ln("  - Check Section 3 PR curve to find a better threshold manually")
        ln("  - Review Section 6 for temporal distribution shift")
        ln("  - If latency gate fails: reduce `n_estimators` or `max_depth`")
    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 9 — Model Configuration
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 9. Model Configuration")
    ln()
    ln("| Parameter              | Value                               |")
    ln("|------------------------|-------------------------------------|")
    ln("| Algorithm              | XGBoost (gradient boosted trees)    |")
    ln("| Regression objective   | reg:squarederror (MSE)              |")
    ln("| Classifier objective   | binary:logistic                     |")
    ln("| Classifier eval metric | aucpr (PR-AUC)                      |")
    ln(f"| WAIT label threshold   | {WAIT_THRESHOLD:.0%} price drop                  |")
    threshold_val = str(CLASSIFY_THRESHOLD) if CLASSIFY_THRESHOLD is not None else "auto (max F1)"
    ln(f"| Decision threshold     | {threshold_val:<35} |")
    ln(f"| Horizons               | {', '.join(str(h) + '-day' for h in horizons)}                           |")
    ln()
    ln("**Base hyperparameters (shared)**")
    ln()
    ln("| Parameter          | Value                               |")
    ln("|--------------------|-------------------------------------|")
    for k, v in BASE_PARAMS.items():
        ln(f"| {k:<18} | {str(v):<35} |")
    ln()
    ln("**Classifier-only hyperparameters**")
    ln()
    ln("| Parameter          | Value                               |")
    ln("|--------------------|-------------------------------------|")
    for k, v in CLS_EXTRA_PARAMS.items():
        ln(f"| {k:<18} | {str(v):<35} |")
    ln()
    ln(f"| CV strategy        | TimeSeriesSplit ({N_CV_FOLDS} folds)           |")
    ln(f"| Early stopping     | {EARLY_STOPPING} rounds                           |")
    ln("| Train/test split   | Time-based (no data leakage)        |")
    ln()

    report_path = _HERE / "MODEL_BENCHMARK_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")