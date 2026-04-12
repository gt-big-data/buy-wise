"""
xgboost_report.py
=================
Generates MODEL_BENCHMARK_REPORT.md from a RunResult produced by
xgboost_price_model.main().

This file is NOT meant to be run directly. It is imported and called
automatically at the end of xgboost_price_model.main().

    from xgboost_report import generate_report
    generate_report(run_result)

Output
------
  MODEL_BENCHMARK_REPORT.md   — full markdown benchmark report
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

# ── Metric gates — edit thresholds here ──────────────────────────────────────
LATENCY_P99_TARGET_MS: float = 200.0   # p99 single-row prediction must be < this

GATES: dict[str, dict] = {
    "r2":         {"target": 0.75,                "op": ">=", "label": "R² > 0.75",          "fmt": ".4f", "unit": ""},
    "mae":        {"target": 2.00,                "op": "<=", "label": "MAE < $2.00",         "fmt": ".4f", "unit": "$"},
    "mape":       {"target": 5.00,                "op": "<=", "label": "MAPE < 5%",           "fmt": ".2f", "unit": "%"},
    "latency_p99":{"target": LATENCY_P99_TARGET_MS, "op": "<=", "label": f"Latency p99 <{LATENCY_P99_TARGET_MS:.0f}ms", "fmt": ".3f", "unit": "ms"},
}


# ── Internal helpers ──────────────────────────────────────────────────────────
def _check_gate(metric: str, value: float) -> tuple[str, bool]:
    """Return formatted result string and pass/fail bool."""
    cfg    = GATES[metric]
    passed = (value >= cfg["target"]) if cfg["op"] == ">=" else (value <= cfg["target"])
    label  = "✅ PASS" if passed else "❌ FAIL"
    return f"{value:{cfg['fmt']}} — {label}", passed


def _early_late_split(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Return (label, mae, mape) for early and late halves of the test period."""
    df = test_df[["date"]].copy().reset_index(drop=True)
    df["abs_err"] = np.abs(y_true - y_pred)
    df["pct_err"] = (
        np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)) * 100
    )
    mid    = df["date"].median()
    rows   = []
    for label, mask in [
        ("Early test period", df["date"] <= mid),
        ("Late test period",  df["date"] > mid),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append((label, float(sub["abs_err"].mean()), float(sub["pct_err"].mean())))
    return rows


# ── Public entry point ────────────────────────────────────────────────────────
def generate_report(run_result: "RunResult") -> None:
    """
    Build MODEL_BENCHMARK_REPORT.md from a RunResult.

    Called automatically by xgboost_price_model.main() — do not call directly.
    """
    from xgboost_price_model import BASE_PARAMS, N_CV_FOLDS, EARLY_STOPPING

    now         = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    horizons    = sorted(run_result.horizons)
    model_files = ", ".join(f"xgb_reg_{h}d.joblib" for h in horizons)

    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    # ═══════════════════════════════════════════════════════════════════════
    # Header
    # ═══════════════════════════════════════════════════════════════════════
    ln("# MODEL_BENCHMARK_REPORT")
    ln()
    ln(f"Generated: {now}")
    ln(f"Model: XGBoost ({model_files})")
    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 0 — Success Metric Gates
    # ═══════════════════════════════════════════════════════════════════════
    ln("## Success Metric Gates")
    ln()

    h_headers = " | ".join(f"{'Result ' + str(h) + 'd':<22}" for h in horizons)
    ln(f"| {'Metric':<20} | {'Target':>8} | {h_headers} |")
    ln(f"|{'-'*22}|{'-'*10}|" + "|".join(["-" * 24] * len(horizons)) + "|")

    all_passed    = True
    failed_gates: list[str] = []

    for metric, cfg in GATES.items():
        row = [f"| {cfg['label']:<20} | {cfg['target']:>8} |"]
        for h in horizons:
            hr  = run_result.horizons[h]
            val = hr.latency["p99"] if metric == "latency_p99" else hr.holdout[metric]
            res_str, passed = _check_gate(metric, val)
            if not passed:
                all_passed = False
                failed_gates.append(
                    f"{cfg['label']} ({h}d) = {val:{cfg['fmt']}}{cfg['unit']}"
                )
            row.append(f" {res_str:<22} |")
        ln("".join(row))

    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1 — Hold-out Regression Metrics
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 1. Regression Metrics — Hold-out Test Set")
    ln()
    ln("| Horizon | MAE ($) | RMSE ($) | R²     | MAPE (%) | R² Gate           |")
    ln("|---------|---------|----------|--------|----------|-------------------|")
    for h in horizons:
        ho      = run_result.horizons[h].holdout
        r2_str, _ = _check_gate("r2", ho["r2"])
        ln(f"| {h}d      | {ho['mae']:.4f}  | {ho['rmse']:.4f}   | "
           f"{ho['r2']:.4f} | {ho['mape']:.2f}%    | {r2_str} |")

    ln()
    ln("**Metric definitions**")
    ln()
    ln("- **MAE** (Mean Absolute Error) — average |predicted − actual| in dollars.  ")
    ln("  Less sensitive to outliers; treats every error with equal weight.")
    ln("- **RMSE** (Root Mean Squared Error) — √(mean squared error) in dollars.  ")
    ln("  Penalises large errors more strongly than MAE while keeping the result")
    ln("  in the original price unit, improving interpretability.")
    ln("- **R²** (Coefficient of Determination) — proportion of price variance explained.  ")
    ln("  1.0 = perfect prediction; 0 = no better than predicting the global mean.")
    ln("  Provides insight into how well the model captures underlying data patterns.")
    ln("- **MAPE** (Mean Absolute Percentage Error) — average |error / actual| × 100.  ")
    ln("  Scale-independent; useful for comparing accuracy across different price ranges.")
    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2 — Cross-Validation
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 2. Cross-Validation Results (TimeSeriesSplit, 5 folds)")
    ln()
    section_labels = iter("abcdefgh")
    for h in horizons:
        label    = next(section_labels)
        cv_list  = run_result.horizons[h].cv_results
        cv_df    = pd.DataFrame(cv_list)

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
    # Section 3 — Latency Benchmarks
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 3. Latency Benchmarks (batch_size=1)")
    ln()
    _lat0 = run_result.horizons[horizons[0]].latency
    ln(f"Measured on {_lat0.get('n_runs', 200)} predictions after "
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
        ln("| Metric | Value    | Target   | Result   |")
        ln("|--------|----------|----------|----------|")
        for row_label, key, target in latency_rows:
            val = lat.get(key, float("nan"))
            if target is not None:
                gate_str, _ = _check_gate("latency_p99", val)
                ln(f"| {row_label:<6} | {val:.3f}ms | <{target:.0f}ms    | {gate_str} |")
            else:
                ln(f"| {row_label:<6} | {val:.3f}ms | —        | —        |")
        ln()
    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 4 — Feature Importance  (was 3)
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 4. Feature Importance (Top 15)")
    ln()
    section_labels = iter("abcdefgh")
    for h in horizons:
        label    = next(section_labels)
        imp_path = _HERE / f"feature_importance_{h}d.csv"

        ln(f"### 4{label}. {h}-Day Horizon")
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
            ln("_Feature importance CSV not found — ensure main() completed successfully._")
        ln()

    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 5 — Temporal Error Analysis  (was 4)
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 5. Temporal Error Analysis")
    ln()
    sub_labels = iter("abcdefghijklmnop")
    for h in horizons:
        hr = run_result.horizons[h]

        monthly_path = _HERE / f"temporal_error_{h}d.csv"
        ln(f"### 5{next(sub_labels)}. Monthly Error Trend — {h}d")
        ln()
        ln("A well-behaved model should show gradually increasing error over time,")
        ln("not random spikes — spikes indicate distribution shift.")
        ln()
        if monthly_path.exists():
            m = pd.read_csv(monthly_path)
            ln("| Month       |    N |  MAE ($) | MAPE (%) |  RMSE ($) |")
            ln("|-------------|------|----------|----------|-----------|")
            for _, row in m.iterrows():
                ln(f"| {str(row['month']):<11} | {int(row['n']):>4} | "
                   f"{row['mae']:>8.4f} | {row['mape']:>7.2f}% | {row['rmse']:>9.4f} |")
        else:
            ln("_temporal_error CSV not found — ensure main() completed._")
        ln()

        tier_path = _HERE / f"price_tier_error_{h}d.csv"
        ln(f"### 5{next(sub_labels)}. Price-Tier Error — {h}d")
        ln()
        ln("Shows whether prediction accuracy degrades for cheap vs expensive products.")
        ln()
        if tier_path.exists():
            t = pd.read_csv(tier_path)
            ln("| Tier              |    N |  MAE ($) | MAPE (%) |")
            ln("|-------------------|------|----------|----------|")
            for _, row in t.iterrows():
                ln(f"| {str(row['price_tier']):<17} | {int(row['n']):>4} | "
                   f"{row['mae']:>8.4f} | {row['mape']:>7.2f}% |")
        else:
            ln("_price_tier_error CSV not found — ensure main() completed._")
        ln()

        ln(f"### 5{next(sub_labels)}. Early vs Late Test Period — {h}d")
        ln()
        ln("Splits the test set at the median date. Growing error in the late period")
        ln("suggests the model struggles as it predicts further from its training window.")
        ln()
        ln("| Period            |  MAE ($) | MAPE (%) |")
        ln("|-------------------|----------|----------|")
        for label, mae, mape in _early_late_split(hr.test_df, hr.y_pred, hr.y_true):
            ln(f"| {label:<17} | {mae:>8.4f} | {mape:>7.2f}% |")
        ln()

    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 6 — Per-ASIN Accuracy  (was 5)
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 6. Per-ASIN Accuracy (Best 5 by RMSE)")
    ln()
    section_labels = iter("abcdefgh")
    for h in horizons:
        label     = next(section_labels)
        asin_path = _HERE / f"per_asin_accuracy_{h}d.csv"

        ln(f"### 6{label}. {h}-Day Horizon")
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
            ln("_per_asin_accuracy CSV not found — ensure main() completed._")
        ln()

    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 7 — Production Recommendation  (was 6)
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 7. Production Recommendation")
    ln()
    if all_passed:
        ln("**Recommendation: READY FOR DEPLOYMENT ✅**")
        ln()
        ln("All metric gates passed:")
        for h in horizons:
            ho  = run_result.horizons[h].holdout
            lat = run_result.horizons[h].latency
            ln(f"  - ✅ R²   ({h}d) = {ho['r2']:.4f}  (target ≥ {GATES['r2']['target']})")
            ln(f"  - ✅ MAE  ({h}d) = ${ho['mae']:.4f}  (target ≤ ${GATES['mae']['target']:.2f})")
            ln(f"  - ✅ MAPE ({h}d) = {ho['mape']:.2f}%   (target ≤ {GATES['mape']['target']:.0f}%)")
            ln(f"  - ✅ Latency p99 ({h}d) = {lat.get('p99', 0):.3f}ms  (target < {LATENCY_P99_TARGET_MS:.0f}ms)")
    else:
        ln("**Recommendation: FURTHER TUNING REQUIRED ❌**")
        ln()
        ln("The following gates were not met:")
        for fg in failed_gates:
            ln(f"  - ❌ {fg}")
        ln()
        ln("Suggested next steps:")
        ln("  - Increase training data (more ASINs or longer history)")
        ln("  - Tune `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight` in BASE_PARAMS")
        ln("  - Add missing features (promotions, competitor pricing, release date)")
        ln("  - Review Section 5 for distribution shift across the test period")
        ln("  - If latency gate fails: reduce `n_estimators`, lower `max_depth`, or limit features")

    ln()
    ln("---")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 8 — Model Configuration  (was 7)
    # ═══════════════════════════════════════════════════════════════════════
    ln("## 8. Model Configuration")
    ln()
    ln("| Parameter          | Value                               |")
    ln("|--------------------|-------------------------------------|")
    ln("| Algorithm          | XGBoost (gradient boosted trees)    |")
    ln("| Objective          | reg:squarederror (MSE)              |")
    ln(f"| Horizons           | {', '.join(str(h) + '-day' for h in horizons)}                       |")
    for k, v in BASE_PARAMS.items():
        ln(f"| {k:<18} | {str(v):<35} |")
    ln(f"| CV strategy        | TimeSeriesSplit ({N_CV_FOLDS} folds)           |")
    ln(f"| Early stopping     | {EARLY_STOPPING} rounds on val RMSE               |")
    ln(f"| Latency n_runs     | {run_result.horizons[horizons[0]].latency.get('n_runs', '—')}                                   |")
    ln(f"| Latency n_warmup   | {run_result.horizons[horizons[0]].latency.get('n_warmup', '—')}                                   |")
    ln("| Train/test split   | Time-based (no data leakage)        |")
    ln()

    # ═══════════════════════════════════════════════════════════════════════
    # Write report
    # ═══════════════════════════════════════════════════════════════════════
    report_path = _HERE / "MODEL_BENCHMARK_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")