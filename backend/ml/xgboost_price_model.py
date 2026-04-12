"""
XGBoost Regression — Amazon Price Prediction
=============================================
Predicts actual future price at 7-day and 14-day horizons.
Run dataset.py first.

At the end of main(), results are passed to xgboost_report.py which writes
MODEL_BENCHMARK_REPORT.md alongside the model files.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).parent

HORIZONS       = [7, 14]
N_CV_FOLDS     = 5
EARLY_STOPPING = 50

SKIP_AS_FEATURE = {
    "asin", "date", "timestamp", "brand", "category", "sales_rank",
    "target_7d", "target_14d",
    "amazon_lag_7d", "amazon_lag_14d",
}

BASE_PARAMS = dict(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 3,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
)


# ── RunResult — passed to xgboost_report.py ───────────────────────────────────
@dataclass
class HorizonResult:
    """All data produced for a single forecast horizon."""
    horizon:      int
    holdout:      dict                  # mae, rmse, r2, mape
    cv_results:   list[dict]            # one dict per fold
    feature_cols: list[str]
    test_df:      pd.DataFrame          # held-out rows (with 'asin', 'date')
    y_pred:       np.ndarray
    y_true:       np.ndarray
    latency:      dict = field(default_factory=dict)  # mean, median, p95, p99 in ms


@dataclass
class RunResult:
    """Aggregated output from main(); consumed by generate_report()."""
    horizons: dict[int, HorizonResult] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SKIP_AS_FEATURE]


def load_splits(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(_HERE / f"train_{horizon}day.csv", parse_dates=["date"])
    test  = pd.read_csv(_HERE / f"test_{horizon}day.csv",  parse_dates=["date"])
    log.info("[%dd] train=%d  test=%d", horizon, len(train), len(test))
    return train, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, tag: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    if tag:
        log.info("[%s]  MAE=%.4f  RMSE=%.4f  R²=%.4f  MAPE=%.2f%%",
                 tag, mae, rmse, r2, mape)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ── CV ────────────────────────────────────────────────────────────────────────
def run_cv(train: pd.DataFrame, feature_cols: list[str], horizon: int) -> list[dict]:
    train  = train.sort_values("date").reset_index(drop=True)
    target = f"target_{horizon}d"
    X, y   = train[feature_cols].values, train[target].values
    tscv   = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        model = XGBRegressor(
            **BASE_PARAMS,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=EARLY_STOPPING,
        )
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        m = regression_metrics(y[val_idx], model.predict(X[val_idx]),
                               tag=f"CV Fold {fold} | {horizon}d")
        m["fold"] = fold
        results.append(m)

    return results


# ── Final model ───────────────────────────────────────────────────────────────
def train_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> tuple[XGBRegressor, np.ndarray, np.ndarray]:
    target = f"target_{horizon}d"
    train  = train.sort_values("date").reset_index(drop=True)
    test   = test.sort_values("date").reset_index(drop=True)

    X_train, y_train = train[feature_cols].values, train[target].values
    X_test,  y_test  = test[feature_cols].values,  test[target].values

    model = XGBRegressor(
        **BASE_PARAMS,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    log.info("[%dd] Best iteration: %d", horizon, model.best_iteration)

    return model, model.predict(X_test), y_test


# ── Feature importance ────────────────────────────────────────────────────────
def save_feature_importance(model: XGBRegressor, feature_cols: list[str], horizon: int) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.nlargest(15)
    log.info("[%dd] Top 15 features:\n%s",
             horizon,
             "\n".join(f"  {f:<40s} {v:.4f}" for f, v in top.items()))
    imp.sort_values(ascending=False).to_csv(
        _HERE / f"feature_importance_{horizon}d.csv", header=["importance"]
    )


# ── Temporal error analysis ───────────────────────────────────────────────────
def temporal_error_analysis(
    test: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    horizon: int,
) -> None:
    df = test[["asin", "date"]].copy().reset_index(drop=True)
    df["y_true"]  = y_true
    df["y_pred"]  = y_pred
    df["abs_err"] = np.abs(y_true - y_pred)
    df["pct_err"] = np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)) * 100

    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("month")
          .agg(
              n        = ("abs_err", "count"),
              mae      = ("abs_err", "mean"),
              mape     = ("pct_err", "mean"),
              rmse     = ("abs_err", lambda x: float(np.sqrt((x**2).mean()))),
          )
          .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    monthly.to_csv(_HERE / f"temporal_error_{horizon}d.csv", index=False)

    df["price_tier"] = pd.qcut(df["y_true"], q=4,
                                labels=["Q1 (cheap)", "Q2", "Q3", "Q4 (expensive)"])
    tier = (
        df.groupby("price_tier", observed=True)
          .agg(
              n    = ("abs_err", "count"),
              mae  = ("abs_err", "mean"),
              mape = ("pct_err", "mean"),
          )
          .reset_index()
    )
    tier.to_csv(_HERE / f"price_tier_error_{horizon}d.csv", index=False)


# ── Per-ASIN breakdown ────────────────────────────────────────────────────────
def per_asin_accuracy(
    test: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray, horizon: int
) -> None:
    df = test[["asin", "date"]].copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    records = []
    for asin, g in df.groupby("asin"):
        if len(g) < 2:
            continue
        yt, yp = g["y_true"].values, g["y_pred"].values
        records.append({
            "asin": asin, "n_obs": len(g),
            "mae":  mean_absolute_error(yt, yp),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "r2":   float(r2_score(yt, yp)),
            "mape": float(np.mean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100),
        })

    result = pd.DataFrame(records).sort_values("rmse")
    result.to_csv(_HERE / f"per_asin_accuracy_{horizon}d.csv", index=False)
    log.info("[%dd] Per-ASIN saved. Best 5:\n%s",
             horizon, result.head(5).to_string(index=False))


# ── Latency benchmark ────────────────────────────────────────────────────────
LATENCY_WARMUP = 10
LATENCY_RUNS   = 200


def benchmark_latency(model: XGBRegressor, feature_cols: list[str], horizon: int) -> dict:
    """
    Measures single-row prediction latency (batch_size=1).

    Runs LATENCY_WARMUP warm-up calls (discarded) then LATENCY_RUNS timed calls,
    each on a single random feature row.  Returns mean, median, p95, and p99
    in milliseconds.
    """
    rng       = np.random.default_rng(42)
    n_feats   = len(feature_cols)
    sample    = rng.standard_normal((1, n_feats)).astype(np.float32)

    # warm-up — fills JIT / internal XGBoost caches
    for _ in range(LATENCY_WARMUP):
        model.predict(sample)

    times_ms = np.empty(LATENCY_RUNS)
    for i in range(LATENCY_RUNS):
        t0          = time.perf_counter()
        model.predict(sample)
        times_ms[i] = (time.perf_counter() - t0) * 1_000

    result = {
        "mean":   float(np.mean(times_ms)),
        "median": float(np.median(times_ms)),
        "p95":    float(np.percentile(times_ms, 95)),
        "p99":    float(np.percentile(times_ms, 99)),
        "n_runs": LATENCY_RUNS,
        "n_warmup": LATENCY_WARMUP,
    }
    log.info(
        "[%dd] Latency (batch=1, n=%d)  mean=%.3fms  median=%.3fms  p95=%.3fms  p99=%.3fms",
        horizon, LATENCY_RUNS,
        result["mean"], result["median"], result["p95"], result["p99"],
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from xgboost_report import generate_report  # imported here to keep files decoupled

    run_result = RunResult()

    for horizon in HORIZONS:
        log.info("\n" + "=" * 60)
        log.info("HORIZON: %d days", horizon)
        log.info("=" * 60)

        train, test  = load_splits(horizon)
        feature_cols = get_feature_cols(train)
        log.info("[%dd] %d features", horizon, len(feature_cols))

        target = f"target_{horizon}d"
        if target not in train.columns:
            log.error("'%s' not found — re-run dataset.py first.", target)
            continue

        # CV
        cv_results = run_cv(train, feature_cols, horizon)
        cv_df      = pd.DataFrame(cv_results)

        # Final model
        model, y_pred, y_true = train_final(train, test, feature_cols, horizon)

        # Metrics
        holdout = regression_metrics(
            y_true, y_pred, tag=f"HOLD-OUT | {horizon}d | n={len(y_true)}"
        )

        # Diagnostics — write CSVs read later by the report
        save_feature_importance(model, feature_cols, horizon)
        per_asin_accuracy(test.reset_index(drop=True), y_pred, y_true, horizon)
        temporal_error_analysis(test.reset_index(drop=True), y_pred, y_true, horizon)

        # Latency benchmark
        latency = benchmark_latency(model, feature_cols, horizon)

        # Save model
        joblib.dump(model, _HERE / f"xgb_reg_{horizon}d.joblib")
        log.info("[%dd] Model saved.", horizon)

        # Collect results for the report
        run_result.horizons[horizon] = HorizonResult(
            horizon      = horizon,
            holdout      = holdout,
            cv_results   = cv_results,
            feature_cols = feature_cols,
            test_df      = test.reset_index(drop=True),
            y_pred       = y_pred,
            y_true       = y_true,
            latency      = latency,
        )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for horizon, hr in run_result.horizons.items():
        ho = hr.holdout
        cv_df = pd.DataFrame(hr.cv_results)
        print(f"\n── {horizon}-Day Horizon ──────────────────────────")
        print(f"  Hold-out MAE  : {ho['mae']:.4f}")
        print(f"  Hold-out RMSE : {ho['rmse']:.4f}")
        print(f"  Hold-out R²   : {ho['r2']:.4f}")
        print(f"  Hold-out MAPE : {ho['mape']:.2f}%")
        print(f"  CV Mean RMSE  : {cv_df['rmse'].mean():.4f}")
        print(f"  CV Mean R²    : {cv_df['r2'].mean():.4f}")

    # ── Generate report ───────────────────────────────────────────────────────
    generate_report(run_result)
    print("\nDone. Outputs written to:", _HERE)


if __name__ == "__main__":
    main()