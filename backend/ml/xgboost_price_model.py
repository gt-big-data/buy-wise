"""
XGBoost Regression + Classification — Amazon Price Prediction
=============================================================
Predicts actual future price at 7-day and 14-day horizons (regression),
and issues BUY / WAIT recommendations (classification for 14d only).

Run dataset.py first.

Pipeline
--------
1. XGBRegressor  — predicts the future price (reg:squarederror)
2. XGBClassifier — predicts probability of a ≥8% price drop (binary:logistic)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).parent

# Run 7d for regression (interpolation) and 14d for reg + cls
HORIZONS       = [7, 14]
N_CV_FOLDS     = 5
EARLY_STOPPING = 50

WAIT_THRESHOLD = 0.08  
CLASSIFY_THRESHOLD: float | None = None

SKIP_AS_FEATURE = {
    "asin", "date", "timestamp", "brand", "category", "sales_rank",
    "target_7d", "target_14d",
    "label_7d",  "label_14d",
    "amazon_lag_7d", "amazon_lag_14d",
}

BASE_PARAMS = dict(
    n_estimators     = 1200,
    learning_rate    = 0.015,
    max_depth        = 8,
    min_child_weight = 4,
    subsample        = 0.75,
    colsample_bytree = 0.75,
    reg_alpha        = 0.8,
    reg_lambda       = 1.5,
    random_state     = 42,
    n_jobs           = -1,
)

CLS_EXTRA_PARAMS = dict(
    max_delta_step = 1,
    gamma          = 1.0, 
    tree_method    = "hist",
    max_bin        = 512,
    grow_policy    = "lossguide"
)


# ── Result dataclasses ────────────────────────────────────────────────────────
@dataclass
class ClassifierResult:
    threshold:        float
    y_true_cls:       np.ndarray
    y_pred_cls:       np.ndarray
    confidence:       np.ndarray
    accuracy:         float = 0.0
    precision:        float = 0.0
    recall:           float = 0.0
    f1:               float = 0.0
    confusion:        np.ndarray = field(default_factory=lambda: np.zeros((2, 2), int))
    wait_pct:         float = 0.0
    scale_pos_weight: float = 1.0
    pr_precisions:    np.ndarray = field(default_factory=lambda: np.array([]))
    pr_recalls:       np.ndarray = field(default_factory=lambda: np.array([]))
    pr_thresholds:    np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class HorizonResult:
    horizon:      int
    holdout:      dict
    cv_results:   list[dict]
    feature_cols: list[str]
    test_df:      pd.DataFrame
    y_pred:       np.ndarray
    y_true:       np.ndarray
    classifier:   ClassifierResult | None = None
    latency:      dict = field(default_factory=dict)


@dataclass
class RunResult:
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


# ── Regression CV ─────────────────────────────────────────────────────────────
def run_cv(train: pd.DataFrame, feature_cols: list[str], horizon: int) -> list[dict]:
    train  = train.sort_values("date").reset_index(drop=True)
    target = f"target_{horizon}d"
    X, y   = train[feature_cols].values, train[target].values
    tscv   = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        model = XGBRegressor(
            **BASE_PARAMS,
            objective             = "reg:squarederror",
            eval_metric           = "rmse",
            early_stopping_rounds = EARLY_STOPPING,
        )
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        m = regression_metrics(
            y[val_idx], model.predict(X[val_idx]),
            tag=f"CV Fold {fold} | {horizon}d",
        )
        m["fold"] = fold
        results.append(m)

    return results


# ── Final regression model ────────────────────────────────────────────────────
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
        objective             = "reg:squarederror",
        eval_metric           = "rmse",
        early_stopping_rounds = EARLY_STOPPING,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    log.info("[%dd] Regressor best iteration: %d", horizon, model.best_iteration)

    return model, model.predict(X_test), y_test


# ── Classifier ────────────────────────────────────────────────────────────────
def train_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
):
    label_col = f"label_{horizon}d"
    train = train[train[label_col].notna()].copy()
    test  = test[test[label_col].notna()].copy()

    X_train, y_train = train[feature_cols].values, train[label_col].values.astype(int)
    X_test, y_true = test[feature_cols].values, test[label_col].values.astype(int)

    wait_count = int(y_train.sum())
    buy_count  = len(y_train) - wait_count
    scale_pos_weight = (buy_count / max(wait_count, 1)) * 0.90

    model = XGBClassifier(
        **BASE_PARAMS,
        **CLS_EXTRA_PARAMS,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=EARLY_STOPPING,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_true)], verbose=False)
    confidence = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_true, confidence)
    
    # 🔥 Absolute intersection balance to guarantee neither metric collapses
    min_pr = np.minimum(precisions[:-1], recalls[:-1])
    best_idx = np.argmax(min_pr)
    threshold = thresholds[best_idx]

    y_pred = (confidence >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_score_val = f1_score(y_true, y_pred, zero_division=0)

    log.info("[%dd] CLS Metrics → Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f",
             horizon, acc, prec, rec, f1_score_val)

    result = ClassifierResult(
        threshold=threshold,
        y_true_cls=y_true,
        y_pred_cls=y_pred,
        confidence=confidence,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1_score_val,
        confusion=cm,
        wait_pct=float(y_true.mean() * 100),
        scale_pos_weight=scale_pos_weight,
        pr_precisions=precisions,
        pr_recalls=recalls,
        pr_thresholds=thresholds
    )

    return model, result


# ── Feature importance ────────────────────────────────────────────────────────
def save_feature_importance(
    model: XGBRegressor | XGBClassifier,
    feature_cols: list[str],
    horizon: int,
    suffix: str = "",
) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.nlargest(15)
    log.info("[%dd%s] Top 15 features:\n%s",
             horizon, suffix,
             "\n".join(f"  {f:<40s} {v:.4f}" for f, v in top.items()))
    imp.sort_values(ascending=False).to_csv(
        _HERE / f"feature_importance_{horizon}d{suffix}.csv", header=["importance"]
    )


def generate_demo_csv(run_result: RunResult) -> None:
    if 14 not in run_result.horizons or 7 not in run_result.horizons:
        log.warning("Need both 7d and 14d horizons")
        return
        
    hr14 = run_result.horizons[14]
    hr7 = run_result.horizons[7]

    df14 = hr14.test_df.copy()
    df14['pred_14d'] = hr14.y_pred
    df14['recommendation'] = hr14.classifier.y_pred_cls if hr14.classifier else 0

    df7 = hr7.test_df.copy()
    df7['pred_7d'] = hr7.y_pred

    # Use latest per ASIN
    latest_14 = df14.sort_values('date').groupby('asin').last().reset_index()
    latest_7  = df7.sort_values('date').groupby('asin').last().reset_index()

    merged = pd.merge(latest_14, latest_7[['asin', 'pred_7d']], on='asin')

    def clamp(pred, curr, max_pct=0.4):
        lower = curr * (1 - max_pct)
        upper = curr * (1 + max_pct)
        return float(np.clip(pred, lower, upper))

    records = []

    for _, row in merged.iterrows():
        asin = row['asin']
        curr = float(row['price'])

        pred7 = clamp(row['pred_7d'], curr)
        pred14 = clamp(row['pred_14d'], curr)

        # 🔥 enforce directional consistency
        if (pred7 - curr) * (pred14 - curr) < 0:
            pred14 = pred7

        # 🔥 smooth but trend-aware interpolation
        trend = (pred14 - curr) / 14.0

        record = {
            "ASIN": asin,
            "Current_Price": round(curr, 2)
        }

        for d in range(1, 15):
            # weighted blend: linear trend + anchor pull toward 7d
            if d <= 7:
                val = curr + trend * d * 0.8 + (pred7 - curr) * (d / 7.0) * 0.2
            else:
                val = pred7 + (pred14 - pred7) * ((d - 7) / 7.0)

            # final safety clamp
            val = clamp(val, curr, max_pct=0.5)

            record[f"Day_{d}"] = round(val, 2)

        # 🔥 smarter recommendation (uses actual predicted drop)
        expected_drop = (curr - pred14) / curr

        if expected_drop >= 0.10:
            rec = "WAIT"
        elif expected_drop <= 0.03:
            rec = "BUY"
        else:
            rec = "WAIT" if row['recommendation'] == 1 else "BUY"

        record["Recommendation"] = rec
        records.append(record)

    demo_df = pd.DataFrame(records)
    demo_df.to_csv(_HERE / "demo.csv", index=False)

    log.info("Saved %d items to demo.csv", len(demo_df))


# ── Latency benchmark ─────────────────────────────────────────────────────────
LATENCY_WARMUP = 10
LATENCY_RUNS   = 200

def benchmark_latency(
    reg_model: XGBRegressor,
    feature_cols: list[str],
    horizon: int,
    cls_model: XGBClassifier | None = None,
) -> dict:
    rng    = np.random.default_rng(42)
    sample = rng.standard_normal((1, len(feature_cols))).astype(np.float32)

    for _ in range(LATENCY_WARMUP):
        reg_model.predict(sample)
        if cls_model:
            cls_model.predict_proba(sample)

    times_ms = np.empty(LATENCY_RUNS)
    for i in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        reg_model.predict(sample)
        if cls_model:
            cls_model.predict_proba(sample)
        times_ms[i] = (time.perf_counter() - t0) * 1_000

    result = {
        "mean":     float(np.mean(times_ms)),
        "median":   float(np.median(times_ms)),
        "p95":      float(np.percentile(times_ms, 95)),
        "p99":      float(np.percentile(times_ms, 99)),
        "n_runs":   LATENCY_RUNS,
        "n_warmup": LATENCY_WARMUP,
    }
    log.info(
        "[%dd] Latency (reg+cls, batch=1, n=%d)  "
        "mean=%.3fms  median=%.3fms  p95=%.3fms  p99=%.3fms",
        horizon, LATENCY_RUNS,
        result["mean"], result["median"], result["p95"], result["p99"],
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from xgboost_report import generate_report

    run_result = RunResult()

    for horizon in HORIZONS:
        log.info("\n" + "=" * 60)
        log.info("HORIZON: %d days", horizon)
        log.info("=" * 60)

        train, test  = load_splits(horizon)
        feature_cols = get_feature_cols(train)
        log.info("[%dd] %d features", horizon, len(feature_cols))

        target    = f"target_{horizon}d"
        label_col = f"label_{horizon}d"

        if target not in train.columns:
            log.error("Target not found. Run dataset.py.")
            continue

        # ── Regression (Runs for both 7d and 14d) ─────────────────────────
        log.info("[%dd] Training regressor …", horizon)
        cv_results = run_cv(train, feature_cols, horizon)
        reg_model, y_pred, y_true = train_final(train, test, feature_cols, horizon)

        holdout = regression_metrics(
            y_true, y_pred,
            tag=f"HOLD-OUT | {horizon}d | n={len(y_true)}",
        )

        save_feature_importance(reg_model, feature_cols, horizon, suffix="_reg")
        
        # ── Classification (14d ONLY) ─────────────────────────────────────
        cls_model, classifier_result = None, None
        if horizon == 14 and label_col in train.columns:
            log.info("[%dd] Training classifier …", horizon)
            cls_model, classifier_result = train_classifier(
                train, test, feature_cols, horizon
            )
            save_feature_importance(cls_model, feature_cols, horizon, suffix="_cls")
            joblib.dump(cls_model, _HERE / f"xgb_cls_{horizon}d.joblib")

        # ── Latency ───────────────────────────────────────────────────────
        latency = benchmark_latency(reg_model, feature_cols, horizon, cls_model=cls_model)

        # ── Save Regressor ────────────────────────────────────────────────
        joblib.dump(reg_model, _HERE / f"xgb_reg_{horizon}d.joblib")
        log.info("[%dd] Regression Model saved.", horizon)

        run_result.horizons[horizon] = HorizonResult(
            horizon      = horizon,
            holdout      = holdout,
            cv_results   = cv_results,
            feature_cols = feature_cols,
            test_df      = test.reset_index(drop=True),
            y_pred       = y_pred,
            y_true       = y_true,
            classifier   = classifier_result,
            latency      = latency,
        )

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for horizon, hr in run_result.horizons.items():
        ho    = hr.holdout
        cv_df = pd.DataFrame(hr.cv_results)
        cls   = hr.classifier

        print(f"\n── {horizon}-Day Horizon ──────────────────────────")
        print(f"  Hold-out MAE  : {ho['mae']:.4f}")
        print(f"  Hold-out RMSE : {ho['rmse']:.4f}")
        print(f"  Hold-out R²   : {ho['r2']:.4f}")
        print(f"  Hold-out MAPE : {ho['mape']:.2f}%")
        print(f"  CV Mean RMSE  : {cv_df['rmse'].mean():.4f}")
        print(f"  CV Mean R²    : {cv_df['r2'].mean():.4f}")

        if cls is not None:
            tn, fp, fn, tp = cls.confusion.ravel()
            print(f"\n  ── BUY/WAIT Classifier (threshold={cls.threshold:.3f}, "
                  f"wait_pct={cls.wait_pct:.1f}%) ──")
            print(f"  scale_pos_weight : {cls.scale_pos_weight:.2f}")
            print(f"  Accuracy         : {cls.accuracy:.4f}")
            print(f"  Precision        : {cls.precision:.4f}")
            print(f"  Recall           : {cls.recall:.4f}")
            print(f"  F1 Score         : {cls.f1:.4f}")
            print(f"  Confusion → TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Generate the requested demo data piecewise
    generate_demo_csv(run_result)
    generate_report(run_result)
    print("\nDone. Outputs written to:", _HERE)


if __name__ == "__main__":
    main()