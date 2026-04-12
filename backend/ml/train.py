"""
backend/ml/train.py

LightGBM binary classification pipeline for 7-day and 14-day price drop prediction.

Labels:
    label_7d / label_14d = 1 if price dropped at that horizon, 0 otherwise

Usage:
    python -m backend.ml.train
    python -m backend.ml.train --horizon 7
    python -m backend.ml.train --horizon 14

PACE usage:
    python -m backend.ml.train --horizon 7 \
        --train_csv $SCRATCH/buy-wise/data/train_7day.csv \
        --test_csv  $SCRATCH/buy-wise/data/test_7day.csv \
        --output_dir $SCRATCH/buy-wise/backend/ml/models

Outputs:
    lgbm_7d.lgb             - Trained 7-day model
    lgbm_14d.lgb            - Trained 14-day model
    scaler_params.json      - Per-ASIN inverse transform params
    training_metadata.json  - Feature list, metrics, hyperparams
"""

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.ml.features import FEATURE_COLS, TARGET_7D, TARGET_14D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = Path(__file__).resolve().parent / "models"

DEFAULT_PATHS = {
    7:  {"train": DATA_DIR / "train_7day.csv",  "test": DATA_DIR / "test_7day.csv"},
    14: {"train": DATA_DIR / "train_14day.csv", "test": DATA_DIR / "test_14day.csv"},
}

# ---------------------------------------------------------------------------
# LightGBM hyperparameters — binary classification
# ---------------------------------------------------------------------------

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            ["binary_logloss", "auc"],
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "verbose":           -1,
    "n_jobs":            -1,
    "seed":              42,
    "is_unbalance":      True,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING  = 50
VALID_FRAC      = 0.15
CLASSIFICATION_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_validate(path: Path, target_col: str) -> pd.DataFrame:
    print(f"  Loading {path} ...")
    df = pd.read_csv(path)

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing feature columns in {path.name}:\n  {missing_features}\n"
            f"Check features.py FEATURE_COLS list."
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path.name}")

    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN target '{target_col}'")

    df[target_col] = df[target_col].astype(int)

    print(f"  Shape: {df.shape}  |  ASINs: {df['asin'].nunique()}")
    pos_rate = df[target_col].mean()
    print(f"  Positive rate (price drops): {pos_rate:.1%}")

    return df


def extract_scaler_params(train_df: pd.DataFrame) -> dict:
    required = ["asin", "global_mean", "global_norm_sd"]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Cannot build scaler_params — missing columns: {missing}")

    scaler = (
        train_df[required]
        .drop_duplicates("asin")
        .set_index("asin")
        .to_dict(orient="index")
    )
    print(f"  Scaler params extracted for {len(scaler)} ASINs")
    return scaler


def time_based_val_split(
    df: pd.DataFrame,
    val_frac: float = VALID_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def classification_metrics(
    y_true: np.ndarray,
    y_prob:  np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    f1        = f1_score(y_true,        y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true,    y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true,  y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    return {
        "f1":           round(float(f1),        4),
        "precision":    round(float(precision), 4),
        "recall":       round(float(recall),    4),
        "accuracy":     round(float(accuracy),  4),
        "auc":          round(float(auc),       4),
        "threshold":    threshold,
        "n_samples":    len(y_true),
        "n_positive":   int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_horizon(
    horizon:     int,
    train_path:  Path,
    test_path:   Path,
    output_dir:  Path,
) -> tuple[dict, dict | None]:

    target_col = TARGET_7D if horizon == 7 else TARGET_14D
    model_path = output_dir / f"lgbm_{horizon}d.lgb"

    print(f"\n{'='*60}")
    print(f"  Training LightGBM — {horizon}-day horizon (binary classification)")
    print(f"{'='*60}")

    train_df = load_and_validate(train_path, target_col)
    test_df  = load_and_validate(test_path,  target_col)

    scaler_params = None
    if horizon == 7:
        scaler_params = extract_scaler_params(train_df)

    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    dropped_features   = [c for c in FEATURE_COLS if c not in train_df.columns]
    if dropped_features:
        print(f"  Warning: skipping {len(dropped_features)} missing features: {dropped_features}")

    X_train_full = train_df[available_features]
    y_train_full = train_df[target_col]
    X_test       = test_df[available_features]
    y_test       = test_df[target_col]

    train_sub, val_sub = time_based_val_split(train_df)
    X_tr  = train_sub[available_features]
    y_tr  = train_sub[target_col]
    X_val = val_sub[available_features]
    y_val = val_sub[target_col]

    print(f"\n  Train (full):  {len(X_train_full):,} rows  |  "
          f"positives: {int(y_train_full.sum()):,} ({y_train_full.mean():.1%})")
    print(f"  Train (sub):   {len(X_tr):,} rows  |  Val: {len(X_val):,} rows")
    print(f"  Test:          {len(X_test):,} rows  |  "
          f"positives: {int(y_test.sum()):,} ({y_test.mean():.1%})")
    print(f"  Features:      {len(available_features)}")

    dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    print(f"\n  Training (max {NUM_BOOST_ROUND} rounds, early stop={EARLY_STOPPING})...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    t0 = time.time()
    model = lgb.train(
        params          = LGBM_PARAMS,
        train_set       = dtrain,
        num_boost_round = NUM_BOOST_ROUND,
        valid_sets      = [dtrain, dval],
        valid_names     = ["train", "val"],
        callbacks       = callbacks,
    )
    train_time = time.time() - t0
    best_round = model.best_iteration
    print(f"\n  Best round: {best_round}  |  Train time: {train_time:.1f}s")

    print(f"\n  Retraining on full train set for {best_round} rounds...")
    dtrain_full = lgb.Dataset(X_train_full, label=y_train_full, free_raw_data=False)
    final_model = lgb.train(
        params          = {**LGBM_PARAMS, "verbose": -1},
        train_set       = dtrain_full,
        num_boost_round = best_round,
    )

    print("\n  Evaluating...")
    y_prob_train = final_model.predict(X_train_full)
    y_prob_test  = final_model.predict(X_test)

    train_metrics = classification_metrics(y_train_full.values, y_prob_train)
    test_metrics  = classification_metrics(y_test.values,       y_prob_test)

    print(f"\n  Train — F1={train_metrics['f1']:.4f}  "
          f"Precision={train_metrics['precision']:.4f}  "
          f"Recall={train_metrics['recall']:.4f}  "
          f"AUC={train_metrics['auc']:.4f}")
    print(f"  Test  — F1={test_metrics['f1']:.4f}  "
          f"Precision={test_metrics['precision']:.4f}  "
          f"Recall={test_metrics['recall']:.4f}  "
          f"AUC={test_metrics['auc']:.4f}")

    f1_flag  = "✓" if test_metrics['f1']  >= 0.70 else "✗"
    auc_flag = "✓" if test_metrics['auc'] >= 0.75 else "✗"
    print(f"\n  F1  >= 0.70:  {f1_flag} ({test_metrics['f1']:.4f})")
    print(f"  AUC >= 0.75:  {auc_flag} ({test_metrics['auc']:.4f})")

    importance_gain = final_model.feature_importance(importance_type="gain")
    importance_df = (
        pd.DataFrame({"feature": available_features, "importance_gain": importance_gain})
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n  Top 10 features by gain:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<35} {row['importance_gain']:.1f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    print(f"\n  Model saved → {model_path}")

    result = {
        "horizon":           horizon,
        "task":              "binary_classification",
        "best_round":        best_round,
        "train_time_sec":    round(train_time, 2),
        "threshold":         CLASSIFICATION_THRESHOLD,
        "features_used":     available_features,
        "features_dropped":  dropped_features,
        "train_metrics":     train_metrics,
        "test_metrics":      test_metrics,
        "top_features":      importance_df.head(20).to_dict(orient="records"),
        "hyperparams":       LGBM_PARAMS,
    }

    return result, scaler_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM binary classification models for price drop prediction"
    )
    parser.add_argument("--horizon",    type=int, choices=[7, 14], default=None)
    parser.add_argument("--train_csv",  type=str, default=None)
    parser.add_argument("--test_csv",   type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons      = [args.horizon] if args.horizon else [7, 14]
    all_results   = {}
    scaler_params = None

    for h in horizons:
        if args.train_csv and args.test_csv:
            train_path = Path(args.train_csv)
            test_path  = Path(args.test_csv)
        else:
            train_path = DEFAULT_PATHS[h]["train"]
            test_path  = DEFAULT_PATHS[h]["test"]

        result, sp = train_horizon(
            horizon    = h,
            train_path = train_path,
            test_path  = test_path,
            output_dir = output_dir,
        )
        all_results[f"{h}d"] = result
        if sp is not None:
            scaler_params = sp

    if scaler_params:
        scaler_path = output_dir / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        print(f"\n  Scaler params saved → {scaler_path}")

    metadata = {
        "trained_horizons": horizons,
        "results":          all_results,
        "lgbm_version":     lgb.__version__,
    }
    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Training metadata saved → {meta_path}")

    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for h_key, res in all_results.items():
        m = res["test_metrics"]
        f1_flag  = "✓" if m["f1"]  >= 0.70 else "✗"
        auc_flag = "✓" if m["auc"] >= 0.75 else "✗"
        print(f"  {h_key}:  F1={m['f1']:.4f} {f1_flag}  "
              f"AUC={m['auc']:.4f} {auc_flag}  "
              f"Precision={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()