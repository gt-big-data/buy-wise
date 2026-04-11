"""
backend/ml/train.py

LightGBM training pipeline for 7-day and 14-day price prediction horizons.

Usage:
    # Train both horizons
    python -m backend.ml.train

    # Train single horizon
    python -m backend.ml.train --horizon 7
    python -m backend.ml.train --horizon 14

    # Custom data paths
    python -m backend.ml.train --horizon 7 \
        --train_csv data/train_7day.csv \
        --test_csv data/test_7day.csv

PACE (SLURM) usage:
    python -m backend.ml.train --horizon 7 \
        --train_csv $SCRATCH/data/train_7day.csv \
        --test_csv $SCRATCH/data/test_7day.csv \
        --output_dir $SCRATCH/models

Outputs (written to backend/ml/models/ by default):
    lgbm_7d.lgb          - Trained 7-day LightGBM model
    lgbm_14d.lgb         - Trained 14-day LightGBM model
    scaler_params.json   - Per-ASIN inverse transform params (global_mean, global_norm_sd)
    training_metadata.json - Feature list, hyperparams, early stopping round
"""

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.ml.features import (
    FEATURE_COLS,
    TARGET_7D,
    TARGET_14D,
    VOLATILITY_FEATURES,
    SEASONAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"

DEFAULT_PATHS = {
    7:  {"train": DATA_DIR / "train_7day.csv",  "test": DATA_DIR / "test_7day.csv"},
    14: {"train": DATA_DIR / "train_14day.csv", "test": DATA_DIR / "test_14day.csv"},
}

# ---------------------------------------------------------------------------
# LightGBM hyperparameters
# ---------------------------------------------------------------------------
# Tuned for structured retail price time-series:
#   - lower learning rate + more rounds = better generalization on seasonal signal
#   - num_leaves=63 balances capacity vs. overfitting on ~300-500 ASINs
#   - feature/bagging fraction adds regularization across correlated price features
#   - poisson objective considered but regression is cleaner for normalized targets

LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           ["mae", "rmse"],
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,             # unconstrained; num_leaves controls complexity
    "min_child_samples": 20,            # prevents overfitting on sparse ASINs
    "feature_fraction": 0.8,            # subsample features per tree
    "bagging_fraction": 0.8,            # subsample rows per iteration
    "bagging_freq":     5,
    "lambda_l1":        0.1,            # mild L1 for sparse feature selection
    "lambda_l2":        0.1,
    "verbose":          -1,
    "n_jobs":           -1,             # use all available CPUs (set to 8 on PACE)
    "seed":             42,
}

NUM_BOOST_ROUND  = 2000   # max rounds; early stopping will cut this down
EARLY_STOPPING   = 50     # stop if no improvement for 50 rounds on val RMSE
VALID_FRAC       = 0.15   # fraction of train set used as LightGBM validation split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_validate(path: Path, target_col: str) -> pd.DataFrame:
    """Load CSV and verify all required feature columns are present."""
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

    # Drop rows where target is NaN (can happen at dataset edges)
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN target '{target_col}'")

    print(f"  Shape: {df.shape}  |  ASINs: {df['asin'].nunique()}")
    return df


def extract_scaler_params(train_df: pd.DataFrame) -> dict:
    """
    Build per-ASIN inverse transform lookup from training data.
    Used in predict.py to convert normalized predictions back to dollar values.

    inverse_transform:
        price_dollars = (label_predicted * global_norm_sd) + global_mean
    """
    required = ["asin", "global_mean", "global_norm_sd"]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(
            f"Cannot build scaler_params — missing columns: {missing}\n"
            f"These should be present in the training CSV."
        )

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
    val_frac: float = VALID_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time rather than randomly.
    Assumes 'date' column is present and sortable (matches PACE/Keepa data format).
    Falls back to row-order split if 'date' is missing.
    """
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - val_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² for model evaluation."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
        "r2":   round(float(r2),   4),
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_horizon(
    horizon: int,
    train_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict:
    """
    Train a LightGBM regression model for a single price horizon.

    Args:
        horizon:     7 or 14 (days)
        train_path:  Path to train CSV
        test_path:   Path to test CSV
        output_dir:  Directory to save model + metadata

    Returns:
        dict with train/test metrics, best round, feature importances
    """
    target_col = TARGET_7D if horizon == 7 else TARGET_14D
    model_path = output_dir / f"lgbm_{horizon}d.lgb"

    print(f"\n{'='*60}")
    print(f"  Training LightGBM — {horizon}-day horizon")
    print(f"{'='*60}")

    # --- Load data ---
    train_df = load_and_validate(train_path, target_col)
    test_df  = load_and_validate(test_path,  target_col)

    # --- Extract scaler params from train set (first horizon only to avoid dupe) ---
    scaler_params = None
    if horizon == 7:
        scaler_params = extract_scaler_params(train_df)

    # --- Feature / target split ---
    # Filter to only columns that exist (handles slight CSV variations gracefully)
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    dropped_features   = [c for c in FEATURE_COLS if c not in train_df.columns]
    if dropped_features:
        print(f"  Warning: {len(dropped_features)} feature(s) not found, skipping: {dropped_features}")

    X_train_full = train_df[available_features]
    y_train_full = train_df[target_col]
    X_test       = test_df[available_features]
    y_test       = test_df[target_col]

    # --- Time-based internal validation split (for early stopping) ---
    train_sub, val_sub = time_based_val_split(train_df)
    X_tr  = train_sub[available_features]
    y_tr  = train_sub[target_col]
    X_val = val_sub[available_features]
    y_val = val_sub[target_col]

    print(f"\n  Train (full):    {len(X_train_full):,} rows")
    print(f"  Train (sub):     {len(X_tr):,} rows  |  Val: {len(X_val):,} rows")
    print(f"  Test:            {len(X_test):,} rows")
    print(f"  Features:        {len(available_features)}")

    # --- LightGBM datasets ---
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    # --- Train with early stopping ---
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

    # --- Retrain on FULL train set using best_round ---
    # Standard practice: early stopping finds optimal rounds on val split,
    # then retrain on all train data for final model
    print(f"\n  Retraining on full train set for {best_round} rounds...")
    dtrain_full = lgb.Dataset(X_train_full, label=y_train_full, free_raw_data=False)

    final_model = lgb.train(
        params          = {**LGBM_PARAMS, "verbose": -1},
        train_set       = dtrain_full,
        num_boost_round = best_round,
    )

    # --- Evaluate ---
    print("\n  Evaluating...")

    y_pred_train = final_model.predict(X_train_full)
    y_pred_test  = final_model.predict(X_test)

    train_metrics = regression_metrics(y_train_full.values, y_pred_train)
    test_metrics  = regression_metrics(y_test.values, y_pred_test)

    print(f"\n  Train metrics: MAE={train_metrics['mae']:.4f}  RMSE={train_metrics['rmse']:.4f}  R²={train_metrics['r2']:.4f}")
    print(f"  Test  metrics: MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}  R²={test_metrics['r2']:.4f}")

    # R² gate check
    if test_metrics["r2"] >= 0.75:
        print(f"  ✓ R² target met: {test_metrics['r2']:.4f} >= 0.75")
    else:
        print(f"  ✗ R² target NOT met: {test_metrics['r2']:.4f} < 0.75  (tune hyperparams or add features)")

    # --- Feature importance ---
    importance_gain = final_model.feature_importance(importance_type="gain")
    importance_df = (
        pd.DataFrame({"feature": available_features, "importance_gain": importance_gain})
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n  Top 10 features by gain:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<35} {row['importance_gain']:.1f}")

    # --- Save model ---
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    print(f"\n  Model saved → {model_path}")

    # --- Build result dict ---
    result = {
        "horizon":          horizon,
        "best_round":       best_round,
        "train_time_sec":   round(train_time, 2),
        "features_used":    available_features,
        "features_dropped": dropped_features,
        "train_metrics":    train_metrics,
        "test_metrics":     test_metrics,
        "top_features":     importance_df.head(20).to_dict(orient="records"),
        "hyperparams":      LGBM_PARAMS,
    }

    return result, scaler_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM price forecast models")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[7, 14],
        default=None,
        help="Which horizon to train (7 or 14). Omit to train both.",
    )
    parser.add_argument("--train_csv",  type=str, default=None, help="Override train CSV path")
    parser.add_argument("--test_csv",   type=str, default=None, help="Override test CSV path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory for model outputs (default: backend/ml/models/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = [args.horizon] if args.horizon else [7, 14]

    all_results    = {}
    scaler_params  = None

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

        # scaler_params extracted from 7d train set (same ASINs across both)
        if sp is not None:
            scaler_params = sp

    # --- Save scaler params ---
    if scaler_params:
        scaler_path = output_dir / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        print(f"\n  Scaler params saved → {scaler_path}")

    # --- Save training metadata ---
    metadata = {
        "trained_horizons": horizons,
        "results": all_results,
        "lgbm_version": lgb.__version__,
    }
    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Training metadata saved → {meta_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for h_key, res in all_results.items():
        m = res["test_metrics"]
        r2_flag = "✓" if m["r2"] >= 0.75 else "✗"
        print(f"  {h_key}:  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f} {r2_flag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()