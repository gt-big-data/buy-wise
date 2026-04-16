from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
_ML_DIR = Path(__file__).parent

try:
    _reg_7d  = joblib.load(_ML_DIR / "xgb_reg_7d.joblib")
    _reg_14d = joblib.load(_ML_DIR / "xgb_reg_14d.joblib")
    _cls_14d = joblib.load(_ML_DIR / "xgb_cls_14d.joblib")
    _scaler  = joblib.load(_ML_DIR / "scaler.joblib")
    _MODELS_LOADED = True
    log.info("ML models loaded from %s", _ML_DIR)
except Exception as _exc:
    _MODELS_LOADED = False
    log.warning("ML models not loaded: %s", _exc)


def predict_for_asin(price_records: list[dict]) -> dict:
    """Run XGBoost inference given DB price records (DESC timestamp order from DB).

    Each record must have 'price' and 'timestamp' keys.
    Returns:
        {
            'pred_7d': float,
            'pred_14d': float,
            'pred_30d': None,
            'recommendation': 'BUY' | 'WAIT',
            'confidence': float  # in [0, 1]
        }
    Raises RuntimeError if models not loaded or too few records.
    """
    if not _MODELS_LOADED:
        raise RuntimeError("ML models not loaded")
    if len(price_records) < 30:
        raise RuntimeError(f"Need ≥30 price records for ML inference, got {len(price_records)}")

    # Import here to avoid circular issues if dataset.py is imported at server startup
    from ml.dataset import (
        FEATURE_COLS,
        SCALE_COLS,
        engineer_features,
        merge_product_features,
    )

    DUMMY_ASIN = "INFERENCE"

    rows = []
    for rec in price_records:
        ts = rec.get("timestamp") or rec.get("date")
        rows.append({
            "asin":  DUMMY_ASIN,
            "date":  pd.Timestamp(ts),
            "price": float(rec["price"]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # Keepa columns the model expects — not available at inference time, fill NaN
    for col in [
        "new_price", "used_price", "list_price",
        "count_new", "count_used", "sales_score",
        "amazon_ma_7d", "amazon_ma_14d",
        "amazon_delta_7d", "amazon_pct_change_7d",
    ]:
        df[col] = np.nan

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month

    df = engineer_features(df)

    # Approximate product-level global stats from available price history
    p = df["price"]
    mean_price = float(p.mean())
    products_df = pd.DataFrame([{
        "asin":           DUMMY_ASIN,
        "global_min":     float(p.min()),
        "global_max":     float(p.max()),
        "global_norm_sd": float(p.std() / (mean_price + 1e-6)),
        "log_mean_price": float(np.log1p(mean_price)),
    }])

    df = merge_product_features(df, products_df)

    # Use the most recent row
    row = df.iloc[-1:].copy()

    feats      = [c for c in FEATURE_COLS if c in row.columns]
    scale_cols = [c for c in SCALE_COLS   if c in row.columns]

    # Fill NaN with column medians derived from the inference window
    for col in feats:
        if row[col].isna().any():
            med = float(df[col].median())
            row[col] = row[col].fillna(med if not np.isnan(med) else 0.0)

    row[scale_cols] = _scaler.transform(row[scale_cols])

    X = row[feats].values

    pred_7d  = float(_reg_7d.predict(X)[0])
    pred_14d = float(_reg_14d.predict(X)[0])

    proba_wait = float(_cls_14d.predict_proba(X)[0][1])
    cls_label  = int(_cls_14d.predict(X)[0])

    # price_records is DESC from DB — index 0 is the most recent price
    curr = float(price_records[0]["price"])
    expected_drop = (curr - pred_14d) / (curr + 1e-6)

    if expected_drop >= 0.10:
        recommendation = "WAIT"
        confidence     = proba_wait
    elif expected_drop <= 0.03:
        recommendation = "BUY"
        confidence     = 1.0 - proba_wait
    else:
        # Classifier tiebreaker in the ambiguous middle band
        if cls_label == 1:
            recommendation = "WAIT"
            confidence     = proba_wait
        else:
            recommendation = "BUY"
            confidence     = 1.0 - proba_wait

    confidence = float(np.clip(confidence, 0.50, 0.97))

    return {
        "pred_7d":        round(pred_7d, 2),
        "pred_14d":       round(pred_14d, 2),
        "pred_30d":       None,
        "recommendation": recommendation,
        "confidence":     confidence,
    }
