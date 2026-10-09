"""Serve the leak-free buy/wait model trained by train_clean.py.

Features come from ml.features, the same code that built the training data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ml.features import REQUIRED, add_features, daily_panel

log = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).parent / "buywise_clean.joblib"

try:
    _bundle = joblib.load(MODEL_PATH)
    _MODELS_LOADED = True
    log.info("clean model loaded (trained %s)", _bundle["meta"]["trained_at"])
except Exception as _exc:
    _bundle = None
    _MODELS_LOADED = False
    log.warning("clean model not loaded: %s", _exc)


def predict_for_asin(price_records: list[dict], today: datetime | None = None) -> dict:
    """Run the model on DB price records (any order; DB returns newest first).

    Each record needs 'price' and 'timestamp'; used_price, count_new and
    count_used are used when present.

    Returns:
        {
            'pred_7d': None,
            'pred_14d': float,          # expected lowest price over the next 14 days
            'pred_30d': None,
            'recommendation': 'BUY' | 'WAIT',
            'confidence': float,        # calibrated, in [0, 1]
            'p_drop': float,            # calibrated P(price falls >= 8% within 14 days)
        }
    Raises RuntimeError if the model isn't loaded or the history is too short.
    """
    if not _MODELS_LOADED:
        raise RuntimeError("ML model not loaded")

    rows = [{
        "asin": "INFERENCE",
        "date": pd.Timestamp(r.get("timestamp") or r.get("date")),
        "price": float(r["price"]),
        "used_price": r.get("used_price"),
        "count_new": r.get("count_new"),
        "count_used": r.get("count_used"),
    } for r in price_records]
    today = pd.Timestamp(today or datetime.now(timezone.utc)).tz_localize(None).normalize()
    # Keepa records price *changes*, so the last price still holds today.
    panel = add_features(daily_panel(pd.DataFrame(rows), until=today))
    row = panel.iloc[[-1]]
    if row[REQUIRED].isna().any(axis=None):
        raise RuntimeError(f"Need ~30+ days of price history, got {len(panel)} days")

    X = row[_bundle["features"]].astype(float)
    raw = float(_bundle["classifier"].predict_proba(X)[0, 1])
    p_drop = float(_bundle["calibrator"].predict([raw])[0])
    wait = raw >= _bundle["threshold"]

    price = float(row.price.iloc[0])
    expected_drop = max(float(_bundle["regressor"].predict(X)[0]), 0.0)

    return {
        "pred_7d": None,
        "pred_14d": round(price * (1 - expected_drop), 2),
        "pred_30d": None,
        "recommendation": "WAIT" if wait else "BUY",
        "confidence": p_drop if wait else 1.0 - p_drop,
        "p_drop": p_drop,
    }
