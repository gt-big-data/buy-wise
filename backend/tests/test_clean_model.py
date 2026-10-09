"""
The clean model's two promises: features never look ahead, and serving builds
the same features training did.

Run: pytest tests/test_clean_model.py -q
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURES, add_features, daily_panel


def _history(days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    # Keepa-style: keep only the days the price changed (~1 in 4)
    keep = np.r_[True, rng.random(days - 1) < 0.25]
    return pd.DataFrame({
        "asin": "A", "date": dates[keep], "price": price[keep].round(2),
        "used_price": (price[keep] * 0.8).round(2), "count_new": 5, "count_used": 2,
    })


def test_features_ignore_the_future():
    h = _history()
    cutoff = pd.Timestamp("2025-09-01")
    full = add_features(daily_panel(h)).set_index("date")
    # Wreck every price after the cutoff; features up to the cutoff must not move.
    tampered = h.copy()
    tampered.loc[tampered.date > cutoff, "price"] *= 3
    cut = add_features(daily_panel(tampered)).set_index("date")
    pd.testing.assert_frame_equal(full.loc[:cutoff, FEATURES], cut.loc[:cutoff, FEATURES])


def test_serving_matches_training_features():
    h = _history()
    train_row = add_features(daily_panel(h)).set_index("date").loc["2025-10-15", FEATURES]
    # Serving only sees records up to that day, newest first, like the DB returns them.
    seen = h[h.date <= "2025-10-15"].iloc[::-1]
    serve_row = add_features(daily_panel(seen, until=pd.Timestamp("2025-10-15"))).iloc[-1][FEATURES]
    pd.testing.assert_series_equal(train_row, serve_row, check_names=False)


def test_predict_for_asin_shape():
    from ml import inference

    if not inference._MODELS_LOADED:
        pytest.skip("run ml/train_clean.py first")
    h = _history()
    records = [{"price": r.price, "timestamp": r.date.to_pydatetime(), "used_price": r.used_price,
                "count_new": 5, "count_used": 2} for r in h.itertuples()][::-1]
    out = inference.predict_for_asin(records, today=datetime(2026, 2, 5))
    assert out["recommendation"] in {"BUY", "WAIT"}
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["pred_14d"] > 0


def test_predict_accepts_db_decimals():
    """MySQL hands back DECIMAL columns as decimal.Decimal, not float."""
    from decimal import Decimal

    from ml import inference

    if not inference._MODELS_LOADED:
        pytest.skip("run ml/train_clean.py first")
    h = _history()
    records = [{"price": Decimal(str(r.price)), "timestamp": r.date.to_pydatetime(),
                "used_price": Decimal(str(r.used_price)), "count_new": 5, "count_used": None}
               for r in h.itertuples()][::-1]
    out = inference.predict_for_asin(records, today=datetime(2026, 2, 5))
    assert out["recommendation"] in {"BUY", "WAIT"}


def test_short_history_is_refused():
    from ml import inference

    if not inference._MODELS_LOADED:
        pytest.skip("run ml/train_clean.py first")
    now = datetime(2026, 2, 5)
    records = [{"price": 50.0 + i, "timestamp": now - timedelta(days=i)} for i in range(10)]
    with pytest.raises(RuntimeError):
        inference.predict_for_asin(records, today=now)
