"""
The clean model's two promises: features never look ahead, and serving builds
the same features training did.

Run: pytest tests/test_clean_model.py -q
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.features import (ALL_FEATURES, EXTRA_FEATURES, FEATURES, HORIZON_DAYS, add_features, add_labels,
                         daily_panel)


def _history(days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    # Keepa-style: keep only the days the price changed (~1 in 4)
    keep = np.r_[True, rng.random(days - 1) < 0.25]
    n = int(keep.sum())
    return pd.DataFrame({
        "asin": "A", "date": dates[keep], "price": price[keep].round(2),
        "used_price": (price[keep] * 0.8).round(2), "count_new": rng.integers(1, 20, n),
        "count_used": 2, "list_price": 130.0, "new_price": (price[keep] * rng.uniform(0.9, 1.1, n)).round(2),
        "sales_rank": rng.integers(100, 5000, n).astype(float),
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


def test_extra_features_ignore_the_future():
    h = _history()
    cutoff = pd.Timestamp("2025-09-01")
    full = add_features(daily_panel(h)).set_index("date")
    # Wreck every series a feature can read, after the cutoff.
    tampered = h.copy()
    late = tampered.date > cutoff
    for c in ["price", "used_price", "new_price", "list_price", "sales_rank", "count_new", "count_used"]:
        tampered.loc[late, c] *= 3
    cut = add_features(daily_panel(tampered)).set_index("date")
    assert full.loc[cutoff, EXTRA_FEATURES].notna().all(), "features should be past warm-up by the cutoff"
    pd.testing.assert_frame_equal(full.loc[:cutoff, ALL_FEATURES], cut.loc[:cutoff, ALL_FEATURES])


def test_past_drop_rate_ignores_its_own_window():
    """past_drop_rate is built from old labels, so it's the easiest feature to leak.
    On day t it may only use drop windows that closed by t-1. Changing prices from
    day t on changes the labels of days t-14..t-1, and must not change it."""
    panel = daily_panel(_history())
    full = add_features(panel).set_index("date")
    lab_full = add_labels(add_features(panel)).set_index("date").y
    yesterday_only = ["past_drop_rate", "days_since_drop", "dcount_new30", "rank_chg30"]
    labels_changed = 0
    # Many cutoffs, so a leak can't hide behind a label that was already 1.
    for cutoff in pd.date_range("2025-06-01", "2025-12-01", freq="5D"):
        tampered = panel.copy()
        tampered.loc[tampered.date >= cutoff, "price"] *= 0.5  # a huge drop starting on day t
        cut = add_features(tampered).set_index("date")
        lab_cut = add_labels(add_features(tampered)).set_index("date").y
        recent = slice(cutoff - pd.Timedelta(days=HORIZON_DAYS), cutoff - pd.Timedelta(days=1))
        labels_changed += int((lab_full.loc[recent] != lab_cut.loc[recent]).any())
        pd.testing.assert_frame_equal(full.loc[:cutoff, yesterday_only], cut.loc[:cutoff, yesterday_only])
    # The tampering really did change recent labels, so the check above meant something.
    assert labels_changed > 20


def test_serving_matches_training_features():
    h = _history()
    train_row = add_features(daily_panel(h)).set_index("date").loc["2025-10-15", ALL_FEATURES]
    # Serving only sees records up to that day, newest first, like the DB returns them.
    seen = h[h.date <= "2025-10-15"].iloc[::-1]
    serve_row = add_features(daily_panel(seen, until=pd.Timestamp("2025-10-15"))).iloc[-1][ALL_FEATURES]
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
