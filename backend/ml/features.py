"""Leak-free features for the buy/wait model.

Training (train_clean.py), scoring (evaluate.py) and serving (inference.py) all
build features through this module, so the model sees the same numbers in
production that it saw in training.

Every feature at day t uses prices from day t-1 and earlier (shift(1) before any
rolling window), plus the price on day t itself. Nothing looks ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DAYS = 14
DROP_THRESHOLD = 0.08  # a WAIT is "right" if the price falls >= 8% within the horizon

# Inputs a price row may carry. Only `price` is required; the rest are optional
# secondary Keepa series and may be missing.
INPUT_COLS = ["price", "used_price", "count_new", "count_used"]

FEATURES = [
    "rel30", "rel90", "pos90", "rel_expmin", "rel_expmax", "rel_min180",
    "vol30", "vol90", "chg7", "chg30", "nchg30", "dsince",
    "logp", "month", "woy", "used_gap", "count_new", "count_used",
]

# Rows without these can't be scored (under ~30 days of history).
REQUIRED = ["med90", "vol30", "pos90"]


def daily_panel(df: pd.DataFrame, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per ASIN per day, forward-filling each price until it changes.

    `df` needs asin, date and price columns. `until` extends every series to that
    date, which is how serving treats "the price hasn't changed since the last
    Keepa event".
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in INPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[df["price"].notna()]

    out = []
    for asin, g in df.sort_values("date").groupby("asin"):
        g = g.groupby("date", as_index=False)[INPUT_COLS].last().set_index("date")
        end = max(g.index.max(), until) if until is not None else g.index.max()
        g = g.reindex(pd.date_range(g.index.min(), end, freq="D")).ffill()
        g["asin"] = asin
        out.append(g.reset_index().rename(columns={"index": "date"}))
    if not out:
        return pd.DataFrame(columns=["asin", "date", *INPUT_COLS])
    return pd.concat(out, ignore_index=True).sort_values(["asin", "date"]).reset_index(drop=True)


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURES to a daily panel from daily_panel()."""
    p = p.copy()
    for c in ["count_new", "count_used"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    g = p.groupby("asin")["price"]

    def past(fn):
        return g.transform(lambda s: fn(s.shift(1)))

    p["med30"] = past(lambda s: s.rolling(30, min_periods=10).median())
    p["med90"] = past(lambda s: s.rolling(90, min_periods=30).median())
    p["min90"] = past(lambda s: s.rolling(90, min_periods=30).min())
    p["max90"] = past(lambda s: s.rolling(90, min_periods=30).max())
    p["min180"] = past(lambda s: s.rolling(180, min_periods=60).min())
    p["exp_min"] = past(lambda s: s.expanding(min_periods=30).min())
    p["exp_max"] = past(lambda s: s.expanding(min_periods=30).max())
    p["vol30"] = past(lambda s: s.pct_change().rolling(30, min_periods=10).std())
    p["vol90"] = past(lambda s: s.pct_change().rolling(90, min_periods=30).std())
    p["chg7"] = past(lambda s: s.pct_change(7))
    p["chg30"] = past(lambda s: s.pct_change(30))
    p["nchg30"] = past(lambda s: (s.diff().abs() > 1e-9).rolling(30, min_periods=10).sum())
    p["dsince"] = past(lambda s: s.groupby((s.diff().abs() > 1e-9).cumsum()).cumcount())

    p["rel30"] = p.price / p.med30
    p["rel90"] = p.price / p.med90
    p["pos90"] = (p.price - p.min90) / (p.max90 - p.min90 + 1e-9)
    p["rel_expmin"] = p.price / p.exp_min
    p["rel_expmax"] = p.price / p.exp_max
    p["rel_min180"] = p.price / p.min180
    p["logp"] = np.log1p(p.price)
    p["month"] = p.date.dt.month
    p["woy"] = p.date.dt.isocalendar().week.astype(int)
    p["used_gap"] = (p.price - p.used_price) / p.price
    return p


def add_labels(p: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """Forward-looking truth, for training and scoring only. Never call at serve time.

    fwd_min:  lowest price over the next `horizon` days (what a WAIT could catch)
    fwd_last: price exactly `horizon` days out (what a WAIT gets with no alert)
    drop_pct: how far the price could fall, as a fraction of today's price
    y:        1 if drop_pct >= DROP_THRESHOLD
    """
    p = p.copy()

    def fwdmin(s):
        return s.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]

    p["fwd_min"] = p.groupby("asin")["price"].transform(fwdmin)
    p["fwd_last"] = p.groupby("asin")["price"].shift(-horizon)
    p = p.dropna(subset=["fwd_min", "fwd_last"])
    p["drop_pct"] = (p.price - p.fwd_min) / p.price
    p = p[p.drop_pct.between(-1, 1)]  # +/-100% moves are Keepa errors, not prices
    p["y"] = (p.drop_pct >= DROP_THRESHOLD).astype(int)
    return p
