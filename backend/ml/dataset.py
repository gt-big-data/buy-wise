"""
backend/ml/dataset.py
─────────────────────
Builds train/test datasets for BuyWise's 7-day and 14-day price-drop forecasts.

Pipeline
--------
1. Pull every product + its full price history from MySQL.
2. Resample to a uniform daily series (fill gaps via forward-fill).
3. Engineer features:  rolling stats, lag prices, calendar signals, deal flags.
4. Label each row:
     horizon=7  → 1 (WAIT) if min price in next 7 days is ≥5% below today
     horizon=14 → 1 (WAIT) if min price in next 14 days is ≥5% below today
5. Time-based split:  train = before Dec 1 2024 | test = Dec 1 2024 onward
6. Normalise numeric features with a StandardScaler fit ONLY on train.
7. Save four CSVs + the scaler to backend/ml/data/.

Usage
-----
    python -m backend.ml.dataset            # uses live DB
    python -m backend.ml.dataset --mock     # generates synthetic data (no DB needed)

The --mock flag lets the visualisation and platform teams keep working before
the DB is populated.  The synthetic data has the same schema as the real data.
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF   = date(2024, 12, 1)   # everything before this → train
DROP_THRESHOLD = 0.05                # 5 % price drop = WAIT label
HORIZONS       = [7, 14]             # forecast windows in days

FEATURE_COLS = [
    # lag prices
    "price_lag1", "price_lag7", "price_lag14",
    # rolling statistics
    "roll7_mean", "roll7_std", "roll7_min", "roll7_max",
    "roll14_mean", "roll14_std", "roll14_min", "roll14_max",
    "roll30_mean", "roll30_std",
    # momentum / trend
    "pct_change_1d", "pct_change_7d",
    "price_vs_30d_mean",      # (price - roll30_mean) / roll30_mean
    # calendar
    "day_of_week", "month", "week_of_year",
    "is_weekend",
    "days_to_prime_day",      # proxy for Amazon summer sale
    "days_to_black_friday",
    # product signals
    "deal_flag",
    "availability",
]

LABEL_COLS = ["label_7d", "label_14d"]
ID_COLS    = ["product_id", "asin", "date"]


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_from_db() -> pd.DataFrame:
    """Pull products + prices from MySQL; return a tidy long DataFrame."""
    # Add backend root to path so the relative import works when called as a script
    backend_root = str(_HERE.parent)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    from db.connection import get_connection  # type: ignore

    log.info("Connecting to database …")
    conn = get_connection()
    query = """
        SELECT
            p.product_id,
            p.asin,
            pr.price,
            pr.timestamp,
            pr.availability,
            pr.deal_flag
        FROM products p
        JOIN prices pr ON p.product_id = pr.product_id
        ORDER BY p.product_id, pr.timestamp
    """
    df = pd.read_sql(query, conn)
    conn.close()
    log.info("Loaded %d price records for %d products.", len(df), df["product_id"].nunique())
    return df


# ── Mock data generator ───────────────────────────────────────────────────────

def _generate_mock_data(
    n_products: int = 500,
    start: str = "2023-01-01",
    end: str = "2025-03-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesise realistic electronics price histories.

    Each product follows:
        price(t) = base * seasonal(t) * trend(t) + noise(t)

    Periodic promotional drops (Prime Day ≈ July, Black Friday ≈ Nov) are
    injected so the label distribution is non-trivial.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows: list[dict] = []

    CATEGORIES = ["Electronics", "Computers", "Smart Home", "Audio", "Cameras"]
    BRANDS     = ["Sony", "Apple", "Samsung", "Logitech", "Amazon", "Anker", "Bose"]

    for pid in range(1, n_products + 1):
        asin      = f"B{pid:09d}"
        base      = rng.uniform(20, 800)
        trend     = rng.uniform(-0.0003, 0.0001)   # slight downward drift (electronics depreciate)
        noise_std = base * rng.uniform(0.01, 0.04)

        for i, d in enumerate(dates):
            # Seasonal multiplier: dip in July (Prime Day) and Nov (Black Friday)
            seasonal = 1.0
            if d.month == 7 and 10 <= d.day <= 17:
                seasonal = rng.uniform(0.82, 0.92)   # Prime Day window
            elif d.month == 11 and 25 <= d.day <= 30:
                seasonal = rng.uniform(0.75, 0.88)   # Black Friday window
            elif d.month == 12 and d.day <= 5:
                seasonal = rng.uniform(0.80, 0.93)   # Cyber Monday tail

            price = max(5.0, base * (1 + trend * i) * seasonal + rng.normal(0, noise_std))
            price = round(price, 2)

            deal   = seasonal < 0.92                         # mark as deal during promo windows
            avail  = rng.random() > 0.02                     # 98 % availability

            rows.append({
                "product_id":   pid,
                "asin":         asin,
                "price":        price,
                "timestamp":    d,
                "availability": bool(avail),
                "deal_flag":    bool(deal),
                "category":     rng.choice(CATEGORIES),
                "brand":        rng.choice(BRANDS),
            })

    df = pd.DataFrame(rows)
    log.info(
        "Generated %d mock price records for %d products (%s → %s).",
        len(df), n_products, start, end,
    )
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def _prime_day_distance(d: pd.Timestamp) -> int:
    """Days until the nearest Prime Day (mid-July) — wraps across years."""
    prime = pd.Timestamp(year=d.year, month=7, day=12)
    diff  = (prime - d).days
    if diff < 0:
        prime = pd.Timestamp(year=d.year + 1, month=7, day=12)
        diff  = (prime - d).days
    return min(diff, 365)


def _black_friday_distance(d: pd.Timestamp) -> int:
    """Days until the nearest Black Friday (4th Thursday of November)."""
    year = d.year
    # 4th Thursday of November
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = nov1 + pd.offsets.Week(weekday=3)
    if first_thursday.month != 11:
        first_thursday += pd.Timedelta(weeks=1)
    bf = first_thursday + pd.Timedelta(weeks=3)
    diff = (bf - d).days
    if diff < 0:
        nov1 = pd.Timestamp(year=year + 1, month=11, day=1)
        first_thursday = nov1 + pd.offsets.Week(weekday=3)
        if first_thursday.month != 11:
            first_thursday += pd.Timedelta(weeks=1)
        bf = first_thursday + pd.Timedelta(weeks=3)
        diff = (bf - d).days
    return min(diff, 365)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw price DataFrame (one row per product-day after resampling),
    add all feature and label columns.  Operates product-by-product to avoid
    cross-contamination of rolling windows.
    """
    groups: list[pd.DataFrame] = []

    for pid, grp in df.groupby("product_id"):
        grp = grp.sort_values("date").copy()
        p   = grp["price"]

        # ── Lag prices ────────────────────────────────────────────────────────
        grp["price_lag1"]  = p.shift(1)
        grp["price_lag7"]  = p.shift(7)
        grp["price_lag14"] = p.shift(14)

        # ── Rolling statistics ────────────────────────────────────────────────
        for w in [7, 14, 30]:
            roll = p.shift(1).rolling(w, min_periods=max(1, w // 2))
            grp[f"roll{w}_mean"] = roll.mean()
            grp[f"roll{w}_std"]  = roll.std().fillna(0)
            if w in (7, 14):
                grp[f"roll{w}_min"] = roll.min()
                grp[f"roll{w}_max"] = roll.max()

        # ── Momentum ──────────────────────────────────────────────────────────
        grp["pct_change_1d"]    = p.pct_change(1)
        grp["pct_change_7d"]    = p.pct_change(7)
        grp["price_vs_30d_mean"] = (p - grp["roll30_mean"]) / grp["roll30_mean"].replace(0, np.nan)

        # ── Calendar ──────────────────────────────────────────────────────────
        grp["day_of_week"]         = grp["date"].dt.dayofweek
        grp["month"]               = grp["date"].dt.month
        grp["week_of_year"]        = grp["date"].dt.isocalendar().week.astype(int)
        grp["is_weekend"]          = (grp["day_of_week"] >= 5).astype(int)
        grp["days_to_prime_day"]   = grp["date"].apply(_prime_day_distance)
        grp["days_to_black_friday"]= grp["date"].apply(_black_friday_distance)

        # ── Labels ────────────────────────────────────────────────────────────
        for h in HORIZONS:
            future_min = (
                p.shift(-1)                         # exclude today
                 .rolling(h, min_periods=1)
                 .min()
                 .shift(-(h - 1))                   # align window to [t+1, t+h]
            )
            drop_pct       = (p - future_min) / p.replace(0, np.nan)
            grp[f"label_{h}d"] = (drop_pct >= DROP_THRESHOLD).astype(int)

        groups.append(grp)

    return pd.concat(groups, ignore_index=True)


# ── Resampling ────────────────────────────────────────────────────────────────

def _resample_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw timestamp-level data to one row per product per calendar day.

    - Multiple readings on the same day → take the last price.
    - Missing days → forward-fill price, carry forward deal_flag / availability.
    """
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["timestamp"]).dt.normalize()

    # Last price of each day
    daily = (
        raw.sort_values("timestamp")
           .groupby(["product_id", "asin", "date"])
           .agg(
               price       =("price",        "last"),
               availability=("availability", "last"),
               deal_flag   =("deal_flag",    "last"),
           )
           .reset_index()
    )

    # Ensure every product has a contiguous daily index (forward-fill gaps)
    complete: list[pd.DataFrame] = []
    global_min = daily["date"].min()
    global_max = daily["date"].max()
    full_idx   = pd.date_range(global_min, global_max, freq="D")

    for (pid, asin), grp in daily.groupby(["product_id", "asin"]):
        grp = grp.set_index("date").reindex(full_idx)
        grp["product_id"] = pid
        grp["asin"]       = asin
        grp[["price", "deal_flag", "availability"]] = (
            grp[["price", "deal_flag", "availability"]].ffill()
        )
        grp = grp.reset_index().rename(columns={"index": "date"})
        complete.append(grp)

    return pd.concat(complete, ignore_index=True)


# ── Train / test split ────────────────────────────────────────────────────────

def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train  = df[df["date"] < cutoff].copy()
    test   = df[df["date"] >= cutoff].copy()
    log.info(
        "Split → train: %d rows (%d products) | test: %d rows (%d products)",
        len(train), train["product_id"].nunique(),
        len(test),  test["product_id"].nunique(),
    )
    return train, test


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train features only; transform both splits."""
    scaler = StandardScaler()

    # Only scale columns that actually exist in this run
    cols = [c for c in FEATURE_COLS if c in train.columns]

    train = train.copy()
    test  = test.copy()

    # Drop rows where any feature is NaN (edges of rolling windows)
    before = len(train)
    train.dropna(subset=cols, inplace=True)
    log.info("Dropped %d train rows with NaN features (rolling-window edges).", before - len(train))

    before = len(test)
    test.dropna(subset=cols, inplace=True)
    log.info("Dropped %d test rows with NaN features.", before - len(test))

    train[cols] = scaler.fit_transform(train[cols])
    test[cols]  = scaler.transform(test[cols])

    return train, test, scaler


# ── Dataset analysis helpers ──────────────────────────────────────────────────

def _analyse(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print a brief dataset summary to stdout."""
    sep = "─" * 60

    print(f"\n{sep}")
    print("  BuyWise Dataset Analysis")
    print(sep)

    for split_name, split_df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{'─'*20} {split_name} {'─'*20}")
        print(f"  Rows      : {len(split_df):,}")
        print(f"  Products  : {split_df['product_id'].nunique():,}")
        if "date" in split_df.columns:
            print(f"  Date range: {split_df['date'].min().date()} → {split_df['date'].max().date()}")
        for h in HORIZONS:
            col = f"label_{h}d"
            if col in split_df.columns:
                wait_pct = split_df[col].mean() * 100
                print(f"  label_{h}d  :  WAIT={wait_pct:.1f}%  BUY={100-wait_pct:.1f}%")

    # Feature NaN check
    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    nan_counts = train[feat_cols].isna().sum()
    if nan_counts.any():
        print(f"\n  [WARNING] NaN counts in train features:")
        print(nan_counts[nan_counts > 0].to_string())
    else:
        print("\n  [OK] No NaN values remain in train features.")

    # Leakage guard
    if "date" in train.columns and "date" in test.columns:
        overlap = set(train["date"].dt.date) & set(test["date"].dt.date)
        if overlap:
            print(f"\n  [WARNING] {len(overlap)} dates appear in BOTH splits — check for leakage!")
        else:
            print("  [OK] No date overlap between train and test.")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_datasets(mock: bool = False) -> dict[str, pd.DataFrame]:
    """
    Full pipeline.  Returns a dict with keys:
        train_7day, test_7day, train_14day, test_14day
    """
    # 1. Load raw data
    if mock:
        log.info("--mock flag set: generating synthetic data.")
        raw = _generate_mock_data()
    else:
        raw = _load_from_db()

    # 2. Resample to daily
    log.info("Resampling to daily frequency …")
    daily = _resample_daily(raw)

    # 3. Feature engineering
    log.info("Engineering features and labels …")
    featured = _engineer_features(daily)

    # 4. Drop rows that can't have a valid label
    #    (last 14 days of each product have no future window to label)
    valid = featured.dropna(subset=["label_7d", "label_14d"]).copy()
    log.info("%d rows after dropping unlabeled tail.", len(valid))

    # 5. Split
    train_all, test_all = _split(valid)

    # 6. Normalise (scaler fit on train only)
    train_norm, test_norm, scaler = _normalise(train_all, test_all)

    # 7. Build per-horizon datasets (keep only relevant label column)
    datasets: dict[str, pd.DataFrame] = {}
    for h in HORIZONS:
        label   = f"label_{h}d"
        keep    = ID_COLS + FEATURE_COLS + [label]
        feat_cols = [c for c in keep if c in train_norm.columns]

        datasets[f"train_{h}day"] = train_norm[feat_cols].copy()
        datasets[f"test_{h}day"]  = test_norm[feat_cols].copy()

    # 8. Save
    for name, df in datasets.items():
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        log.info("Saved %s  →  %s  (%d rows)", name, out, len(df))

    scaler_path = DATA_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info("Saved scaler → %s", scaler_path)

    # 9. Analysis summary
    _analyse(datasets["train_7day"], datasets["test_7day"])

    return datasets


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BuyWise train/test datasets.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic data instead of reading from the database.",
    )
    args = parser.parse_args()
    build_datasets(mock=args.mock)
