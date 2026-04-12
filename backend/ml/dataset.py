from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_HERE         = Path(__file__).parent
DATA_DIR      = _HERE
PRICES_PATH   = _HERE / "Prices.csv"
PRODUCTS_PATH = _HERE / "Product.csv"

HORIZONS = [7, 14]

KEEPA_FEATURE_COLS = [
    "price",
    "new_price", "used_price", "list_price",
    "count_new", "count_used",
    "sales_score",
    "amazon_ma_7d", "amazon_ma_14d",
    "amazon_delta_7d", "amazon_pct_change_7d",
    "day_of_week", "month",
]

ENGINEERED_COLS = [
    "price_lag1", "price_lag30",
    "roll7_std", "roll7_min", "roll7_max",
    "roll14_mean", "roll14_std", "roll14_min", "roll14_max",
    "roll30_mean", "roll30_std", "roll30_min", "roll30_max",
    "pct_change_1d", "pct_change_14d",
    "price_vs_7d_mean", "price_vs_14d_mean", "price_vs_30d_mean",
    "week_of_year", "is_weekend",
    "price_vs_new", "price_vs_used", "price_vs_list",
]

PRODUCT_COLS = [
    "global_min", "global_max", "global_norm_sd", "log_mean_price",
]

DERIVED_GLOBAL_COLS = [
    "price_vs_global_min", "price_vs_global_max", "price_vs_global_mean",
]

FEATURE_COLS = KEEPA_FEATURE_COLS + ENGINEERED_COLS + PRODUCT_COLS + DERIVED_GLOBAL_COLS

SKIP_SCALE = {"day_of_week", "month", "week_of_year", "is_weekend"}
SCALE_COLS = [c for c in FEATURE_COLS if c not in SKIP_SCALE]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _ensure_cols(df: pd.DataFrame, cols: list[str], fill=np.nan) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            log.warning("Column '%s' missing — filling with %s", c, fill)
            df[c] = fill
    return df


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_PATH, parse_dates=["datetime"])
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={
        "datetime":  "date",
        "amazon":    "price",
        "new":       "new_price",
        "used":      "used_price",
        "listprice": "list_price",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["price"].notna()].copy()

    optional = [
        "new_price", "used_price", "list_price",
        "count_new", "count_used", "sales_score",
        "amazon_lag_7d", "amazon_lag_14d",
        "amazon_ma_7d", "amazon_ma_14d",
        "amazon_delta_7d", "amazon_pct_change_7d",
    ]
    return _ensure_cols(df, optional)


def load_products() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_PATH)
    df.columns = df.columns.str.lower().str.strip()
    required = ["asin", "global_min", "global_max", "global_norm_sd", "log_mean_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Product.csv is missing columns: {missing}")
    return df[required]


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.sort_values("date")
          .groupby(["asin", "date"], as_index=False)
          .last()
    )
    out = []
    for asin, g in df.groupby("asin"):
        g = g.sort_values("date").set_index("date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_idx).ffill()
        g["asin"] = asin
        g = g.reset_index().rename(columns={"index": "date"})
        out.append(g)
    return pd.concat(out, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = []

    for asin, g in df.groupby("asin"):
        g = g.sort_values("date").copy()
        p = g["price"]

        g["price_lag1"]  = p.shift(1)
        g["price_lag30"] = p.shift(30)

        r7 = p.shift(1).rolling(7, min_periods=7)
        g["roll7_std"] = r7.std()
        g["roll7_min"] = r7.min()
        g["roll7_max"] = r7.max()

        for w, col_mean in [(14, "roll14_mean"), (30, "roll30_mean")]:
            r = p.shift(1).rolling(w, min_periods=w)
            g[col_mean]       = r.mean()
            g[f"roll{w}_std"] = r.std()
            g[f"roll{w}_min"] = r.min()
            g[f"roll{w}_max"] = r.max()

        g["pct_change_1d"]  = p.pct_change(1)
        g["pct_change_14d"] = p.pct_change(14)

        g["price_vs_7d_mean"]  = _safe_ratio(p - g["amazon_ma_7d"],  g["amazon_ma_7d"])
        g["price_vs_14d_mean"] = _safe_ratio(p - g["amazon_ma_14d"], g["amazon_ma_14d"])
        g["price_vs_30d_mean"] = _safe_ratio(p - g["roll30_mean"],   g["roll30_mean"])

        g["week_of_year"] = g["date"].dt.isocalendar().week.astype(int)
        g["is_weekend"]   = (g["day_of_week"] >= 5).astype(int)

        g["price_vs_new"]  = _safe_ratio(p - g["new_price"],  g["new_price"])
        g["price_vs_used"] = _safe_ratio(p - g["used_price"], g["used_price"])
        g["price_vs_list"] = _safe_ratio(p - g["list_price"], g["list_price"])

        # Regression targets only — actual price h days ahead
        for h in HORIZONS:
            g[f"target_{h}d"] = p.shift(-h)

        result.append(g)

    return pd.concat(result, ignore_index=True)


def merge_product_features(prices_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.merge(products_df, on="asin", how="left")

    for col in ["global_min", "global_max"]:
        df[col] = df[col].fillna(df["price"])
    df["global_norm_sd"] = df["global_norm_sd"].fillna(0.0)
    df["log_mean_price"] = df["log_mean_price"].fillna(np.log1p(df["price"]))
    df["global_mean"]    = np.expm1(df["log_mean_price"])

    df["price_vs_global_min"]  = _safe_ratio(df["price"] - df["global_min"],  df["global_min"])
    df["price_vs_global_max"]  = _safe_ratio(df["price"] - df["global_max"],  df["global_max"])
    df["price_vs_global_mean"] = _safe_ratio(df["price"] - df["global_mean"], df["global_mean"])

    return df


def get_split_date(df: pd.DataFrame) -> pd.Timestamp:
    dates = sorted(df["date"].unique())
    return pd.Timestamp(dates[int(len(dates) * 0.8)])


def split(df: pd.DataFrame, horizon: int, split_date: pd.Timestamp):
    target   = f"target_{horizon}d"
    max_date = df["date"].max()

    valid = df[df[target].notna()].copy()

    train = valid[
        (valid["date"] < split_date) &
        ((valid["date"] + pd.to_timedelta(horizon, "D")) < split_date)
    ].copy()

    test = valid[
        (valid["date"] >= split_date) &
        ((valid["date"] + pd.to_timedelta(horizon, "D")) <= max_date)
    ].copy()

    return train, test


def scale(train_7, test_7, train_14, test_14):
    feats      = [c for c in FEATURE_COLS if c in train_7.columns]
    scale_cols = [c for c in SCALE_COLS   if c in train_7.columns]

    medians = train_7[feats].median().fillna(0)
    for df in [train_7, test_7, train_14, test_14]:
        df[feats] = df[feats].fillna(medians)

    scaler = StandardScaler()
    scaler.fit(train_7[scale_cols])
    for df in [train_7, test_7, train_14, test_14]:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return train_7, test_7, train_14, test_14, scaler


def main():
    log.info("Loading Prices.csv …")
    prices = load_prices()

    log.info("Loading Product.csv …")
    products = load_products()

    log.info("Resampling to daily …")
    prices = resample_daily(prices)

    log.info("Engineering features …")
    prices = engineer_features(prices)

    log.info("Merging product metadata …")
    prices = merge_product_features(prices, products)

    split_date = get_split_date(prices)
    log.info("Split date: %s  (80/20 by date)", split_date.date())

    t7,  v7  = split(prices,  7, split_date)
    t14, v14 = split(prices, 14, split_date)

    log.info("Scaling (fit on train_7 only) …")
    t7, v7, t14, v14, scaler = scale(t7, v7, t14, v14)

    t7.to_csv( DATA_DIR / "train_7day.csv",  index=False)
    v7.to_csv( DATA_DIR / "test_7day.csv",   index=False)
    t14.to_csv(DATA_DIR / "train_14day.csv", index=False)
    v14.to_csv(DATA_DIR / "test_14day.csv",  index=False)
    joblib.dump(scaler, DATA_DIR / "scaler.joblib")

    log.info("Done — train_7=%d  test_7=%d  train_14=%d  test_14=%d",
             len(t7), len(v7), len(t14), len(v14))


if __name__ == "__main__":
    main()