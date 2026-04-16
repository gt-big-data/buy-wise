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
WAIT_THRESHOLD = 0.08  

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
    "price_trend_7d",
    "price_trend_14d",
    "volatility_7d",
    "volatility_30d",
    "price_vs_30d_min",
    "price_vs_30d_max",
    "zscore_30d",
    "rsi_7d", "rsi_14d",          
    "dist_from_30d_high",        
    "dist_from_30d_low",
    "velocity_3d",               
    "day_of_year_sin",           
    "day_of_year_cos"
]

PRODUCT_COLS = [
    "global_min", "global_max", "global_norm_sd", "log_mean_price",
]

DERIVED_GLOBAL_COLS = [
    "price_vs_global_min", "price_vs_global_max", "price_vs_global_mean",
]

FEATURE_COLS = KEEPA_FEATURE_COLS + ENGINEERED_COLS + PRODUCT_COLS + DERIVED_GLOBAL_COLS

SKIP_SCALE = {"day_of_week", "month", "week_of_year", "is_weekend", "day_of_year_sin", "day_of_year_cos"}
SCALE_COLS = [c for c in FEATURE_COLS if c not in SKIP_SCALE and c != "price"]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _ensure_cols(df: pd.DataFrame, cols: list[str], fill=np.nan) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
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

        # RSI 
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        g["rsi_14d"] = 100 - (100 / (1 + rs))
        
        gain7 = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs7 = gain7 / (loss7 + 1e-6)
        g["rsi_7d"] = 100 - (100 / (1 + rs7))

        # Extreme Positioning
        high_30 = p.rolling(30).max()
        low_30 = p.rolling(30).min()
        g["dist_from_30d_high"] = (high_30 - p) / (high_30 + 1e-6)
        g["dist_from_30d_low"] = (p - low_30) / (low_30 + 1e-6)

        # Acceleration & Temporals
        g["velocity_3d"] = p.pct_change(3)
        day_of_year = g["date"].dt.dayofyear
        g["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        g["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Legacy Features
        g["price_z_asin"] = (p - p.mean()) / (p.std() + 1e-6)
        g["price_minmax_asin"] = (p - p.min()) / (p.max() - p.min() + 1e-6)
        
        roll7 = p.shift(1).rolling(7, min_periods=7)
        g["roll7_mean"] = roll7.mean()
        g["roll7_std"] = roll7.std()
        g["roll7_min"] = roll7.min()
        g["roll7_max"] = roll7.max()

        g["price_vs_roll7_z"] = (p - g["roll7_mean"]) / (g["roll7_std"] + 1e-6)
        g["price_lag1"] = p.shift(1)
        g["price_lag30"] = p.shift(30)
        g["pct_change_1d"] = p.pct_change(1)
        g["pct_change_14d"] = p.pct_change(14)
        
        # TARGET GENERATION
        for h in HORIZONS:
            # Regressor stays exact
            future_exact = p.shift(-h)
            g[f"target_{h}d"] = future_exact
            
            # Classifier gets a 3-day windowed minimum (e.g., Days 12, 13, 14)
            future_window_min = p.shift(-h).rolling(window=3, min_periods=1).min()
            
            drop_pct = _safe_ratio(p - future_window_min, p)
            
            is_wait = drop_pct >= WAIT_THRESHOLD
            is_buy = drop_pct <= 0.06
            
            labels = pd.Series(np.nan, index=g.index, dtype="Int8")
            labels.loc[is_wait] = 1
            labels.loc[is_buy] = 0
            
            g[f"label_{h}d"] = labels
            g.loc[g[f"target_{h}d"].isna(), f"label_{h}d"] = pd.NA

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
    target = f"target_{horizon}d"
    label  = f"label_{horizon}d"
    max_date = df["date"].max()
    valid = df[df[target].notna() & df[label].notna()].copy()
    train = valid[(valid["date"] < split_date) &
                  ((valid["date"] + pd.to_timedelta(horizon, "D")) < split_date)]
    test = valid[(valid["date"] >= split_date) &
                 ((valid["date"] + pd.to_timedelta(horizon, "D")) <= max_date)]
    return train.copy(), test.copy()


def scale(train_7, test_7, train_14, test_14):
    feats = [c for c in FEATURE_COLS if c in train_7.columns]
    scale_cols = [c for c in SCALE_COLS if c in train_7.columns]
    medians = train_7[feats].median().fillna(0)
    for df in [train_7, test_7, train_14, test_14]:
        df[feats] = df[feats].fillna(medians)
    scaler = StandardScaler()
    scaler.fit(train_7[scale_cols])
    for df in [train_7, test_7, train_14, test_14]:
        df[scale_cols] = scaler.transform(df[scale_cols])
    return train_7, test_7, train_14, test_14, scaler


def main():
    log.info("Loading data …")
    prices = load_prices()
    products = load_products()
    prices = resample_daily(prices)
    prices = engineer_features(prices)
    prices = merge_product_features(prices, products)
    split_date = get_split_date(prices)
    t7, v7 = split(prices, 7, split_date)
    t14, v14 = split(prices, 14, split_date)
    log.info("Scaling …")
    t7, v7, t14, v14, scaler = scale(t7, v7, t14, v14)
    t7.to_csv(DATA_DIR / "train_7day.csv", index=False)
    v7.to_csv(DATA_DIR / "test_7day.csv", index=False)
    t14.to_csv(DATA_DIR / "train_14day.csv", index=False)
    v14.to_csv(DATA_DIR / "test_14day.csv", index=False)
    joblib.dump(scaler, DATA_DIR / "scaler.joblib")
    log.info("Done")

if __name__ == "__main__":
    main()