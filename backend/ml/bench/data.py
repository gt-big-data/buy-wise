"""Load prices and build the table every model is scored on.

Today the source is studies/Prices.csv. The Keepa snapshot (T1, #83) will plug
in here as another source, so switching data is one argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.features import REQUIRED, add_features, add_labels, daily_panel

STUDIES = Path(__file__).resolve().parents[3] / "studies"
DEFAULT_SOURCE = STUDIES / "Prices.csv"
MIN_SPAN_DAYS = 270  # ASINs with less history can't fill the 180-day features


@dataclass
class Dataset:
    rows: pd.DataFrame    # one row per ASIN per day: features + labels
    panel: pd.DataFrame   # raw daily prices behind those rows (for PriceHistory)
    source: str


def load_prices(path: Path = DEFAULT_SOURCE) -> pd.DataFrame:
    """Read the Keepa export into the column names the backend DB uses."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"datetime": "date", "amazon": "price", "used": "used_price"})
    return df[["asin", "date", "price", "used_price", "count_new", "count_used"]]


def _usable(prices: pd.DataFrame) -> pd.DataFrame:
    span = prices.dropna(subset=["price"]).groupby("asin")["date"].agg(lambda s: (s.max() - s.min()).days)
    return prices[prices.asin.isin(span[span >= MIN_SPAN_DAYS].index)]


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Feature + label table (kept for train_clean.py and older callers)."""
    p = add_labels(add_features(daily_panel(_usable(prices))))
    return p.dropna(subset=REQUIRED).reset_index(drop=True)


def load_dataset(source: str | Path = DEFAULT_SOURCE) -> Dataset:
    """The one place data enters the benchmark. `source` is a path to a
    Prices.csv-style file. Keepa snapshots (`keepa:<id>`) arrive with T1."""
    s = str(source)
    if s.startswith("keepa:"):
        raise NotImplementedError("Keepa snapshots load through T1 (#83), which hasn't landed yet.")
    panel = daily_panel(_usable(load_prices(Path(s))))
    rows = add_labels(add_features(panel)).dropna(subset=REQUIRED).reset_index(drop=True)
    return Dataset(rows=rows, panel=panel, source=s)
