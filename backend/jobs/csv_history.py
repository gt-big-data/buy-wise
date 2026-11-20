"""Price history for any product in the repo's datasets, in place of Keepa.

BuyWise no longer buys Keepa access. The two price datasets in studies/ cover 649
products between them, the same data every model and study was built on:

  studies/snapshots/apr2026_500/Prices.csv.gz   500 products, denser (preferred)
  studies/Prices.csv                            312 products

When the extension opens one of these products and the database doesn't have it yet,
the backend loads its history from here, the way it used to call Keepa. Both datasets
end in April 2026, and the dates stay real: a history ends in April, not today. The
model forward-fills the last April price to today, and the decision guard
(analysis.signals.history_out_of_date) turns WAIT off when the page price has moved
more than 10% since then.

In Docker, studies/ is mounted at /studies (docker-compose.yml). Set STUDIES_DIR to
read it from somewhere else.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

STUDIES = Path(os.getenv("STUDIES_DIR", Path(__file__).resolve().parents[2] / "studies"))
SOURCES = [STUDIES / "snapshots" / "apr2026_500" / "Prices.csv.gz", STUDIES / "Prices.csv"]
EXTRA = ("used_price", "list_price", "sales_rank", "count_new", "count_used")
INTS = {"sales_rank", "count_new", "count_used"}


@lru_cache(maxsize=None)
def _by_asin(path: Path) -> dict:
    """One dataset, split by ASIN, oldest first. Loaded once per process."""
    from ml.bench.data import load_prices
    if not path.exists():
        log.warning("price dataset missing: %s", path)
        return {}
    df = load_prices(path).dropna(subset=["price"])
    df = df[df.price > 0].sort_values(["asin", "date"])
    return {asin: g for asin, g in df.groupby("asin")}


@lru_cache(maxsize=None)
def _brands() -> dict:
    import pandas as pd
    out = {}
    for path in (STUDIES / "Product.csv", STUDIES / "snapshots" / "apr2026_500" / "Product.csv"):
        if path.exists():
            out.update(pd.read_csv(path, usecols=["asin", "brand"]).dropna().set_index("asin").brand.to_dict())
    return out


def warm() -> None:
    """Load both datasets now (a few seconds) instead of on the first request."""
    try:
        for path in SOURCES:
            _by_asin(path)
        _brands()
    except Exception as exc:
        log.warning("couldn't preload the price datasets: %s", exc)


def last_price(asin: str):
    """The last price the datasets have for this product (April 2026), or None."""
    for path in SOURCES:
        g = _by_asin(path).get(asin)
        if g is not None and not g.empty:
            return float(g.price.iloc[-1])
    return None


def known(asin: str) -> bool:
    return any(asin in _by_asin(p) for p in SOURCES)


def rows(asin: str, path: Path):
    """One product's rows in one dataset (asin, date, price, ...), oldest first, or None."""
    return _by_asin(path).get(asin)


def fetch_price_history(asin: str) -> list[dict]:
    """The product's price records, with their real dates. [] if no dataset has it.

    Same record shape as jobs.keepa_fetch: price, timestamp (datetime), used_price,
    list_price, sales_rank, count_new, count_used."""
    import pandas as pd
    for path in SOURCES:
        g = _by_asin(path).get(asin)
        if g is None or g.empty:
            continue
        records = []
        for row in g.itertuples(index=False):
            rec = {"price": round(float(row.price), 2), "timestamp": row.date.to_pydatetime()}
            for c in EXTRA:
                v = getattr(row, c)
                rec[c] = None if pd.isna(v) else int(v) if c in INTS else round(float(v), 2)
            records.append(rec)
        log.info("loaded %d prices for %s from %s, through %s", len(records), asin, path.name,
                 records[-1]["timestamp"].date())
        return records
    return []


def save(asin: str, records: list[dict]) -> bool:
    """Insert the product and its history. Returns False if there's nothing to save."""
    if not records:
        return False
    from db.connection import get_product, insert_prices, insert_product
    brand = _brands().get(asin)
    title = f"{brand} product {asin}" if brand else f"Product {asin}"
    insert_product(asin, title, brand=brand, category=None)
    product = get_product(asin)
    if not product:
        return False
    insert_prices(product["product_id"], records)
    return True
