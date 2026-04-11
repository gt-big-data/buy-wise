import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db.connection import get_product, insert_product, insert_price

load_dotenv()
API_KEY = os.getenv("KEEPA_API_KEY")

# Keepa encodes timestamps as minutes since 2011-01-01 00:00 UTC.
# Adding 21564000 minutes converts to minutes since Unix epoch (1970-01-01),
# then multiplying by 60 gives Unix seconds.
_KEEPA_EPOCH_OFFSET_MINUTES = 21564000

# Keepa csv array indices for price types
_CSV_AMAZON_PRICE = 0       # Amazon-fulfilled price (preferred)
_CSV_MARKETPLACE_NEW = 1    # Third-party new (fallback when Amazon doesn't sell directly)

# Rate limit: max retries before giving up on a throttled request
_MAX_RETRIES = 3


def fetch_price_history(asin: str, days: int = 30) -> list[dict]:
    """
    Fetch price history for an ASIN from Keepa and write records to the DB.

    Calls the Keepa /product endpoint, parses the Amazon price history from
    the csv array, converts timestamps to Unix seconds, and inserts each
    price record via insert_price(). Skips entries where price == -1
    (Keepa's sentinel for "out of stock / no listing").

    Rate limiting: Keepa returns a `refillIn` field (milliseconds) when the
    token bucket is empty. We sleep for that duration and retry up to
    _MAX_RETRIES times before raising. Keepa's free tier refills every 5 min;
    the paid key refills much faster. If we hit retries we raise so the caller
    can schedule a retry rather than blocking indefinitely.

    Args:
        asin: 10-character Amazon ASIN.
        days: How many days of history to request (default 30).

    Returns:
        List of dicts with keys: asin, name, timestamp (Unix seconds), price (USD float).

    Raises:
        RuntimeError: If the Keepa API returns an unexpected error response.
        RuntimeError: If rate limit retries are exhausted.
    """
    url = "https://api.keepa.com/product"
    params = {
        "key": API_KEY,
        "domain": 1,       # 1 = amazon.com
        "asin": asin,
        "history": 1,
        "days": days,
    }

    product = _fetch_with_retry(url, params)

    csv_data = product.get("csv") or []
    amazon_prices = csv_data[_CSV_AMAZON_PRICE] if len(csv_data) > _CSV_AMAZON_PRICE else []
    marketplace_prices = csv_data[_CSV_MARKETPLACE_NEW] if len(csv_data) > _CSV_MARKETPLACE_NEW else []
    # Prefer Amazon-fulfilled; fall back to marketplace new for third-party-only products
    new_prices = amazon_prices if amazon_prices else marketplace_prices
    name = product.get("title", "Unknown Product")

    records = _parse_price_records(asin, name, new_prices)
    _write_to_db(asin, name, records)

    return records


def _fetch_with_retry(url: str, params: dict) -> dict:
    """Call Keepa API, sleeping and retrying on rate-limit responses."""
    for attempt in range(_MAX_RETRIES):
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        if "products" in data:
            return data["products"][0]

        if "refillIn" in data:
            wait_seconds = data["refillIn"] / 1000
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait_seconds)
                continue
            else:
                raise RuntimeError(
                    f"Keepa rate limit exhausted after {_MAX_RETRIES} retries. "
                    f"refillIn={data['refillIn']}ms"
                )

        raise RuntimeError(f"Unexpected Keepa response: {data}")

    raise RuntimeError(f"Failed to fetch from Keepa after {_MAX_RETRIES} attempts")


def _parse_price_records(asin: str, name: str, raw: list) -> list[dict]:
    """
    Parse Keepa's flat [timestamp, price, timestamp, price, ...] csv array.

    Keepa timestamps are minutes since 2011-01-01 UTC. Prices are in cents;
    -1 means no listing / out of stock.
    """
    records = []
    for i in range(0, len(raw) - 1, 2):
        t = raw[i]
        price_cents = raw[i + 1]

        if price_cents == -1:
            continue

        unix_seconds = (t + _KEEPA_EPOCH_OFFSET_MINUTES) * 60
        records.append({
            "asin": asin,
            "name": name,
            "timestamp": unix_seconds,
            "price": price_cents / 100.0,
        })
    return records


def _write_to_db(asin: str, name: str, records: list[dict]) -> None:
    """Upsert the product and insert all price records into the DB."""
    if not records:
        return

    insert_product(asin, name, brand=None, category=None)
    product = get_product(asin)
    if not product:
        return

    product_id = product["product_id"]
    for record in records:
        ts = datetime.utcfromtimestamp(int(record["timestamp"]))
        insert_price(
            product_id,
            record["price"],
            availability=True,
            deal_flag=False,
            recorded_at=ts,
        )
