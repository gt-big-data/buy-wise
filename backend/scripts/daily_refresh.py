"""Fetch and score every tracked Keepa product once per UTC day.

Run from ``backend`` with the same environment used by the API.  The process
exits non-zero when any product fails so a scheduler can alert the owner; a
same-day retry is safe because price snapshots are upserted by UTC date.
"""

import logging
import sys
from datetime import datetime, timezone

from db.connection import (
    get_all_products,
    get_price_history,
    insert_prediction,
    upsert_daily_price_snapshot,
)
from jobs.keepa_fetch import fetch_price_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def refresh_product(product: dict) -> None:
    """Capture today's current price and produce a new prediction."""
    asin = product["asin"]
    # A short window keeps Keepa usage bounded.  We store only the newest price
    # as today's observation; complete history was collected at onboarding.
    records = fetch_price_history(asin, days=2, persist=False)
    if not records:
        raise RuntimeError("Keepa returned no usable price")

    latest = max(records, key=lambda record: record["timestamp"])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    upsert_daily_price_snapshot(
        product["product_id"], latest["price"], recorded_at=now,
        used_price=latest.get("used_price"), list_price=latest.get("list_price"),
        sales_rank=latest.get("sales_rank"), count_new=latest.get("count_new"),
        count_used=latest.get("count_used"),
    )

    prices = get_price_history(product["product_id"], limit=1000)
    from ml import inference
    prediction = inference.predict_for_asin(prices)
    insert_prediction(
        product["product_id"], prediction["pred_7d"], prediction["pred_14d"],
        prediction["pred_30d"], prediction["recommendation"], prediction["confidence"],
    )
    logger.info("refreshed and scored %s", asin)


def main() -> int:
    products = get_all_products()
    if not products:
        logger.warning("No tracked products: nothing to refresh")
        return 0

    failures = []
    for product in products:
        try:
            refresh_product(product)
        except Exception as exc:
            # Do not print request URLs or environment values: they can contain
            # the Keepa key.  The product identifier is safe and actionable.
            logger.exception("refresh failed for %s: %s", product["asin"], type(exc).__name__)
            failures.append(product["asin"])

    logger.info("daily refresh complete: %d succeeded, %d failed", len(products) - len(failures), len(failures))
    if failures:
        logger.error("failed ASINs: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
