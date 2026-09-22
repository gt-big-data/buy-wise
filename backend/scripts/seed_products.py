"""
Bulk-seed a curated list of popular products into the BuyWise DB.
Fetches price history from Keepa and generates ML predictions for each ASIN.

Usage:
    cd backend && .venv/bin/python scripts/seed_products.py
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from jobs.keepa_fetch import fetch_price_history
from db.connection import get_product, insert_prediction, get_price_history as db_get_price_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Products chosen for high Keepa coverage: sold on Amazon for 2+ years,
# frequent price changes (competitive categories, Prime Day/BF sales).
ASINS = [
    # Smart home & plugs — volatile pricing, Amazon-tracked
    ("B01MZEEFNX", "TP-Link Kasa Smart Plug HS103"),
    ("B086G3WJHR", "Amazon Smart Plug"),
    ("B082YD279H", "Kasa Smart Light Bulbs 4-Pack"),

    # Storage — highly competitive, daily price changes
    ("B07H289S7C", "Seagate Portable 2TB External Hard Drive"),
    ("B01LQQH86O", "SanDisk 128GB Ultra USB 3.0 Flash Drive"),
    ("B09CLG4JLY", "Samsung T7 Portable SSD 1TB"),

    # Amazon devices — Amazon sets/changes price frequently
    ("B08C1W5N87", "Fire TV Stick 4K"),
    ("B08L5NP6NG", "Echo Dot 4th Gen"),
    ("B09B8YWXDF", "Echo Dot 5th Gen"),

    # Networking
    ("B08H95TKR9", "TP-Link AX1800 WiFi 6 Router"),
    ("B003U0JGUM", "TP-Link N300 Wi-Fi Extender"),

    # Kitchen appliances — major brand, heavy sale history
    ("B00FLYWNYQ", "Instant Pot Duo 6 Qt 7-in-1"),
    ("B07GJBBGHG", "Ninja AF101 Air Fryer 4 Qt"),
    ("B0088LR592", "Hamilton Beach 12-Cup Coffee Maker"),
    ("B07THGGYZZ", "BLACK+DECKER 5-Cup Coffee Maker"),

    # Vacuums & cleaning
    ("B07ZD3K3DG", "iRobot Roomba 675 Robot Vacuum"),
    ("B08P2CMTSL", "Eufy RoboVac 11S Robot Vacuum"),

    # Headphones & audio
    ("B09XS7JWHH", "Sony WH-1000XM5 Wireless Headphones"),
    ("B07G9WHRLG", "JBL Charge 4 Bluetooth Speaker"),

    # Cables & accessories — high volume, frequent deals
    ("B09C4KKP8C", "Anker USB-C to USB-C Cable 6ft"),
    ("B07THHQMHM", "Instant Vortex 6-Qt Air Fryer"),
]

# 365 days to maximize records — ML needs ≥30
_FETCH_DAYS = 365
# Pause between Keepa calls to stay within free-tier token budget
_SLEEP_BETWEEN = 2.5
_ML_MIN_RECORDS = 30


def seed_asin(asin: str, label: str) -> bool:
    logger.info("Seeding %s (%s)", asin, label)
    try:
        records = fetch_price_history(asin, days=_FETCH_DAYS)
        logger.info("  Keepa returned %d price records", len(records))
    except Exception as exc:
        logger.warning("  Keepa fetch failed: %s", exc)
        return False

    if not records:
        logger.warning("  No price records — skipping")
        return False

    product = get_product(asin)
    if not product:
        logger.warning("  Product not in DB after fetch — skipping")
        return False

    prices = db_get_price_history(product["product_id"], limit=1000)
    if len(prices) < _ML_MIN_RECORDS:
        logger.warning(
            "  Only %d price records in DB (need %d) — skipping (ML would fail)",
            len(prices), _ML_MIN_RECORDS,
        )
        return False

    try:
        from ml import inference as _ml
        result = _ml.predict_for_asin(prices)
        insert_prediction(
            product["product_id"],
            result["pred_7d"],
            result["pred_14d"],
            result["pred_30d"],
            result["recommendation"],
            result["confidence"],
        )
        logger.info(
            "  ML prediction: %s  confidence=%.0f%%  pred_14d=$%.2f",
            result["recommendation"],
            result["confidence"] * 100,
            result["pred_14d"] or 0,
        )
        return True
    except Exception as exc:
        logger.warning("  ML inference failed: %s — skipping (no heuristic fallback)", exc)
        return False


def main():
    succeeded, failed = 0, 0
    for asin, label in ASINS:
        ok = seed_asin(asin, label)
        if ok:
            succeeded += 1
        else:
            failed += 1
        time.sleep(_SLEEP_BETWEEN)

    logger.info("Done. %d succeeded, %d failed.", succeeded, failed)


if __name__ == "__main__":
    main()
