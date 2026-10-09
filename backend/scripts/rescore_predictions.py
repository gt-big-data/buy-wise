"""
Re-run the current model on every product in the DB and store a fresh prediction.

Run after retraining (ml/train_clean.py) so the extension stops serving
predictions from an older model. Products with under ~30 days of price
history are skipped and keep whatever prediction they had.

Usage:
    cd backend && .venv/bin/python scripts/rescore_predictions.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from db.connection import get_connection, insert_prediction, get_price_history
from ml.inference import predict_for_asin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT product_id, asin FROM products")
        products = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    scored = skipped = 0
    for product in products:
        prices = get_price_history(product["product_id"], limit=5000)
        try:
            r = predict_for_asin(prices)
        except RuntimeError as exc:
            logger.info("skip %s: %s", product["asin"], exc)
            skipped += 1
            continue
        insert_prediction(product["product_id"], r["pred_7d"], r["pred_14d"], r["pred_30d"],
                          r["recommendation"], r["confidence"])
        logger.info("%s -> %s (P(drop) %.0f%%)", product["asin"], r["recommendation"], r["p_drop"] * 100)
        scored += 1

    logger.info("rescored %d products, skipped %d", scored, skipped)


if __name__ == "__main__":
    main()
