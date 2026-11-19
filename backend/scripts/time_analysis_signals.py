"""
How long analysis_signals() + decide() take per product page (#87).

Runs every product in studies/Prices.csv through the function with its real price
history, a cached prediction (the endpoint's normal path), and a page of offers, then
reports the median, 95th percentile and worst case in milliseconds. The first call in
a process also loads the results file; that's reported separately.

Usage:
    cd backend
    python scripts/time_analysis_signals.py              # 20 offers per page
    python scripts/time_analysis_signals.py --offers 60  # a crowded offer list
    python scripts/time_analysis_signals.py --model      # run the model instead of the cached prediction
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.signals import analysis_signals, decide, load_results
from ml.bench.data import load_prices

PRICES = Path(__file__).resolve().parents[2] / "studies" / "Prices.csv"


def fake_offers(price: float, n: int, today: date, rng: random.Random) -> list[dict]:
    offers = [{"seller": "Amazon.com", "price": price, "condition": "New", "pinned": True,
               "deliveryDate": (today + timedelta(days=2)).isoformat()}]
    for i in range(n - 1):
        offers.append({"seller": f"Seller {i}", "price": round(price * rng.uniform(0.7, 1.2), 2),
                       "condition": rng.choice(["New", "Used - Like New", "Used - Good"]),
                       "deliveryDate": (today + timedelta(days=rng.randint(1, 14))).isoformat(),
                       "stars": round(rng.uniform(3, 5), 1), "years": rng.randint(0, 10),
                       "freeReturns": rng.random() < 0.5})
    return offers


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offers", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=5, help="calls per product")
    ap.add_argument("--model", action="store_true", help="no cached prediction: run the served model")
    args = ap.parse_args(argv)

    df = load_prices(PRICES)  # same column names as the DB
    rng = random.Random(0)
    load_results.cache_clear()
    cold_start = time.perf_counter()
    load_results()
    cold_ms = (time.perf_counter() - cold_start) * 1000

    times, statuses = [], {}
    for asin, g in df.groupby("asin"):
        g = g.sort_values("date")
        today = g.date.max().date() + timedelta(days=1)
        prices = [{"price": float(p), "timestamp": t.to_pydatetime()}
                  for t, p in zip(g.date, g.price) if pd.notna(p)]
        if not prices:
            continue
        price = prices[-1]["price"]
        prediction = None if args.model else {
            "recommendation": rng.choice(["BUY", "WAIT"]), "confidence_score": rng.uniform(0.5, 0.9),
            "pred_14d": round(price * rng.uniform(0.8, 1.0), 2)}
        offers = fake_offers(price, args.offers, today, rng)
        for _ in range(args.repeat):
            start = time.perf_counter()
            d = decide(analysis_signals(asin, offers, {"condition": "any_good"}, prices, prediction, today=today))
            times.append((time.perf_counter() - start) * 1000)
        statuses[d["status"]] = statuses.get(d["status"], 0) + 1

    q = statistics.quantiles(times, n=100)
    print(f"{len(times):,} calls on {df.asin.nunique()} products, {args.offers} offers per page, "
          f"{'model' if args.model else 'cached prediction'} path")
    print(f"  median {q[49]:.2f} ms   p95 {q[94]:.2f} ms   worst {max(times):.2f} ms")
    print(f"  first call also loads the results file: +{cold_ms:.2f} ms, once per process")
    print(f"  answers: {statuses}")


if __name__ == "__main__":
    main()
