"""Track record for the extension dashboard (#93).

Serves the files `scripts/track_record.py replay` writes (#85). No database: the
numbers come from track_record_out/summary.json and resolved.csv. Rerun the script
to refresh them.
"""

import csv
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["track record"])

TRACK_RECORD_DIR = Path(os.getenv("TRACK_RECORD_DIR", Path(__file__).resolve().parents[1] / "track_record_out"))
TOP_N = 5


def _call(row: dict, amount: float) -> dict:
    return {
        "asin": row["asin"],
        "predicted_on": row["predicted_on"],
        "said": row["recommendation"],
        "price": float(row["price_at_prediction"]),
        "low": float(row["actual_min_price"]),
        "end": float(row["actual_end_price"]),
        "amount": round(amount, 2),
    }


def _wins_and_misses(path: Path) -> tuple[list[dict], list[dict]]:
    wins = []
    misses = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "resolved":
                continue
            price = float(row["price_at_prediction"])
            low = float(row["actual_min_price"])
            end = float(row["actual_end_price"])
            right = row["was_correct"] == "True"
            if right and row["recommendation"] == "WAIT":
                wins.append(_call(row, price - low))  # saved by waiting
            elif not right and row["recommendation"] == "WAIT":
                misses.append(_call(row, max(end - price, 0)))  # paid more by waiting
            elif not right:
                misses.append(_call(row, price - low))  # said buy, then it dropped
    return _top(wins), _top(misses)


def _top(calls: list[dict]) -> list[dict]:
    """Biggest first, one row per product (the same call often repeats week to week)."""
    calls.sort(key=lambda c: c["amount"], reverse=True)
    top = []
    seen = set()
    for c in calls:
        if c["asin"] in seen:
            continue
        seen.add(c["asin"])
        top.append(c)
        if len(top) == TOP_N:
            break
    return top


@router.get("/track-record")
def get_track_record() -> dict:
    summary_path = TRACK_RECORD_DIR / "summary.json"
    resolved_path = TRACK_RECORD_DIR / "resolved.csv"
    if not summary_path.exists() or not resolved_path.exists():
        raise HTTPException(status_code=404, detail="no_track_record")

    s = json.loads(summary_path.read_text())
    wins, misses = _wins_and_misses(resolved_path)
    return {
        "products": s["products"],
        "first_day": s["first_day"],
        "last_day": s["last_day"],
        "calls": s["resolved"],
        "hit_rate": s["hit_rate %"] / 100,
        "base_rate": s["base_rate %"] / 100,  # how often prices dropped anyway
        "wait_calls": s["wait_calls"],
        "wait_right": s["wait_right %"] / 100,
        "buy_right": s["buy_right %"] / 100,
        "saved": s["$ saved by right WAITs"],
        "lost_waiting": s["$ lost by wrong WAITs"],
        "missed_drops": s["$ missed by wrong BUYs"],
        "wins": wins,
        "misses": misses,
    }