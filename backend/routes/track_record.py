"""Track record for the extension dashboard (#93).

Serves the files `scripts/track_record.py replay` writes (#85). No database: the
numbers come from track_record_out/summary.json and resolved.csv. Rerun the script
to refresh them. A fresh clone has no track_record_out/, so it falls back to the
replay committed in analysis/results/track_record/.

Replay (/track-record/replay) picks a few of those calls by a fixed rule and, for one
call, returns the real prices around it from studies/Prices.csv, the file the replay
ran on, with their real dates (no shift to today).
"""

import csv
import json
import os
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["track record"])

TRACK_RECORD_DIR = Path(os.getenv("TRACK_RECORD_DIR", Path(__file__).resolve().parents[1] / "track_record_out"))
COMMITTED_DIR = Path(__file__).resolve().parents[1] / "analysis" / "results" / "track_record"
TOP_N = 5


def _call(row: dict, amount: float) -> dict:
    from jobs import csv_history
    return {
        "asin": row["asin"],
        "brand": csv_history._brands().get(row["asin"]),
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


def _source_dir() -> Path | None:
    for d in (TRACK_RECORD_DIR, COMMITTED_DIR):
        if (d / "summary.json").exists() and (d / "resolved.csv").exists():
            return d
    return None


@router.get("/track-record")
def get_track_record() -> dict:
    d = _source_dir()
    if d is None:
        raise HTTPException(status_code=404, detail="no_track_record")
    summary_path = d / "summary.json"
    resolved_path = d / "resolved.csv"

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

# ── Replay a past call ───────────────────────────────────────────────────────

REPLAY_RULE = ("The 3 biggest savings from right Wait calls, the biggest overpay from a wrong Wait, "
               "and the 2 biggest price rises after a right Buy. One call per product.")
REPLAY_PICKS = (("wait_right", 3), ("wait_wrong", 1), ("buy_right", 2))
DAYS_BEFORE = 60


def _kind(row: dict) -> str:
    said = "wait" if row["recommendation"] == "WAIT" else "buy"
    return f"{said}_{'right' if row['was_correct'] == 'True' else 'wrong'}"


def _replay_call(row: dict) -> dict:
    from jobs import csv_history
    price = float(row["price_at_prediction"])
    low = float(row["actual_min_price"])
    end = float(row["actual_end_price"])
    kind = _kind(row)
    # what the call was worth: saved by waiting, paid extra by waiting, missed, or avoided by buying
    amount = {"wait_right": price - low, "wait_wrong": max(end - price, 0),
              "buy_wrong": price - low, "buy_right": max(end - price, 0)}[kind]
    return {
        "asin": row["asin"],
        "brand": csv_history._brands().get(row["asin"]),
        "predicted_on": row["predicted_on"],
        "said": row["recommendation"],
        "kind": kind,
        "right": row["was_correct"] == "True",
        "price": price,
        "drop_chance": round(float(row["p_drop"]), 4),
        "low": low,
        "end": end,
        "amount": round(amount, 2),
    }


def _resolved_rows(d: Path) -> list[dict]:
    with open(d / "resolved.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "resolved"]


def replay_calls(rows: list[dict]) -> list[dict]:
    """REPLAY_RULE, applied to the resolved calls."""
    calls = [_replay_call(r) for r in rows]
    picked = []
    seen = set()
    for kind, n in REPLAY_PICKS:
        group = sorted((c for c in calls if c["kind"] == kind), key=lambda c: c["amount"], reverse=True)
        for c in group:
            if len([p for p in picked if p["kind"] == kind]) == n:
                break
            if c["asin"] not in seen and c["amount"] > 0:
                seen.add(c["asin"])
                picked.append(c)
    return picked


def _replay_prices():
    from jobs import csv_history
    return csv_history.STUDIES / "Prices.csv"


def _expected_low(hist, day, row: dict):
    """The served model's expected 14-day low that day, if the loaded model is the one
    that made the call (its p_drop comes out the same). Otherwise None."""
    try:
        from ml import inference
        from ml.track_record import _records
        r = inference.predict_for_asin(_records(hist), today=day)
    except Exception:
        return None
    if abs(float(r["p_drop"]) - float(row["p_drop"])) > 1e-6:
        return None
    return round(min(float(r["pred_14d"]), float(row["price_at_prediction"])), 2)


@router.get("/track-record/replay")
def get_replay_list() -> dict:
    d = _source_dir()
    if d is None:
        raise HTTPException(status_code=404, detail="no_track_record")
    return {"rule": REPLAY_RULE, "calls": replay_calls(_resolved_rows(d))}


@router.get("/track-record/replay/{asin}/{day}")
def get_replay(asin: str, day: str) -> dict:
    """One past call: what the panel would have said that day, the real daily prices
    from 60 days before to 14 days after, and what happened."""
    import pandas as pd
    from analysis.signals import DROP_MONTHS, count_drops
    from jobs import csv_history
    from ml.features import daily_panel

    d = _source_dir()
    if d is None:
        raise HTTPException(status_code=404, detail="no_track_record")
    row = next((r for r in _resolved_rows(d) if r["asin"] == asin and r["predicted_on"] == day), None)
    g = csv_history.rows(asin, _replay_prices())
    if row is None or g is None:
        raise HTTPException(status_code=404, detail="no_such_call")

    call = _replay_call(row)
    horizon = int(row["horizon_days"])
    on = pd.Timestamp(day)
    hist = g[g.date <= on]
    daily = daily_panel(g[["asin", "date", "price"]], until=on + pd.Timedelta(days=horizon))
    daily = daily[(daily.date >= on - pd.Timedelta(days=DAYS_BEFORE)) & (daily.date <= on + pd.Timedelta(days=horizon))]
    after = daily[daily.date > on]
    low_on = after[after.price <= call["low"] + 0.005].date.min() if call["low"] < call["price"] else None

    return {
        **call,
        "horizon_days": horizon,
        "expected_low": _expected_low(hist, on, row),
        "drops": count_drops([{"price": float(p), "timestamp": t} for t, p in zip(hist.date, hist.price)],
                             on.date()),
        "months": DROP_MONTHS,
        "low_on": low_on.date().isoformat() if low_on is not None and not pd.isna(low_on) else None,
        "end_on": (date.fromisoformat(day) + timedelta(days=horizon)).isoformat(),
        "prices": [[t.date().isoformat(), round(float(p), 2)] for t, p in zip(daily.date, daily.price)],
        "model_version": row["model_version"],
    }
