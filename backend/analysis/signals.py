"""Hand analysis results to the decision endpoint (#87).

One function, analysis_signals(), returns every analysis signal for one product page:

  price       price-drop chance and expected low, from the served model (ml.inference)
  quiet       whether a WAIT is worth the shopper's attention (R4, #81)
  best_offer  the best other offer and why the rest were skipped (T4, #86)
  version     whether reviews say this version is worse than its sibling (T2, #84)

Each signal is {"status": "ok", ...} or {"status": "missing", "why": "..."}. A missing
signal never raises; the decision just goes on without it, and the response says why.

decide() turns the signals into the decision endpoint's response format (the
Decision type in extension/src/content/decision.ts), so #92 can return it as-is.

Precomputed results (version verdicts, thresholds) come from a versioned file,
analysis/results/v<N>.json, built by scripts/build_analysis_results.py and loaded
once per process. Every answer says which version it used.

Inputs are plain data, so this module needs no database and no web framework:
offers and profile can be dicts or the #92 pydantic models (read by attribute).
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from analysis import text as copy  # mirrors website/app/preview/copy.ts

RESULTS_DIR = Path(__file__).parent / "results"
DROP_MONTHS = 4
DROP_SIZE = 0.08  # ml.features.DROP_THRESHOLD; repeated so this module never imports pandas


# ── Results file ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def load_results(version: Optional[str] = None) -> dict:
    """The newest results file, or a specific one ("v1"). Cached per process."""
    if version is None:
        files = sorted(RESULTS_DIR.glob("v*.json"), key=lambda p: int(p.stem[1:]))
        if not files:
            raise FileNotFoundError(f"no results file in {RESULTS_DIR}; run scripts/build_analysis_results.py")
        path = files[-1]
    else:
        path = RESULTS_DIR / f"{version}.json"
    results = json.loads(path.read_text())
    # index version pairs by ASIN once, so lookups are O(1) per request
    index: dict[str, list[dict]] = {}
    for p in results["version_pairs"]["pairs"]:
        index.setdefault(p["asin_a"], []).append(p)
        index.setdefault(p["asin_b"], []).append(p)
    results["_pairs_by_asin"] = index
    return results


# ── Reading inputs (dicts or #92's pydantic models) ──────────────────────────

def _get(obj: Any, *names: str, default=None):
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _day(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _total(offer) -> float:
    return round(float(_get(offer, "price")) + float(_get(offer, "shipping_cost", "shippingCost", default=0)), 2)


def _condition_ok(condition: Optional[str], wanted: str) -> bool:
    """#92's _condition_matches_profile, but reading Amazon's own wording: the page says
    "Used - Like New" / "Used - Good", which #92's version never matches."""
    c = (condition or "").strip().lower()
    c = re.sub(r"^used\s*[-–:]\s*", "", c)  # "used - very good" -> "very good"
    if wanted == "like_new":
        return c in {"new", "like new", "new - open box"}
    if wanted == "any_good":
        return any(c == v or c.startswith(v + " ") for v in ("new", "like new", "very good", "good"))
    return c == "new"  # new_only, and the default for anything unknown


# ── Signals ──────────────────────────────────────────────────────────────────

def _missing(why: str, **extra) -> dict:
    return {"status": "missing", "why": why, **extra}


def count_drops(prices: Iterable, today: date, months: int = DROP_MONTHS) -> int:
    """Price falls of 8%+ from one recorded price to the next, over the last `months`."""
    cutoff = today - timedelta(days=months * 30)
    pts = sorted(((_day(_get(p, "timestamp", "date")), float(_get(p, "price"))) for p in prices),
                 key=lambda t: t[0] or date.min)
    pts = [(d, v) for d, v in pts if d is not None and cutoff <= d <= today]
    return sum(1 for (_, a), (_, b) in zip(pts, pts[1:]) if a > 0 and b <= a * (1 - DROP_SIZE))


def price_signal(prices: list, prediction: Optional[dict], today: date,
                 predict_fn: Optional[Callable] = None) -> dict:
    """Drop chance and expected 14-day low. Uses the cached prediction when there is
    one (fast), else runs the model on the price history."""
    if not prices:
        return _missing("no price history for this product yet")
    latest = max(prices, key=lambda p: _day(_get(p, "timestamp", "date")) or date.min)
    price = float(_get(latest, "price"))
    drops = count_drops(prices, today)

    if prediction:
        rec = str(prediction.get("recommendation", "")).upper()
        p_drop = prediction.get("p_drop")
        if p_drop is None:
            # Rows from before migration 003 only store confidence, which is the
            # chance the recommendation is right: P(drop) for WAIT, 1 - P(drop) for BUY.
            conf = prediction.get("confidence_score")
            if conf is None or rec not in ("BUY", "WAIT"):
                return _missing("cached prediction has no drop chance")
            p_drop = float(conf) if rec == "WAIT" else 1.0 - float(conf)
        low = prediction.get("pred_14d")
        source = f"cached prediction ({prediction.get('model_version') or 'model version not recorded'})"
    else:
        if predict_fn is None:
            try:
                from ml.inference import predict_for_asin as predict_fn
            except Exception as exc:  # model or its libraries not installed
                return _missing(f"price model unavailable: {exc}")
        try:
            r = predict_fn(prices, today=datetime.combine(today, datetime.min.time()))
        except RuntimeError as exc:  # model not loaded, or too little history
            return _missing(str(exc))
        p_drop, low, rec = r["p_drop"], r.get("pred_14d"), r["recommendation"]
        source = "model"

    p_drop = max(0.0, min(1.0, float(p_drop)))
    low = round(min(float(low), price), 2) if low is not None else price
    return {"status": "ok", "price": round(price, 2), "drop_chance": round(p_drop, 4),
            "expected_low": low, "recommendation": rec, "drops": drops, "months": DROP_MONTHS,
            "source": source}


def quiet_signal(price: dict, profile, rules: dict) -> dict:
    """Is a WAIT worth showing? Placeholder for R4 (#81), thresholds from the results file."""
    if price["status"] != "ok":
        return _missing("needs the price signal", wait_worth_showing=False)
    if _get(profile, "patience", default="week_or_two") == "soon":
        return {"status": "ok", "wait_worth_showing": False, "why": "shopper needs it soon",
                "source": rules["source"]}
    if price["price"] < rules["min_price_for_wait"]:
        return {"status": "ok", "wait_worth_showing": False,
                "why": f"under {copy.money(rules['min_price_for_wait'])}: waiting saves too little",
                "source": rules["source"]}
    if price["drop_chance"] < rules["min_drop_chance"]:
        return {"status": "ok", "wait_worth_showing": False,
                "why": f"drop chance under {copy.pct(rules['min_drop_chance'])}", "source": rules["source"]}
    return {"status": "ok", "wait_worth_showing": True, "why": "likely drop on an item worth waiting for",
            "source": rules["source"]}


def best_offer_signal(offers: list, profile, rules: dict, min_saving: float) -> dict:
    """The cheapest other offer that fits the shopper, beats Amazon's pick by at least
    min_saving, and has the seller details the panel shows. Placeholder for T4 (#86)."""
    amazon = next((o for o in offers if _get(o, "pinned")), None)
    if amazon is None:
        return _missing("no Amazon pick (pinned offer) on the page")
    amazon_day = _day(_get(amazon, "delivery_date", "deliveryDate"))
    if amazon_day is None:
        return _missing("the page didn't show when Amazon's pick arrives")
    others = [o for o in offers if o is not amazon]
    if not others:
        return {"status": "ok", "best": None, "next_best": None, "skipped": [],
                "why": "only one offer on the page", "source": rules["source"]}
    if all(_get(o, "stars") is None for o in others):
        return _missing("the page didn't give seller ratings for the other offers")

    wanted = _get(profile, "condition", default="new_only")
    amazon_total = _total(amazon)
    eligible, slow, skipped = [], [], []
    for o in others:
        day = _day(_get(o, "delivery_date", "deliveryDate"))
        stars, years = _get(o, "stars"), _get(o, "years")
        if not _condition_ok(_get(o, "condition"), wanted):
            skipped.append({"seller": _get(o, "seller"), "reason": "condition_profile"})
        elif stars is None or years is None or stars < rules["min_stars"] or years < rules["min_years"]:
            skipped.append({"seller": _get(o, "seller"), "reason": "low_ratings"})
        elif day is None or (day - amazon_day).days > rules["max_days_later"]:
            skipped.append({"seller": _get(o, "seller"), "reason": "slow_shipping"})
            if day is not None:
                slow.append((o, day))
        else:
            eligible.append((o, day))
    for s in skipped:
        s["text"] = copy.SKIP[s["reason"]]

    def rank(item):  # cheapest first, then soonest
        return (_total(item[0]), item[1])

    cheaper = sorted([e for e in eligible if _total(e[0]) <= amazon_total - min_saving], key=rank)
    best = None
    if cheaper:
        o, day = cheaper[0]
        best = {"seller": _get(o, "seller"), "condition": _get(o, "condition") or "New",
                "price": _total(o), "arrives": day.isoformat(), "stars": float(_get(o, "stars")),
                "years": float(_get(o, "years")),
                "free_returns": bool(_get(o, "free_returns", "freeReturns", default=False)),
                "full_warranty": bool(_get(o, "full_warranty", "fullWarranty", default=False)),
                "saving": round(amazon_total - _total(o), 2)}
    later = sorted([s for s in slow if _total(s[0]) <= amazon_total - min_saving], key=rank)
    next_best = ({"price": _total(later[0][0]), "days_later": (later[0][1] - amazon_day).days}
                 if later else None)
    return {"status": "ok", "best": best, "next_best": next_best, "skipped": skipped,
            "eligible": len(eligible), "source": rules["source"]}


def version_signal(asin: str, results: dict) -> dict:
    """What reviews say about this version vs its sibling (T2, #84)."""
    pairs = results["_pairs_by_asin"].get(asin)
    if not pairs:
        return _missing("no labeled version pair for this product")
    p = pairs[0]
    this_is_a = p["asin_a"] == asin
    key = {"No real difference": "same", "Mixed": "mixed",
           "A is worse": "this_worse" if this_is_a else "other_worse",
           "B is worse": "other_worse" if this_is_a else "this_worse"}[p["verdict"]]
    return {"status": "ok", "verdict": key, "text": copy.VERSION_LINE[key],
            "other_asin": p["asin_b"] if this_is_a else p["asin_a"], "topics": p["topics"],
            "source": p["source"]}


# ── The function #92 calls ───────────────────────────────────────────────────

def analysis_signals(asin: str, offers: list, profile=None, prices: Optional[list] = None,
                     prediction: Optional[dict] = None, *, today: Optional[date] = None,
                     results: Optional[dict] = None, predict_fn: Optional[Callable] = None) -> dict:
    """Every analysis signal for one product page, each timed.

    asin        the page's ASIN
    offers      the offers read off the page; Amazon's pick has pinned=True
    profile     the shopper's profile (condition, patience); never stored
    prices      price history records ({price, timestamp}), any order
    prediction  the cached DB prediction row, if any (skips running the model)
    """
    today = today or date.today()
    profile = profile or {}
    offers = list(offers or [])
    t0 = time.perf_counter()
    timing = {}

    def timed(name, fn):
        start = time.perf_counter()
        try:
            out = fn()
        except Exception as exc:  # a broken signal must not break the endpoint
            out = _missing(f"error: {type(exc).__name__}: {exc}")
        timing[name] = round((time.perf_counter() - start) * 1000, 3)
        return out

    results = results or timed("load_results", load_results)
    price = timed("price", lambda: price_signal(prices or [], prediction, today, predict_fn))
    quiet = timed("quiet", lambda: quiet_signal(price, profile, results["quiet"]))
    best = timed("best_offer", lambda: best_offer_signal(offers, profile, results["offers"],
                                                         results["quiet"]["min_saving"]))
    version = timed("version", lambda: version_signal(asin, results))
    timing["total"] = round((time.perf_counter() - t0) * 1000, 3)
    return {"asin": asin, "results_version": results["version"], "offers_checked": len(offers),
            "amazon": _amazon_response(offers), "price": price, "quiet": quiet,
            "best_offer": best, "version": version,
            "min_price": results["quiet"]["min_price_for_wait"], "timing_ms": timing}


def _amazon_response(offers: list) -> Optional[dict]:
    a = next((o for o in offers if _get(o, "pinned")), None)
    day = _day(_get(a, "delivery_date", "deliveryDate")) if a is not None else None
    if a is None or day is None:
        return None
    return {"seller": _get(a, "seller"), "condition": _get(a, "condition") or "New",
            "price": _total(a), "arrives": day.isoformat()}


def decide(signals: dict) -> dict:
    """Combine the signals into the decision endpoint's response (decision.ts Decision).

    When signals conflict:
      1. A better offer and a likely drop: take whichever is cheaper in expectation.
         Waiting is worth p * expected_low + (1 - p) * price; the offer is certain.
      2. Otherwise a better offer wins (instant, no uncertainty).
      3. Otherwise WAIT, if the quiet rule says it's worth showing.
      4. Otherwise "nothing better". Stay hidden on cheap items unless a cheaper
         offer arrives later (next_best), since there's nothing worth saying.
    """
    price, quiet, offer = signals["price"], signals["quiet"], signals["best_offer"]
    best = offer.get("best") if offer["status"] == "ok" else None
    can_wait = quiet["status"] == "ok" and quiet["wait_worth_showing"]
    base = {"offers_checked": signals["offers_checked"], "amazon": signals["amazon"], "best": None,
            "wait": None, "next_best": None}

    if best and can_wait:
        p = price["drop_chance"]
        if p * price["expected_low"] + (1 - p) * price["price"] < best["price"]:
            best = None  # waiting beats the offer

    if best and signals["amazon"]:
        return {**base, "show": True, "status": "better_offer",
                "best": {k: best[k] for k in ("seller", "condition", "price", "arrives", "stars",
                                              "years", "free_returns", "full_warranty")},
                "reason": copy.headline_better_offer(best["saving"], best["condition"])}
    if can_wait:
        return {**base, "show": True, "status": "wait",
                "reason": copy.headline_wait(price["drop_chance"]),
                "wait": {k: price[k] for k in ("price", "expected_low", "drop_chance", "drops", "months")}}

    next_best = offer.get("next_best") if offer["status"] == "ok" else None
    cheap = price["status"] == "ok" and price["price"] < signals["min_price"]
    return {**base, "show": bool(next_best) or not cheap, "status": "nothing_better",
            "reason": copy.headline_buy_here(), "next_best": next_best}
