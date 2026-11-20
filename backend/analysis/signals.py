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
RECENT_DAYS = 30  # stored history older than this can't say what the price did "in the last 4 months"
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
            "vol30": vol30(prices, today), "source": source}


def vol30(prices: list, today: date) -> Optional[float]:
    """The model's own 30-day volatility feature (ml.features) as of today, or None.
    R4's wait economics differ by volatility band, so the quiet rule needs it."""
    try:
        import pandas as pd
        from ml.features import add_features, daily_panel
        df = pd.DataFrame({"asin": "X", "price": [float(_get(p, "price")) for p in prices],
                           "date": [pd.Timestamp(_day(_get(p, "timestamp", "date"))) for p in prices]})
        v = add_features(daily_panel(df, until=pd.Timestamp(today))).vol30.iloc[-1]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


BANDS = ("calm", "medium", "twitchy")


def volatility_band(v: Optional[float], cuts: list) -> str:
    """R4's bands: vol30 tercile cut-offs from train. Unknown volatility counts as medium."""
    if v is None:
        return "medium"
    return BANDS[0] if v <= cuts[0] else BANDS[1] if v <= cuts[1] else BANDS[2]


def quiet_signal(price: dict, profile, rules: dict) -> dict:
    """Is a WAIT worth showing? R4 (#81, Study 024): only when BOTH

        calibrated drop chance  >= min_drop_chance
        expected net saving     >= min_saving, where
        net = price * (p * G_right[band] + (1 - p) * G_wrong[band])

    G_wrong is negative: a wrong WAIT often means paying more on day 14, so the rule
    stays quiet on cheap, calm items by itself, without a hard price floor."""
    if price["status"] != "ok":
        return _missing("needs the price signal", wait_worth_showing=False)
    p, amount = price["drop_chance"], price["price"]
    band = volatility_band(price.get("vol30"), rules["vol30_cuts"]) if "bands" in rules else None
    if band:
        g = rules["bands"][band]
        net = round(amount * (p * g["G_right"] + (1 - p) * g["G_wrong"]), 2)
    else:  # v1 placeholder rules: no economics, the drop chance is the whole bar
        net = round(amount - price["expected_low"], 2)
    out = {"status": "ok", "band": band, "net_saving": net, "source": rules["source"]}
    if _get(profile, "patience", default="week_or_two") == "soon":
        return {**out, "wait_worth_showing": False, "why": "shopper needs it soon"}
    if amount < rules["min_price_for_wait"]:
        return {**out, "wait_worth_showing": False,
                "why": f"under {copy.money(rules['min_price_for_wait'])}: waiting saves too little"}
    if p < rules["min_drop_chance"]:
        return {**out, "wait_worth_showing": False, "why": f"drop chance under {copy.pct(rules['min_drop_chance'])}"}
    if band and net < rules["min_saving"]:
        return {**out, "wait_worth_showing": False,
                "why": f"expected saving {copy.money(net)} is under {copy.money(rules['min_saving'])}"}
    return {**out, "wait_worth_showing": True,
            "why": f"{copy.pct(p)} drop chance, worth {copy.money(net)} on average"}


PATIENCE_DAYS = {"soon": 3, "week_or_two": 14, "deal": None}  # how soon it must arrive


def _amazon_run(o) -> bool:
    from ml.offer_ranking import is_amazon_run
    return is_amazon_run(_get(o, "seller"))


def _trusted(o, rules: dict) -> bool:
    """The panel names a seller only with all three numbers on the page, all high enough.
    Amazon-run sellers (Amazon.com, Amazon Resale, ...) show no ratings and pass as is."""
    if _amazon_run(o):
        return True
    stars = _get(o, "stars")
    count = _get(o, "rating_count", "ratingCount")
    positive = _get(o, "positive_percent", "positivePercent")
    return (stars is not None and count is not None and positive is not None
            and stars >= rules["min_stars"] and count >= rules["min_ratings"]
            and positive >= rules["min_positive"])


def best_offer_signal(offers: list, profile, rules: dict, today: date) -> dict:
    """The best other offer for this shopper, ranked by T4 (#86, ml.offer_ranking).

    An offer counts only if its condition fits the profile, it arrives within the
    shopper's patience, and its seller clears the trust bar. T4 then ranks what's
    left against Amazon's pick on the condition-adjusted price (a "Used - Very Good"
    copy must be 30% cheaper to be worth as much as new) and keeps Amazon's pick
    unless the winner saves at least max(min_saving_usd, min_saving_pct of its price)."""
    from ml.offer_ranking import rank_offers

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
    if all(_get(o, "stars") is None and not _amazon_run(o) for o in others):
        return _missing("the page didn't give seller ratings for the other offers")

    wanted = _get(profile, "condition", default="new_only")
    max_days = PATIENCE_DAYS.get(_get(profile, "patience", default="week_or_two"), 14)
    amazon_total = _total(amazon)
    eligible, slow, skipped = [], [], []
    for o in others:
        day = _day(_get(o, "delivery_date", "deliveryDate"))
        if not _condition_ok(_get(o, "condition"), wanted):
            skipped.append({"seller": _get(o, "seller"), "reason": "condition_profile"})
        elif not _trusted(o, rules):
            skipped.append({"seller": _get(o, "seller"), "reason": "low_ratings"})
        elif day is None or day < today or (max_days is not None and (day - today).days > max_days):
            skipped.append({"seller": _get(o, "seller"), "reason": "slow_shipping"})
            if day is not None and day >= today:
                slow.append((o, day))
        else:
            eligible.append((o, day))
    for s in skipped:
        s["text"] = copy.SKIP[s["reason"]]

    def needed(price: float) -> float:
        return max(rules["min_saving_usd"], rules["min_saving_pct"] * price)

    def adjusted(o) -> float:
        c = str(_get(o, "condition") or "new").strip().lower()
        d = rules["condition_discount"]
        rung = d["new"] if c == "new" else next(
            (d[k] for k in ("like new", "very good", "good", "acceptable", "refurbished") if k in c),
            rules["unknown_condition_discount"])
        return _total(o) / (1 - rung)

    # T4 ranks every eligible offer against Amazon's pick. Cheapest, then soonest, goes in
    # first so ties in T4's score break the same way.
    eligible.sort(key=lambda e: (_total(e[0]), e[1]))
    # Built the way T4's hand check built them (studies/analysis_scripts/s020): each offer
    # with its own seller signals; T4 fills them in for Amazon-run sellers.
    candidates = [{"source": _get(amazon, "seller"), "price": amazon_total,
                   "condition": _get(amazon, "condition") or "New",
                   "feedback_score": _get(amazon, "positive_percent", "positivePercent"),
                   "rating_count": _get(amazon, "rating_count", "ratingCount"),
                   "is_fba": bool(_get(amazon, "ships_from_amazon", "shipsFromAmazon", default=False)),
                   "is_atc": True, "_offer": amazon}]
    def num(v, cast):
        return None if v is None else cast(v)

    for o, day in eligible:
        candidates.append({"source": _get(o, "seller"), "price": _total(o),
                           "condition": _get(o, "condition") or "New",
                           "feedback_score": num(_get(o, "positive_percent", "positivePercent"), float),
                           "rating_count": num(_get(o, "rating_count", "ratingCount"), int),
                           "is_fba": bool(_get(o, "ships_from_amazon", "shipsFromAmazon", default=False)),
                           "_offer": o, "_day": day})
    ranked = rank_offers(candidates, typical_new=None, typical_used=None,
                         discounts=rules["condition_discount"])
    winner = ranked[0] if ranked else None

    best = None
    if winner is not None and not winner.get("is_atc") \
            and amazon_total - winner["effective_price"] >= needed(amazon_total):
        o, day = winner["_offer"], winner["_day"]
        best = {"seller": _get(o, "seller"), "condition": _get(o, "condition") or "New",
                "price": _total(o), "arrives": day.isoformat(), "stars": num(_get(o, "stars"), float),
                "rating_count": num(_get(o, "rating_count", "ratingCount"), int),
                "positive_percent": num(_get(o, "positive_percent", "positivePercent"), float),
                "free_returns": _get(o, "free_returns", "freeReturns"),
                "full_warranty": _get(o, "full_warranty", "fullWarranty"),
                "saving": round(amazon_total - _total(o), 2),
                "t4_score": winner["score"]}
    later = sorted([s for s in slow if amazon_total - adjusted(s[0]) >= needed(amazon_total)],
                   key=lambda s: (_total(s[0]), s[1]))
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


STALE_GAP = 0.10  # page price this far from the datasets' last price: history too old to judge


def history_out_of_date(page_price: Optional[float], dataset_price: Optional[float]) -> Optional[str]:
    """Our price history ends in April 2026. When today's page price has moved more than
    10% since then, the model would read the gap as a one-day spike or crash and call it a
    drop or a bargain. So no WAIT on those pages; the offers are still compared."""
    if not page_price or not dataset_price:
        return None
    gap = page_price / dataset_price - 1
    if abs(gap) <= STALE_GAP:
        return None
    return (f"price history is out of date: {copy.money(dataset_price)} in our data, "
            f"{copy.money(page_price)} on the page ({gap:+.0%})")


PAGE_PRICE_TOLERANCE = 0.01  # under 1% apart, the cached prediction still holds


def with_page_price(prices: list, prediction: Optional[dict], page_price: Optional[float],
                    today: date) -> tuple[list, Optional[dict]]:
    """The page price is today's price (#104). Stored prices can be days or months old,
    so unless the cached prediction was scored at this price (within 1%), today's price
    joins the history and the model scores it fresh."""
    if not page_price or page_price <= 0:
        return prices, prediction
    stamp = datetime.combine(today, datetime.min.time())
    scored_at = (prediction or {}).get("price_at_prediction")
    if scored_at is not None and abs(page_price - float(scored_at)) <= PAGE_PRICE_TOLERANCE * float(scored_at):
        return prices, prediction
    return [*prices, {"price": float(page_price), "timestamp": stamp}], None


# ── The function #92 calls ───────────────────────────────────────────────────

def analysis_signals(asin: str, offers: list, profile=None, prices: Optional[list] = None,
                     prediction: Optional[dict] = None, *, page_price: Optional[float] = None,
                     dataset_price: Optional[float] = None,
                     today: Optional[date] = None,
                     results: Optional[dict] = None, predict_fn: Optional[Callable] = None) -> dict:
    """Every analysis signal for one product page, each timed.

    asin        the page's ASIN
    offers      the offers read off the page; Amazon's pick has pinned=True
    profile     the shopper's profile (condition, patience); never stored
    prices      price history records ({price, timestamp}), any order
    prediction  the cached DB prediction row, if any (skips running the model)
    page_price  the one-time price on the page the shopper is looking at (#104)
    dataset_price  the last price in the price datasets, which end in April 2026
    """
    today = today or date.today()
    profile = profile or {}
    offers = list(offers or [])
    stored_end = max((d for d in (_day(_get(p, "timestamp", "date")) for p in prices or []) if d), default=None)
    prices, prediction = with_page_price(prices or [], prediction, page_price, today)
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
    stale = history_out_of_date(page_price, dataset_price)
    price = (_missing(stale) if stale else
             timed("price", lambda: price_signal(prices or [], prediction, today, predict_fn)))
    if price["status"] == "ok" and (stored_end is None or (today - stored_end).days > RECENT_DAYS):
        # The datasets end in April 2026: a count of "drops in the last 4 months" would
        # describe months that aren't recent. The panel leaves the line out.
        price = {**price, "drops": None}
    quiet = timed("quiet", lambda: quiet_signal(price, profile, results["quiet"]))
    best = timed("best_offer", lambda: best_offer_signal(offers, profile, results["offers"], today))
    version = timed("version", lambda: version_signal(asin, results))
    timing["total"] = round((time.perf_counter() - t0) * 1000, 3)
    return {"asin": asin, "results_version": results["version"], "offers_checked": len(offers),
            "amazon": _amazon_response(offers), "price": price, "quiet": quiet,
            "best_offer": best, "version": version, "page_price": page_price,
            "min_price": results["offers"].get("hide_nothing_better_below",
                                               results["quiet"]["min_price_for_wait"]),
            "timing_ms": timing}


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
      3. Otherwise WAIT, if R4's quiet rule says it's worth showing.
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
        best_response = {
            k: best[k]
            for k in ("seller", "condition", "price", "arrives", "stars")
        }

        for key in (
            "rating_count",
            "positive_percent",
            "years",
            "free_returns",
            "full_warranty",
        ):
            if key in best:
                best_response[key] = best[key]

        return {
            **base,
            "show": True,
            "status": "better_offer",
            "best": best_response,
            "reason": (
                copy.headline_better_offer(best["saving"], best["condition"]) + " "
                + copy.trust_line(best["stars"], best["rating_count"], best["positive_percent"])
                if all(best.get(k) is not None for k in ("stars", "rating_count", "positive_percent"))
                else copy.headline_better_offer(best["saving"], best["condition"])
            ),
        }

    if can_wait:
        return {**base, "show": True, "status": "wait",
                "reason": copy.headline_wait(price["drop_chance"]),
                "wait": {k: price[k] for k in ("price", "expected_low", "drop_chance", "drops", "months")}}

    next_best = offer.get("next_best") if offer["status"] == "ok" else None
    # Without a price signal (e.g. stale history), the page's own price still says whether it's cheap.
    item_price = (price["price"] if price["status"] == "ok"
                  else (signals["amazon"] or {}).get("price") or signals.get("page_price"))
    cheap = item_price is not None and item_price < signals["min_price"]
    return {**base, "show": bool(next_best) or not cheap, "status": "nothing_better",
            "reason": copy.headline_buy_here(), "next_best": next_best}
