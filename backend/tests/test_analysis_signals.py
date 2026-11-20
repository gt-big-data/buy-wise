"""Analysis signals for the decision endpoint (#87), on fake pages with hand-checked answers.

The promises:
  - each signal works alone, and any of them can be missing without breaking the rest
  - conflicts resolve the documented way (decide's docstring)
  - the output is exactly the panel's Decision format (extension/src/content/decision.ts)
  - shopper-facing text matches website/app/preview/copy.ts word for word
  - results come from the versioned file, and every answer names the version

Run: pytest tests/test_analysis_signals.py -q
"""
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from analysis import text
from analysis.signals import (analysis_signals, best_offer_signal, count_drops, decide,
                              load_results)

T = date(2026, 11, 13)
D = lambda n: (T + timedelta(days=n)).isoformat()  # noqa: E731
ROOT = Path(__file__).resolve().parents[2]


def amazon(price=449.99, arrives=2):
    return {"seller": "Amazon.com", "price": price, "condition": "New", "pinned": True, "deliveryDate": D(arrives)}


def seller(name, price, condition="New", arrives=4, stars=4.8, ratings=390, positive=94, **kw):
    return {"seller": name, "price": price, "condition": condition, "deliveryDate": D(arrives),
            "stars": stars, "ratingCount": ratings, "positivePercent": positive, "freeReturns": True, **kw}


def history(*prices, start=100):
    """Prices recorded every 10 days, ending yesterday."""
    n = len(prices)
    return [{"price": p, "timestamp": T - timedelta(days=start - 10 * i if i < n - 1 else 1)}
            for i, p in enumerate(prices)]


BUY = {"recommendation": "BUY", "confidence_score": 0.9, "pred_14d": 440.0}  # 10% drop chance
WAIT = {"recommendation": "WAIT", "confidence_score": 0.74, "pred_14d": 285.0}


def run(offers, prediction=BUY, prices=None, profile=None, **kw):
    s = analysis_signals("B000TEST01", offers, profile or {"condition": "new_only"},
                         prices if prices is not None else history(449.99, 449.99), prediction, today=T, **kw)
    return s, decide(s)


# ── panel format ─────────────────────────────────────────────────────────────

def _decision_ts_fields():
    """Field names of the Decision type the panel reads."""
    ts = (ROOT / "extension/src/content/decision.ts").read_text()
    body = re.search(r"export type Decision = \{(.*?)\n\};", ts, re.S).group(1)
    return set(re.findall(r"^\s+(\w+)\??:", body, re.M))


def test_output_is_exactly_the_panel_format():
    _, d = run([amazon(), seller("TrailTech Outfitters", 369.0)])
    assert set(d) == _decision_ts_fields()
    assert d["status"] in ("better_offer", "wait", "nothing_better")


# ── better offer (T4, #86) ───────────────────────────────────────────────────

def test_better_offer_matches_the_panels_own_fake_example():
    s, d = run([amazon(), seller("TrailTech Outfitters", 369.0)])
    assert d["status"] == "better_offer" and d["show"]
    assert d["best"]["seller"] == "TrailTech Outfitters" and d["best"]["price"] == 369.0
    assert d["reason"] == "Same item, new, $80.99 less. 4.8 stars from 390 ratings, 94% positive."
    assert d["amazon"] == {"seller": "Amazon.com", "condition": "New", "price": 449.99, "arrives": D(2)}


def test_skipped_offers_say_why():
    s, _ = run([amazon(),
                seller("Used", 300.0, condition="Used - Good"),
                seller("Unrated", 310.0, stars=3.2),
                seller("Slow", 320.0, arrives=20),
                seller("Few ratings", 330.0, ratings=28),
                seller("Good", 400.0)])
    reasons = {x["seller"]: x["reason"] for x in s["best_offer"]["skipped"]}
    assert reasons == {"Used": "condition_profile", "Unrated": "low_ratings", "Slow": "slow_shipping",
                       "Few ratings": "low_ratings"}
    assert s["best_offer"]["best"]["seller"] == "Good"
    assert s["best_offer"]["next_best"] == {"price": 320.0, "days_later": 18}  # cheaper, but 18 days later
    assert all(x["text"] == text.SKIP[x["reason"]] for x in s["best_offer"]["skipped"])


def test_profile_changes_which_offers_count():
    offers = [amazon(), seller("Used", 250.0, condition="Used - Good"), seller("New", 400.0)]
    assert run(offers, profile={"condition": "new_only"})[1]["best"]["seller"] == "New"
    assert run(offers, profile={"condition": "any_good"})[1]["best"]["seller"] == "Used"


@pytest.mark.parametrize("condition, new_only, like_new, any_good", [
    ("New", True, True, True),
    ("Used - Like New", False, True, True),
    ("Used - Very Good", False, False, True),
    ("Used - Good", False, False, True),
    ("Used - Acceptable", False, False, False),
    ("Renewed", False, False, False),
])
def test_amazon_condition_wording(condition, new_only, like_new, any_good):
    from analysis.signals import _condition_ok
    assert [_condition_ok(condition, p) for p in ("new_only", "like_new", "any_good")] == [new_only, like_new, any_good]


def test_shipping_counts_and_ties_go_to_sooner_delivery():
    s, _ = run([amazon(), seller("Far", 400.0, arrives=7), seller("Near", 400.0, arrives=3),
                seller("Shipping", 395.0, shippingCost=10.0)])
    assert s["best_offer"]["best"]["seller"] == "Near"  # Shipping costs $405 all in


def test_tiny_savings_are_not_a_better_offer():
    _, d = run([amazon(price=100.0), seller("Penny", 99.50)])
    assert d["status"] == "nothing_better"
    # T4: the saving must be at least max($2, 3% of Amazon's price), $3 here
    assert run([amazon(price=100.0), seller("Close", 97.50)])[1]["status"] == "nothing_better"
    assert run([amazon(price=100.0), seller("Enough", 96.50)])[1]["status"] == "better_offer"


def test_used_copies_must_be_cheaper_by_their_condition_discount():
    """T4's ladder: "Very Good" is worth 30% less than new, so 18% off isn't a deal."""
    profile = {"condition": "any_good"}
    meh = seller("Meh", 369.0, condition="Used - Very Good")
    deal = seller("Deal", 250.0, condition="Used - Very Good")  # $357 as-new, $93 under
    assert run([amazon(), meh], profile=profile)[1]["status"] == "nothing_better"
    _, d = run([amazon(), meh, deal], profile=profile)
    assert d["status"] == "better_offer" and d["best"]["seller"] == "Deal"
    assert d["reason"].startswith("Same item, used - very good, $199.99 less.")


def test_amazon_shipping_its_own_pick_needs_a_bigger_saving():
    """T4 scores trust and fulfilment too: a 94%-positive seller $3.50 under an
    Amazon-fulfilled $100 pick isn't worth switching for; $6 under is."""
    fba = amazon(price=100.0) | {"shipsFromAmazon": True}
    assert run([fba, seller("Small", 96.50)])[1]["status"] == "nothing_better"
    assert run([fba, seller("Real", 94.00)])[1]["status"] == "better_offer"


def test_patience_sets_how_late_an_offer_may_arrive():
    offers = [amazon(), seller("Five days", 369.0, arrives=5)]
    assert run(offers, profile={"patience": "week_or_two"})[1]["status"] == "better_offer"
    s, d = run(offers, profile={"patience": "soon"})
    assert d["status"] == "nothing_better"
    assert s["best_offer"]["skipped"][0]["reason"] == "slow_shipping"


def test_seller_trust_bar():
    for bad in (dict(stars=3.9), dict(ratings=99), dict(positive=89)):
        s, d = run([amazon(), seller("X", 300.0, **bad)])
        assert d["status"] == "nothing_better", bad


def test_amazon_run_sellers_need_no_ratings():
    """Amazon Resale and Amazon.com show no seller ratings on the page; T4 trusts them anyway."""
    resale = {"seller": "Amazon Resale", "price": 360.0, "condition": "Used - Like New", "deliveryDate": D(3)}
    _, d = run([amazon(), resale], profile={"condition": "like_new"})
    assert d["status"] == "better_offer" and d["best"]["seller"] == "Amazon Resale"
    assert d["best"]["stars"] is None and d["best"]["rating_count"] is None
    assert d["reason"] == "Same item, used - like new, $89.99 less."
    shop = resale | {"seller": "Some Shop"}
    s, d = run([amazon(), shop, seller("Rated", 445.0)], profile={"condition": "like_new"})
    assert d["status"] == "nothing_better"
    assert {x["seller"]: x["reason"] for x in s["best_offer"]["skipped"]} == {"Some Shop": "low_ratings"}


def test_pydantic_style_offer_objects_work():
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    offers = [Obj(seller="Amazon.com", price=449.99, condition="New", pinned=True, delivery_date=D(2)),
              Obj(seller="X", price=369.0, condition="New", delivery_date=D(4), stars=4.8, rating_count=390,
                  positive_percent=94.0, free_returns=True, full_warranty=True, shipping_cost=None)]
    _, d = run(offers)
    assert d["status"] == "better_offer" and d["best"]["full_warranty"] is True


# ── wait and staying quiet (R4, #81) ─────────────────────────────────────────

def test_wait_when_a_drop_is_likely():
    prices = history(349.0, 300.0, 349.0, 310.0, 349.0)  # dropped twice in the window
    _, d = run([amazon(price=349.0)], prediction=WAIT, prices=prices)
    assert d["status"] == "wait"
    assert d["wait"] == {"price": 349.0, "expected_low": 285.0, "drop_chance": 0.74, "drops": 2, "months": 4}
    assert d["reason"] == "Worth waiting if you can. It could drop to $321.08 or less in the next 14 days."


def test_no_drop_count_when_the_stored_history_is_months_old():
    """History ending in April can't say what the price did in the last 4 months."""
    prices = [{**p, "timestamp": p["timestamp"] - timedelta(days=200)} for p in history(349.0, 300.0, 349.0)]
    _, d = run([amazon(price=349.0)], prediction=WAIT, prices=prices)
    assert d["status"] == "wait" and d["wait"]["drops"] is None


@pytest.mark.parametrize("why, kw", [
    ("cheap item: 74% of a $9.99 drop is worth under $2", dict(prices=history(9.99, 9.99), offers=[amazon(price=9.99)])),
    ("needs it soon", dict(profile={"patience": "soon"})),
    ("drop unlikely: under R4's 15%", dict(prediction={**WAIT, "p_drop": 0.10})),
])
def test_no_wait_when_not_worth_it(why, kw):
    kw.setdefault("offers", [amazon(price=349.0)])
    kw.setdefault("prices", history(349.0, 349.0))
    s, d = run(prediction=kw.pop("prediction", WAIT), **kw)
    assert d["status"] != "wait", why
    assert s["quiet"]["wait_worth_showing"] is False


def test_r4_net_saving_by_volatility_band():
    """net = price * (p * G_right + (1 - p) * G_wrong), with the band's gains from Study 024."""
    from analysis.signals import quiet_signal, volatility_band
    rules = load_results()["quiet"]
    assert volatility_band(None, rules["vol30_cuts"]) == "medium"
    assert volatility_band(0.0, rules["vol30_cuts"]) == "calm"
    assert volatility_band(0.2, rules["vol30_cuts"]) == "twitchy"
    price = {"status": "ok", "price": 100.0, "drop_chance": 0.3, "expected_low": 90.0, "vol30": 0.2}
    q = quiet_signal(price, {}, rules)
    g = rules["bands"]["twitchy"]
    assert q["band"] == "twitchy" and q["net_saving"] == round(100 * (0.3 * g["G_right"] + 0.7 * g["G_wrong"]), 2)
    # a wrong WAIT costs more on twitchy items, so the same chance is worth less there
    calm = quiet_signal({**price, "vol30": 0.0}, {}, rules)
    assert calm["net_saving"] > q["net_saving"]


def test_r4_dollar_test_silences_small_expected_savings():
    from analysis.signals import quiet_signal
    rules = load_results()["quiet"]
    base = {"status": "ok", "drop_chance": 0.5, "expected_low": 0.0, "vol30": 0.0}
    assert not quiet_signal({**base, "price": 20.0}, {}, rules)["wait_worth_showing"]  # ~$1.67
    assert quiet_signal({**base, "price": 40.0}, {}, rules)["wait_worth_showing"]      # ~$3.33


def test_stays_hidden_on_cheap_items_with_nothing_to_say():
    _, d = run([amazon(price=12.0)], prices=history(12.0, 12.0))
    assert d == {**d, "show": False, "status": "nothing_better"}
    _, d = run([amazon(price=349.0)], prices=history(349.0, 349.0))
    assert d["show"] is True and d["reason"] == "This is the best offer for you."


def test_cheap_items_stay_hidden_when_the_price_signal_is_missing():
    """Stale history drops the price signal; the page price still marks a $9.99 item cheap."""
    s, d = run([amazon(price=9.99)], prediction=None, prices=history(29.0, 29.0), page_price=9.99,
               dataset_price=29.0)
    assert s["price"]["status"] == "missing" and d["status"] == "nothing_better" and d["show"] is False
    _, d = run([], prediction=None, prices=[], page_price=9.99, dataset_price=29.0)
    assert d["show"] is False  # no Amazon offer either: falls back to the page price
    s, d = run([amazon(price=349.0)], prediction=None, prices=history(29.0, 29.0), page_price=349.0,
               dataset_price=29.0)
    assert s["price"]["status"] == "missing" and d["show"] is True


def test_buy_confidence_is_turned_into_drop_chance():
    """Rows from before migration 003 store confidence = 1 - P(drop) for BUY."""
    s, _ = run([amazon()], prediction={"recommendation": "BUY", "confidence_score": 0.9, "pred_14d": 440.0})
    assert s["price"]["drop_chance"] == pytest.approx(0.1)
    s, _ = run([amazon()], prediction={**BUY, "p_drop": 0.33})
    assert s["price"]["drop_chance"] == 0.33  # migration-003 rows store it directly


def test_count_drops_only_counts_8_percent_falls_in_the_window():
    prices = [{"price": 100, "timestamp": T - timedelta(days=200)},  # outside 4 months
              {"price": 80, "timestamp": T - timedelta(days=100)},   # 20% drop, but from outside
              {"price": 75, "timestamp": T - timedelta(days=60)},    # 6%: too small
              {"price": 60, "timestamp": T - timedelta(days=30)},    # 20%: counts
              {"price": 70, "timestamp": T - timedelta(days=10)}]
    assert count_drops(prices, T) == 1


# ── conflicts ────────────────────────────────────────────────────────────────

def test_conflict_takes_whichever_is_cheaper_in_expectation():
    prices = history(449.99, 449.99)
    # waiting is worth 0.74 * 285 + 0.26 * 449.99 = $327.90 in expectation
    _, d = run([amazon(), seller("Meh", 400.0)], prediction=WAIT, prices=prices)
    assert d["status"] == "wait"
    _, d = run([amazon(), seller("Great", 300.0)], prediction=WAIT, prices=prices)
    assert d["status"] == "better_offer" and d["best"]["seller"] == "Great"


# ── missing signals ──────────────────────────────────────────────────────────

def test_every_signal_can_be_missing_without_breaking():
    s, d = run([], prediction=None, prices=[])
    for name in ("price", "quiet", "best_offer", "version"):
        assert s[name]["status"] == "missing" and s[name]["why"], name
    assert d["status"] == "nothing_better" and d["amazon"] is None


def test_missing_seller_ratings_say_so():
    s, d = run([amazon(), {"seller": "X", "price": 300.0, "condition": "New", "deliveryDate": D(3)}])
    assert s["best_offer"] == {"status": "missing",
                               "why": "the page didn't give seller ratings for the other offers"}
    assert d["status"] == "nothing_better"


def test_short_history_and_broken_model_are_reported_not_raised():
    def short(records, today):
        raise RuntimeError("Need ~30+ days of price history, got 3 days")

    def broken(records, today):
        raise ValueError("bad input")

    s, _ = run([amazon()], prediction=None, predict_fn=short)
    assert s["price"] == {"status": "missing", "why": "Need ~30+ days of price history, got 3 days"}
    s, d = run([amazon(), seller("Good", 300.0)], prediction=None, predict_fn=broken)
    assert s["price"]["status"] == "missing" and "ValueError" in s["price"]["why"]
    assert d["status"] == "better_offer"  # the other signals still work


def test_model_runs_when_there_is_no_cached_prediction():
    def fake_model(records, today):
        return {"p_drop": 0.6, "pred_14d": 300.0, "recommendation": "WAIT"}

    s, d = run([amazon(price=349.0)], prediction=None, prices=history(349.0, 349.0), predict_fn=fake_model)
    assert s["price"]["source"] == "model" and d["status"] == "wait"


# ── version verdicts (T2) and the results file ───────────────────────────────

def test_version_verdict_from_the_results_file_both_directions():
    pair = next(p for p in load_results()["version_pairs"]["pairs"]
                if p["verdict"] == "A is worse" and p["source"] == "human label")
    a =analysis_signals(pair["asin_a"], [], {}, [], None, today=T)["version"]
    b = analysis_signals(pair["asin_b"], [], {}, [], None, today=T)["version"]
    assert a["verdict"] == "this_worse" and a["other_asin"] == pair["asin_b"]
    assert b["verdict"] == "other_worse" and b["other_asin"] == pair["asin_a"]
    assert a["source"] == "human label"


def test_results_file_is_versioned_and_named_in_every_answer():
    r = load_results()
    assert r["version"] == "v2" and load_results("v1")["version"] == "v1"
    assert len(r["version_pairs"]["pairs"]) == 105
    s, _ = run([amazon()])
    assert s["results_version"] == "v2"


def test_results_file_matches_the_studies_it_was_built_from():
    """Rebuilding v1 from studies/ gives the same pairs, so nobody edited it by hand."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build", ROOT / "backend/scripts/build_analysis_results.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    pairs, inputs = build.version_pairs()
    r = load_results("v1")
    assert pairs == r["version_pairs"]["pairs"] and inputs == r["inputs"]
    # v2's rules come straight from R4's result file and T4's module
    (quiet, q_in), (offers, o_in) = build.quiet_v2(), build.offers_v2()
    v2 = load_results("v2")
    assert v2["quiet"] == quiet and v2["offers"] == offers
    assert v2["inputs"] == {**inputs, **q_in, **o_in}


# ── shopper-facing text matches copy.ts ──────────────────────────────────────

def test_text_matches_copy_ts():
    ts = (ROOT / "website/app/preview/copy.ts").read_text()
    for key, line in text.SKIP.items():
        assert f'{key}: "{line}"' in ts, key
    assert "`Same item, ${condition.toLowerCase()}, ${money(savings)} less.`" in ts
    assert "`Worth waiting if you can. It could drop to ${money(dropPrice(price))} or less in the next 14 days.`" in ts
    assert "DROP_SIZE = 0.08" in ts and text.DROP_SIZE == 0.08
    assert "Math.floor(price * (1 - DROP_SIZE) * 100) / 100" in ts and text.drop_price(299.0) == 275.08
    assert '"This is the best offer for you."' in ts
    assert ("`${r.stars.toFixed(1)} stars from ${r.count.toLocaleString(\"en-US\")} ratings, "
            "${Math.round(r.positive)}% positive.`") in ts
    assert text.trust_line(4.7, 21295, 92.5) == "4.7 stars from 21,295 ratings, 93% positive."
    assert f'SOLD_BY_AMAZON = "{text.SOLD_BY_AMAZON}"' in ts
    assert "MIN_WAIT_CHANCE = 0.15" in ts and load_results()["quiet"]["min_drop_chance"] == 0.15
    assert text.pct(0.745) == "75%" and text.pct(0.005) == "1%"  # rounds like Math.round


# ── speed ────────────────────────────────────────────────────────────────────

def test_fast_enough_for_the_panel():
    offers = [amazon()] + [seller(f"S{i}", 300.0 + i) for i in range(50)]
    s, _ = run(offers, prices=history(*([449.99] * 200)))
    assert s["timing_ms"]["total"] < 50  # milliseconds, cached prediction path


# ── the page price is today's price (#104) ───────────────────────────────────

def test_page_price_rescores_when_the_stored_price_is_stale():
    seen = {}

    def model(records, today):
        seen["latest"] = max(records, key=lambda r: str(r["timestamp"])[:10])["price"]
        return {"p_drop": 0.6, "pred_14d": 250.0, "recommendation": "WAIT"}

    cached = {**WAIT, "price_at_prediction": 349.0}
    s, d = run([amazon(price=299.0)], prediction=cached, prices=history(349.0, 349.0),
               page_price=299.0, predict_fn=model)
    assert seen["latest"] == 299.0 and s["price"]["source"] == "model"
    assert d["wait"]["price"] == 299.0 and d["wait"]["expected_low"] == 250.0


def test_cached_prediction_holds_when_it_was_scored_at_the_page_price():
    def model(records, today):
        raise AssertionError("should use the cached prediction")

    cached = {**WAIT, "price_at_prediction": 349.0}
    s, _ = run([amazon(price=350.0)], prediction=cached, prices=history(349.0, 349.0),
               page_price=350.0, predict_fn=model)
    assert s["price"]["source"].startswith("cached")


def test_no_wait_when_the_page_price_has_moved_far_from_our_history():
    """Datasets end in April 2026. A Roku that was $29 then and is $49 today must not get a
    WAIT from the model reading that gap as a spike."""
    def model(records, today):
        return {"p_drop": 0.75, "pred_14d": 40.0, "recommendation": "WAIT"}

    stale, d = run([amazon(price=49.0)], prediction=None, prices=history(29.0, 29.0), page_price=49.0,
                   dataset_price=29.0, predict_fn=model)
    assert stale["price"]["status"] == "missing" and "out of date" in stale["price"]["why"]
    assert d["status"] != "wait"
    fresh, d = run([amazon(price=30.0)], prediction=None, prices=history(29.0, 29.0), page_price=30.0,
                   dataset_price=29.0, predict_fn=model)
    assert fresh["price"]["status"] == "ok" and d["status"] == "wait"


def test_offers_still_compared_when_history_is_out_of_date():
    _, d = run([amazon(price=449.99), seller("Good", 369.0)], prices=history(300.0), page_price=449.99,
               dataset_price=300.0)
    assert d["status"] == "better_offer"
