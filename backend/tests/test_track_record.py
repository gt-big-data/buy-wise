"""Track record (#85): logging, resolving and scoring, on faked prices and a fake model.

The promises:
  - a replayed prediction never sees prices after the day it was made
  - WAIT is right when an 8% drop came within 14 days, BUY is right when it didn't
  - predictions whose 14 days haven't passed stay pending, not scored
  - live numbers use the same definitions as the offline benchmark

Run: pytest tests/test_track_record.py -q
"""
import numpy as np
import pandas as pd
import pytest

from ml.bench.metrics import _policy_row
from ml.features import add_labels, daily_panel
from ml.track_record import misses, replay_predictions, report, resolve, score

DAY0 = pd.Timestamp("2025-01-01")
D = lambda n: DAY0 + pd.Timedelta(days=n)  # noqa: E731


def _prices():
    """Keepa-style: one row per price *change*. Calls are made on day 40.

    A  110, drops to 95 on day 45        -> 13.6% drop: a WAIT is right
    B   50, rises to 60 on day 45        -> no drop: a BUY is right
    C  120, rises to 130 on day 47       -> no drop: a WAIT is wrong, shopper pays $10 more
    E   80, drops to 70 on day 50        -> 12.5% drop: a BUY is wrong, missed $10
    """
    rows = [("A", 0, 110), ("A", 45, 95), ("B", 0, 50), ("B", 45, 60),
            ("C", 0, 120), ("C", 47, 130), ("E", 0, 80), ("E", 50, 70), ("E", 70, 70)]
    # every series runs to day 70 so the panel covers each 14-day window
    rows += [(a, 70, p) for a, p in (("A", 95), ("B", 60), ("C", 130))]
    return pd.DataFrame([{"asin": a, "date": D(d), "price": float(p)} for a, d, p in rows])


def fake_model(records, today):
    """Says WAIT when the price is $100+, unless E (forced BUY). Refuses <30 days of history."""
    dates = [pd.Timestamp(r["timestamp"]) for r in records]
    assert max(dates) <= today, "the model was shown a price from the future"
    if (today - min(dates)).days < 30:
        raise RuntimeError("too little history")
    price = sorted(records, key=lambda r: r["timestamp"])[-1]["price"]
    wait = price >= 100
    return {"recommendation": "WAIT" if wait else "BUY", "p_drop": 0.7 if wait else 0.2}


@pytest.fixture
def resolved():
    log = replay_predictions(_prices(), [D(40)], fake_model, "test@1")
    return resolve(log, _prices())


def test_replay_never_sees_the_future_and_skips_short_history():
    log = replay_predictions(_prices(), [D(10), D(40)], fake_model, "test@1")
    assert set(log.predicted_on) == {D(40)}  # day 10 has under 30 days of history
    assert log.set_index("asin").loc["A", "price_at_prediction"] == 110  # not the day-45 price
    assert (log.model_version == "test@1").all() and (log.horizon_days == 14).all()


def test_right_and_wrong_for_buy_and_wait(resolved):
    r = resolved.set_index("asin")
    assert (r.status == "resolved").all()
    assert r.loc["A", "recommendation"] == "WAIT" and r.loc["A", "was_correct"] == True   # noqa: E712
    assert r.loc["B", "recommendation"] == "BUY" and r.loc["B", "was_correct"] == True    # noqa: E712
    assert r.loc["C", "recommendation"] == "WAIT" and r.loc["C", "was_correct"] == False  # noqa: E712
    assert r.loc["E", "recommendation"] == "BUY" and r.loc["E", "was_correct"] == False   # noqa: E712
    assert r.loc["A", "actual_min_price"] == 95 and r.loc["C", "actual_end_price"] == 130


def test_dollars_and_rates(resolved):
    s = score(resolved)
    assert s["hit_rate %"] == 50
    assert s["wait_right %"] == 50 and s["buy_right %"] == 50
    assert s["$ saved by right WAITs"] == pytest.approx(15)   # A: 110 -> 95
    assert s["$ lost by wrong WAITs"] == pytest.approx(10)    # C: 120 -> 130
    assert s["$ missed by wrong BUYs"] == pytest.approx(10)   # E: 80 -> 70
    m = misses(resolved)
    assert list(m.asin) == ["C", "E"] or list(m.asin) == ["E", "C"]
    assert m.cost_usd.tolist() == [10, 10]


def test_unresolved_predictions_stay_pending():
    log = replay_predictions(_prices(), [D(40), D(65)], fake_model, "test@1")
    r = resolve(log, _prices())  # prices end on day 70, so day-65 calls aren't due until day 79
    assert (r[r.predicted_on == D(65)].status == "pending").all()
    assert r[r.predicted_on == D(65)].was_correct.isna().all()
    assert score(r)["pending"] == 4 and score(r)["resolved"] == 4
    # live use: the same day-40 calls are still pending if we only know prices to day 50
    early = resolve(log[log.predicted_on == D(40)], _prices(), as_of=D(50))
    assert (early.status == "pending").all()


def test_same_definitions_as_offline_benchmark():
    """Score random calls on random prices both ways: the track record and the
    benchmark's own labels + _policy_row must agree to the cent."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(5):
        p, t = 100.0, 0
        while t < 200:
            rows.append({"asin": f"X{i}", "date": D(t), "price": round(p, 2)})
            p *= float(rng.choice([0.85, 0.95, 1.0, 1.05, 1.1]))
            t += int(rng.integers(1, 6))
    prices = pd.DataFrame(rows)

    def random_model(records, today):
        return {"recommendation": "WAIT" if rng.random() < 0.3 else "BUY", "p_drop": 0.5}

    log = replay_predictions(prices, pd.date_range(D(30), D(180), freq="3D"), random_model, "rand")
    r = resolve(log, prices)
    s = score(r)

    lab = add_labels(daily_panel(prices)).rename(columns={"date": "predicted_on"})
    te = log.merge(lab, on=["asin", "predicted_on"])
    assert len(te) == s["resolved"]
    assert (te.price == te.price_at_prediction).all()
    want = _policy_row(te, (te.recommendation == "WAIT").values)
    for k in ("precision %", "lift", "$ caught", "$ no alert"):
        assert s[k] == pytest.approx(want[k]), k
    assert s["base_rate %"] == pytest.approx(te.y.mean() * 100)


def test_report_shows_misses_and_offline_column(resolved):
    md = report(resolved, offline={"precision %": 40.0, "lift": 2.0, "$ caught": 1.5, "$ no alert": 0.5})
    assert "Biggest misses" in md and "| C |" in md and "| E |" in md
    assert "Offline test set" in md and "40.0%" in md
    assert "Nothing has resolved yet" in report(resolved.assign(status="pending"))
