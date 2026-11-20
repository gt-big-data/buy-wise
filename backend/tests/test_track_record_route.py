import json

from fastapi.testclient import TestClient

import routes.track_record as tr
from main import app

SUMMARY = {
    "predictions": 4, "resolved": 4, "pending": 0, "bad_data": 0, "products": 2,
    "first_day": "2025-08-28", "last_day": "2025-09-04",
    "hit_rate %": 50.0, "base_rate %": 50.0, "wait_calls": 2, "buy_calls": 2,
    "wait_right %": 50.0, "buy_right %": 50.0,
    "$ saved by right WAITs": 20.0, "$ lost by wrong WAITs": 5.0, "$ missed by wrong BUYs": 30.0,
}
HEADER = "asin,predicted_on,price_at_prediction,p_drop,recommendation,horizon_days,drop_threshold,model_version,actual_min_price,actual_end_price,drop_pct,status,dropped,was_correct\n"
ROWS = [
    "A1,2025-08-28,100,0.6,WAIT,14,0.08,v,80,85,0.2,resolved,True,True",    # right wait, saved 20
    "A1,2025-09-04,100,0.6,WAIT,14,0.08,v,98,105,0.02,resolved,False,False", # wrong wait, paid 5 more
    "A2,2025-08-28,200,0.1,BUY,14,0.08,v,170,180,0.15,resolved,True,False",  # wrong buy, missed 30
    "A2,2025-09-04,200,0.1,BUY,14,0.08,v,200,200,0.0,resolved,False,True",   # right buy
]


def test_track_record(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text(json.dumps(SUMMARY))
    (tmp_path / "resolved.csv").write_text(HEADER + "\n".join(ROWS) + "\n")
    monkeypatch.setattr(tr, "TRACK_RECORD_DIR", tmp_path)

    body = TestClient(app).get("/track-record").json()
    assert body["calls"] == 4
    assert body["wait_right"] == 0.5
    assert [w["amount"] for w in body["wins"]] == [20.0]
    assert [(m["asin"], m["said"], m["amount"]) for m in body["misses"]] == [("A2", "BUY", 30.0), ("A1", "WAIT", 5.0)]


def test_track_record_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "TRACK_RECORD_DIR", tmp_path)
    monkeypatch.setattr(tr, "COMMITTED_DIR", tmp_path / "none")
    assert TestClient(app).get("/track-record").status_code == 404


def test_track_record_falls_back_to_committed_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "TRACK_RECORD_DIR", tmp_path / "empty")
    assert (tr.COMMITTED_DIR / "summary.json").exists()
    summary = json.loads((tr.COMMITTED_DIR / "summary.json").read_text())

    res = TestClient(app).get("/track-record")
    assert res.status_code == 200
    body = res.json()
    assert body["calls"] == summary["resolved"] > 0
    assert body["products"] == summary["products"]


def test_track_record_prefers_fresh_output(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text(json.dumps(SUMMARY))
    (tmp_path / "resolved.csv").write_text(HEADER + "\n".join(ROWS) + "\n")
    monkeypatch.setattr(tr, "TRACK_RECORD_DIR", tmp_path)
    assert TestClient(app).get("/track-record").json()["calls"] == 4

def test_replay_list_follows_the_stated_rule(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text(json.dumps(SUMMARY))
    (tmp_path / "resolved.csv").write_text(HEADER + "\n".join(ROWS) + "\n")
    monkeypatch.setattr(tr, "TRACK_RECORD_DIR", tmp_path)

    body = TestClient(app).get("/track-record/replay").json()
    assert body["rule"] == tr.REPLAY_RULE
    # A1's wrong Wait is skipped (one call per product); A2's right Buy saw no price rise
    assert [(c["asin"], c["kind"], c["amount"]) for c in body["calls"]] == [("A1", "wait_right", 20.0),
                                                                          ("A2", "buy_wrong", 30.0)]
    assert body["calls"][0]["drop_chance"] == 0.6


def test_replay_of_a_real_call_has_the_real_prices_around_it(monkeypatch):
    """First listed call from the committed replay: prices with real dates, the 14-day low matches."""
    from datetime import date, timedelta

    monkeypatch.setattr(tr, "TRACK_RECORD_DIR", tr.COMMITTED_DIR / "none")
    api = TestClient(app)
    call = api.get("/track-record/replay").json()["calls"][0]
    assert call["kind"] == "wait_right"

    r = api.get(f"/track-record/replay/{call['asin']}/{call['predicted_on']}").json()
    on = date.fromisoformat(call["predicted_on"])
    days = [date.fromisoformat(d) for d, _ in r["prices"]]
    assert days[0] == on - timedelta(days=60) and days[-1] == on + timedelta(days=14) == date.fromisoformat(r["end_on"])
    assert dict(r["prices"])[call["predicted_on"]] == call["price"]
    after = [p for d, p in r["prices"] if d > call["predicted_on"]]
    assert min(after) == r["low"] < r["price"] and dict(r["prices"])[r["low_on"]] == r["low"]
    assert r["drop_chance"] >= 0.4 and r["drops"] >= 0


def test_replay_of_an_unknown_call_is_404():
    assert TestClient(app).get("/track-record/replay/B0NOTREAL0/2025-09-25").status_code == 404
