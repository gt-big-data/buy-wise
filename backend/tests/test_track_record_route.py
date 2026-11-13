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
    assert TestClient(app).get("/track-record").status_code == 404