"""
Regression: first-time product load should expose ~30 daily chart points with calendar
labels (not weekly -4w buckets).

Run: pytest tests/test_price_history_chart_fix.py -q
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import main

    return TestClient(main.app)


def _synthetic_prices_desc(product_id: int, anchor: datetime):
    """30 days of prices, newest first (matches DB ORDER BY timestamp DESC)."""
    rows = []
    for i in range(30):
        ts = anchor.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=i)
        rows.append(
            {
                "price_id": i,
                "product_id": product_id,
                "price": 100.0 + i * 0.5,
                "timestamp": ts,
                "availability": True,
                "deal_flag": False,
            }
        )
    return rows


def _install_price_history_stubs(main, prices_desc, prediction, product_row):
    """Works when MySQL import failed: endpoint still needs these names on main."""

    calls = {"gp": 0}

    def get_product(asin):
        calls["gp"] += 1
        if calls["gp"] == 1:
            return None
        return product_row

    main.get_product = get_product
    main._fetch_and_seed = lambda asin: None
    main.db_get_price_history = MagicMock(return_value=prices_desc)
    main.get_latest_prediction = MagicMock(return_value=prediction)
    main._require_db = lambda: None


def test_price_history_endpoint_daily_labels_and_count(client):
    import main

    anchor = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    product_row = {"product_id": 42, "asin": "B0VERIFYNEW1"}
    prices_desc = _synthetic_prices_desc(42, anchor)
    prediction = {"pred_7d": 95.0, "pred_14d": 94.0, "pred_30d": 92.0}

    _install_price_history_stubs(main, prices_desc, prediction, product_row)

    r = client.get("/price-history/B0VERIFYNEW1")
    assert r.status_code == 200, r.text
    body = r.json()
    labels = [p["label"] for p in body["points"]]
    assert not any(l.startswith("-") and l.endswith("w") for l in labels), labels
    assert any(c.isalpha() for c in labels[0]), labels
    forecast_only = [
        p for p in body["points"] if p.get("predicted") is not None and p.get("actual") is None
    ]
    assert len(forecast_only) == 3
    assert len(body["points"]) >= 30
    assert "30 days" in body["chart_title"] or "UTC" in body["chart_title"]


def test_bucket_never_emits_weekly_minus_w_labels():
    from main import _bucket_price_history

    anchor = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    prices_desc = _synthetic_prices_desc(1, anchor)
    pred = {"pred_7d": 90.0, "pred_14d": 89.0, "pred_30d": 88.0}
    out = _bucket_price_history(prices_desc, pred, "B0X")
    labels = [p.label for p in out.points]
    assert all(not (l.startswith("-") and l.endswith("w")) for l in labels)
    assert len(out.points) >= 30


def test_keepa_passes_recorded_at_to_insert_price():
    from pathlib import Path

    text = Path(__file__).resolve().parents[1] / "jobs" / "keepa_fetch.py"
    src = text.read_text(encoding="utf-8")
    assert "recorded_at=ts" in src
    assert "insert_price(product_id, record[\"price\"], availability=True, deal_flag=False)" not in src


def test_insert_price_accepts_recorded_at():
    from pathlib import Path

    text = Path(__file__).resolve().parents[1] / "db" / "connection.py"
    src = text.read_text(encoding="utf-8")
    assert "recorded_at" in src
    assert "recorded_at if recorded_at is not None else datetime.utcnow()" in src


def test_extension_price_chart_keeps_full_month_range():
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "extension" / "src" / "components" / "PriceChart.tsx"
    text = src.read_text(encoding="utf-8")
    assert "firstForecastOnlyIndex" in text
    assert 'range === "1M"' in text
    # Old bug: 1M sliced to 6 points (weekly assumption)
    assert "Math.min(points.length, 6)" not in text


def test_main_uses_limit_1000_for_history():
    from pathlib import Path
    import re

    text = Path(__file__).resolve().parents[1] / "main.py"
    src = text.read_text(encoding="utf-8")
    assert src.count("db_get_price_history(product[\"product_id\"], limit=1000)") >= 2
    assert re.search(r"db_get_price_history\([^)]*limit=100\)", src) is None
