from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
import main


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(main, "_require_db", lambda: None)
    monkeypatch.setattr(main, "get_product", Mock(return_value={"product_id": 42}), raising=False)
    monkeypatch.setattr(main, "insert_price", Mock(), raising=False)
    return TestClient(main.app)


def test_saves_live_price_and_exposes_it_to_alert_reads(api, monkeypatch):
    response = api.post("/prices", json={"asin": "B08N5WRWNW", "price": 19.99})
    assert response.status_code == 201
    main.insert_price.assert_called_once_with(42, 19.99)
    monkeypatch.setattr(main, "db_get_price_history", Mock(return_value=[
        {"price": 19.99, "timestamp": datetime.utcnow()}
    ]), raising=False)
    monkeypatch.setattr(main, "get_latest_prediction", Mock(return_value=None), raising=False)
    assert api.get("/price-history/B08N5WRWNW").json()["current_price"] == 19.99


@pytest.mark.parametrize("price", [0, -1, 100000000, "NaN", "Infinity", None])
def test_rejects_invalid_prices(api, price):
    assert api.post("/prices", json={"asin": "B08N5WRWNW", "price": price}).status_code == 422
    main.insert_price.assert_not_called()


def test_unknown_product_is_not_inserted(api, monkeypatch):
    main.get_product.return_value = None
    assert api.post("/prices", json={"asin": "B08N5WRWNW", "price": 10}).status_code == 404
    main.insert_price.assert_not_called()


def test_rejects_invalid_asin(api):
    assert api.post("/prices", json={"asin": "bad", "price": 10}).status_code == 422


def test_write_failure_does_not_report_success(api):
    main.insert_price.side_effect = RuntimeError("database write failed")
    with TestClient(main.app, raise_server_exceptions=False) as client:
        assert client.post("/prices", json={"asin": "B08N5WRWNW", "price": 10}).status_code == 500
