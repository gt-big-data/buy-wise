"""
Keepa fetching: one fetch per product even under concurrent requests, and the API
key never appears in error messages.

Run: pytest tests/test_keepa_fetch.py -q
"""

import threading
import time
from unittest.mock import MagicMock

import pytest


def test_concurrent_requests_fetch_once(monkeypatch):
    import main

    seen = {"product": None, "calls": 0}

    def fake_keepa(asin):
        seen["calls"] += 1
        time.sleep(0.05)  # long enough for the other thread to arrive
        seen["product"] = {"product_id": 1, "asin": asin}
        return []

    monkeypatch.setattr(main, "keepa_fetch", fake_keepa)
    monkeypatch.setattr(main, "get_product", lambda asin: seen["product"])
    monkeypatch.setattr(main, "_score_and_store", lambda product, asin: None)

    threads = [threading.Thread(target=main._fetch_and_seed, args=("B000TEST01",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["calls"] == 1


def test_keepa_errors_hide_the_key(monkeypatch):
    from jobs import keepa_fetch

    monkeypatch.setattr(keepa_fetch, "API_KEY", "SECRETKEY123")
    res = MagicMock(ok=False, status_code=402)
    monkeypatch.setattr(keepa_fetch.requests, "get", lambda *a, **k: res)
    with pytest.raises(RuntimeError) as err:
        keepa_fetch._fetch_with_retry("https://api.keepa.com/product", {"key": "SECRETKEY123"})
    assert "SECRETKEY123" not in str(err.value)
    assert "402" in str(err.value)
