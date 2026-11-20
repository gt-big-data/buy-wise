"""
Keepa fetching: the API key never appears in error messages. (The backend now seeds
products from the repo's datasets; see test_csv_history.py.)

Run: pytest tests/test_keepa_fetch.py -q
"""

from unittest.mock import MagicMock

import pytest


def test_keepa_errors_hide_the_key(monkeypatch):
    from jobs import keepa_fetch

    monkeypatch.setattr(keepa_fetch, "API_KEY", "SECRETKEY123")
    res = MagicMock(ok=False, status_code=402)
    monkeypatch.setattr(keepa_fetch.requests, "get", lambda *a, **k: res)
    with pytest.raises(RuntimeError) as err:
        keepa_fetch._fetch_with_retry("https://api.keepa.com/product", {"key": "SECRETKEY123"})
    assert "SECRETKEY123" not in str(err.value)
    assert "402" in str(err.value)
