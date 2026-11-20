"""Price history from the repo's datasets, in place of Keepa (jobs/csv_history.py).

Run: pytest tests/test_csv_history.py -q
"""
import threading
import time
from datetime import datetime

from jobs import csv_history

TODAY = datetime(2026, 11, 20, 15, 30)


def test_histories_keep_their_real_dates_and_prices():
    """No shift to today: the history ends in April 2026, where the dataset does."""
    from ml.bench.data import load_prices
    raw = load_prices(csv_history.SOURCES[0]).dropna(subset=["price"])
    asin = raw.asin.iloc[0]
    raw = raw[(raw.asin == asin) & (raw.price > 0)].sort_values("date")

    records = csv_history.fetch_price_history(asin)
    assert [r["timestamp"] for r in records] == [d.to_pydatetime() for d in raw.date]
    assert [r["price"] for r in records] == [round(p, 2) for p in raw.price]
    assert records[-1]["timestamp"] < datetime(2026, 5, 1)


def test_newer_dataset_wins_and_both_are_covered():
    april = set(csv_history._by_asin(csv_history.SOURCES[0]))
    older = set(csv_history._by_asin(csv_history.SOURCES[1]))
    assert len(april | older) == 649
    only_older = next(iter(older - april))
    assert csv_history.known(only_older) and csv_history.fetch_price_history(only_older)


def test_unknown_products_have_no_history():
    assert csv_history.fetch_price_history("B0NOTREAL0") == []
    assert csv_history.save("B0NOTREAL0", []) is False


def test_enough_history_for_the_model_on_almost_every_product():
    """The served model needs ~30+ days. Count products with at least 60 days of history."""
    long_enough = 0
    asins = set(csv_history._by_asin(csv_history.SOURCES[0])) | set(csv_history._by_asin(csv_history.SOURCES[1]))
    for asin in asins:
        r = csv_history.fetch_price_history(asin)
        long_enough += (r[-1]["timestamp"] - r[0]["timestamp"]).days >= 60
    assert long_enough / len(asins) > 0.9


def test_concurrent_requests_load_once(monkeypatch):
    import main

    seen = {"product": None, "calls": 0}

    def fake_fetch(asin):
        seen["calls"] += 1
        time.sleep(0.05)  # long enough for the other threads to arrive
        return [{"price": 10.0, "timestamp": TODAY}]

    def fake_save(asin, records):
        seen["product"] = {"product_id": 1, "asin": asin}
        return True

    monkeypatch.setattr(main.csv_history, "fetch_price_history", fake_fetch)
    monkeypatch.setattr(main.csv_history, "save", fake_save)
    monkeypatch.setattr(main, "get_product", lambda asin: seen["product"])
    monkeypatch.setattr(main, "_score_and_store", lambda product, asin: None)

    threads = [threading.Thread(target=main._fetch_and_seed, args=("B000TEST01",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["calls"] == 1
