from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import main

SOON = (date.today() + timedelta(days=3)).isoformat()
AMAZON = {"seller": "Amazon.com", "price": 449.99, "condition": "New", "deliveryDate": SOON,
          "pinned": True, "shipsFromAmazon": True}
# Seller numbers taken from a real offer list (Bose QuietComfort, Nov 13).
TRUSTED = {"seller": "ADV Sellers", "price": 369.0, "condition": "New", "deliveryDate": SOON,
           "stars": 4.5, "ratingCount": 390, "positivePercent": 94}
FEW_RATINGS = {**TRUSTED, "seller": "Win-Plus shop", "ratingCount": 28}
LOW_STARS = {**TRUSTED, "seller": "Silmarils", "stars": 2.5, "ratingCount": 2255, "positivePercent": 34}
NO_RATING = {"seller": "Unrated", "price": 369.0, "condition": "New", "deliveryDate": SOON}


@pytest.fixture
def api(monkeypatch):
    prediction = {"recommendation": "BUY", "confidence_score": 0.9, "pred_7d": 440, "pred_14d": 445, "pred_30d": 450}
    monkeypatch.setattr(main, "_require_db", lambda: None)
    monkeypatch.setattr(main, "get_product", lambda asin: {"product_id": 1})
    monkeypatch.setattr(main, "get_latest_prediction", lambda product_id: prediction)
    monkeypatch.setattr(main, "db_get_price_history",
                        lambda product_id, limit=1000: [{"price": 449.99, "timestamp": datetime(2026, 11, 13)}])
    return TestClient(main.app)


def decide(api, *offers):
    return api.post("/decision", json={"asin": "B0CCZ26B5V", "offers": [AMAZON, *offers]}).json()


def test_trusted_cheaper_seller_is_recommended(api):
    answer = decide(api, TRUSTED)
    assert answer["status"] == "better_offer"
    assert answer["best"]["seller"] == "ADV Sellers"
    assert answer["best"]["rating_count"] == 390
    assert answer["best"]["positive_percent"] == 94
    assert "4.5 stars from 390 ratings, 94% positive" in answer["reason"]


@pytest.mark.parametrize("offer", [FEW_RATINGS, LOW_STARS, NO_RATING])
def test_untrusted_sellers_are_not_recommended(api, offer):
    assert decide(api, offer)["status"] == "nothing_better"


def test_amazon_resale_is_recommended_without_ratings(api):
    resale = {"seller": "Amazon Resale", "price": 360.0, "condition": "Used - Like New", "deliveryDate": SOON}
    answer = api.post("/decision", json={"asin": "B0CCZ26B5V", "offers": [AMAZON, resale],
                                         "profile": {"condition": "like_new"}}).json()
    assert answer["status"] == "better_offer"
    assert answer["best"]["seller"] == "Amazon Resale"
    assert answer["best"]["stars"] is None and answer["best"]["rating_count"] is None
    assert answer["reason"] == "Same item, used - like new, $89.99 less."


def test_snake_case_field_names_also_work(api):
    offer = {"seller": "ADV Sellers", "price": 369.0, "condition": "New", "delivery_date": SOON,
             "stars": 4.5, "rating_count": 390, "positive_percent": 94}
    assert decide(api, offer)["status"] == "better_offer"


def test_cheapest_trusted_seller_wins(api):
    cheaper = {**TRUSTED, "seller": "Cheaper Co", "price": 350.0}
    assert decide(api, TRUSTED, cheaper)["best"]["seller"] == "Cheaper Co"


def test_used_conditions_match_amazons_wording(api):
    used = {**TRUSTED, "condition": "Used - Very Good", "price": 250.0}  # T4: Very Good is worth 30% less than new
    body = {"asin": "B0CCZ26B5V", "offers": [AMAZON, used], "profile": {"condition": "any_good"}}
    assert api.post("/decision", json=body).json()["status"] == "better_offer"
    body["profile"]["condition"] = "new_only"
    assert api.post("/decision", json=body).json()["status"] == "nothing_better"


def test_seller_shipping_through_amazon_still_counts(api):
    fba = {**TRUSTED, "shipsFromAmazon": True}
    assert decide(api, fba)["status"] == "better_offer"


def test_unseen_products_load_from_the_datasets_then_answer(api, monkeypatch):
    seeded = []
    monkeypatch.setattr(main, "get_product", lambda asin: {"product_id": 1} if seeded else None)
    monkeypatch.setattr(main, "_fetch_and_seed", lambda asin: seeded.append(asin))
    assert decide(api, TRUSTED)["status"] == "better_offer"
    assert seeded == ["B0CCZ26B5V"]


def test_products_outside_the_datasets_still_get_their_sellers_compared(api, monkeypatch):
    monkeypatch.setattr(main, "get_product", lambda asin: None)
    monkeypatch.setattr(main, "_fetch_and_seed", lambda asin: None)
    assert decide(api, TRUSTED)["status"] == "better_offer"
    answer = decide(api)
    assert answer["status"] == "nothing_better" and answer["wait"] is None


def test_too_little_history_means_no_wait_not_a_made_up_one(api, monkeypatch):
    """No cached prediction and 8 days of prices: the model can't run, so no WAIT."""
    monkeypatch.setattr(main, "get_latest_prediction", lambda product_id: None)
    monkeypatch.setattr(main, "db_get_price_history", lambda product_id, limit=1000: [
        {"price": 549.0 - i, "timestamp": datetime(2026, 11, 12 + i)} for i in range(8)])
    answer = decide(api)
    assert answer["status"] == "nothing_better" and answer["wait"] is None


def _watch(api, monkeypatch, **body):
    saved = {}
    monkeypatch.setattr(main, "add_to_watchlist", lambda **kw: saved.update(kw))
    r = api.post("/watchlist", json={"user_id": "u1", "asin": "B0CCZ26B5V", "target_price": 99.0, **body})
    assert r.status_code == 200
    return saved["recommendation_at_add"]


def test_watch_stores_what_the_panel_said(api, monkeypatch):
    assert _watch(api, monkeypatch, recommendation_at_add="WAIT") == "WAIT"  # model says BUY


def test_watch_without_the_panel_status_falls_back_to_the_model(api, monkeypatch):
    assert _watch(api, monkeypatch) == "BUY"


def test_better_offer_passes_returns_and_warranty_through(api):
    best = decide(api, {**TRUSTED, "freeReturns": True, "fullWarranty": False})["best"]
    assert best["free_returns"] is True and best["full_warranty"] is False
    best = decide(api, TRUSTED)["best"]
    assert best["free_returns"] is None and best["full_warranty"] is None
