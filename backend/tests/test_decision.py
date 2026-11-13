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
    prediction = {"recommendation": "BUY", "confidence_score": 0.4, "pred_7d": 440, "pred_14d": 445, "pred_30d": 450}
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


def test_snake_case_field_names_also_work(api):
    offer = {"seller": "ADV Sellers", "price": 369.0, "condition": "New", "delivery_date": SOON,
             "stars": 4.5, "rating_count": 390, "positive_percent": 94}
    assert decide(api, offer)["status"] == "better_offer"


def test_cheapest_trusted_seller_wins(api):
    cheaper = {**TRUSTED, "seller": "Cheaper Co", "price": 350.0}
    assert decide(api, TRUSTED, cheaper)["best"]["seller"] == "Cheaper Co"


def test_used_conditions_match_amazons_wording(api):
    used = {**TRUSTED, "condition": "Used - Very Good"}
    body = {"asin": "B0CCZ26B5V", "offers": [AMAZON, used], "profile": {"condition": "any_good"}}
    assert api.post("/decision", json=body).json()["status"] == "better_offer"
    body["profile"]["condition"] = "new_only"
    assert api.post("/decision", json=body).json()["status"] == "nothing_better"


def test_seller_shipping_through_amazon_still_counts(api):
    fba = {**TRUSTED, "shipsFromAmazon": True}
    assert decide(api, fba)["status"] == "better_offer"
