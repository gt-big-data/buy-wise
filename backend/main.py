from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Optional, Literal
import logging
import threading

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from routes.dashboard import router as dashboard_router
from routes.track_record import router as track_record_router
from schemas.common import RecommendationDirection

logger = logging.getLogger(__name__)

# Keepa can be imported independently of database availability.
keepa_fetch = None
try:
    from jobs.keepa_fetch import fetch_price_history as keepa_fetch
except Exception as _exc:
    logger.warning("Keepa fetch unavailable at startup: %s", _exc)

# Database imports are separate so a DB failure doesn't disable Keepa imports.
try:
    from db.connection import (
        get_product,
        get_latest_prediction,
        get_price_history as db_get_price_history,
        insert_prediction,
        add_to_watchlist,
        remove_from_watchlist,
        get_watchlist,
        insert_user_activity,
        get_recent_user_activity,
    )
    _DB_AVAILABLE = True
except Exception as _exc:
    logger.warning("DB unavailable at startup: %s", _exc)
    _DB_AVAILABLE = False

    # Define DB functions so the module remains importable without MySQL.
    get_product = None
    get_latest_prediction = None
    db_get_price_history = None
    insert_prediction = None
    add_to_watchlist = None
    remove_from_watchlist = None
    get_watchlist = None
    insert_user_activity = None
    get_recent_user_activity = None


app = FastAPI(
    title="BuyWise API",
    description="Backend for the BuyWise Chrome extension and web dashboard.",
    version="0.1.0",
)

app.include_router(dashboard_router)
app.include_router(track_record_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# Chrome blocks HTTPS→localhost fetches (Private Network Access policy) unless the
# server returns Access-Control-Allow-Private-Network: true on OPTIONS preflight.
@app.middleware("http")
async def private_network_access(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=204)
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


# ── Existing API models ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    started_at: datetime = Field(..., example="2026-03-16T00:00:00Z")


class PredictResponse(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    recommendation: RecommendationDirection = Field(..., example="WAIT")
    confidence: float = Field(..., ge=0, le=100, description="0–100", example=87.0)
    drop_chance: float = Field(
        ...,
        ge=0,
        le=100,
        description="0–100: chance the price falls 8%+ within horizon_days",
        example=41.0,
    )
    higher_after_wait: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="0–100: share of past WAITs where the price was higher after horizon_days.",
        example=13.7,
    )
    predicted_price: float = Field(..., example=169.99)
    potential_savings: float = Field(..., example=30.00)
    horizon_days: int = Field(..., ge=1, example=30)
    why: str = Field(
        ...,
        example="Our model is 87% confident this item will drop in price.",
    )


class ProductInfoResponse(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    title: str = Field(..., example="Wireless Noise-Cancelling Headphones")
    brand: Optional[str] = Field(None, example="Sony")
    category: Optional[str] = Field(None, example="Electronics")
    current_price: float = Field(..., example=199.99)
    currency: str = Field(..., example="USD")
    url: str = Field(..., example="https://www.amazon.com/dp/B08N5WRWNW")


class PricePoint(BaseModel):
    label: str
    actual: Optional[float] = None
    predicted: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    asin: str
    chart_title: str
    current_price: float
    predicted_best_price: float
    points: List[PricePoint]


class ActivityAction(str, Enum):
    purchased = "purchased"
    dismissed = "dismissed"
    added_to_watchlist = "added_to_watchlist"


class ActivityRequest(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    action: ActivityAction = Field(..., example="dismissed")
    recommendation_shown: Optional[RecommendationDirection] = Field(
        None,
        example="WAIT",
    )
    user_id: Optional[str] = Field(None, example="user_123")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = Field(None, example={"price": 199.99})


class ActivityResponse(BaseModel):
    status: str = Field(..., example="logged")
    received_at: datetime = Field(..., example="2026-03-16T00:00:00Z")
    details: Optional[str] = Field(None)


class WatchlistAddRequest(BaseModel):
    user_id: str
    asin: str
    target_price: Optional[float] = None


class ActivityRecord(BaseModel):
    activity_id: int = Field(..., example=42)
    asin: str = Field(..., example="B08N5WRWNW")
    recommendation_shown: RecommendationDirection = Field(..., example="WAIT")
    action: ActivityAction = Field(..., example="dismissed")
    user_id: Optional[str] = Field(None, example="user_123")
    timestamp: datetime = Field(..., example="2026-03-16T00:00:00Z")


class RecentActivityResponse(BaseModel):
    items: List[ActivityRecord]
    total_count: int = Field(..., ge=0, example=20)
    limit: int = Field(..., ge=1, example=20)


# ── Decision endpoint models ─────────────────────────────────────────────────

class DecisionProfile(BaseModel):
    condition: Literal["new_only", "like_new", "any_good"] = "new_only"
    patience: Literal["soon", "week_or_two", "deal"] = "week_or_two"
    alerts: Literal["on", "when_asked", "off"] = "on"


class DecisionOffer(BaseModel):
    seller: str
    seller_id: Optional[str] = Field(None, alias="sellerId")
    price: float = Field(..., gt=0)
    condition: Optional[str] = None

    shipping_cost: Optional[float] = Field(None, alias="shippingCost")
    delivery_date: Optional[str] = Field(None, alias="deliveryDate")
    fastest_delivery: Optional[str] = Field(None, alias="fastestDelivery")
    ships_from: Optional[str] = Field(None, alias="shipsFrom")
    ships_from_amazon: bool = Field(False, alias="shipsFromAmazon")

    pinned: bool = False

    # These are optional because the current Amazon offer scraper does not
    # extract all seller-quality information yet.
    stars: Optional[float] = Field(None, ge=0, le=5)
    years: Optional[float] = Field(None, ge=0)
    rating_count: Optional[int] = Field(None, ge=0)
    free_returns: Optional[bool] = Field(None, alias="freeReturns")
    full_warranty: Optional[bool] = Field(None, alias="fullWarranty")

    class Config:
        allow_population_by_field_name = True


class DecisionRequest(BaseModel):
    asin: str
    offers: List[DecisionOffer] = Field(default_factory=list)
    profile: DecisionProfile = Field(default_factory=DecisionProfile)


class DecisionOfferResponse(BaseModel):
    seller: str
    condition: str
    price: float
    arrives: str


class QualifiedDecisionOfferResponse(DecisionOfferResponse):
    stars: float
    years: float
    free_returns: bool
    full_warranty: bool


class DecisionWaitResponse(BaseModel):
    price: float
    expected_low: float
    drop_chance: float
    drops: int
    months: int


class DecisionNextBestResponse(BaseModel):
    price: float
    days_later: int


class DecisionResponse(BaseModel):
    show: bool
    status: Literal["better_offer", "wait", "nothing_better"]
    offers_checked: int
    amazon: Optional[DecisionOfferResponse] = None
    best: Optional[QualifiedDecisionOfferResponse] = None
    reason: str
    wait: Optional[DecisionWaitResponse] = None
    next_best: Optional[DecisionNextBestResponse] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _require_db() -> None:
    if not _DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")


def _higher_after_wait() -> Optional[float]:
    """Share (0–100) of past WAITs where the price was higher 14 days later."""
    from ml import inference as _ml

    if not _ml._MODELS_LOADED:
        return None

    return round(_ml._bundle["wait_outcomes"]["higher_share"] * 100, 1)


def _generate_why(
    recommendation: str,
    confidence: float,
    potential_savings: float,
    current_price: float,
    pred_7d: Optional[float],
    pred_14d: Optional[float],
) -> str:
    pct = int(round(confidence))
    drop = pct if recommendation == "WAIT" else 100 - pct
    article = "an" if str(drop).startswith("8") or drop in (11, 18) else "a"

    if recommendation == "WAIT":
        lines = [
            f"Based on this product's price history, there's {article} {drop}% chance it drops 8% or more in the next 14 days."
        ]

        if pred_14d and pred_14d < current_price:
            lines.append(
                f"The expected low over that window is about ${pred_14d:.2f}, versus ${current_price:.2f} today."
            )

        if potential_savings > 0:
            lines.append(
                f"Waiting could save you approximately ${potential_savings:.0f}."
            )

        higher = _higher_after_wait()
        if higher is not None:
            lines.append(
                f"If it doesn't drop, it's usually about the same price: only {higher:.0f}% of past waits ended higher."
            )

        return " ".join(lines)

    lines = [
        f"There's only {article} {drop}% chance of a drop of 8% or more in the next 14 days, so buying now is reasonable."
    ]

    if pred_7d and pred_7d > current_price:
        rise_pct = round(
            (pred_7d - current_price) / current_price * 100,
            1,
        )
        lines.append(
            f"The 7-day forecast is ${pred_7d:.2f}, suggesting prices may rise {rise_pct}% — buying now locks in today's rate."
        )
    else:
        lines.append(
            f"At ${current_price:.2f}, the price is in line with its recent history."
        )

    return " ".join(lines)


def _resolve_activity_recommendation(
    request: ActivityRequest,
) -> RecommendationDirection:
    if request.recommendation_shown:
        return request.recommendation_shown

    product = get_product(request.asin)
    if not product:
        raise HTTPException(
            status_code=400,
            detail="recommendation_shown is required when no product prediction exists",
        )

    prediction = get_latest_prediction(product["product_id"])
    if not prediction:
        raise HTTPException(
            status_code=400,
            detail="recommendation_shown is required when no product prediction exists",
        )

    return RecommendationDirection(prediction["recommendation"])


def _day_label_utc(d: datetime) -> str:
    """Chart x-axis: month abbreviation + day (UTC calendar day)."""
    return f"{d.strftime('%b')} {d.day}"


def _bucket_price_history(
    prices: list,
    prediction: Optional[dict],
    asin: str,
) -> PriceHistoryResponse:
    """~30 daily actuals (forward-filled EOD per UTC day) plus forecast points."""
    now = datetime.utcnow()
    prices_asc = sorted(prices, key=lambda p: p["timestamp"])

    current_price = float(prices[0]["price"]) if prices else 0.0

    points: List[PricePoint] = []

    if prices_asc:
        now_date = now.date()
        last_close: Optional[float] = None
        p_i = 0
        n = len(prices_asc)

        for days_back in range(29, -1, -1):
            day = now_date - timedelta(days=days_back)
            next_day = day + timedelta(days=1)
            day_end_excl = datetime.combine(
                next_day,
                datetime.min.time(),
            )

            while p_i < n and prices_asc[p_i]["timestamp"] < day_end_excl:
                last_close = float(prices_asc[p_i]["price"])
                p_i += 1

            if last_close is None:
                continue

            label = _day_label_utc(
                datetime.combine(day, datetime.min.time())
            )
            is_today = days_back == 0

            if is_today and prediction:
                points.append(
                    PricePoint(
                        label=label,
                        actual=round(last_close, 2),
                        predicted=round(last_close, 2),
                    )
                )
            else:
                points.append(
                    PricePoint(
                        label=label,
                        actual=round(last_close, 2),
                    )
                )

        if prediction:
            if prediction.get("pred_7d") is not None:
                d = now_date + timedelta(days=7)
                points.append(
                    PricePoint(
                        label=_day_label_utc(
                            datetime.combine(d, datetime.min.time())
                        ),
                        predicted=round(float(prediction["pred_7d"]), 2),
                    )
                )

            if prediction.get("pred_14d") is not None:
                d = now_date + timedelta(days=14)
                points.append(
                    PricePoint(
                        label=_day_label_utc(
                            datetime.combine(d, datetime.min.time())
                        ),
                        predicted=round(float(prediction["pred_14d"]), 2),
                    )
                )

            if prediction.get("pred_30d") is not None:
                d = now_date + timedelta(days=30)
                points.append(
                    PricePoint(
                        label=_day_label_utc(
                            datetime.combine(d, datetime.min.time())
                        ),
                        predicted=round(float(prediction["pred_30d"]), 2),
                    )
                )

    else:
        if prediction:
            points.append(
                PricePoint(
                    label=_day_label_utc(now),
                    actual=current_price,
                    predicted=current_price,
                )
            )

            if prediction.get("pred_7d") is not None:
                d = now.date() + timedelta(days=7)
                points.append(
                    PricePoint(
                        label=_day_label_utc(
                            datetime.combine(d, datetime.min.time())
                        ),
                        predicted=round(float(prediction["pred_7d"]), 2),
                    )
                )

            if prediction.get("pred_14d") is not None:
                d = now.date() + timedelta(days=14)
                points.append(
                    PricePoint(
                        label=_day_label_utc(
                            datetime.combine(d, datetime.min.time())
                        ),
                        predicted=round(float(prediction["pred_14d"]), 2),
                    )
                )

            if prediction.get("pred_30d") is not None:
                d = now.date() + timedelta(days=30)
                points.append(
                    PricePoint(
                        label=_day_label_utc(
                            datetime.combine(d, datetime.min.time())
                        ),
                        predicted=round(float(prediction["pred_30d"]), 2),
                    )
                )
        else:
            points.append(
                PricePoint(
                    label=_day_label_utc(now),
                    actual=current_price,
                )
            )

    future_preds = [
        p.predicted
        for p in points
        if p.predicted is not None
    ]
    predicted_best = min(future_preds) if future_preds else current_price

    return PriceHistoryResponse(
        asin=asin,
        chart_title="Price history (30 days, UTC) & forecast",
        current_price=current_price,
        predicted_best_price=round(predicted_best, 2),
        points=points,
    )


def _condition_matches_profile(
    condition: Optional[str],
    profile_condition: str,
) -> bool:
    """Check whether an offer's condition is allowed by the shopper profile."""
    if not condition:
        return False

    normalized = condition.strip().lower()

    if profile_condition == "new_only":
        return normalized == "new"

    if profile_condition == "like_new":
        return normalized in {"new", "like new", "new - open box"}

    if profile_condition == "any_good":
        allowed = (
            "new",
            "like new",
            "very good",
            "good",
        )
        return any(
            normalized == value or normalized.startswith(value + " ")
            for value in allowed
        )

    return False


def _parse_delivery_date(value: Optional[str]) -> Optional[date]:
    """Parse a delivery date only when the supplied value is an actual date."""
    if not value:
        return None

    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _delivery_matches_patience(
    delivery: date,
    patience: str,
) -> bool:
    days_until_delivery = (delivery - date.today()).days

    if days_until_delivery < 0:
        return False

    if patience == "soon":
        return days_until_delivery <= 3

    if patience == "week_or_two":
        return days_until_delivery <= 14

    # "deal" means the shopper is willing to wait.
    return True


def _count_recent_price_drops(prices: list, months: int = 4) -> int:
    """Count recorded price decreases of at least 8% over the recent period."""
    if not prices:
        return 0

    cutoff = datetime.utcnow() - timedelta(days=months * 30)

    recent = [
        p for p in prices
        if p.get("timestamp") and p["timestamp"] >= cutoff
    ]

    recent_asc = sorted(
        recent,
        key=lambda p: p["timestamp"],
    )

    drops = 0
    previous_price = None

    for point in recent_asc:
        price = float(point["price"])

        if (
            previous_price is not None
            and previous_price > 0
            and price <= previous_price * 0.92
        ):
            drops += 1

        previous_price = price

    return drops


def _offer_total_price(offer: DecisionOffer) -> float:
    shipping = float(offer.shipping_cost or 0)
    return round(float(offer.price) + shipping, 2)


def _offer_to_response(
    offer: DecisionOffer,
) -> Optional[DecisionOfferResponse]:
    delivery = _parse_delivery_date(offer.delivery_date)

    if delivery is None:
        return None

    return DecisionOfferResponse(
        seller=offer.seller,
        condition=offer.condition or "Unknown",
        price=_offer_total_price(offer),
        arrives=delivery.isoformat(),
    )


def _find_qualified_alternative(
    request: DecisionRequest,
    amazon_offer: Optional[DecisionOffer],
) -> Optional[tuple]:
    """
    Find a cheaper third-party offer only when the request contains the
    seller-quality and delivery details required by the panel.

    Returns (source_offer, response_offer), or None.
    """
    if amazon_offer is None:
        return None

    amazon_total = _offer_total_price(amazon_offer)
    qualified = []

    for offer in request.offers:
        if offer.pinned:
            continue

        if offer.ships_from_amazon:
            continue

        if not _condition_matches_profile(
            offer.condition,
            request.profile.condition,
        ):
            continue

        delivery = _parse_delivery_date(offer.delivery_date)
        if delivery is None:
            continue

        if not _delivery_matches_patience(
            delivery,
            request.profile.patience,
        ):
            continue

        # Do not claim that a seller is well-rated unless the rating,
        # seller tenure, returns, and warranty information are supplied.
        if offer.stars is None or offer.stars < 4.0:
            continue

        if offer.years is None or offer.years < 1:
            continue

        if offer.free_returns is not True:
            continue

        if offer.full_warranty is not True:
            continue

        total_price = _offer_total_price(offer)

        if total_price >= amazon_total:
            continue

        response_offer = QualifiedDecisionOfferResponse(
            seller=offer.seller,
            condition=offer.condition or "New",
            price=total_price,
            arrives=delivery.isoformat(),
            stars=offer.stars,
            years=offer.years,
            free_returns=offer.free_returns,
            full_warranty=offer.full_warranty,
        )

        qualified.append((offer, response_offer))

    if not qualified:
        return None

    qualified.sort(key=lambda item: item[1].price)
    return qualified[0]


_seed_locks: dict[str, threading.Lock] = {}
_seed_locks_guard = threading.Lock()


def _fetch_and_seed(asin: str) -> None:
    """Fetch and score an unseen ASIN once, even when several endpoints ask at once."""
    with _seed_locks_guard:
        lock = _seed_locks.setdefault(asin, threading.Lock())

    with lock:
        if get_product(asin):
            return

        _fetch_and_seed_unlocked(asin)


def _fetch_and_seed_unlocked(asin: str) -> None:
    """Pull price history from Keepa for an unseen ASIN and score it."""
    try:
        records = keepa_fetch(asin)
        logger.info("Keepa returned %d records for %s", len(records), asin)
    except Exception as exc:
        logger.error("Keepa fetch failed for %s: %s", asin, exc)
        return

    product = get_product(asin)
    if not product:
        return

    _score_and_store(product, asin)


def _score_and_store(product: dict, asin: str) -> None:
    """Run the model on stored price history and save a prediction."""
    prices = db_get_price_history(product["product_id"], limit=1000)

    if not prices:
        return

    try:
        from ml import inference as _ml

        result = _ml.predict_for_asin(prices)

        pred_7d = result["pred_7d"]
        pred_14d = result["pred_14d"]
        pred_30d = result["pred_30d"]
        recommendation = result["recommendation"]
        confidence = result["confidence"]

        logger.info(
            "ML prediction for %s: %s %.0f%%",
            asin,
            recommendation,
            confidence * 100,
        )

    except Exception as ml_exc:
        logger.warning(
            "ML inference failed for %s (%s); falling back to heuristic",
            asin,
            ml_exc,
        )

        current = float(prices[0]["price"])
        avg = sum(float(p["price"]) for p in prices) / len(prices)
        diff = (current - avg) / avg if avg else 0

        if diff > 0.05:
            recommendation = "WAIT"
            confidence = min(0.90, 0.55 + abs(diff))
            pred_7d = round(avg + (current - avg) * 0.6, 2)
            pred_14d = round(avg + (current - avg) * 0.3, 2)
            pred_30d = round(avg, 2)
        else:
            recommendation = "BUY"
            confidence = min(0.90, 0.55 + abs(diff))
            pred_7d = round(current * 1.02, 2)
            pred_14d = round(current * 1.04, 2)
            pred_30d = round(current * 1.06, 2)

    insert_prediction(
        product["product_id"],
        pred_7d,
        pred_14d,
        pred_30d,
        recommendation,
        confidence,
    )

    logger.info(
        "seeded prediction for %s: %s %.0f%%",
        asin,
        recommendation,
        confidence * 100,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        started_at=datetime.utcnow(),
    )


@app.get("/predict/{asin}", response_model=PredictResponse)
def get_prediction(asin: str) -> PredictResponse:
    _require_db()

    product = get_product(asin)

    if not product:
        _fetch_and_seed(asin)
        product = get_product(asin)

    if not product:
        raise HTTPException(status_code=404, detail="not_tracked")

    prediction = get_latest_prediction(product["product_id"])

    if not prediction:
        _score_and_store(product, asin)
        prediction = get_latest_prediction(product["product_id"])

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for {asin}",
        )

    prices = db_get_price_history(product["product_id"], limit=1)
    current_price = float(prices[0]["price"]) if prices else 0.0

    recommendation = prediction["recommendation"]

    pred_7d = (
        float(prediction["pred_7d"])
        if prediction.get("pred_7d") is not None
        else None
    )
    pred_14d = (
        float(prediction["pred_14d"])
        if prediction.get("pred_14d") is not None
        else None
    )
    pred_30d = (
        float(prediction["pred_30d"])
        if prediction.get("pred_30d") is not None
        else None
    )

    confidence = round(
        float(prediction["confidence_score"]) * 100,
        1,
    )

    future_preds = [
        p for p in [pred_7d, pred_14d, pred_30d]
        if p is not None
    ]

    best_pred = min(future_preds) if future_preds else current_price

    potential_savings = (
        max(0.0, round(current_price - best_pred, 2))
        if recommendation == "WAIT"
        else 0.0
    )

    return PredictResponse(
        asin=asin,
        recommendation=RecommendationDirection(recommendation),
        confidence=confidence,
        drop_chance=(
            confidence
            if recommendation == "WAIT"
            else round(100 - confidence, 1)
        ),
        higher_after_wait=(
            _higher_after_wait()
            if recommendation == "WAIT"
            else None
        ),
        predicted_price=round(best_pred, 2),
        potential_savings=potential_savings,
        horizon_days=14,
        why=_generate_why(
            recommendation,
            confidence,
            potential_savings,
            current_price,
            pred_7d,
            pred_14d,
        ),
    )


@app.post("/decision", response_model=DecisionResponse)
def get_decision(request: DecisionRequest) -> DecisionResponse:
    """
    Unified decision endpoint for the browser extension.

    Receives the current page's offers and the shopper's profile.
    The profile is used only for this request and is never persisted.

    This endpoint intentionally uses cached product/prediction data so
    the extension does not wait for a Keepa fetch.
    """
    _require_db()

    product = get_product(request.asin)

    if not product:
        raise HTTPException(
            status_code=404,
            detail="Product is not tracked yet. Load its price history first.",
        )

    prediction = get_latest_prediction(product["product_id"])

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="No cached prediction is available for this product yet.",
        )

    prices = db_get_price_history(product["product_id"], limit=1000)

    if not prices:
        raise HTTPException(
            status_code=404,
            detail="No price history is available for this product.",
        )

    current_price = float(prices[0]["price"])

    # Use the page's pinned offer as the current Amazon offer when available.
    amazon_source = next(
        (offer for offer in request.offers if offer.pinned),
        None,
    )

    amazon_response = (
        _offer_to_response(amazon_source)
        if amazon_source is not None
        else None
    )

    # The model's existing BUY/WAIT prediction is the placeholder decision rule.
    recommendation = str(prediction["recommendation"]).upper()

    confidence = float(prediction["confidence_score"])

    pred_7d = (
        float(prediction["pred_7d"])
        if prediction.get("pred_7d") is not None
        else None
    )

    pred_14d = (
        float(prediction["pred_14d"])
        if prediction.get("pred_14d") is not None
        else None
    )

    pred_30d = (
        float(prediction["pred_30d"])
        if prediction.get("pred_30d") is not None
        else None
    )

    future_predictions = [
        value
        for value in [pred_7d, pred_14d, pred_30d]
        if value is not None
    ]

    expected_low = (
        min(future_predictions)
        if future_predictions
        else current_price
    )

    if recommendation == "WAIT":
        drop_chance = max(0.0, min(1.0, confidence))

        return DecisionResponse(
            show=True,
            status="wait",
            offers_checked=len(request.offers),
            amazon=amazon_response,
            best=None,
            reason=(
                f"The price model recommends waiting. "
                f"It estimates a {drop_chance * 100:.0f}% chance "
                f"of a price drop based on this product's price history."
            ),
            wait=DecisionWaitResponse(
                price=round(current_price, 2),
                expected_low=round(expected_low, 2),
                drop_chance=round(drop_chance, 4),
                drops=_count_recent_price_drops(prices),
                months=4,
            ),
            next_best=None,
        )

    # If the model says BUY, check whether a verifiably qualified alternative
    # seller is cheaper than the pinned Amazon offer.
    alternative = _find_qualified_alternative(
        request,
        amazon_source,
    )

    if alternative is not None:
        source_offer, best_response = alternative

        amazon_price = _offer_total_price(amazon_source)
        savings = round(
            amazon_price - best_response.price,
            2,
        )

        return DecisionResponse(
            show=True,
            status="better_offer",
            offers_checked=len(request.offers),
            amazon=amazon_response,
            best=best_response,
            reason=(
                f"{best_response.seller} offers the same eligible condition "
                f"for ${savings:.2f} less than the Amazon offer. "
                f"The seller has a {best_response.stars:.1f}/5 rating, "
                f"at least one year of history, free returns, and a full warranty."
            ),
            wait=None,
            next_best=None,
        )

    return DecisionResponse(
        show=True,
        status="nothing_better",
        offers_checked=len(request.offers),
        amazon=amazon_response,
        best=None,
        reason=(
            "The price model does not currently recommend waiting, "
            "and no cheaper alternative seller could be verified against "
            "your condition, delivery, and seller-quality preferences."
        ),
        wait=None,
        next_best=None,
    )


@app.get("/product-info/{asin}", response_model=ProductInfoResponse)
def get_product_info(asin: str) -> ProductInfoResponse:
    _require_db()

    product = get_product(asin)

    if not product:
        raise HTTPException(
            status_code=404,
            detail=f"Product {asin} not found",
        )

    prices = db_get_price_history(
        product["product_id"],
        limit=1,
    )

    current_price = float(prices[0]["price"]) if prices else 0.0

    return ProductInfoResponse(
        asin=asin,
        title=product.get("title") or "Unknown Product",
        brand=product.get("brand"),
        category=product.get("category"),
        current_price=current_price,
        currency="USD",
        url=f"https://www.amazon.com/dp/{asin}",
    )


@app.get("/price-history/{asin}", response_model=PriceHistoryResponse)
def get_price_history(asin: str) -> PriceHistoryResponse:
    _require_db()

    product = get_product(asin)

    if not product:
        _fetch_and_seed(asin)
        product = get_product(asin)

    if not product:
        raise HTTPException(
            status_code=404,
            detail="not_tracked",
        )

    prices = db_get_price_history(
        product["product_id"],
        limit=1000,
    )

    prediction = get_latest_prediction(product["product_id"])

    return _bucket_price_history(
        prices,
        prediction,
        asin,
    )


@app.post("/activity", response_model=ActivityResponse)
def log_activity(request: ActivityRequest) -> ActivityResponse:
    _require_db()

    recommendation_shown = _resolve_activity_recommendation(request)

    activity_id = insert_user_activity(
        request.asin,
        recommendation_shown.value,
        request.action.value,
        request.timestamp,
        request.user_id,
    )

    logger.info(
        "activity saved: id=%s asin=%s recommendation=%s action=%s user=%s",
        activity_id,
        request.asin,
        recommendation_shown.value,
        request.action.value,
        request.user_id,
    )

    return ActivityResponse(
        status="saved",
        received_at=datetime.utcnow(),
        details=f"Action '{request.action.value}' on ASIN {request.asin}",
    )


@app.post("/watchlist")
def add_watchlist_item(req: WatchlistAddRequest):
    _require_db()

    try:
        product = get_product(req.asin)

        if not product:
            raise HTTPException(
                status_code=404,
                detail=f"Product {req.asin} not found",
            )

        latest = get_latest_prediction(product["product_id"])

        if not latest:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction found for {req.asin}",
            )

        add_to_watchlist(
            user_id=req.user_id,
            product_id=product["product_id"],
            recommendation_at_add=latest["recommendation"],
            target_price=req.target_price,
        )

        return {
            "status": "added",
            "asin": req.asin,
            "user_id": req.user_id,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )


@app.delete("/watchlist/{user_id}/{product_id}")
def remove_watchlist_item(user_id: str, product_id: int):
    _require_db()

    try:
        remove_from_watchlist(
            user_id=user_id,
            product_id=product_id,
        )

        return {
            "status": "removed",
            "product_id": product_id,
            "user_id": user_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )


@app.get("/watchlist/{user_id}")
def get_user_watchlist(user_id: str):
    _require_db()

    try:
        items = get_watchlist(user_id=user_id)

        return {
            "user_id": user_id,
            "watchlist": items,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )


@app.get("/activity/recent", response_model=RecentActivityResponse)
def get_recent_activity(
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[str] = Query(None),
) -> RecentActivityResponse:
    _require_db()

    rows = get_recent_user_activity(
        limit=limit,
        user_id=user_id,
    )

    items = [
        ActivityRecord(
            activity_id=row["activity_id"],
            asin=row["asin"],
            recommendation_shown=RecommendationDirection(
                row["recommendation_shown"]
            ),
            action=ActivityAction(row["action"]),
            user_id=row.get("user_id"),
            timestamp=row["timestamp"],
        )
        for row in rows
    ]

    return RecentActivityResponse(
        items=items,
        total_count=len(items),
        limit=limit,
    )