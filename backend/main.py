from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from routes.dashboard import router as dashboard_router
from schemas.common import RecommendationDirection

logger = logging.getLogger(__name__)

# Lazy DB import — server stays up even if DB is unavailable at startup
try:
    from db.connection import (
        get_product,
        get_latest_prediction,
        get_price_history as db_get_price_history,
        insert_prediction,
    )
    from jobs.keepa_fetch import fetch_price_history as keepa_fetch
    _DB_AVAILABLE = True
except Exception as _exc:
    logger.warning("DB unavailable at startup: %s", _exc)
    _DB_AVAILABLE = False

app = FastAPI(
    title="BuyWise API",
    description="Backend for the BuyWise Chrome extension and web dashboard.",
    version="0.1.0",
)

app.include_router(dashboard_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chrome extension content scripts run under the page origin
    allow_methods=["GET", "POST"],
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


# ── Models ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    started_at: datetime = Field(..., example="2026-03-16T00:00:00Z")


class PredictResponse(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    recommendation: RecommendationDirection = Field(..., example="WAIT")
    confidence: float = Field(..., ge=0, le=100, description="0–100", example=87.0)
    predicted_price: float = Field(..., example=169.99)
    potential_savings: float = Field(..., example=30.00)
    horizon_days: int = Field(..., ge=1, example=30)
    why: str = Field(..., example="Our model is 87% confident this item will drop in price.")


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


class ActivityRequest(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    action: ActivityAction = Field(..., example="dismissed")
    user_id: Optional[str] = Field(None, example="user_123")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = Field(None, example={"price": 199.99})


class ActivityResponse(BaseModel):
    status: str = Field(..., example="logged")
    received_at: datetime = Field(..., example="2026-03-16T00:00:00Z")
    details: Optional[str] = Field(None)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _require_db() -> None:
    if not _DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")


def _generate_why(
    recommendation: str, confidence: float, potential_savings: float, horizon_days: int
) -> str:
    pct = int(round(confidence))
    if recommendation == "WAIT":
        return (
            f"Our model is {pct}% confident this item will drop in price within "
            f"{horizon_days} days, suggesting potential savings of ${potential_savings:.0f}. "
            f"Recent price movement and historical trends support waiting."
        )
    return (
        f"Our model is {pct}% confident this is a good time to buy. "
        f"Based on recent price history, the current price is competitive "
        f"and unlikely to drop significantly in the next {horizon_days} days."
    )


def _bucket_price_history(
    prices: list, prediction: Optional[dict], asin: str
) -> PriceHistoryResponse:
    now = datetime.utcnow()
    prices_asc = list(reversed(prices))  # prices come in DESC from DB

    points: List[PricePoint] = []
    for weeks_back in range(4, 0, -1):
        start = now - timedelta(weeks=weeks_back)
        end = now - timedelta(weeks=weeks_back - 1)
        bucket = [
            float(p["price"]) for p in prices_asc
            if start <= p["timestamp"] < end
        ]
        if bucket:
            avg = round(sum(bucket) / len(bucket), 2)
            points.append(PricePoint(label=f"-{weeks_back}w", actual=avg))

    current_price = float(prices[0]["price"]) if prices else 0.0

    if prediction:
        points.append(PricePoint(label="Today", actual=current_price, predicted=current_price))
        if prediction.get("pred_7d") is not None:
            points.append(PricePoint(label="+1w", predicted=float(prediction["pred_7d"])))
        if prediction.get("pred_14d") is not None:
            points.append(PricePoint(label="+2w", predicted=float(prediction["pred_14d"])))
        if prediction.get("pred_30d") is not None:
            points.append(PricePoint(label="+1m", predicted=float(prediction["pred_30d"])))
    else:
        points.append(PricePoint(label="Today", actual=current_price))

    future_preds = [p.predicted for p in points if p.predicted is not None]
    predicted_best = min(future_preds) if future_preds else current_price

    return PriceHistoryResponse(
        asin=asin,
        chart_title="Price history & 30-day forecast",
        current_price=current_price,
        predicted_best_price=round(predicted_best, 2),
        points=points,
    )


def _fetch_and_seed(asin: str) -> None:
    """Pull price history from Keepa for an unseen ASIN and generate a stub prediction.

    Called the first time any endpoint is hit for an ASIN not yet in the DB.
    Keepa populates products + prices. We then compute a simple trend-based
    prediction so the extension has something to show immediately. The real ML
    model will overwrite this row once it runs.
    """
    try:
        records = keepa_fetch(asin)
        logger.info("Keepa returned %d records for %s", len(records), asin)
    except Exception as exc:
        logger.error("Keepa fetch failed for %s: %s", asin, exc)
        return

    product = get_product(asin)
    if not product:
        return

    prices = db_get_price_history(product["product_id"], limit=100)
    if not prices:
        return

    current = float(prices[0]["price"])
    avg = sum(float(p["price"]) for p in prices) / len(prices)
    diff = (current - avg) / avg if avg else 0

    if diff > 0.05:
        # Price is elevated — likely to come down
        recommendation = "WAIT"
        confidence = min(0.90, 0.55 + abs(diff))
        pred_7d  = round(avg + (current - avg) * 0.6, 2)
        pred_14d = round(avg + (current - avg) * 0.3, 2)
        pred_30d = round(avg, 2)
    else:
        # Price is at or below average — good time to buy
        recommendation = "BUY"
        confidence = min(0.90, 0.55 + abs(diff))
        pred_7d  = round(current * 1.02, 2)
        pred_14d = round(current * 1.04, 2)
        pred_30d = round(current * 1.06, 2)

    insert_prediction(
        product["product_id"], pred_7d, pred_14d, pred_30d, recommendation, confidence
    )
    logger.info("seeded prediction for %s: %s %.0f%%", asin, recommendation, confidence * 100)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", started_at=datetime.utcnow())


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
        raise HTTPException(status_code=404, detail=f"No prediction found for {asin}")

    prices = db_get_price_history(product["product_id"], limit=1)
    current_price = float(prices[0]["price"]) if prices else 0.0

    recommendation = prediction["recommendation"]  # "BUY" or "WAIT" from DB enum
    confidence = round(float(prediction["confidence_score"]) * 100, 1)

    pred_values = [
        float(prediction[k])
        for k in ("pred_7d", "pred_14d", "pred_30d")
        if prediction.get(k) is not None
    ]
    best_pred = min(pred_values) if pred_values else current_price
    potential_savings = max(0.0, round(current_price - best_pred, 2))
    horizon_days = 30

    return PredictResponse(
        asin=asin,
        recommendation=RecommendationDirection(recommendation),
        confidence=confidence,
        predicted_price=round(best_pred, 2),
        potential_savings=potential_savings,
        horizon_days=horizon_days,
        why=_generate_why(recommendation, confidence, potential_savings, horizon_days),
    )


@app.get("/product-info/{asin}", response_model=ProductInfoResponse)
def get_product_info(asin: str) -> ProductInfoResponse:
    _require_db()

    product = get_product(asin)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {asin} not found")

    prices = db_get_price_history(product["product_id"], limit=1)
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
        raise HTTPException(status_code=404, detail="not_tracked")

    prices = db_get_price_history(product["product_id"], limit=100)
    prediction = get_latest_prediction(product["product_id"])

    return _bucket_price_history(prices, prediction, asin)


@app.post("/activity", response_model=ActivityResponse)
def log_activity(request: ActivityRequest) -> ActivityResponse:
    logger.info(
        "activity: asin=%s action=%s user=%s",
        request.asin, request.action, request.user_id,
    )
    return ActivityResponse(
        status="logged",
        received_at=datetime.utcnow(),
        details=f"Action '{request.action}' on ASIN {request.asin}",
    )
