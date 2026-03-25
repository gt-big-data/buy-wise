from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="BuyWise Backend Mock API",
    description="Mock backend skeleton for the Chrome extension.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    started_at: datetime = Field(..., example="2026-03-16T00:00:00Z")


class RecommendationDirection(str, Enum):
    buy = "buy"
    wait = "wait"

class PredictResponse(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    recommendation: RecommendationDirection = Field(..., example="wait")
    confidence: float = Field(..., ge=0, le=1, example=0.87)
    predicted_price: float = Field(..., example=169.99)
    potential_savings: float = Field(..., example=30.00)
    horizon_days: int = Field(..., ge=1, example=14)

class ProductInfoResponse(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    title: str = Field(..., example="Wireless Noise-Cancelling Headphones")
    brand: str = Field(..., example="BuyWise")
    category: str = Field(..., example="Electronics")
    current_price: float = Field(..., example=199.99)
    currency: str = Field(..., example="USD")
    url: str = Field(..., example="https://www.amazon.com/dp/B08N5WRWNW")
    rating: float = Field(..., ge=0, le=5, example=4.5)
    review_count: int = Field(..., example=1234)


class ActivityAction(str, Enum):
    purchased = "purchased"
    dismissed = "waited"


class ActivityRequest(BaseModel):
    asin: str = Field(..., example="B08N5WRWNW")
    action: ActivityAction = Field(..., example="waited")
    user_id: Optional[str] = Field(None, example="user_123")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = Field(None, example={"price": 199.99})


class ActivityResponse(BaseModel):
    status: str = Field(..., example="logged")
    received_at: datetime = Field(..., example="2026-03-16T00:00:00Z")
    details: Optional[str] = Field(None, example="Mock event recorded")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", started_at=datetime.utcnow())


@app.get("/predict/{asin}", response_model=PredictResponse)
def get_prediction(asin: str):
    if not asin:
        raise HTTPException(status_code=400, detail="ASIN is required")

    return PredictResponse(
        asin=asin,
        recommendation=RecommendationDirection.wait,
        confidence=0.79,
        predicted_price=159.99,
        potential_savings=40.00,
        horizon_days=14,
    )


@app.get("/product-info/{asin}", response_model=ProductInfoResponse)
def get_product_info(asin: str):
    if not asin:
        raise HTTPException(status_code=400, detail="ASIN is required")

    return ProductInfoResponse(
        asin=asin,
        title="Mock Product Title",
        brand="BuyWise Brand",
        category="Electronics",
        current_price=199.99,
        currency="USD",
        url=f"https://www.amazon.com/dp/{asin}",
        rating=4.4,
        review_count=875,
    )


@app.post("/activity", response_model=ActivityResponse)
def log_activity(request: ActivityRequest):
    #Here we should have 
    return ActivityResponse(
        status="logged",
        received_at=datetime.utcnow(),
        details=f"Activity received for ASIN {request.asin} with action {request.action}",
    )

