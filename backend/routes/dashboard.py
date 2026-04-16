"""Dashboard API routes — aggregate history and stats for the web dashboard (mock data).

Extension endpoints live on the root app in `main.py`; dashboard routes are prefixed
with `/dashboard` and do not require the database.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from schemas.common import RecommendationDirection

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class OutcomeStatus(str, Enum):
    """Whether the eventual price movement matched our recommendation."""

    pending = "pending"
    correct = "correct"
    incorrect = "incorrect"
    partial = "partial"


# ── Summary ─────────────────────────────────────────────────────────────────


class DashboardSummaryResponse(BaseModel):
    """Headline KPIs for the dashboard hero section."""

    user_id: str = Field(..., example="user_demo_001")
    total_predictions: int = Field(..., ge=0, example=47)
    recommendations_followed: int = Field(..., ge=0, example=31)
    estimated_savings_usd: float = Field(..., example=284.5)
    accuracy_when_followed_pct: float = Field(..., ge=0, le=100, example=82.3)
    avg_confidence_when_correct_pct: float = Field(..., ge=0, le=100, example=79.1)
    watchlist_count: int = Field(..., ge=0, example=8)
    last_active_at: datetime = Field(..., example="2026-03-28T14:22:00Z")
    period_label: str = Field(..., example="Last 90 days")


# ── Prediction history ────────────────────────────────────────────────────────


class PredictionHistoryItem(BaseModel):
    prediction_id: str = Field(..., example="pred_20260315_01")
    asin: str = Field(..., example="B08N5WRWNW")
    product_title: str = Field(..., example="Wireless Noise-Cancelling Headphones")
    recommendation: RecommendationDirection = Field(..., example="WAIT")
    confidence_pct: float = Field(..., ge=0, le=100, example=87.0)
    price_at_prediction_usd: float = Field(..., example=199.99)
    predicted_best_price_usd: float = Field(..., example=169.99)
    horizon_days: int = Field(..., ge=1, example=14)
    predicted_at: datetime = Field(..., example="2026-03-01T18:30:00Z")
    outcome: OutcomeStatus = Field(..., example="correct")
    actual_lowest_price_usd: Optional[float] = Field(None, example=164.5)
    outcome_resolved_at: Optional[datetime] = Field(None, example="2026-03-14T09:00:00Z")
    savings_realized_usd: Optional[float] = Field(None, example=35.49)


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    total_count: int = Field(..., ge=0, example=47)
    limit: int = Field(..., ge=1, example=20)
    offset: int = Field(..., ge=0, example=0)


# ── Watchlist ─────────────────────────────────────────────────────────────────


class WatchlistItem(BaseModel):
    asin: str = Field(..., example="B0BSHF7WHW")
    title: str = Field(..., example="USB-C Laptop Docking Station")
    brand: Optional[str] = Field(None, example="Anker")
    category: Optional[str] = Field(None, example="Electronics")
    current_price_usd: float = Field(..., example=89.99)
    currency: str = Field(..., example="USD")
    target_price_usd: Optional[float] = Field(None, example=74.99)
    added_at: datetime = Field(..., example="2026-02-10T12:00:00Z")
    last_price_change_pct: Optional[float] = Field(None, description="vs. price when added", example=-3.2)
    url: str = Field(..., example="https://www.amazon.com/dp/B0BSHF7WHW")


class WatchlistResponse(BaseModel):
    items: List[WatchlistItem]
    total_count: int = Field(..., ge=0, example=8)


# ── Market context (categories / segments user cares about) ─────────────────


class CategoryMarketTrend(BaseModel):
    category: str = Field(..., example="Electronics › Headphones")
    median_price_change_7d_pct: float = Field(..., example=-2.1)
    median_price_change_30d_pct: float = Field(..., example=-5.4)
    deal_intensity_index: float = Field(
        ...,
        ge=0,
        le=100,
        description="Heuristic 0–100: higher = more promotional activity in segment",
        example=62.0,
    )
    sample_product_count: int = Field(..., ge=0, example=128)


class SegmentMover(BaseModel):
    asin: str = Field(..., example="B09V3KXJPB")
    title: str = Field(..., example="Compact Mechanical Keyboard")
    category: str = Field(..., example="Electronics › Keyboards")
    price_change_7d_pct: float = Field(..., example=-11.4)
    current_price_usd: float = Field(..., example=79.99)


class MarketContextResponse(BaseModel):
    """Benchmark-style view around categories the user commonly interacts with."""

    user_top_categories: List[str] = Field(
        ...,
        example=["Electronics", "Computer Accessories", "Travel Gear"],
    )
    category_trends: List[CategoryMarketTrend]
    notable_movers: List[SegmentMover]
    narrative: str = Field(
        ...,
        example=(
            "Deals in your usual electronics categories are slightly stronger than "
            "last month; headphones and docks show the steepest week-over-week drops."
        ),
    )
    as_of: datetime = Field(..., example="2026-04-01T08:00:00Z")


# ── Performance over time (for charts) ──────────────────────────────────────


class PerformancePeriodPoint(BaseModel):
    period_start: datetime = Field(..., example="2026-01-06T00:00:00Z")
    period_end: datetime = Field(..., example="2026-01-12T23:59:59Z")
    predictions_made: int = Field(..., ge=0, example=6)
    predictions_followed: int = Field(..., ge=0, example=4)
    accuracy_pct: Optional[float] = Field(None, ge=0, le=100, example=75.0)
    savings_usd: float = Field(..., example=42.0)


class PerformanceTimeseriesResponse(BaseModel):
    granularity: str = Field(..., example="week")
    points: List[PerformancePeriodPoint]


class OutcomeBreakdownSlice(BaseModel):
    outcome: OutcomeStatus
    count: int = Field(..., ge=0, example=12)
    share_pct: float = Field(..., ge=0, le=100, example=25.5)


class OutcomeBreakdownResponse(BaseModel):
    slices: List[OutcomeBreakdownSlice]
    total_resolved: int = Field(..., ge=0, example=39)


# ── Mock payloads ─────────────────────────────────────────────────────────────

_MOCK_HISTORY: List[PredictionHistoryItem] = [
    PredictionHistoryItem(
        prediction_id="pred_20260328_01",
        asin="B08N5WRWNW",
        product_title="Wireless Noise-Cancelling Headphones",
        recommendation=RecommendationDirection.WAIT,
        confidence_pct=87.0,
        price_at_prediction_usd=199.99,
        predicted_best_price_usd=169.99,
        horizon_days=14,
        predicted_at=datetime(2026, 3, 15, 18, 30, 0),
        outcome=OutcomeStatus.correct,
        actual_lowest_price_usd=164.5,
        outcome_resolved_at=datetime(2026, 3, 28, 9, 0, 0),
        savings_realized_usd=35.49,
    ),
    PredictionHistoryItem(
        prediction_id="pred_20260320_02",
        asin="B0BSHF7WHW",
        product_title="USB-C Laptop Docking Station",
        recommendation=RecommendationDirection.BUY,
        confidence_pct=76.0,
        price_at_prediction_usd=89.99,
        predicted_best_price_usd=92.5,
        horizon_days=14,
        predicted_at=datetime(2026, 3, 10, 12, 15, 0),
        outcome=OutcomeStatus.correct,
        actual_lowest_price_usd=94.0,
        outcome_resolved_at=datetime(2026, 3, 22, 16, 0, 0),
        savings_realized_usd=0.0,
    ),
    PredictionHistoryItem(
        prediction_id="pred_20260305_03",
        asin="B09V3KXJPB",
        product_title="Compact Mechanical Keyboard",
        recommendation=RecommendationDirection.WAIT,
        confidence_pct=71.0,
        price_at_prediction_usd=99.99,
        predicted_best_price_usd=84.0,
        horizon_days=7,
        predicted_at=datetime(2026, 2, 28, 9, 0, 0),
        outcome=OutcomeStatus.incorrect,
        actual_lowest_price_usd=97.5,
        outcome_resolved_at=datetime(2026, 3, 7, 9, 0, 0),
        savings_realized_usd=0.0,
    ),
    PredictionHistoryItem(
        prediction_id="pred_20260401_04",
        asin="B0CQMK4QTP",
        product_title="Portable SSD 1TB",
        recommendation=RecommendationDirection.WAIT,
        confidence_pct=81.0,
        price_at_prediction_usd=129.99,
        predicted_best_price_usd=109.99,
        horizon_days=14,
        predicted_at=datetime(2026, 4, 1, 10, 0, 0),
        outcome=OutcomeStatus.pending,
        actual_lowest_price_usd=None,
        outcome_resolved_at=None,
        savings_realized_usd=None,
    ),
]

_MOCK_WATCHLIST: List[WatchlistItem] = [
    WatchlistItem(
        asin="B0BSHF7WHW",
        title="USB-C Laptop Docking Station",
        brand="Anker",
        category="Electronics",
        current_price_usd=89.99,
        currency="USD",
        target_price_usd=74.99,
        added_at=datetime(2026, 2, 10, 12, 0, 0),
        last_price_change_pct=-3.2,
        url="https://www.amazon.com/dp/B0BSHF7WHW",
    ),
    WatchlistItem(
        asin="B0CQMK4QTP",
        title="Portable SSD 1TB",
        brand="Samsung",
        category="Electronics",
        current_price_usd=129.99,
        currency="USD",
        target_price_usd=109.99,
        added_at=datetime(2026, 3, 22, 8, 30, 0),
        last_price_change_pct=1.5,
        url="https://www.amazon.com/dp/B0CQMK4QTP",
    ),
    WatchlistItem(
        asin="B07ZPKBL9V",
        title="Carry-On Spinner Luggage 21\"",
        brand=None,
        category="Travel Gear",
        current_price_usd=149.0,
        currency="USD",
        target_price_usd=129.0,
        added_at=datetime(2026, 1, 5, 19, 0, 0),
        last_price_change_pct=-6.0,
        url="https://www.amazon.com/dp/B07ZPKBL9V",
    ),
]


# ── Handlers ──────────────────────────────────────────────────────────────────


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary() -> DashboardSummaryResponse:
    return DashboardSummaryResponse(
        user_id="user_demo_001",
        total_predictions=47,
        recommendations_followed=31,
        estimated_savings_usd=284.5,
        accuracy_when_followed_pct=82.3,
        avg_confidence_when_correct_pct=79.1,
        watchlist_count=len(_MOCK_WATCHLIST),
        last_active_at=datetime(2026, 3, 28, 14, 22, 0),
        period_label="Last 90 days",
    )


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> PredictionHistoryResponse:
    total = len(_MOCK_HISTORY)
    slice_ = _MOCK_HISTORY[offset : offset + limit]
    return PredictionHistoryResponse(
        items=slice_,
        total_count=total,
        limit=limit,
        offset=offset,
    )


@router.get("/watchlist", response_model=WatchlistResponse)
def get_watchlist() -> WatchlistResponse:
    return WatchlistResponse(items=_MOCK_WATCHLIST, total_count=len(_MOCK_WATCHLIST))


@router.get("/market", response_model=MarketContextResponse)
def get_market_context() -> MarketContextResponse:
    return MarketContextResponse(
        user_top_categories=["Electronics", "Computer Accessories", "Travel Gear"],
        category_trends=[
            CategoryMarketTrend(
                category="Electronics › Audio",
                median_price_change_7d_pct=-2.1,
                median_price_change_30d_pct=-5.4,
                deal_intensity_index=62.0,
                sample_product_count=128,
            ),
            CategoryMarketTrend(
                category="Electronics › Storage",
                median_price_change_7d_pct=-1.3,
                median_price_change_30d_pct=-3.8,
                deal_intensity_index=55.0,
                sample_product_count=96,
            ),
            CategoryMarketTrend(
                category="Travel › Luggage",
                median_price_change_7d_pct=0.4,
                median_price_change_30d_pct=-2.0,
                deal_intensity_index=48.0,
                sample_product_count=54,
            ),
        ],
        notable_movers=[
            SegmentMover(
                asin="B09V3KXJPB",
                title="Compact Mechanical Keyboard",
                category="Electronics › Keyboards",
                price_change_7d_pct=-11.4,
                current_price_usd=79.99,
            ),
            SegmentMover(
                asin="B0BSHF7WHW",
                title="USB-C Laptop Docking Station",
                category="Electronics › Docks",
                price_change_7d_pct=-3.2,
                current_price_usd=89.99,
            ),
        ],
        narrative=(
            "Deals in your usual electronics categories are slightly stronger than "
            "last month; headphones and docks show the steepest week-over-week drops."
        ),
        as_of=datetime(2026, 4, 1, 8, 0, 0),
    )


@router.get("/performance/timeseries", response_model=PerformanceTimeseriesResponse)
def get_performance_timeseries() -> PerformanceTimeseriesResponse:
    return PerformanceTimeseriesResponse(
        granularity="week",
        points=[
            PerformancePeriodPoint(
                period_start=datetime(2026, 1, 6, 0, 0, 0),
                period_end=datetime(2026, 1, 12, 23, 59, 59),
                predictions_made=6,
                predictions_followed=4,
                accuracy_pct=75.0,
                savings_usd=42.0,
            ),
            PerformancePeriodPoint(
                period_start=datetime(2026, 1, 13, 0, 0, 0),
                period_end=datetime(2026, 1, 19, 23, 59, 59),
                predictions_made=8,
                predictions_followed=5,
                accuracy_pct=80.0,
                savings_usd=61.5,
            ),
            PerformancePeriodPoint(
                period_start=datetime(2026, 1, 20, 0, 0, 0),
                period_end=datetime(2026, 1, 26, 23, 59, 59),
                predictions_made=5,
                predictions_followed=3,
                accuracy_pct=66.7,
                savings_usd=28.0,
            ),
            PerformancePeriodPoint(
                period_start=datetime(2026, 1, 27, 0, 0, 0),
                period_end=datetime(2026, 2, 2, 23, 59, 59),
                predictions_made=7,
                predictions_followed=6,
                accuracy_pct=83.3,
                savings_usd=55.0,
            ),
            PerformancePeriodPoint(
                period_start=datetime(2026, 2, 3, 0, 0, 0),
                period_end=datetime(2026, 2, 9, 23, 59, 59),
                predictions_made=9,
                predictions_followed=7,
                accuracy_pct=85.7,
                savings_usd=98.0,
            ),
        ],
    )


@router.get("/performance/outcomes", response_model=OutcomeBreakdownResponse)
def get_outcome_breakdown() -> OutcomeBreakdownResponse:
    return OutcomeBreakdownResponse(
        slices=[
            OutcomeBreakdownSlice(outcome=OutcomeStatus.correct, count=26, share_pct=66.7),
            OutcomeBreakdownSlice(outcome=OutcomeStatus.incorrect, count=8, share_pct=20.5),
            OutcomeBreakdownSlice(outcome=OutcomeStatus.partial, count=3, share_pct=7.7),
            OutcomeBreakdownSlice(outcome=OutcomeStatus.pending, count=2, share_pct=5.1),
        ],
        total_resolved=37,
    )
