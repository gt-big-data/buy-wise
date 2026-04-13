from dataclasses import dataclass
from backend.ml.volatility import compute_volatility

@dataclass
class ClassificationResult:
    recommendation: str
    confidence: float
    savings: float
    threshold: float
    drop_pct: float
    sufficient_data: bool
    days_to_wait: int   
    earliest_horizon: int   

class PriceRecommendationClassifier:

    """
    Converts multi-horizon future price forecasts into BUY or WAIT
    recommendations.

    Expected inputs:
        - predicted_price_7d
        - predicted_price_14d
        - predicted_price_30d
        - price_history (newest first)

    Returns:
        ClassificationResult with:
            recommendation
            confidence
            savings
            threshold
            drop_pct
            sufficient_data
            days_to_wait
            earliest_horizon
    """
    def __init__(
        self,
        threshold: float = 0.05,
        min_confidence: float = 0.70
    ):
        self.threshold = threshold
        self.min_confidence = min_confidence

    def relative_drop(
        self,
        current_price: float,
        predicted_price: float
    ) -> float:
        if current_price <= 0:
            return 0.0
        return (current_price - predicted_price) / current_price

    def confidence_score(
        self,
        drop_pct: float,
        price_history: list[dict]
    ) -> tuple[float, bool]:
        vol = compute_volatility(price_history, window_days=365)
        margin = drop_pct - self.threshold
        volatility = vol.value if vol.value > 0 else 0.01
        confidence = margin / volatility
        confidence = max(0.0, min(1.0, confidence))
        return round(confidence, 3), vol.sufficient_data

    def classify(
        self,
        predicted_price_7d: float,
        predicted_price_14d: float,
        predicted_price_30d: float,
        price_history: list[dict]
    ) -> ClassificationResult:
        available = [r for r in price_history if r["availability"]]

        if not available:
            return ClassificationResult(
                recommendation="BUY",
                confidence=0.0,
                savings=0.0,
                threshold=self.threshold,
                drop_pct=0.0,
                sufficient_data=False,
                days_to_wait=0,
                earliest_horizon=0
            )

        if available[0]["deal_flag"]:
            return ClassificationResult(
                recommendation="BUY",
                confidence=0.0,
                savings=0.0,
                threshold=self.threshold,
                drop_pct=0.0,
                sufficient_data=False,
                days_to_wait=0,
                earliest_horizon=0
            )

        current_price = float(available[0]["price"])

        horizons = {
            7:  float(predicted_price_7d),
            14: float(predicted_price_14d),
            30: float(predicted_price_30d)
        }

        qualifying = []
        for days, predicted_price in horizons.items():
            drop_pct = self.relative_drop(current_price, predicted_price)
            savings = max(current_price - predicted_price, 0.0)
            confidence, sufficient_data = self.confidence_score(drop_pct, price_history)

            if drop_pct >= self.threshold and confidence >= self.min_confidence:
                qualifying.append({
                    "days": days,
                    "predicted_price": predicted_price,
                    "drop_pct": drop_pct,
                    "savings": savings,
                    "confidence": confidence,
                    "sufficient_data": sufficient_data
                })

        if not qualifying:
            drop_pct = self.relative_drop(current_price, float(predicted_price_7d))
            confidence, sufficient_data = self.confidence_score(drop_pct, price_history)
            return ClassificationResult(
                recommendation="BUY",
                confidence=0.0,
                savings=0.0,
                threshold=self.threshold,
                drop_pct=round(drop_pct, 4),
                sufficient_data=sufficient_data,
                days_to_wait=0,
                earliest_horizon=0
            )

        best = max(qualifying, key=lambda x: x["savings"])

        earliest = min(qualifying, key=lambda x: x["days"])

        return ClassificationResult(
            recommendation="WAIT",
            confidence=best["confidence"],
            savings=round(best["savings"], 2),
            threshold=self.threshold,
            drop_pct=round(best["drop_pct"], 4),
            sufficient_data=best["sufficient_data"],
            days_to_wait=best["days"],
            earliest_horizon=earliest["days"]
        )


""'Test""'
if __name__ == "__main__":
    clf = PriceRecommendationClassifier()

    sample_history = [
        {
            "price": 100.0,
            "availability": True,
            "deal_flag": False,
            "timestamp": "2025-01-01"
        },
        {
            "price": 98.0,
            "availability": True,
            "deal_flag": False,
            "timestamp": "2024-12-01"
        }
    ]

    result = clf.classify(
        predicted_price_7d=97.0,
        predicted_price_14d=94.0,
        predicted_price_30d=90.0,
        price_history=sample_history
    )

    print(result)
        