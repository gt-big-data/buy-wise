import pytest
from ml.classifier import PriceRecommendationClassifier, ClassificationResult
from ml.volatility import compute_volatility, VolatilityResult
from ml.threshold_optimizer import optimize_threshold


def make_price_history(prices: list[float], available: bool = True) -> list[dict]:
    """Helper to build price history dicts matching get_price_history() format."""
    from datetime import datetime, timedelta
    rows = []
    for i, price in enumerate(reversed(prices)):  # newest first
        rows.append({
            "price": price,
            "timestamp": datetime.now() - timedelta(days=i),
            "availability": available,
            "deal_flag": False
        })
    return rows


class TestComputeVolatility:
    def test_stable_product(self):
        """Product with flat pricing should have near-zero volatility."""
        history = make_price_history([100.0] * 30)
        result = compute_volatility(history, window_days=30)
        assert result.value == 0.0
        assert result.sufficient_data is True

    def test_volatile_product(self):
        """Product with large swings should have high volatility."""
        prices = [100.0, 80.0, 120.0, 75.0, 130.0] * 6
        history = make_price_history(prices)
        result = compute_volatility(history, window_days=30)
        assert result.value > 0.1

    def test_insufficient_data(self):
        """Single price point should return fallback volatility."""
        history = make_price_history([99.99])
        result = compute_volatility(history, window_days=90)
        assert result.value == 0.05
        assert result.sufficient_data is False

    def test_filters_unavailable_prices(self):
        """Unavailable prices should be excluded from calculation."""
        history = make_price_history([100.0] * 10, available=False)
        result = compute_volatility(history, window_days=10)
        assert result.sufficient_data is False

    def test_decimal_prices(self):
        """Should handle Decimal type from mysql.connector."""
        from decimal import Decimal
        history = [
            {"price": Decimal("149.99"), "timestamp": None, "availability": True, "deal_flag": False},
            {"price": Decimal("139.99"), "timestamp": None, "availability": True, "deal_flag": False},
            {"price": Decimal("159.99"), "timestamp": None, "availability": True, "deal_flag": False},
        ]
        result = compute_volatility(history, window_days=90)
        assert isinstance(result.value, float)


class TestPriceRecommendationClassifier:
    def setup_method(self):
        self.clf = PriceRecommendationClassifier(threshold=0.05, min_confidence=0.75)

    def test_clear_wait_recommendation(self):
        """Large drop on stable product should return WAIT."""
        history = make_price_history([100.0] * 90)
        result = self.clf.classify(
            predicted_price_7d=98.0,
            predicted_price_14d=85.0,
            predicted_price_30d=80.0,
            price_history=history
        )
        assert result.recommendation == "WAIT"
        assert result.savings > 0
        assert result.confidence > 0.75
        assert result.days_to_wait > 0

    def test_max_savings_horizon_selected(self):
        """Should pick 30d over 14d if 30d saves more."""
        history = make_price_history([100.0] * 90)
        result = self.clf.classify(
            predicted_price_7d=96.0,
            predicted_price_14d=88.0,
            predicted_price_30d=82.0,
            price_history=history
        )
        assert result.days_to_wait == 30
        assert result.savings == 18.0

    def test_earliest_horizon_tracked(self):
        """earliest_horizon should be smallest qualifying days."""
        history = make_price_history([100.0] * 90)
        result = self.clf.classify(
            predicted_price_7d=93.0,
            predicted_price_14d=88.0,
            predicted_price_30d=82.0,
            price_history=history
        )
        assert result.earliest_horizon == 7

    def test_small_drop_returns_buy(self):
        """Drop below threshold should return BUY."""
        history = make_price_history([100.0] * 90)
        result = self.clf.classify(
            predicted_price_7d=98.0,
            predicted_price_14d=97.0,
            predicted_price_30d=96.0,
            price_history=history
        )
        assert result.recommendation == "BUY"
        assert result.confidence == 0.0

    def test_volatile_product_returns_buy(self):
        """High volatility should suppress WAIT even with large drop."""
        prices = [100.0, 60.0, 140.0, 55.0, 145.0] * 18
        history = make_price_history(prices)
        result = self.clf.classify(
            predicted_price_7d=98.0,
            predicted_price_14d=88.0,
            predicted_price_30d=82.0,
            price_history=history
        )
        assert result.recommendation == "BUY"

    def test_deal_flag_returns_buy(self):
        """If current price already has deal_flag, should return BUY."""
        history = make_price_history([100.0] * 90)
        history[0]["deal_flag"] = True
        result = self.clf.classify(
            predicted_price_7d=85.0,
            predicted_price_14d=80.0,
            predicted_price_30d=75.0,
            price_history=history
        )
        assert result.recommendation == "BUY"

    def test_unavailable_product_returns_buy(self):
        """No available prices should return BUY."""
        history = make_price_history([100.0] * 10, available=False)
        result = self.clf.classify(
            predicted_price_7d=85.0,
            predicted_price_14d=80.0,
            predicted_price_30d=75.0,
            price_history=history
        )
        assert result.recommendation == "BUY"

    def test_returns_classification_result(self):
        """classify() should always return ClassificationResult."""
        history = make_price_history([100.0] * 90)
        result = self.clf.classify(
            predicted_price_7d=85.0,
            predicted_price_14d=80.0,
            predicted_price_30d=75.0,
            price_history=history
        )
        assert isinstance(result, ClassificationResult)


class TestOptimizeThreshold:
    def setup_method(self):
        self.sample_rows = [
            {"current_price": 100, "predicted_price": 88, "actual_future_price": 85},
            {"current_price": 200, "predicted_price": 198, "actual_future_price": 199},
            {"current_price": 150, "predicted_price": 130, "actual_future_price": 128},
            {"current_price": 80,  "predicted_price": 78,  "actual_future_price": 79},
            {"current_price": 120, "predicted_price": 105, "actual_future_price": 102},
            {"current_price": 300, "predicted_price": 270, "actual_future_price": 265},
            {"current_price": 50,  "predicted_price": 49,  "actual_future_price": 48},
            {"current_price": 90,  "predicted_price": 80,  "actual_future_price": 82},
        ]

    def test_returns_best_threshold(self):
        """Should return a threshold from the tested values."""
        result = optimize_threshold(self.sample_rows)
        assert result["best_threshold"] in [0.03, 0.05, 0.07, 0.10]

    def test_all_thresholds_evaluated(self):
        """Should evaluate all four thresholds."""
        result = optimize_threshold(self.sample_rows)
        thresholds = [r["threshold"] for r in result["all_results"]]
        assert set(thresholds) == {0.03, 0.05, 0.07, 0.10}

    def test_precision_guard(self):
        """Best threshold should meet min_precision requirement."""
        result = optimize_threshold(self.sample_rows, min_precision=0.75)
        best = next(
            r for r in result["all_results"]
            if r["threshold"] == result["best_threshold"]
        )
        assert best["precision"] >= 0.75 or all(
            r["precision"] < 0.75 for r in result["all_results"]
        )

    def test_metrics_in_results(self):
        """Each threshold result should have precision, recall, f1."""
        result = optimize_threshold(self.sample_rows)
        for r in result["all_results"]:
            assert "precision" in r
            assert "recall" in r
            assert "f1" in r