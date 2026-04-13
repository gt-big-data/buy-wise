from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score

from backend.ml.classifier import PriceRecommendationClassifier


def optimize_threshold(
    validation_rows: List[Dict],
    thresholds: List[float] = [0.03, 0.05, 0.07, 0.10],
    min_precision: float = 0.75,
    evaluation_horizon: int = 14
) -> Dict:
    """
    Optimize WAIT threshold using walk-forward validation snapshots.

    Args:
        validation_rows:
            List of historical Keepa validation snapshots.
            Each row must contain:
                - price_history
                - predicted_price_7d
                - predicted_price_14d
                - predicted_price_30d
                - actual_price_7d
                - actual_price_14d
                - actual_price_30d

        thresholds:
            Candidate relative drop thresholds.

        min_precision:
            Minimum precision constraint to preserve user trust.

        evaluation_horizon:
            Which realized horizon to optimize against.
            Default = 14 (recommended MVP primary horizon)

    Returns:
        Dict with:
            - best_threshold
            - best_metrics
            - all_results
    """
    results = []

    horizon_key = f"actual_price_{evaluation_horizon}d"

    for threshold in thresholds:
        clf = PriceRecommendationClassifier(threshold=threshold)

        y_true = []
        y_pred = []

        for row in validation_rows:
            price_history = row["price_history"]

            available = [r for r in price_history if r["availability"]]
            if not available:
                continue

            current_price = float(available[0]["price"])

            pred_result = clf.classify(
                predicted_price_7d=row["predicted_price_7d"],
                predicted_price_14d=row["predicted_price_14d"],
                predicted_price_30d=row["predicted_price_30d"],
                price_history=price_history
            )

            actual_future_price = float(row[horizon_key])

            actual_drop_pct = (
                (current_price - actual_future_price) / current_price
                if current_price > 0 else 0.0
            )

            y_true.append(actual_drop_pct >= threshold)
            y_pred.append(pred_result.recommendation == "WAIT")

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        })

    valid = [r for r in results if r["precision"] >= min_precision]

    best = (
        max(valid, key=lambda x: x["f1"])
        if valid else max(results, key=lambda x: x["f1"])
    )

    return {
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "all_results": results
    }


