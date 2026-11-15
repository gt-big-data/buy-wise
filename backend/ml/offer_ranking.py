"""Offer ranking, risk filtering, and alternative savings analysis (Study 009).

Evaluates offers, applies dynamic relative score clustering, enforces condition 
preferences, and performs hard risk checks.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from ml.evaluate import STUDIES


def filter_by_condition_preference(condition_str: str, preference: str) -> bool:
    """Filters offers based on the user's explicit profile condition preference.
    
    Options:
    - 'new or like new only': Accepts only New or Like New / Open Box.
    - 'new or barely used': Accepts New, Like New, or Very Good.
    - 'any acceptable condition': Accepts any condition.
    """
    if not isinstance(condition_str, str):
        cond_lower = "new"
    else:
        cond_lower = condition_str.strip().lower()
        
    is_new = "new" in cond_lower and "used" not in cond_lower and "refurbished" not in cond_lower
    is_like_new = "like new" in cond_lower or "open box" in cond_lower
    is_very_good = "very good" in cond_lower
    
    if preference == "new or like new only":
        return is_new or is_like_new
    elif preference == "new or barely used":
        return is_new or is_like_new or is_very_good
    elif preference == "any acceptable condition":
        return True
    else:
        return True  # Default fallback


def check_offer_risk(
    feedback_score: float, 
    rating_count: int, 
    min_feedback: float = 90.0, 
    min_ratings: int = 5
) -> Dict[str, bool]:
    """Applies hard risk filters and returns boolean test results."""
    feedback_ok = pd.isna(feedback_score) or (feedback_score >= min_feedback)
    ratings_ok = pd.isna(rating_count) or (rating_count >= min_ratings)
    passed_all = bool(feedback_ok and ratings_ok)
    
    return {
        "feedback_ok": feedback_ok,
        "ratings_ok": ratings_ok,
        "passed_all": passed_all
    }


def compute_offer_score(
    price: float,
    baseline_price: float,
    feedback_score: float,
    rating_count: int,
    is_fba: bool,
    product_purchases: int
) -> float:
    """Computes a standalone raw composite score for ranking offers."""
    if pd.isna(price) or pd.isna(baseline_price) or baseline_price <= 0:
        return 0.0
        
    savings_ratio = max(0.0, (baseline_price - price) / baseline_price)
    fb_norm = (feedback_score / 100.0) if not pd.isna(feedback_score) else 0.9
    rating_weight = min(1.0, np.log1p(rating_count if not pd.isna(rating_count) else 1) / np.log1p(100))
    fba_bonus = 0.1 if is_fba else 0.0
    popularity_weight = min(1.0, np.log1p(product_purchases if not pd.isna(product_purchases) else 1) / np.log1p(500))
    
    score = (
        (savings_ratio * 0.50) +
        (fb_norm * 0.20 * rating_weight) +
        (fba_bonus * 0.15) +
        (popularity_weight * 0.15)
    ) * 100.0
    
    return round(score, 2)


GREAT_SAVINGS = 0.10   # >= 10% below the product's typical price
OKAY_SAVINGS = -0.03   # within 3% above typical still counts as okay


def assign_dynamic_tiers(scored_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tiers each offer by its savings against its own typical price.

    Absolute, not relative to the page: ranking within a page always made the
    top offer "great", so the winner's label carried no information. Each
    candidate carries a `reference_price` for its condition (new vs used), so
    used offers aren't "great" just for being used. No history -> "okay".
    """
    for c in scored_candidates:
        if not c["passed_tests"]["passed_all"]:
            c["tier"] = "bad offer"
            continue
        reference_price = c.get("reference_price")
        if reference_price is None or pd.isna(reference_price) or reference_price <= 0:
            c["tier"] = "okay offer"
            continue

        savings = (reference_price - c["price"]) / reference_price
        if savings >= GREAT_SAVINGS:
            c["tier"] = "great offer"
        elif savings >= OKAY_SAVINGS:
            c["tier"] = "okay offer"
        else:
            c["tier"] = "bad offer"

    return scored_candidates

def evaluate_single_product_offers(
    row: pd.Series, 
    user_preferred_condition: str = "any acceptable condition"
) -> Dict[str, Any]:
    """Evaluates offers for a single product row with dynamic clustering and fallbacks."""
    asin = row.get("asin", "UNKNOWN")
    amazon_price = row.get("amazon")
    new_price = row.get("new")
    used_price = row.get("used")
    
    baseline_source = "Amazon (Add to Cart)"
    # Typical prices from earlier observations only (see add_typical_prices).
    typical_new = row.get("typical_new")
    typical_used = row.get("typical_used")

    # Fallback if official Amazon Buy Box price is missing
    if pd.isna(amazon_price):
        if not pd.isna(new_price):
            amazon_price = new_price
            baseline_source = "Lowest New (Fallback Baseline)"
        elif not pd.isna(used_price):
            amazon_price = used_price
            baseline_source = "Lowest Used (Fallback Baseline)"
        else:
            return {
                "asin": asin,
                "status": "reader_failure",
                "winning_source": "None",
                "winning_price": None,
                "category": "bad offer",
                "reason": "Reader failed: missing baseline price and all alternative offers."
            }
        
    candidates = [{
        "source": baseline_source,
        "price": amazon_price,
        "condition": "New",
        "feedback_score": 100.0,
        "rating_count": 1000,
        "is_fba": True,
        "product_purchases": row.get("purchases", 50),
        "reference_price": typical_used if baseline_source.startswith("Lowest Used") else typical_new,
        "reason": f"Baseline reference ({baseline_source})"
    }]
    
    # Only add alternative New if it's different from the baseline
    if not pd.isna(new_price) and new_price != amazon_price:
        candidates.append({
            "source": "Alternative New Seller",
            "price": new_price,
            "condition": "New",
            "feedback_score": row.get("new_feedback", 95.0),
            "rating_count": row.get("new_ratings", 50),
            "is_fba": row.get("new_fba", True),
            "product_purchases": row.get("purchases", 50),
            "reference_price": typical_new,
            "reason": f"New offer priced at ${new_price:.2f}"
        })
        
    if not pd.isna(used_price) and used_price != amazon_price:
        cond_str = row.get("used_condition", "Used - Very Good")
        candidates.append({
            "source": "Used Offer",
            "price": used_price,
            "condition": cond_str,
            "feedback_score": row.get("used_feedback", 92.0),
            "rating_count": row.get("used_ratings", 20),
            "is_fba": row.get("used_fba", False),
            "product_purchases": row.get("purchases", 50),
            "reference_price": typical_used,
            "reason": f"({cond_str}) offer priced at ${used_price:.2f}"
        })

    if len(candidates) == 1:
        only = assign_dynamic_tiers([{
            **candidates[0],
            "passed_tests": check_offer_risk(candidates[0]["feedback_score"], candidates[0]["rating_count"]),
        }])[0]
        return {
            "asin": asin,
            "status": "single_offer",
            "winning_source": baseline_source,
            "winning_price": amazon_price,
            "category": only["tier"],
            "passed_tests": only["passed_tests"],
            "reason": "Only one valid offer available on page."
        }
        
    scored_candidates = []
    for c in candidates:
        if not filter_by_condition_preference(c["condition"], user_preferred_condition):
            continue
            
        tests = check_offer_risk(c["feedback_score"], c["rating_count"])
        score = compute_offer_score(
            price=c["price"],
            baseline_price=amazon_prices_ref if 'amazon_prices_ref' in locals() else amazon_price,
            feedback_score=c["feedback_score"],
            rating_count=c["rating_count"],
            is_fba=c["is_fba"],
            product_purchases=c["product_purchases"]
        )
        
        scored_candidates.append({
            **c,
            "score": score,
            "passed_tests": tests
        })
        
    if not scored_candidates:
        scored_candidates = [{
            **candidates[0],
            "score": 0.0,
            "tier": "okay offer",
            "passed_tests": {"passed_all": True},
            "reason": "All alternative offers filtered out by user condition preference."
        }]
    else:
        scored_candidates = assign_dynamic_tiers(scored_candidates)
        
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    winner = scored_candidates[0]
    
    return {
        "asin": asin,
        "status": "success",
        "winning_source": winner["source"],
        "winning_price": winner["price"],
        "winning_condition": winner["condition"],
        "score": winner["score"],
        "category": winner.get("tier", "okay offer"),
        "passed_tests": winner["passed_tests"],
        "reason": winner["reason"]
    }

def export_results_to_csv(output_filename: str = "studies/offer_ranking_results.csv", user_preference: str = "any acceptable condition") -> Path:
    """Runs batch evaluation and saves the resulting dataframe to a CSV file."""
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_df = run_batch_evaluation(user_preference=user_preference)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Exported {len(result_df):,} evaluated rows to {output_path}")
    return output_path


TYPICAL_WINDOW = 14  # previous observations averaged into the "typical" price


def add_typical_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Adds typical_new / typical_used: per-ASIN mean of the previous
    TYPICAL_WINDOW observations, excluding the current row.

    The export's amazon_ma_* columns include the current price, so comparing
    a price against them partly compares it with itself (lookahead).
    """
    df = df.sort_values(["asin", "datetime"]).reset_index(drop=True)
    by_asin = df.groupby("asin")
    trailing = lambda s: s.shift(1).rolling(TYPICAL_WINDOW, min_periods=1).mean()
    df["typical_new"] = by_asin["amazon"].transform(trailing)
    df["typical_used"] = by_asin["used"].transform(trailing)
    return df


def run_batch_evaluation(prices_path: Path = STUDIES / "Prices.csv", user_preference: str = "any acceptable condition") -> pd.DataFrame:
    print(f"Running batch offer ranking on {prices_path} (Preference: {user_preference})...")
    # Read the raw export: load_prices renames amazon/used and drops `new`,
    # which this module needs under their original names.
    df = pd.read_csv(prices_path)
    
    # Normalize column names to lowercase to match lookups
    df.columns = df.columns.str.lower().str.strip()
    df = add_typical_prices(df)

    results = []
    for _, row in df.iterrows():
        res = evaluate_single_product_offers(row, user_preferred_condition=user_preference)
        results.append(res)
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    summary_df = run_batch_evaluation()
    print("\nBatch Evaluation Summary Preview:")
    print(summary_df["category"].value_counts().to_string())
    export_results_to_csv("../studies/offer_rankings/output.csv", "new or like new only")