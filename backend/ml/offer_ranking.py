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
    """Applies hard risk filters and returns boolean test results.

    A missing signal fails: we can't explain why an unknown seller is safe.
    Amazon-run sellers get their signals from `amazon_signals` first.
    """
    feedback_ok = not pd.isna(feedback_score) and feedback_score >= min_feedback
    ratings_ok = not pd.isna(rating_count) and rating_count >= min_ratings
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


def is_amazon_run(seller: Any) -> bool:
    """Amazon.com, Amazon Resale, Amazon Warehouse, Amazon <country>."""
    return isinstance(seller, str) and seller.strip().lower().startswith("amazon")


def amazon_signals(offer: Dict[str, Any]) -> Dict[str, Any]:
    """Amazon's own offers show no seller feedback or rating count on the page.
    Treat them as fully trusted instead of as an unrated seller."""
    if not is_amazon_run(offer.get("source")):
        return offer
    return {
        **offer,
        "feedback_score": 100.0 if pd.isna(offer.get("feedback_score")) else offer["feedback_score"],
        "rating_count": 1000 if pd.isna(offer.get("rating_count")) else offer["rating_count"],
    }


def rank_offers(
    offers: List[Dict[str, Any]],
    typical_new: float,
    typical_used: float,
    user_preferred_condition: str = "any acceptable condition",
) -> List[Dict[str, Any]]:
    """Ranks any number of offers on one page, best first.

    Each offer: source (seller name), price (total incl. shipping), condition,
    feedback_score, rating_count, is_fba; optional is_atc, product_purchases,
    reference_price, reason. Savings are measured against the Add to Cart
    offer (else the cheapest new, else the cheapest offer). Offers that fail
    the risk checks rank below every offer that passes. Returns [] if the
    condition preference filters out everything.
    """
    if not offers:
        return []
    atc = [o["price"] for o in offers if o.get("is_atc")]
    new = [o["price"] for o in offers if str(o.get("condition", "")).strip().lower() == "new"]
    baseline = atc[0] if atc else min(new) if new else min(o["price"] for o in offers)

    ranked = []
    for o in map(amazon_signals, offers):
        if not filter_by_condition_preference(o.get("condition"), user_preferred_condition):
            continue
        is_new = str(o.get("condition", "")).strip().lower() == "new"
        ranked.append({
            **o,
            "reference_price": o.get("reference_price", typical_new if is_new else typical_used),
            "reason": o.get("reason") or f"{o.get('source')}: {o.get('condition')}, ${o['price']:.2f}",
            "score": compute_offer_score(
                price=o["price"],
                baseline_price=baseline,
                feedback_score=o.get("feedback_score"),
                rating_count=o.get("rating_count"),
                is_fba=bool(o.get("is_fba")),
                product_purchases=o.get("product_purchases", 50),
            ),
            "passed_tests": check_offer_risk(o.get("feedback_score"), o.get("rating_count")),
        })

    ranked = assign_dynamic_tiers(ranked)
    ranked.sort(key=lambda c: (c["passed_tests"]["passed_all"], c["score"]), reverse=True)
    return ranked


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
        
    # Prices.csv carries no seller signals, so the candidates above use
    # placeholder ones. Real offer lists (#69) should call rank_offers directly.
    candidates[0]["is_atc"] = True
    scored_candidates = rank_offers(candidates, typical_new, typical_used, user_preferred_condition)

    if not scored_candidates:
        scored_candidates = [{
            **candidates[0],
            "score": 0.0,
            "tier": "okay offer",
            "passed_tests": {"passed_all": True},
            "reason": "All alternative offers filtered out by user condition preference."
        }]

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