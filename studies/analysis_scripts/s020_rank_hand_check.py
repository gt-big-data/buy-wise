"""Run the offer ranker (backend/ml/offer_ranking.py) on the hand-checked
pages and fill `ranker_pick` in studies/hand_check/pages.csv (#86).

The ranker sees every offer recorded for the page in
studies/hand_check/offers.csv (one row per offer, real seller signals,
total = price + shipping). ranker_pick is "atc" if it keeps the Add to Cart
offer, "pick" if it matches the human pick, else the offer it chose; plus
its tier. Prints how often it agrees on pages where it had a real choice.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.offer_ranking import rank_offers  # noqa: E402

PAGES = ROOT / "studies/hand_check/pages.csv"
sheet = pd.read_csv(PAGES, dtype=str, keep_default_na=False)
offers = pd.read_csv(ROOT / "studies/hand_check/offers.csv", dtype={"asin": str})


def num(x):
    return float(x) if x != "" else float("nan")


def to_offer(o):
    return {
        "source": o.seller,
        "price": o.total,
        "condition": o.condition,
        "feedback_score": o.feedback_pct,
        "rating_count": o.ratings,
        "is_fba": o.fba == "yes",
        "is_atc": o.is_atc == 1,
    }


def same_offer(o, r):
    """The human recorded seller, condition and total; prices can move by cents."""
    return (o["source"] == r.pick_seller and o["condition"] == r.pick_condition
            and abs(o["price"] - num(r.pick_total)) < 1.0)


live = (sheet.checked_on != "") & (sheet.role != "dead")
for i, r in sheet[live].iterrows():
    page = [to_offer(o) for o in offers[offers.asin == r.asin].itertuples()]
    ranked = rank_offers(page, num(r.typical_new_price), num(r.typical_used_price))
    if not ranked:
        sheet.at[i, "ranker_pick"] = "none"
        continue
    w = ranked[0]
    if w["is_atc"]:
        label = "atc"
    elif same_offer(w, r):
        label = "pick"
    else:
        label = f"other: {w['source']}, {w['condition']}, ${w['price']:.2f}"
    sheet.at[i, "ranker_pick"] = f"{label} ({w['tier']})"
for i in sheet.index[sheet.role == "dead"]:
    sheet.at[i, "ranker_pick"] = "none"

sheet.to_csv(PAGES, index=False)

done = sheet[live].copy()
n_offers = offers.groupby("asin").size()
done["offers"] = done.asin.map(n_offers).fillna(0).astype(int)
picked_atc = done.atc_seller.eq(done.pick_seller) & done.atc_total.eq(done.pick_total) & done.atc_total.ne("")
done["agree"] = done.ranker_pick.str.startswith("atc") & picked_atc | done.ranker_pick.str.startswith("pick")
print(done[["asin", "brand", "offers", "amazon_already_best", "ranker_pick", "agree"]].to_string(index=False))
choice = done[done.offers > 1]
print(f"\nPages with a real choice (2+ offers): {len(choice)}, ranker agrees on {choice.agree.sum()}")
