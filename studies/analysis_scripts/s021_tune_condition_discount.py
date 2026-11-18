"""Tune the ranker's condition discount ladder on the hand-check dev set (#86).

Dev set = hand-check rows 1-25 (the first 25 main pages checked, dead pages
swapped for backups). Rows 26-50 are the held-out test: this script never
reads them. Like New is fixed at the Keepa anchor (CONDITION_DISCOUNT);
Very Good / Good / Acceptable are searched on a coarse grid, kept monotonic.
Prints agreement with the human pick on pages with 2+ offers.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.offer_ranking import CONDITION_DISCOUNT, rank_offers  # noqa: E402

sheet = pd.read_csv(ROOT / "studies/hand_check/pages.csv", dtype=str, keep_default_na=False)
offers = pd.read_csv(ROOT / "studies/hand_check/offers.csv", dtype={"asin": str})
DEV = offers.asin.unique()  # offers.csv only holds the dev pages


def num(x):
    return float(x) if x != "" else float("nan")


pages = []
for r in sheet[sheet.asin.isin(DEV)].itertuples():
    page = [{"source": o.seller, "price": o.total, "condition": o.condition,
             "feedback_score": o.feedback_pct, "rating_count": o.ratings,
             "is_fba": o.fba == "yes", "is_atc": o.is_atc == 1}
            for o in offers[offers.asin == r.asin].itertuples()]
    if len(page) > 1:
        pages.append((r, page))


def agreement(discounts):
    hits, misses = 0, []
    for r, page in pages:
        w = rank_offers(page, num(r.typical_new_price), num(r.typical_used_price), discounts=discounts)[0]
        if (w["source"], w["condition"]) == (r.pick_seller, r.pick_condition) and abs(w["price"] - num(r.pick_total)) < 1:
            hits += 1
        else:
            misses.append(f"{r.brand}: ranker {w['source']}, {w['condition']}, ${w['price']:.2f} | "
                          f"human {r.pick_seller}, {r.pick_condition}, ${r.pick_total}")
    return hits, misses


hits, misses = agreement(CONDITION_DISCOUNT)
print(f"Current CONDITION_DISCOUNT: {hits}/{len(pages)}")
for m in misses:
    print("  miss:", m)

print("\nGrid (Like New fixed at %.2f):" % CONDITION_DISCOUNT["like new"])
rows = []
steps = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
for vg, g, a in itertools.product(steps, steps, steps):
    if not (CONDITION_DISCOUNT["like new"] <= vg <= g <= a):
        continue
    d = {**CONDITION_DISCOUNT, "very good": vg, "good": g, "acceptable": a}
    rows.append((vg, g, a, agreement(d)[0]))
grid = pd.DataFrame(rows, columns=["very_good", "good", "acceptable", "agree"])
best = grid.agree.max()
print(f"best {best}/{len(pages)}; {(grid.agree == best).sum()} of {len(grid)} ladders reach it")
print("range of each rung among the best ladders:")
print(grid[grid.agree == best][["very_good", "good", "acceptable"]].agg(["min", "max"]).to_string())

print("\nLike New sensitivity (other rungs at current values):")
for ln in [0.0, 0.05, 0.10, 0.13, 0.15, 0.20, 0.25]:
    print(f"  like new {ln:.2f}: {agreement({**CONDITION_DISCOUNT, 'like new': ln})[0]}/{len(pages)}")
