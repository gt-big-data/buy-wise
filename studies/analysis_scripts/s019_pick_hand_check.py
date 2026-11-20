"""Pick the 50 product pages for the T4 hand check (#86), plus 10 backups,
and write the fill-in sheet studies/hand_check/pages.csv.

Stratified by how many offers the page had at last sighting, overweighting
pages with many offers (where ranking matters). Pages not seen since
2026-01 are dropped as likely dead. At most 2 per brand. Seeded.

The fill-in columns record the same things the ranker
(backend/ml/offer_ranking.py) judges an offer on, for two offers per page:
the Add to Cart offer and the one a careful person would pick. That lets us
compare the human pick with the ranker's on identical inputs.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.offer_ranking import add_typical_prices  # noqa: E402

SEED = 42
QUOTA = {"0-1 offers": 8, "2-4 offers": 12, "5-12 offers": 15, "13+ offers": 15}
BACKUPS = 10

d = pd.read_csv(ROOT / "studies/Prices.csv")
d.columns = d.columns.str.lower()
d = add_typical_prices(d)
last = d.sort_values("datetime").groupby("asin").tail(1)
last = last[last.datetime >= "2026-01-01"].copy()
offers = last.count_new.fillna(0) + last.count_used.fillna(0)
last["stratum"] = pd.cut(offers, [-1, 1, 4, 12, 1e9], labels=list(QUOTA))

shuffled = last.sample(frac=1, random_state=SEED)
shuffled = shuffled[shuffled.groupby("brand").cumcount() < 2]
picked = pd.concat([shuffled[shuffled.stratum == s].head(q) for s, q in QUOTA.items()])
# A stratum can run short (few live single-offer pages); top up from the rest.
shortfall = sum(QUOTA.values()) - len(picked)
picked = pd.concat([picked, shuffled.drop(picked.index).head(shortfall)])
backup = shuffled.drop(picked.index).head(BACKUPS)

out = pd.concat([picked.assign(role="main"), backup.assign(role="backup")])
products = pd.read_csv(ROOT / "studies/Product.csv").set_index("asin")
out = out.join(products[["global_min", "global_max"]], on="asin")

# The ranker's criteria, recorded for one offer: price incl. shipping,
# condition, seller feedback %, number of seller ratings, Fulfilled by Amazon.
def offer_cols(prefix):
    return {f"{prefix}_{k}": "" for k in
            ["seller", "total", "condition", "feedback_pct", "ratings", "fba"]}

sheet = pd.DataFrame({
    "asin": out.asin,
    "url": "https://www.amazon.com/dp/" + out.asin,
    "brand": out.brand,
    "role": out.role,
    "stratum": out.stratum.astype(str),
    # reference from the data (last seen <= Apr 2026), to judge today's price:
    "typical_new_price": out.typical_new.round(2),
    "typical_used_price": out.typical_used.round(2),
    "all_time_low": out.global_min,
    "all_time_high": out.global_max,
    # filled in by hand, on the live page:
    "checked_on": "",
    **offer_cols("atc"),    # the Add to Cart offer
    **offer_cols("pick"),   # what a careful person would buy
    "ranker_pick": "",
    "amazon_already_best": "",
    "notes": "",
})
sheet.to_csv(ROOT / "studies/hand_check/pages.csv", index=False)
print(sheet.groupby(["role", "stratum"]).size())
