"""Study 016: is the April 500-product pull usable as a second dataset?

Compares snapshots/apr2026_500/ (500 products, recovered from the April
xgboost_model branch) with the 312-product Prices.csv on the checks Study 000
raised: gaps, weekly sampling, duplicates, impossible moves, short histories.
Then compares the two files on the dates they share.

    cd studies && ../backend/.venv/bin/python s016_april_snapshot.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "current (312)": os.path.join(HERE, "Prices.csv"),
    "april (500)": os.path.join(HERE, "snapshots", "apr2026_500", "Prices.csv.gz"),
}


def load(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.columns = df.columns.str.lower().str.strip()
    return df.sort_values(["asin", "datetime"]).reset_index(drop=True)


def checks(df):
    gap = df.groupby("asin")["datetime"].diff().dt.days.dropna()
    gap = gap[gap > 0]
    move = df.groupby("asin")["amazon"].pct_change().abs()
    span = df.groupby("asin")["datetime"].agg(lambda s: (s.max() - s.min()).days)
    repeats = df.groupby("asin")["amazon"].diff() == 0
    changes = df[~repeats].groupby("asin").size()
    return {
        "rows": len(df),
        "products": df.asin.nunique(),
        "date span": f"{df.datetime.min().date()} to {df.datetime.max().date()}",
        "median gap (days)": gap.median(),
        "gaps that are multiples of 7 (%)": round((gap % 7 == 0).mean() * 100, 1),
        "duplicate asin+date rows": int(df.duplicated(["asin", "datetime"]).sum()),
        "rows repeating the previous price (%)": round(repeats.mean() * 100, 1),
        "real price changes per product (median)": int(changes.median()),
        "moves over 100% (count)": int((move > 1).sum()),
        "marketplace new >40% above Amazon (%)": round((df.new > df.amazon * 1.4).mean() * 100, 2),
        "products with <30 rows": int((df.groupby("asin").size() < 30).sum()),
        "products spanning a year+": int((span >= 365).sum()),
    }


cur, apr = (load(p) for p in FILES.values())
print(pd.DataFrame({name: checks(df) for name, df in zip(FILES, (cur, apr))}).to_string())

shared = cur.merge(apr, on=["asin", "datetime"], suffixes=("_cur", "_apr"))
same = np.isclose(shared.amazon_cur, shared.amazon_apr, atol=0.01)
gap = ((shared.amazon_cur - shared.amazon_apr).abs() / shared.amazon_apr)[~same]
print(f"\nproducts in both: {len(set(cur.asin) & set(apr.asin))} | only in april: {len(set(apr.asin) - set(cur.asin))}")
print(f"shared asin+date rows: {len(shared)} | same Amazon price: {same.mean():.1%}")
print(f"when they differ: median {gap.median():.1%}, 90th percentile {gap.quantile(.9):.1%}")
