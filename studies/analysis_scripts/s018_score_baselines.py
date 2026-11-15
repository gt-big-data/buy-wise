import pandas as pd, os
here = os.path.dirname(__file__)
keys = ["parent_asin", "asin_a", "asin_b"]

gold = pd.read_csv(f"{here}/labels_gold.csv", dtype=str, keep_default_na=False)
star = pd.read_csv(f"{here}/star_baseline.csv")[keys + ["star_verdict"]]
sent = pd.read_csv(f"{here}/sentiment_baseline.csv")[keys + ["sentiment_verdict"]]
d = gold.merge(star, on=keys).merge(sent, on=keys)
d = d[d.gold.isin(["A is worse", "B is worse", "No real difference"])]

for name, subset in [("all pairs", d), ("agreed pairs only", d[d.source == "agreed"])]:
    print(f"\n=== {name}: {len(subset)} pairs ===")
    print("always 'No real difference':", round((subset.gold == "No real difference").mean(), 3))
    for col in ["star_verdict", "sentiment_verdict"]:
        print(f"{col}: {round((subset[col] == subset.gold).mean(), 3)}")

for col in ["star_verdict", "sentiment_verdict"]:
    print(f"\n{col} (all pairs)")
    print(pd.crosstab(d.gold, d[col]))
