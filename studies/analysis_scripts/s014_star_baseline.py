import pandas as pd, numpy as np, os
here = os.path.dirname(__file__)
reviews = pd.read_json(f"{here}/reviews_pulled.jsonl", lines=True)
pairs = pd.read_csv(f"{here}/pairs.csv")

s = reviews.groupby("asin").rating.agg(["mean", "var", "count"])
p = pairs.merge(s, left_on="asin_a", right_index=True)
p = p.merge(s, left_on="asin_b", right_index=True, suffixes=("_a", "_b"))

p["diff"] = p.mean_a - p.mean_b
p["z"] = p["diff"] / np.sqrt(p.var_a / p.count_a + p.var_b / p.count_b)

real = (p.z.abs() > 1.96) & (p["diff"].abs() >= 0.2)
p["star_verdict"] = "No real difference"
p.loc[real & (p["diff"] < 0), "star_verdict"] = "A is worse"
p.loc[real & (p["diff"] > 0), "star_verdict"] = "B is worse"

cols = ["parent_asin", "asin_a", "asin_b", "group", "mean_a", "mean_b", "z", "star_verdict"]
p[cols].to_csv(f"{here}/star_baseline.csv", index=False)
print(p.star_verdict.value_counts())
