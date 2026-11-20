import pandas as pd, os
here = os.path.dirname(__file__)
ours = set(pd.read_csv(f"{here}/Product.csv").asin)
c = pd.read_csv(f"{here}/review_counts.csv")

c = c[c.n_reviews >= 30]
c = c[c.groupby("parent_asin").asin.transform("nunique") >= 2].copy()
c["ours"] = c.asin.isin(ours)
c = c.sort_values(["parent_asin", "ours", "n_reviews"], ascending=[True, False, False])
c["slot"] = c.groupby("parent_asin").cumcount()

a = c[c.slot == 0].set_index("parent_asin")
b = c[c.slot == 1].set_index("parent_asin")
pairs = pd.DataFrame({"asin_a": a.asin, "asin_b": b.asin,
                      "n_a": a.n_reviews, "n_b": b.n_reviews, "ours": a.ours}).reset_index()
print(len(pairs), "eligible families")

mine = pairs[pairs.ours]
rest = pairs[~pairs.ours].sample(47, random_state=42)
out = pd.concat([rest.head(5).assign(group="trial"),
                 mine.assign(group="label"),
                 rest.tail(42).assign(group="label")])
out.to_csv(f"{here}/pairs.csv", index=False)
print(out.group.value_counts())
