import pandas as pd, os
here = os.path.dirname(__file__)
ours = set(pd.read_csv(f"{here}/Product.csv").asin)
counts = pd.read_csv(f"{here}/review_counts.csv")

print("total reviews", counts.n_reviews.sum())
print("families", counts.parent_asin.nunique())

versions = counts.groupby("parent_asin").asin.nunique()
multi = set(versions[versions >= 2].index)
print("families with 2+ versions", len(multi))

print("ours found as asin", len(ours & set(counts.asin)))
print("ours found as parent_asin", len(ours & set(counts.parent_asin)))

found = counts[counts.asin.isin(ours) | counts.parent_asin.isin(ours)]
any_match = (set(found.asin) | set(found.parent_asin)) & ours
print("ours found at all", len(any_match))

in_multi = found[found.parent_asin.isin(multi)]
with_sibling = (set(in_multi.asin) | set(in_multi.parent_asin)) & ours
print("ours with another version", len(with_sibling))

print(in_multi.n_reviews.describe(percentiles=[.5, .9]))
