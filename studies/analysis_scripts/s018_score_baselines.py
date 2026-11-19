import pandas as pd, os
here = os.path.dirname(__file__)
keys = ["parent_asin", "asin_a", "asin_b"]
methods = ["star_verdict", "sentiment_verdict", "topic_verdict", "llm_verdict"]

gold = pd.read_csv(f"{here}/labels_gold.csv", dtype=str, keep_default_na=False)
star = pd.read_csv(f"{here}/star_baseline.csv")[keys + ["star_verdict"]]
sent = pd.read_csv(f"{here}/sentiment_baseline.csv")[keys + ["sentiment_verdict"]]
topic = pd.read_csv(f"{here}/topic_baseline.csv")[keys + ["topic_verdict"]]
order = pd.read_csv(f"{here}/pairs.csv").reset_index()[keys + ["index"]]
llm = pd.read_csv(f"{here}/llm_baseline.csv").reset_index()[keys + ["llm_verdict"]]
d = gold.merge(star, on=keys).merge(sent, on=keys).merge(topic, on=keys).merge(llm, on=keys).merge(order, on=keys)
d = d[d.gold.isin(["A is worse", "B is worse", "No real difference"])]

# The topic method was tuned on pairs 5-54, so 55-104 is the fair comparison.
for name, subset in [("all pairs", d), ("held-out pairs 55-104", d[d["index"] >= 55]),
                     ("agreed pairs only", d[d.source == "agreed"])]:
    print(f"\n=== {name}: {len(subset)} pairs ===")
    print("always 'No real difference':", round((subset.gold == "No real difference").mean(), 3))
    for col in methods:
        print(f"{col}: {round((subset[col] == subset.gold).mean(), 3)}")

for col in methods:
    print(f"\n{col} (all pairs)")
    print(pd.crosstab(d.gold, d[col]))
