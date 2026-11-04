import pandas as pd, os, sys
here = os.path.dirname(__file__)
reviews = pd.read_json(f"{here}/reviews_pulled.jsonl", lines=True)
pairs = pd.read_csv(f"{here}/pairs.csv")

n = int(sys.argv[1])
row = pairs.iloc[n]
print(f"PAIR {n} ({row.group})  {row.parent_asin}\n")
for name, asin in [("A", row.asin_a), ("B", row.asin_b)]:
    g = reviews[(reviews.asin == asin) & (reviews.parent_asin == row.parent_asin)]
    g = g.sample(min(30, len(g)), random_state=42)
    print(f"=== VERSION {name} ({asin}), {len(g)} reviews ===")
    for r in g.itertuples():
        print(f"\n[{r.rating} stars] {r.title}\n{r.text}".replace("<br />", "\n"))
