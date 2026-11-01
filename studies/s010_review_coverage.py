import pandas as pd, os
import json, urllib.request
from collections import Counter
products = pd.read_csv(f"{os.path.dirname(__file__)}/Product.csv")
our_asins = set(products.asin)
print(len(our_asins))

REVIEWS_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl"

counts = Counter()
with urllib.request.urlopen(REVIEWS_URL) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        counts[(row["parent_asin"], row["asin"])] += 1
        if i % 1_000_000 == 0:
            print(i, flush=True)

rows = [(p, a, n) for (p, a), n in counts.items()]
out = pd.DataFrame(rows, columns=["parent_asin", "asin", "n_reviews"])
out.to_csv(f"{os.path.dirname(__file__)}/review_counts.csv", index=False)
print(len(out), "versions")
