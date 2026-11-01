import pandas as pd, numpy as np, os
from transformers import pipeline
here = os.path.dirname(__file__)
SCORES = f"{here}/sentiment_scores.csv"

reviews = pd.read_json(f"{here}/reviews_pulled.jsonl", lines=True)
pairs = pd.read_csv(f"{here}/pairs.csv")
texts = (reviews.title.fillna("") + ". " + reviews.text.fillna("")).tolist()

done = len(pd.read_csv(SCORES)) if os.path.exists(SCORES) else 0
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device="mps")

for start in range(done, len(texts), 5000):
    out = clf(texts[start:start + 5000], batch_size=8, truncation=True)
    labels = [int(x["label"] == "POSITIVE") for x in out]
    pd.DataFrame({"positive": labels}).to_csv(SCORES, mode="a", header=start == 0, index=False)
    print(start + len(labels), "of", len(texts), flush=True)

reviews["positive"] = pd.read_csv(SCORES).positive.values
s = reviews.groupby("asin").positive.agg(["mean", "var", "count"])
p = pairs.merge(s, left_on="asin_a", right_index=True)
p = p.merge(s, left_on="asin_b", right_index=True, suffixes=("_a", "_b"))

p["diff"] = p.mean_a - p.mean_b
p["z"] = p["diff"] / np.sqrt(p.var_a / p.count_a + p.var_b / p.count_b)

real = (p.z.abs() > 1.96) & (p["diff"].abs() >= 0.05)
p["sentiment_verdict"] = "No real difference"
p.loc[real & (p["diff"] < 0), "sentiment_verdict"] = "A is worse"
p.loc[real & (p["diff"] > 0), "sentiment_verdict"] = "B is worse"

cols = ["parent_asin", "asin_a", "asin_b", "group", "mean_a", "mean_b", "z", "sentiment_verdict"]
p[cols].to_csv(f"{here}/sentiment_baseline.csv", index=False)
print(p.sentiment_verdict.value_counts())
