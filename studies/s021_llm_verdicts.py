import pandas as pd, os, json
from collections import Counter
here = os.path.dirname(__file__)
keys = ["parent_asin", "asin_a", "asin_b"]

with open(f"{here}/llm_tags.jsonl") as f:
    tags = {(j["parent_asin"], j["asin"]): j["topics"] for j in map(json.loads, f)}

print(len(tags))
print(tags[("B01GJ3WA36", "B008F3QGK2")][:3])

def counts(parent, asin):
    c = Counter()
    for review in tags[(parent, asin)]:
        c.update(set(review))
    return c

pairs = pd.read_csv(f"{here}/pairs.csv")

rows = []
for p in pairs.itertuples():
    a = counts(p.parent_asin, p.asin_a)
    b = counts(p.parent_asin, p.asin_b)
    topics = sorted((set(a) | set(b)))
    against_a = [t for t in topics if a[t] >= 4 and a[t] >= 2 * b[t]]
    against_b = [t for t in topics if b[t] >= 4 and b[t] >= 2 * a[t]]
    if against_a and against_b:
        verdict = "Mixed"
    elif against_a:
        verdict = "A is worse"
    elif against_b:
        verdict = "B is worse"
    else:
        verdict = "No real difference"
    rows.append([p.parent_asin, p.asin_a, p.asin_b, p.group, verdict,
                 "; ".join(against_a), "; ".join(against_b)])

out = pd.DataFrame(rows, columns=keys + ["group", "llm_verdict", "topics_against_a", "topics_against_b"])
out.to_csv(f"{here}/llm_baseline.csv", index=False)
print(out.llm_verdict.value_counts())


