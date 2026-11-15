import pandas as pd, json, os, urllib.request
here = os.path.dirname(__file__)
OUT = f"{here}/reviews_pulled.jsonl"
STATE = f"{here}/reviews_pulled.state"
REVIEWS_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl"

pairs = pd.read_csv(f"{here}/pairs.csv")
wanted = set(zip(pairs.parent_asin, pairs.asin_a)) | set(zip(pairs.parent_asin, pairs.asin_b))

def save(offset, size):
    with open(f"{STATE}.tmp", "w") as s:
        json.dump([offset, size], s)
    os.replace(f"{STATE}.tmp", STATE)

offset, size = json.load(open(STATE)) if os.path.exists(STATE) else (0, 0)
req = urllib.request.Request(REVIEWS_URL, headers={"Range": f"bytes={offset}-"})

with urllib.request.urlopen(req) as f, open(OUT, "ab") as out:
    out.truncate(size)
    for i, line in enumerate(f):
        offset += len(line)
        row = json.loads(line)
        if (row["parent_asin"], row["asin"]) in wanted:
            out.write(line)
        if i % 1_000_000 == 0:
            out.flush()
            size = out.tell()
            save(offset, size)
            print(offset // 1_000_000_000, "GB of 22", flush=True)
    out.flush()
    save(offset, out.tell())
print("done", size)
