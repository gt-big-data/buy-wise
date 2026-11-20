from groq import Groq
from dotenv import load_dotenv
import pandas as pd, os, json
import time, groq

here = os.path.dirname(__file__)
reviews = pd.read_json(f"{here}/reviews_pulled.jsonl", lines=True)
pairs = pd.read_csv(f"{here}/pairs.csv")


load_dotenv(f"{here}/..//.env")
client = Groq(max_retries=10)
MODEL = "openai/gpt-oss-120b"

PROMPT = """You read Amazon reviews of one product version and tag the complaints in each review.

Topics (use only these names):
- durability: stops working, dies, is dead on arrival, breaks, or a part cracks or comes loose
- connectivity: wireless pairing problems, Bluetooth dropouts, disconnects, or won't stay connected to a phone, computer or network
- battery: short battery life, drains fast, or won't hold a charge
- charging: charges slowly, won't charge, or charging or USB ports don't work
- sound: poor audio such as weak or no bass, tinny, static, crackling, buzzing, distortion, muffled, or too quiet
- mic: callers can't hear the user, or the microphone is muffled, noisy or doesn't work
- noise_cancelling: blocks little or no outside noise
- fit: uncomfortable, painful, or doesn't stay in place
- controls: buttons or touch controls that are hard to use, too sensitive, or unresponsive
- software: app, firmware, syncing or setup problems
- build: feels cheap, flimsy or poorly made, without actually breaking
- handling: physically hard to work with, such as a stiff cable, hard to strip, or tight outlets
- safety: overheating, smoke, sparks, or other electrical hazards
- other: a real complaint about the product that fits none of the above

If a complaint fits a specific topic, use it instead of durability. For example, USB ports that stopped working are charging, and an earbud that died is durability.

Rules:
- Tag a topic only when the reviewer complains about it. Praise is not a complaint.
- Only complaints about the product itself. Ignore shipping, packaging, seller, price and missing accessories.
- A review can have several topics or none.

Reply in JSON: {"reviews": [{"n": 1, "topics": ["durability"]}, ...]} with one entry for every review number."""

def sample(parent, asin):
    g = reviews[(reviews.asin == asin) & (reviews.parent_asin == parent)]
    return g.sample(min(30, len(g)), random_state=42).fillna("")

def format_reviews(g):
    lines = []
    for i, r in enumerate(g.itertuples(), 1):
        text = f"{r.title}. {r.text}".replace("<br />", " ")
        lines.append(f"{i}. {text}")
    return "\n".join(lines)

def tag(text, seed=0):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": text}],
        reasoning_effort="low",
        response_format={"type": "json_object"},
        temperature=0,
        seed=seed,
    )
    return json.loads(resp.choices[0].message.content), resp.usage

TAGS = f"{here}/llm_tags.jsonl"

def done_keys():
    if not os.path.exists(TAGS):
        return set()
    with open(TAGS) as f:
        return {(j["parent_asin"], j["asin"]) for j in map(json.loads, f)}

def tag_version(parent, asin):
    g = sample(parent, asin)
    for attempt in range(3):
        start = time.time()
        try:
            tags, usage = tag(format_reviews(g), seed=attempt)
        except groq.BadRequestError as e:
            print(f"  {asin}: bad JSON, retrying ({e})")
            continue
        got = {int(r["n"]): r["topics"] for r in tags.get("reviews", [])}
        if sorted(got) == list(range(1, len(g) + 1)):
            return {"parent_asin": parent, "asin": asin,
                    "topics": [got[n] for n in range(1, len(g) + 1)],
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "seconds": round(time.time() - start, 2)}
        print(f"  {asin}: got {len(got)} of {len(g)} reviews, retrying")
    raise RuntimeError(f"{asin}: failed after 3 tries")

done = done_keys()
todo = [(p.parent_asin, a) for p in pairs.itertuples()
        for a in (p.asin_a, p.asin_b) if (p.parent_asin, a) not in done]
print(len(done), "done,", len(todo), "to go")

with open(TAGS, "a") as out:
    for i, (parent, asin) in enumerate(todo, 1):
        out.write(json.dumps(tag_version(parent, asin)) + "\n")
        out.flush()
        print(i, "of", len(todo), flush=True)

