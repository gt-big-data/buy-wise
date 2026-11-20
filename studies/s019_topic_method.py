import pandas as pd, os
here = os.path.dirname(__file__)
keys = ["parent_asin", "asin_a", "asin_b"]

# Complaint phrases per topic. Tuned on pairs 5-54 only; 55-104 are held out.
# "Fit" was dropped: it flagged 9 pairs on 5-54 and was right on 2.
TOPICS = {
    "durability": r"stopped working|stop working|quit working|quit on me|\bdied\b|\bdead\b|\bbroke\b|broken|won'?t turn on|no longer work|fell apart|cracked|defective|lasted (?:only |about |less than )?(?:a |\d+ |one |two |three |few )?(?:day|week|month)s?|doesn'?t work anymore",
    "connectivity": r"disconnect|drops? (?:the )?connection|dropped connection|won'?t pair|wouldn'?t pair|trouble pairing|pairing (?:issue|problem)|cuts? (?:in and )?out|loses? (?:the )?connection|lost connection|keeps dropping|connection (?:issue|problem)|won'?t connect|wouldn'?t connect|doesn'?t connect|not connect",
    "battery": r"battery (?:life )?(?:is |was )?(?:short|poor|bad|terrible|awful|drain)|battery (?:dies|died|drains)|(?:won'?t|doesn'?t|does not|will not) hold (?:a )?charge|short battery|battery doesn'?t last",
    "charging": r"slow(?:ly)? charg|charg(?:es|ing) (?:very |super |so )?slow|usb (?:port|ports)? ?(?:don'?t|doesn'?t|stopped|won'?t|not)|won'?t charge|wouldn'?t charge|not charging|stopped charging",
    "sound": r"no bass|lack(?:s|ing)? (?:of )?bass|tinny|static|crackl|buzz|hiss|distort|muffled|too quiet|not (?:very )?loud|low volume",
    "mic": r"can'?t hear me|cannot hear me|could not hear me|couldn'?t hear me|(?:mic|microphone) (?:is |was )?(?:bad|poor|terrible|awful|doesn'?t|does not|not work)",
    "build": r"\bflimsy\b|cheaply made|poor(?:ly)? (?:quality|made)|cheap plastic",
    "handling": r"\bstiff\b|hard to strip|difficult to strip",
}

reviews = pd.read_json(f"{here}/reviews_pulled.jsonl", lines=True)
pairs = pd.read_csv(f"{here}/pairs.csv")
text = (reviews.title.fillna("") + ". " + reviews.text.fillna("")).str.lower().str.replace("<br />", " ")
for topic, pattern in TOPICS.items():
    reviews[topic] = text.str.contains(pattern)

def counts(parent, asin):
    g = reviews[(reviews.asin == asin) & (reviews.parent_asin == parent)]
    return g.sample(min(30, len(g)), random_state=42)[list(TOPICS)].sum()

rows = []
for p in pairs.itertuples():
    a, b = counts(p.parent_asin, p.asin_a), counts(p.parent_asin, p.asin_b)
    against_a = [t for t in TOPICS if a[t] >= 3 and a[t] >= 2 * b[t]]
    against_b = [t for t in TOPICS if b[t] >= 3 and b[t] >= 2 * a[t]]
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

out = pd.DataFrame(rows, columns=keys + ["group", "topic_verdict", "topics_against_a", "topics_against_b"])
out.to_csv(f"{here}/topic_baseline.csv", index=False)
print(out.topic_verdict.value_counts())
