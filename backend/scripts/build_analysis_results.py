"""
Build the versioned file the decision endpoint reads analysis results from (#87).

The endpoint never computes these per request. It loads one JSON file once, and
every answer says which version it used, so a number on the panel can always be
traced back to the study that produced it.

What goes in:
  version_pairs  T2 (#84): for each labeled pair of product versions, which one is
                 worse. A human label (studies/labels_gold.csv) wins; otherwise the
                 sentiment baseline (studies/sentiment_baseline.csv), the best
                 automatic verdict so far (s018: 55.9% vs 49.5% for always saying
                 "No real difference").
  quiet          R4 (#81) placeholder: when to stay quiet, from FINDINGS.md section 3.
  offers         T4 (#86) placeholder: which other offers count, matching the #92 rules.

Usage:
    cd backend
    python scripts/build_analysis_results.py              # writes analysis/results/v<N>.json
    python scripts/build_analysis_results.py --version 2  # when anything above changes

Bump --version whenever an input changes. Never edit an existing file in place:
old answers in the logs point at it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "studies"
OUT_DIR = ROOT / "backend" / "analysis" / "results"
KEYS = ["parent_asin", "asin_a", "asin_b"]
VERDICTS = {"A is worse", "B is worse", "No real difference", "Mixed"}

# R4 placeholder (#81). FINDINGS.md section 3: perfect foresight is worth ~$0.13 per
# purchase on calm items under $25, and ~$23 on volatile items over $200. Below this
# price a WAIT isn't worth the shopper's attention.
QUIET = {
    "source": "placeholder until #81 (FINDINGS.md section 3)",
    "min_price_for_wait": 25.0,
    "min_drop_chance": 0.40,  # copy.ts MIN_WAIT_CHANCE
    "min_saving": 1.00,       # an offer must beat Amazon's pick by at least this much
}

# T4 placeholder (#86). Same bar as the #92 placeholder, so the panel never claims a
# seller is well rated without the numbers to show.
OFFERS = {
    "source": "placeholder until #86 (matches the #92 rules)",
    "min_stars": 4.0,
    "min_years": 1.0,
    "max_days_later": 7,  # copy.ts SKIP.slow_shipping: "Arrives more than a week later"
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def version_pairs() -> tuple[list[dict], dict]:
    gold_path, sent_path = STUDIES / "labels_gold.csv", STUDIES / "sentiment_baseline.csv"
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)[KEYS + ["gold", "topics"]]
    sent = pd.read_csv(sent_path, dtype=str, keep_default_na=False)[KEYS + ["sentiment_verdict"]]
    d = sent.merge(gold, on=KEYS, how="left").fillna("")

    pairs = []
    for r in d.itertuples(index=False):
        if r.gold in VERDICTS:
            verdict, source = r.gold, "human label"
        elif r.sentiment_verdict in VERDICTS:
            verdict, source = r.sentiment_verdict, "sentiment baseline"
        else:
            continue
        pairs.append({"parent_asin": r.parent_asin, "asin_a": r.asin_a, "asin_b": r.asin_b,
                      "verdict": verdict, "source": source,
                      "topics": [t.strip() for t in r.topics.split(";") if t.strip()]})
    inputs = {gold_path.name: _sha(gold_path), sent_path.name: _sha(sent_path)}
    return pairs, inputs


def build(version: int) -> Path:
    pairs, inputs = version_pairs()
    out = {
        "version": f"v{version}",
        "built_on": date.today().isoformat(),
        "inputs": inputs,
        "quiet": QUIET,
        "offers": OFFERS,
        "version_pairs": {"source": "T2 (#84): human label, else sentiment baseline", "pairs": pairs},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"v{version}.json"
    if path.exists() and json.loads(path.read_text()) | {"built_on": None} != out | {"built_on": None}:
        raise SystemExit(f"{path} exists with different contents. Use --version {version + 1}.")
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", type=int, default=1)
    path = build(ap.parse_args().version)
    n = len(json.loads(path.read_text())["version_pairs"]["pairs"])
    print(f"wrote {path.relative_to(ROOT)} ({n} version pairs)")
