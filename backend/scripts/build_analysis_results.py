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
  quiet          R4 (#81): when a WAIT is worth showing. Calibrated drop chance and the
                 expected net saving per volatility band, from Study 024 (v1: placeholder).
  offers         T4 (#86): which other offers count and how they rank. Condition
                 discounts and the minimum saving from ml/offer_ranking.py, plus the
                 seller trust bar the panel needs (v1: placeholder).

Usage:
    cd backend
    python scripts/build_analysis_results.py --version 2  # writes analysis/results/v2.json
    python scripts/build_analysis_results.py --version 3  # when anything above changes

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

# v1 placeholders, kept so `--version 1` still rebuilds v1 exactly.
QUIET_V1 = {
    "source": "placeholder until #81 (FINDINGS.md section 3)",
    "min_price_for_wait": 25.0,
    "min_drop_chance": 0.40,
    "min_saving": 1.00,
}
OFFERS_V1 = {
    "source": "placeholder until #86 (matches the #92 rules)",
    "min_stars": 4.0,
    "min_years": 1.0,
    "max_days_later": 7,
}

# R4 (#81), Study 024. The served model (ml/buywise_clean.joblib) predicts an 8% drop
# within 14 days, so this is the 14-day rule: FINDINGS.md "If 14 days stays".
R4_RULE = STUDIES / "results" / "r4_quiet_rule_Prices_walk_r4_h14_t0.08.json"

# The panel only names a seller when it can show all three numbers, and they clear
# this bar (#129, measured on real offer lists).
SELLER_TRUST = {"min_stars": 4.0, "min_ratings": 100, "min_positive": 90.0}

# Study 006 / FINDINGS.md section 3: on cheap items there's nothing worth saying, so a
# "this is the best offer" panel stays hidden below this price.
HIDE_NOTHING_BETTER_BELOW = 25.0


def quiet_v2() -> tuple[dict, dict]:
    r = json.loads(R4_RULE.read_text())
    keep = ["horizon_days", "drop_threshold", "min_drop_chance", "min_saving", "min_price_for_wait",
            "vol30_cuts", "bands", "test"]
    return ({"source": f"{r['source']}; {r['horizon_days']}-day rule", "model": r["model"],
             **{k: r[k] for k in keep}},
            {R4_RULE.name: _sha(R4_RULE)})


def offers_v2() -> tuple[dict, dict]:
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from ml import offer_ranking as t4
    path = ROOT / "backend" / "ml" / "offer_ranking.py"
    return ({"source": "T4 (#86): condition discounts and minimum saving tuned on hand-check "
                       "rows 1-25; seller trust bar from #129",
             **SELLER_TRUST,
             "condition_discount": dict(t4.CONDITION_DISCOUNT),
             "unknown_condition_discount": t4.UNKNOWN_CONDITION_DISCOUNT,
             "min_saving_usd": t4.MIN_SAVINGS_USD,
             "min_saving_pct": t4.MIN_SAVINGS_PCT,
             "hide_nothing_better_below": HIDE_NOTHING_BETTER_BELOW},
            {path.name: _sha(path)})


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
    if version == 1:
        quiet, offers = QUIET_V1, OFFERS_V1
    else:
        (quiet, q_in), (offers, o_in) = quiet_v2(), offers_v2()
        inputs = {**inputs, **q_in, **o_in}
    out = {
        "version": f"v{version}",
        "built_on": date.today().isoformat(),
        "inputs": inputs,
        "quiet": quiet,
        "offers": offers,
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
