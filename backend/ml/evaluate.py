"""The shared scoring tool. Any model or rule goes in, one standard report comes out.

Every policy is compared at the SAME wait rate: each one says WAIT on exactly the
top N% of decisions by its own score. Without that, a policy can look better just
by saying WAIT more often (the April model did exactly this).

    cd backend && python -m ml.evaluate                          # today's leaderboard
    python -m ml.evaluate --models top90,rel90,november,xgb      # pick models (see ml/bench/models.py)
    python -m ml.evaluate --data path/to/Prices.csv              # different data, same scoring

The framework itself lives in ml/bench/. To add a model, see ml/bench/models.py.
Metric definitions are in ml/bench/metrics.py.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ml.bench.data import DEFAULT_SOURCE, MIN_SPAN_DAYS, STUDIES, build_dataset, load_dataset, load_prices
from ml.bench.metrics import WAIT_RATES, _policy_row, oracle, score_decision, score_policies
from ml.bench.models import DEFAULT_MODELS, MODELS, RuleNovember, RuleRel90, RuleTop90, april_model_scores
from ml.bench.runner import run_fold
from ml.bench.splits import Split, chronological_split, legacy_fold

__all__ = [
    "STUDIES", "WAIT_RATES", "MIN_SPAN_DAYS", "Split", "load_prices", "build_dataset", "chronological_split",
    "baseline_scores", "score_policies", "score_decision", "oracle", "leaderboard",
    "deployed_april_model_scores", "_policy_row",
]


def baseline_scores(te: pd.DataFrame) -> dict[str, np.ndarray]:
    """The rules every model has to beat. Higher score = more reason to WAIT."""
    return {r.name: r.score(te, None) for r in (RuleTop90(), RuleRel90(), RuleNovember())}


def deployed_april_model_scores(te: pd.DataFrame) -> np.ndarray | None:
    """P(WAIT) from the April 2026 model, or None if its files are gone."""
    try:
        return april_model_scores(te)
    except FileNotFoundError:
        return None


def leaderboard(te: pd.DataFrame, scores: dict[str, np.ndarray],
                probabilistic: set[str] | frozenset[str] = frozenset()) -> str:
    r = score_policies(te, scores, probabilistic=probabilistic)
    lines = [
        f"test set: {len(te):,} decisions, {te.asin.nunique()} products, "
        f"{te.date.min().date()} to {te.date.max().date()}",
        f"base rate P(drop >= 8% in 14d) = {te.y.mean() * 100:.1f}%   "
        f"mean price ${te.price.mean():.2f}   perfect-foresight ceiling ${oracle(te):.2f}/decision",
    ]
    for q in sorted(r["target wait %"].unique()):
        block = r[r["target wait %"] == q].drop(columns="target wait %").sort_values("$ caught", ascending=False)
        lines += ["", f"=== everyone says WAIT on {q}% of decisions ===",
                  block.round(3).to_string(index=False, na_rep="n/a")]
    lines += ["", "Brier and ECE only apply to models whose score is a probability (n/a for rules)."]
    return "\n".join(lines)


def parse_models(arg: str) -> list[str]:
    keys = list(DEFAULT_MODELS) if arg == "default" else list(MODELS) if arg == "all" else arg.split(",")
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}. Known: {', '.join(MODELS)}")
    return keys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ml.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_SOURCE), help="Prices.csv-style file (Keepa snapshots: T1)")
    ap.add_argument("--models", default="default", help=f"'default', 'all', or a comma list of: {', '.join(MODELS)}")
    args = ap.parse_args(argv)

    ds = load_dataset(args.data)
    fold = legacy_fold(ds.rows, ds.panel)
    models = [MODELS[k](ds.source) for k in parse_models(args.models)]
    print(fold.summary(), file=sys.stderr)
    res = run_fold(models, fold, verbose=False)
    for name, why in res.skipped.items():
        print(f"skipped {name}: {why}", file=sys.stderr)
    print(leaderboard(fold.test, res.scores, res.probabilistic))


if __name__ == "__main__":
    main()
