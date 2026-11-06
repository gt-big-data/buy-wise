"""The shared scoring tool. Any model or rule goes in, one standard report comes out.

Every policy is compared at the SAME wait rate: each one says WAIT on exactly the
top N% of decisions by its own score. Without that, a policy can look better just
by saying WAIT more often (the April model did exactly this).

    cd backend && python -m ml.evaluate                          # today's leaderboard
    python -m ml.evaluate --models top90,rel90,november,xgb      # pick models (see ml/bench/models.py)
    python -m ml.evaluate --data path/to/Prices.csv              # different data, same scoring
    python -m ml.evaluate --models r2 --folds walk --ci          # 4 test windows, error bars
    python -m ml.evaluate --models r2 --folds walk --by volatility   # results by kind of product

The framework itself lives in ml/bench/. To add a model, see ml/bench/models.py.
Metric definitions are in ml/bench/metrics.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.bench.data import DEFAULT_SOURCE, MIN_SPAN_DAYS, STUDIES, build_dataset, load_dataset, load_prices
from ml.bench.metrics import WAIT_RATES, _policy_row, oracle, score_decision, score_policies
from ml.bench.models import (DEFAULT_MODELS, MODELS, R2_MODELS, RULES, RuleNovember, RuleRel90, RuleTop90,
                             april_model_scores)
from ml.bench.runner import run_fold
from ml.bench.splits import Split, chronological_split, legacy_fold, walk_forward_folds

RESULTS = STUDIES / "results"

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
    presets = {"default": list(DEFAULT_MODELS), "all": list(MODELS), "r2": [*RULES, "served", "xgb", *R2_MODELS]}
    keys = presets.get(arg) or arg.split(",")
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}. Known: {', '.join(MODELS)}")
    return keys


def ci_leaderboard(results, n_boot: int, reference: str | None):
    """Leaderboard pooled over folds, with 95% product-level intervals, plus paired
    differences: every model against `reference`, and each tuned model against
    its defaults. Returns (text, leaderboard table, paired table)."""
    from ml.bench.ci import bootstrap, compare, summary

    boot = bootstrap(results, n_boot=n_boot)
    tab = summary(boot)
    lines = [f"pooled over {len(results)} fold(s); 95% intervals from {n_boot} product-level resamples"]
    fmt = lambda r, m, d=3: f"{r[m]:.{d}f} [{r[m + ' lo']:.{d}f}, {r[m + ' hi']:.{d}f}]"
    for q in sorted(tab["target wait %"].unique()):
        block = tab[tab["target wait %"] == q].sort_values("$ caught", ascending=False)
        show = pd.DataFrame({
            "policy": block.policy, "folds": block.folds,
            "precision %": [fmt(r, "precision %", 1) for _, r in block.iterrows()],
            "$ caught": [fmt(r, "$ caught", 2) for _, r in block.iterrows()],
            "$ no alert": [fmt(r, "$ no alert", 2) for _, r in block.iterrows()],
            "PR-AUC": [fmt(r, "PR-AUC") for _, r in block.iterrows()],
            "AUC": [fmt(r, "AUC") for _, r in block.iterrows()],
        })
        lines += ["", f"=== everyone says WAIT on {q}% of decisions ===", show.to_string(index=False)]
    pairs = []
    if reference and reference in boot.point:
        pairs += [compare(boot, m, reference) for m in boot.point if m != reference]
    # Each tuned model against its own defaults: what tuning bought, paired.
    for m in boot.point:
        if "(tuned" in m and m.replace("(tuned", "(default") in boot.point:
            pairs.append(compare(boot, m, m.replace("(tuned", "(default")))
    paired = (pd.concat(pairs, ignore_index=True).drop_duplicates(["a", "b", "metric"]) if pairs
              else pd.DataFrame())
    if len(paired):
        key = ["PR-AUC", "$ caught@10", "$ caught@20", "$ caught@30"]
        d = paired[paired.metric.isin(key)].copy()
        d["diff [95%]"] = [f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for _, r in d.iterrows()]
        for b, block in d.groupby("b", sort=False):
            wide = block.pivot(index="a", columns="metric", values="diff [95%]")[key]
            lines += ["", f"=== paired difference vs {b} (positive = better; interval excludes 0 = real) ===",
                      wide.to_string()]
    return "\n".join(lines), tab, paired


def main(argv: list[str] | None = None) -> None:
    from ml.bench import tuning

    ap = argparse.ArgumentParser(prog="python -m ml.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_SOURCE), help="Prices.csv-style file (Keepa snapshots: T1)")
    ap.add_argument("--models", default="default",
                    help=f"'default', 'all', 'r2', or a comma list of: {', '.join(MODELS)}")
    ap.add_argument("--folds", choices=["legacy", "walk"], default="legacy",
                    help="legacy: the spring 60/20/20 split. walk: 4 back-to-back 91-day test windows")
    ap.add_argument("--ci", type=int, nargs="?", const=1000, default=0, metavar="N",
                    help="add 95%% intervals from N product-level resamples (default 1000)")
    ap.add_argument("--by", help="break results down by: price band, volatility, product seen in training, "
                                 "history, season (and category/brand when the data has them)")
    ap.add_argument("--budget", type=int, default=tuning.TUNE_BUDGET, help="tuning trials per model per fold")
    ap.add_argument("--reference", default="XGBoost (default)", help="model the paired differences are against")
    ap.add_argument("--save", action="store_true", help=f"write tables to {RESULTS}")
    ap.add_argument("--tag", default="", help="suffix for saved file names, so separate experiments don't overwrite")
    args = ap.parse_args(argv)
    tuning.TUNE_BUDGET = args.budget

    ds = load_dataset(args.data)
    folds = [legacy_fold(ds.rows, ds.panel)] if args.folds == "legacy" else walk_forward_folds(ds.rows, ds.panel)
    keys = parse_models(args.models)
    results = []
    for fold in folds:
        models = [MODELS[k](ds.source) for k in keys]
        print(fold.summary(), file=sys.stderr)
        res = run_fold(models, fold, verbose=False)
        for name, why in res.skipped.items():
            print(f"  skipped {name}: {why}", file=sys.stderr)
        res.models = models
        results.append(res)

    tag = f"{Path(ds.source).name.split('.')[0]}_{args.folds}" + (f"_{args.tag}" if args.tag else "")
    if args.ci or len(folds) > 1:
        text, tab, paired = ci_leaderboard(results, args.ci or 1000, args.reference)
        print(text)
    else:
        print(leaderboard(folds[0].test, results[0].scores, results[0].probabilistic))
        tab, paired = results[0].table(), pd.DataFrame()

    if args.by:
        from ml.bench.breakdown import breakdown, segments

        first_seen = ds.panel.groupby("asin").date.min()
        bd = breakdown(results, args.by, lambda f: segments(f.test, f.train, first_seen))
        cols = ["policy", args.by, "rows", "base rate %", "PR-AUC", "PR-AUC lo", "PR-AUC hi",
                "precision %", "$ caught", "$ caught lo", "$ caught hi", "ceiling $"]
        print(f"\n=== by {args.by}: everyone says WAIT on 20% of all decisions ===")
        print(bd[cols].round(3).to_string(index=False))

    if args.save:
        RESULTS.mkdir(exist_ok=True)
        tab.to_csv(RESULTS / f"leaderboard_{tag}.csv", index=False)
        if len(paired):
            paired.to_csv(RESULTS / f"paired_{tag}.csv", index=False)
        # Every test score, so tables and charts can be rebuilt without refitting.
        pd.concat([r.fold.test[["asin", "date", "y", "price", "fwd_min", "fwd_last"]]
                   .assign(fold=r.fold.name, **{m: s for m, s in r.scores.items()}) for r in results]
                  ).to_csv(RESULTS / f"scores_{tag}.csv.gz", index=False)
        if args.by:
            bd.to_csv(RESULTS / f"breakdown_{tag}_{args.by.replace(' ', '_')}.csv", index=False)
        trials = [m.trials.assign(fold=r.fold.name, model=m.name)
                  for r in results for m in r.models if getattr(m, "trials", None) is not None]
        if trials:
            pd.concat(trials).to_csv(RESULTS / f"tuning_{tag}.csv", index=False)
        print(f"\nsaved to {RESULTS}", file=sys.stderr)


if __name__ == "__main__":
    main()
