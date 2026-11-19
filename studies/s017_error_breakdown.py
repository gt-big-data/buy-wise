"""Study 017: where accuracy is worst, and why (extends Study 006).

Models: the served recipe (XGBoost default), XGBoost and CatBoost with the
settings they picked on each fold's validation rows (26 features), and the best
simple rule. Four walk-forward test windows, product-level 95% intervals.

For each segment it reports two things, to tell "hard to predict" from
"nothing to predict":
  - how well the model ranks decisions inside the segment (PR-AUC), against
    the segment's own base rate and against the rule (lift over rule)
  - what the model's overall 20% WAIT policy is worth there, against the most
    a perfect forecaster could get in that segment (ceiling $)

    cd studies && python s017_error_breakdown.py [prices|apr2026_500]
"""
import os
import sys

import pandas as pd

from r2_common import RESULTS, load, tuned_models

from ml.bench.breakdown import breakdown, segments  # noqa: E402
from ml.bench.models import XGB, RuleTop90  # noqa: E402
from ml.bench.runner import run_fold  # noqa: E402
from ml.features import ALL_FEATURES  # noqa: E402

SEGMENTS = ["price band", "volatility", "product seen in training", "history", "season", "brand", "category"]


def main(data="prices"):
    ds, folds, tag = load(data)
    raw = pd.read_csv(ds.source, usecols=lambda c: c.lower() in ("asin", "brand", "category"))
    raw.columns = raw.columns.str.lower()
    meta = raw.drop_duplicates("asin").set_index("asin")
    if "brand" in meta:  # brands with too few products to say anything become "other"
        top = meta.brand.value_counts()
        meta["brand"] = meta.brand.where(meta.brand.map(top) >= 8, "other")
    first_seen = ds.panel.groupby("asin").date.min()

    results = []
    for fold in folds:
        print(fold.summary(), file=sys.stderr)
        models = [RuleTop90(), XGB(), *tuned_models(tag, fold.name, ALL_FEATURES, suffix=" (tuned, 26)")]
        fold.test = fold.test.assign(**{c: fold.test.asin.map(meta[c]) for c in meta.columns})
        results.append(run_fold(models, fold, verbose=False))

    seg_fn = lambda f: segments(f.test, f.train, first_seen)  # noqa: E731
    tables = []
    pd.set_option("display.width", 220)
    for by in SEGMENTS:
        if by not in seg_fn(folds[0]).columns:
            print(f"\n(no {by} in this data)")
            continue
        bd = breakdown(results, by, seg_fn)
        rule = bd[bd.policy == RuleTop90.name].set_index(by)["PR-AUC"]
        bd["PR-AUC / base rate"] = bd["PR-AUC"] / (bd["base rate %"] / 100)
        bd["PR-AUC vs rule"] = bd["PR-AUC"] - bd[by].map(rule)
        bd["share of ceiling"] = bd["$ caught"] / bd["ceiling $"]
        tables.append(bd.rename(columns={by: "segment"}).assign(by=by))
        show = bd[bd.policy != RuleTop90.name]
        print(f"\n=== by {by} (20% WAIT policy over all decisions) ===")
        print(show[["policy", by, "rows", "base rate %", "PR-AUC", "PR-AUC lo", "PR-AUC hi", "PR-AUC / base rate",
                    "PR-AUC vs rule", "precision %", "$ caught", "ceiling $", "share of ceiling"]]
              .round(3).to_string(index=False))

    os.makedirs(RESULTS, exist_ok=True)
    pd.concat(tables).to_csv(os.path.join(RESULTS, f"s017_breakdown_{tag}.csv"), index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
