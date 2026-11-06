"""Study 003: which kinds of signal carry the prediction, and how much do they overlap?

Seven feature groups (ml.features.FEATURE_GROUPS). For XGBoost and CatBoost,
with the settings each picked on each fold's validation rows (held fixed, not
re-tuned per variant), over the four walk-forward test windows:

  drop-one   all 26 features minus one group: what we'd lose without it
  add-one    price position alone, plus one group: what it adds on its own
             (groups overlap, so drop-one alone under-sells correlated groups)
  shuffle    the full model with one group's columns shuffled together on test:
             a cheap cross-check that needs no retraining

Differences are paired product-level bootstraps against the full model.

    cd studies && python s003_signal_inventory.py [prices|apr2026_500]
"""
import os
import sys

import numpy as np
import pandas as pd

from r2_common import RESULTS, load, tuned_models

from ml.bench.ci import bootstrap, compare  # noqa: E402
from ml.bench.runner import FoldResult, run_model  # noqa: E402
from ml.features import ALL_FEATURES, FEATURE_GROUPS  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

BASE = "price position"
METRICS = ["PR-AUC", "$ caught@10", "$ caught@20", "$ caught@30"]


def variants():
    yield "all 26", ALL_FEATURES
    for g, cols in FEATURE_GROUPS.items():
        yield f"without {g}", [c for c in ALL_FEATURES if c not in cols]
    yield f"{BASE} only", FEATURE_GROUPS[BASE]
    for g, cols in FEATURE_GROUPS.items():
        if g != BASE:
            yield f"{BASE} + {g}", FEATURE_GROUPS[BASE] + cols


def main(data="prices"):
    ds, folds, tag = load(data)
    results, shuffle_rows = [], []
    for fold in folds:
        print(fold.summary(), file=sys.stderr)
        res = FoldResult(fold)
        for label, cols in variants():
            for m in tuned_models(tag, fold.name, cols, suffix=f" | {label}"):
                res.scores[m.name] = run_model(m, fold)
                if label == "all 26":
                    # shuffle each group's columns together, on test only
                    te = fold.test.reset_index(drop=True)
                    base = average_precision_score(te.y, res.scores[m.name])
                    for g, gcols in FEATURE_GROUPS.items():
                        X = te[ALL_FEATURES].copy()
                        X[gcols] = X[gcols].to_numpy()[np.random.default_rng(0).permutation(len(X))]
                        drop = base - average_precision_score(te.y, m.score(X))
                        shuffle_rows.append({"model": m.name.split(" |")[0], "fold": fold.name, "group": g,
                                             "PR-AUC drop when shuffled": drop})
        results.append(res)

    boot = bootstrap(results, n_boot=1000)
    rows = []
    for key in ("XGBoost", "CatBoost"):
        full = f"{key} | all 26"
        base_only = f"{key} | {BASE} only"
        for name in boot.point:
            if not name.startswith(key + " |") or name in (full,):
                continue
            ref = base_only if " + " in name else full
            c = compare(boot, name, ref).set_index("metric").loc[METRICS]
            row = {"model": key, "variant": name.split("| ")[1], "vs": ref.split("| ")[1],
                   "PR-AUC": boot.point[name]["PR-AUC"]}
            for m in METRICS:
                row[f"d {m}"], row[f"d {m} lo"], row[f"d {m} hi"] = c.loc[m, "diff"], c.loc[m, "lo"], c.loc[m, "hi"]
            rows.append(row)
        rows.append({"model": key, "variant": "all 26", "vs": "-", "PR-AUC": boot.point[full]["PR-AUC"]})
    out = pd.DataFrame(rows)
    shuf = pd.DataFrame(shuffle_rows).groupby(["model", "group"])["PR-AUC drop when shuffled"].mean().unstack(0)

    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, f"s003_groups_{tag}.csv"), index=False)
    shuf.to_csv(os.path.join(RESULTS, f"s003_shuffle_{tag}.csv"))
    pd.set_option("display.width", 200)
    show = out.copy()
    for m in ["PR-AUC", "$ caught@20"]:
        show[f"d {m} [95%]"] = [f"{r[f'd {m}']:+.4f} [{r[f'd {m} lo']:+.4f}, {r[f'd {m} hi']:+.4f}]"
                                if pd.notna(r.get(f"d {m}")) else "" for _, r in show.iterrows()]
    print(show[["model", "variant", "vs", "PR-AUC", "d PR-AUC [95%]", "d $ caught@20 [95%]"]].round(4)
          .to_string(index=False))
    print("\nPR-AUC lost when a group is shuffled on test (full model, mean over folds):")
    print(shuf.round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
