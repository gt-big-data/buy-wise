"""Study 018: getting dollars, not just ranking.

Tuned models rank better (PR-AUC) but don't catch more dollars than the served
recipe. Two ways to point them at the money, both on the saved test scores of
`python -m ml.evaluate ... --tag weighted --save` (no refitting, nothing chosen on test):

  weighted   models trained and tuned with each decision weighted by price or
             sqrt(price) (rows already in the saved scores)
  x price    rank decisions by P(drop) x today's price: WAIT where the most money
             is at stake. A fixed formula, not tuned.

Every comparison is paired, on the same product resamples.

    cd studies && python s018_dollar_policies.py
"""
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd

from r2_common import RESULTS

from ml.bench.ci import bootstrap, compare, summary  # noqa: E402

SERVED = "XGBoost (default)"
PLAIN = ["XGBoost (default)", "XGBoost (tuned, 26 features)", "CatBoost (tuned, 26 features)"]
WEIGHTED = ["XGBoost (tuned, sqrt-price-weighted, 26 features)", "XGBoost (tuned, price-weighted, 26 features)",
            "CatBoost (tuned, sqrt-price-weighted, 26 features)", "CatBoost (tuned, price-weighted, 26 features)"]
KEY = ["PR-AUC", "precision %@20", "$ caught@10", "$ caught@20", "$ caught@30"]


def main(tag="Prices_walk_weighted"):
    s = pd.read_csv(os.path.join(RESULTS, f"scores_{tag}.csv.gz"), parse_dates=["date"])
    results = []
    for fold, g in s.groupby("fold", sort=False):
        te = g.reset_index(drop=True)
        scores = {m: te[m].to_numpy() for m in PLAIN + WEIGHTED}
        for m in PLAIN:
            scores[f"{m} x price"] = te[m].to_numpy() * te.price.to_numpy()
        results.append(NS(fold=NS(name=fold, test=te), scores=scores))
    boot = bootstrap(results, n_boot=1000)

    rows = []
    for a, b in ([(w, w.replace("sqrt-price-weighted, ", "").replace("price-weighted, ", "")) for w in WEIGHTED]
                 + [(f"{m} x price", m) for m in PLAIN]
                 + [(m, SERVED) for m in [f"{p} x price" for p in PLAIN] + WEIGHTED]):
        c = compare(boot, a, b).set_index("metric").loc[KEY]
        rows.append({"a": a, "b": b, **{f"{k}": f"{c.loc[k, 'diff']:+.3f} [{c.loc[k, 'lo']:+.3f}, {c.loc[k, 'hi']:+.3f}]"
                                         for k in KEY}})
    out = pd.DataFrame(rows).drop_duplicates(["a", "b"])
    tab = summary(boot)
    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "s018_dollar_policies_paired.csv"), index=False)
    tab.to_csv(os.path.join(RESULTS, "s018_dollar_policies.csv"), index=False)
    pd.set_option("display.width", 250)
    print(tab[tab["target wait %"] == 20][["policy", "precision %", "$ caught", "$ caught lo", "$ caught hi",
                                            "PR-AUC"]].round(3).to_string(index=False))
    print("\npaired differences, a - b (95%):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
