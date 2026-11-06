"""Error bars and "does A really beat B?" (#78).

Rows from the same product move together (a product that drops once tends to
drop again), so resampling rows would make every interval too narrow. We
resample *products* instead: each draw picks products with replacement, and a
product drawn twice counts twice. The same draw is used for every model and
every fold, so differences between models are paired, which is much tighter
than comparing two separate intervals.

Across folds, a metric is pooled by averaging the per-fold values weighted by
each fold's (resampled) number of decisions. Each fold keeps its own wait-rate
cut, so every fold's policy still says WAIT on its own top N%.

Implemented with weights so 1000 draws over ~80k decisions take seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.bench.metrics import WAIT_RATES

N_BOOT = 1000
CHUNK = 100  # draws processed at once (memory: CHUNK x rows floats)


def _metric_names(wait_rates):
    names = ["AUC", "PR-AUC"]
    for q in wait_rates:
        p = int(round(q * 100))
        names += [f"precision %@{p}", f"lift@{p}", f"$ caught@{p}", f"$ no alert@{p}"]
    return names


def weighted_stats(te: pd.DataFrame, s: np.ndarray, W: np.ndarray,
                   wait_rates=WAIT_RATES) -> dict[str, np.ndarray]:
    """Every metric for one model on one fold, under each row of weights W [B, n].
    With W all ones this matches metrics.score_policies (up to tie handling)."""
    s = np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    y = te.y.to_numpy(float)[order]
    gain = (te.price - te.fwd_min).to_numpy(float)[order]
    gain_noalert = (te.price - te.fwd_last).to_numpy(float)[order]
    # Last index of each run of equal scores: thresholds sit between distinct scores.
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    group_end = np.repeat(last, np.diff(np.r_[-1, last]))

    Ws = W[:, order]
    cw = np.cumsum(Ws, axis=1)
    cwy = np.cumsum(Ws * y, axis=1)
    tot, pos = cw[:, -1], cwy[:, -1]
    neg = tot - pos
    out = {}

    tp, fp = cwy[:, last], (cw - cwy)[:, last]
    tpr = np.c_[np.zeros(len(W)), tp / pos[:, None]]
    fpr = np.c_[np.zeros(len(W)), fp / neg[:, None]]
    out["AUC"] = np.trapezoid(tpr, fpr, axis=1)
    prec = tp / np.maximum(cw[:, last], 1e-12)
    out["PR-AUC"] = (np.diff(tpr, axis=1) * prec).sum(axis=1)

    cg = np.cumsum(Ws * gain, axis=1)
    cgn = np.cumsum(Ws * gain_noalert, axis=1)
    rows = np.arange(len(W))
    for q in wait_rates:
        p = int(round(q * 100))
        # Same cut as metrics.score_policies (s >= np.quantile(s, 1 - q)), counted in weights:
        # everything above the ceil((1-q)(n-1))-th lowest row says WAIT.
        need = tot - np.ceil((1 - q) * (tot - 1) - 1e-9)
        k = np.argmax(cw >= need[:, None] - 1e-9, axis=1)
        ke = group_end[k]  # a WAIT at a score means WAIT on every row tied with it
        precision = cwy[rows, ke] / cw[rows, ke]
        out[f"precision %@{p}"] = precision * 100
        out[f"lift@{p}"] = precision / (pos / tot)
        out[f"$ caught@{p}"] = cg[rows, ke] / tot
        out[f"$ no alert@{p}"] = cgn[rows, ke] / tot
    out["_n"] = tot
    return out


@dataclass
class Boot:
    """Bootstrap draws for every model, pooled over the folds it ran on."""
    point: dict[str, dict[str, float]]                  # model -> metric -> estimate
    draws: dict[str, dict[str, np.ndarray]]             # model -> metric -> [n_boot]
    folds: dict[str, list[str]]                         # model -> folds it was scored on
    per_fold: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    point_fold: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    wait_rates: tuple = WAIT_RATES
    n_boot: int = N_BOOT


def _pool(per_fold: dict[str, dict[str, np.ndarray]], folds: list[str], names) -> dict[str, np.ndarray]:
    n = sum(per_fold[f]["_n"] for f in folds)
    return {m: sum(per_fold[f][m] * per_fold[f]["_n"] for f in folds) / n for m in names}


def bootstrap(results, n_boot: int = N_BOOT, seed: int = 0, wait_rates=WAIT_RATES) -> Boot:
    """`results`: list of runner.FoldResult. Returns pooled point estimates and draws."""
    names = _metric_names(wait_rates)
    products = sorted(set().union(*(set(r.fold.test.asin) for r in results)))
    pidx = {a: i for i, a in enumerate(products)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(products), np.full(len(products), 1 / len(products)), size=n_boot).astype(float)

    per_fold: dict[str, dict[str, dict]] = {}   # model -> fold -> metric -> draws
    point_fold: dict[str, dict[str, dict]] = {}
    for r in results:
        te = r.fold.test.reset_index(drop=True)
        cols = te.asin.map(pidx).to_numpy()
        ones = np.ones((1, len(te)))
        for model, s in r.scores.items():
            chunks = [weighted_stats(te, s, counts[i:i + CHUNK][:, cols], wait_rates)
                      for i in range(0, n_boot, CHUNK)]
            per_fold.setdefault(model, {})[r.fold.name] = {k: np.concatenate([c[k] for c in chunks])
                                                           for k in chunks[0]}
            point_fold.setdefault(model, {})[r.fold.name] = weighted_stats(te, s, ones, wait_rates)

    point, draws, folds = {}, {}, {}
    for model, pf in per_fold.items():
        folds[model] = list(pf)
        draws[model] = _pool(pf, folds[model], names)
        point[model] = {k: float(v[0]) for k, v in _pool(point_fold[model], folds[model], names).items()}
    return Boot(point, draws, folds, per_fold, point_fold, tuple(wait_rates), n_boot)


def summary(boot: Boot, level: float = 0.95) -> pd.DataFrame:
    """One row per (model, wait rate): estimate and interval for each metric."""
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    rows = []
    for model, pt in boot.point.items():
        d = boot.draws[model]
        for q in boot.wait_rates:
            p = int(round(q * 100))
            row = {"policy": model, "target wait %": p, "folds": len(boot.folds[model])}
            for m in ["AUC", "PR-AUC", f"precision %@{p}", f"lift@{p}", f"$ caught@{p}", f"$ no alert@{p}"]:
                base = m.split("@")[0]
                lo, hi = np.nanpercentile(d[m], [lo_q, hi_q])
                row[base], row[f"{base} lo"], row[f"{base} hi"] = pt[m], lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def compare(boot: Boot, a: str, b: str, level: float = 0.95) -> pd.DataFrame:
    """Paired difference a - b on the folds both ran on, with an interval and the
    share of draws where a is not better (a one-sided bootstrap p-value)."""
    common = [f for f in boot.folds[a] if f in boot.folds[b]]
    if not common:
        raise ValueError(f"{a} and {b} share no folds")
    names = _metric_names(boot.wait_rates)
    da, db = _pool(boot.per_fold[a], common, names), _pool(boot.per_fold[b], common, names)
    pa, pb = _pool(boot.point_fold[a], common, names), _pool(boot.point_fold[b], common, names)
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    rows = []
    for m in names:
        diff = da[m] - db[m]
        lo, hi = np.nanpercentile(diff, [lo_q, hi_q])
        rows.append({"metric": m, "a": a, "b": b, "folds": len(common), "diff": float(pa[m][0] - pb[m][0]),
                     "lo": lo, "hi": hi, "p(a<=b)": float((diff <= 0).mean())})
    return pd.DataFrame(rows)
