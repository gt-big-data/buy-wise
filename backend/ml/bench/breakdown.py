"""Results by kind of product (#78). One overall number hides where we're good
and where we're useless: Study 006 found a 180x spread in value per decision.

Two views per segment, both with product-level error bars:
  within    how well the model ranks decisions *inside* the segment (AUC, PR-AUC)
  as run    the model's top-N% WAIT policy over the whole test set, then looked
            at inside the segment: how often it says WAIT there, how often
            that WAIT is right, and what it's worth per decision

Segments are built from what is known on the decision day (today's price,
past volatility, history so far), never from the outcome. Tercile cut-offs come
from the fold's training rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.bench.ci import weighted_stats

PRICE_BANDS = [0, 25, 75, 200, np.inf]
PRICE_LABELS = ["<$25", "$25-75", "$75-200", ">$200"]


def segments(test: pd.DataFrame, train: pd.DataFrame, first_seen: pd.Series | None = None) -> pd.DataFrame:
    """Segment columns for each test row. `first_seen`: each ASIN's first date in
    the price history (for history length)."""
    out = pd.DataFrame(index=test.index)
    out["price band"] = pd.cut(test.price, PRICE_BANDS, labels=PRICE_LABELS, right=False).astype(str)
    edges = train.vol30.quantile([1 / 3, 2 / 3]).to_numpy()
    out["volatility"] = pd.cut(test.vol30, [-np.inf, *edges, np.inf], labels=["calm", "medium", "twitchy"]).astype(str)
    out["product seen in training"] = np.where(test.asin.isin(set(train.asin)), "seen", "unseen")
    if first_seen is not None:
        days = (test.date - test.asin.map(first_seen)).dt.days
        out["history"] = pd.cut(days, [-1, 365, 730, np.inf], labels=["<1y", "1-2y", "2y+"]).astype(str)
    out["season"] = np.where(test.date.dt.month.isin([11, 12]), "Nov-Dec", "rest of year")
    for col in ("category", "brand"):
        if col in test.columns:
            out[col] = test[col].astype(str)
    return out


def _as_run(te, wait, W):
    """Weighted sums for a fixed WAIT set inside a segment, under weights W [B, n]."""
    y = te.y.to_numpy(float)
    gain = (te.price - te.fwd_min).to_numpy(float)
    return {"n": W.sum(1), "waits": (W * wait).sum(1), "hits": (W * (wait * y)).sum(1),
            "drops": (W * y).sum(1), "gain": (W * (gain * wait)).sum(1), "ceiling": (W * gain).sum(1)}


def breakdown(results, by: str, seg_fn, wait_rate: float = 0.20, n_boot: int = 500,
              seed: int = 0, min_rows: int = 200) -> pd.DataFrame:
    """Per (model, segment value): base rate, within-segment AUC / PR-AUC, and the
    as-run policy at `wait_rate`, pooled over folds, with 95% intervals.
    `seg_fn(fold)` returns segments() for that fold's test rows."""
    rng = np.random.default_rng(seed)
    products = sorted(set().union(*(set(r.fold.test.asin) for r in results)))
    pidx = {a: i for i, a in enumerate(products)}
    counts = rng.multinomial(len(products), np.full(len(products), 1 / len(products)), size=n_boot).astype(float)

    acc: dict[tuple, list[dict]] = {}
    for r in results:
        te = r.fold.test.reset_index(drop=True)
        seg = seg_fn(r.fold).reset_index(drop=True)[by]
        cols = te.asin.map(pidx).to_numpy()
        for model, s in r.scores.items():
            s = np.asarray(s, float)
            wait = s >= np.quantile(s, 1 - wait_rate)
            for value in seg.unique():
                m = (seg == value).to_numpy()
                t = te[m].reset_index(drop=True)
                if len(t) < min_rows or t.y.nunique() < 2:
                    continue
                W = np.vstack([np.ones(len(t)), counts[:, cols[m]]])  # row 0: the estimate itself
                st = weighted_stats(t, s[m], W, (wait_rate,))
                sums = _as_run(t, wait[m], W)
                acc.setdefault((model, value), []).append({
                    **sums, "auc": st["AUC"] * sums["n"], "ap": st["PR-AUC"] * sums["n"],
                    "products": t.asin.nunique(), "rows": len(t)})

    rows = []
    for (model, value), parts in acc.items():
        tot = {k: sum(p[k] for p in parts) for k in ["n", "waits", "hits", "drops", "gain", "ceiling", "auc", "ap"]}
        with np.errstate(invalid="ignore", divide="ignore"):
            stats = {
                "base rate %": tot["drops"] / tot["n"] * 100,
                "AUC": tot["auc"] / tot["n"],          # per-fold values, weighted by decisions
                "PR-AUC": tot["ap"] / tot["n"],
                "wait %": tot["waits"] / tot["n"] * 100,
                "precision %": tot["hits"] / tot["waits"] * 100,
                "$ caught": tot["gain"] / tot["n"],
                "ceiling $": tot["ceiling"] / tot["n"],
            }
        row = {"policy": model, by: value, "folds": len(parts), "rows": sum(p["rows"] for p in parts)}
        for k, v in stats.items():
            lo, hi = np.nanpercentile(v[1:], [2.5, 97.5]) if np.isfinite(v[1:]).any() else (np.nan, np.nan)
            row[k], row[f"{k} lo"], row[f"{k} hi"] = v[0], lo, hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["policy", by]).reset_index(drop=True)
