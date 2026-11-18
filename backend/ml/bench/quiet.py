"""When to stay quiet (R4, #81).

The rule decided on #81: show WAIT only when BOTH

    the calibrated chance of a drop  >= min_drop_chance   (protects precision)
    the expected net saving          >= min_saving ($)    (stops 13-cent interruptions)

Expected net saving counts what a wrong WAIT costs. With p the calibrated chance:

    net = price * (p * G_right + (1 - p) * G_wrong)

where G_right / G_wrong are the average day-N saving, as a fraction of today's
price, when the drop did / didn't come. G_wrong is usually negative: after a
wrong WAIT the shopper often pays more on day N. Both are measured on *train*
rows, per volatility band (cut-offs from train), so jumpy products, which gain
more when right and lose more when wrong, get their own numbers.

"Day N" means the shopper just buys on the last day of the window, with no
alert catching the low. That is the measure #81 chose; `$ at low` is reported
next to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = ("calm", "medium", "twitchy")


def band_edges(train: pd.DataFrame) -> np.ndarray:
    """vol30 tercile cut-offs from train (same as breakdown.segments)."""
    return train.vol30.quantile([1 / 3, 2 / 3]).to_numpy()


def bands(rows: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    return pd.cut(rows.vol30, [-np.inf, *edges, np.inf], labels=list(BANDS)).astype(str).to_numpy()


def day_n_gain(rows: pd.DataFrame) -> np.ndarray:
    """Saving if the shopper waits and buys on day N, as a fraction of today's price.
    Negative = paid more. Clipped to +/-100%, the rule features.add_labels uses for
    Keepa errors (0.2% of wrong WAITs on Prices.csv move more than that)."""
    return ((rows.price - rows.fwd_last) / rows.price).clip(-1, 1).to_numpy(float)


def wait_economics(train: pd.DataFrame, edges: np.ndarray | None = None) -> pd.DataFrame:
    """Per volatility band, from train rows only: what a right and a wrong WAIT is worth.

    G_right       mean day-N saving when the drop came (fraction of price)
    G_right_low   mean saving at the window's low when it came (an alert catches it)
    G_wrong       mean day-N saving when it didn't (usually < 0: paid more)
    higher_share  share of wrong WAITs where day N cost more than today
    higher_by     mean extra paid on those, as a fraction of price
    """
    edges = band_edges(train) if edges is None else edges
    t = train.assign(band=bands(train, edges), g=day_n_gain(train))
    rows = []
    for band in BANDS:
        b = t[t.band == band]
        right, wrong = b[b.y == 1], b[b.y == 0]
        higher = wrong[wrong.g < 0]
        rows.append({"band": band, "rows": len(b), "base rate": b.y.mean(),
                     "G_right": right.g.mean(), "G_right_low": right.drop_pct.mean(),
                     "G_wrong": wrong.g.mean(), "higher_share": (wrong.g < 0).mean(),
                     "higher_by": -higher.g.mean() if len(higher) else 0.0})
    return pd.DataFrame(rows).set_index("band")


def net_saving(p, rows: pd.DataFrame, econ: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Expected day-N dollars from a WAIT on each row: price * (p*G_right + (1-p)*G_wrong)."""
    p = np.asarray(p, float)
    b = bands(rows, edges)
    g_right = econ.G_right.reindex(b).to_numpy(float)
    g_wrong = econ.G_wrong.reindex(b).to_numpy(float)
    return rows.price.to_numpy(float) * (p * g_right + (1 - p) * g_wrong)


def jumpiness(rows: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """How jumpy a product is, 0..1: the mean of two percentiles among train rows,
    its volatility (vol30) and how often it has dropped. The drop measure is
    past_drop_rate (share of past windows with an 8% drop, from #79) when the
    rows have it, else nchg30 (price changes in the last 30 days). A missing
    value counts as average."""
    drops = "past_drop_rate" if "past_drop_rate" in rows.columns and "past_drop_rate" in train.columns else "nchg30"
    out = []
    for col in ("vol30", drops):
        ref = np.sort(train[col].dropna().to_numpy(float))
        x = rows[col].to_numpy(float)
        pct = np.searchsorted(ref, x, side="right") / max(len(ref), 1)
        out.append(np.where(np.isnan(x), 0.5, pct))
    return np.mean(out, axis=0)


# ---- the curve -----------------------------------------------------------------

def outcome(rows: pd.DataFrame, speak: np.ndarray) -> dict:
    """What a set of shown WAITs is worth. Dollars per decision divide by every
    decision (quiet ones save $0); per shown WAIT divides by WAITs shown."""
    speak = np.asarray(speak, bool)
    n, k = len(rows), int(speak.sum())
    s = rows[speak]
    day_n = (s.price - s.fwd_last).to_numpy(float)
    low = (s.price - s.fwd_min).to_numpy(float)
    wrong = s[s.y == 0]
    extra = (wrong.fwd_last - wrong.price).clip(lower=0)
    return {
        "quiet %": (1 - k / n) * 100 if n else np.nan,
        "shown": k,
        "precision %": s.y.mean() * 100 if k else np.nan,
        "$ day N per WAIT": day_n.mean() if k else np.nan,
        "$ at low per WAIT": low.mean() if k else np.nan,
        "$ day N per decision": day_n.sum() / n if n else np.nan,
        "$ at low per decision": low.sum() / n if n else np.nan,
        "wrong WAITs %": (1 - s.y.mean()) * 100 if k else np.nan,
        "wrong: paid more %": (wrong.fwd_last > wrong.price).mean() * 100 if len(wrong) else np.nan,
        "wrong: extra paid $": extra.mean() if len(wrong) else np.nan,
    }


def single_knob_curve(rows: pd.DataFrame, score: np.ndarray, name: str,
                      quiet_levels=np.arange(0, 100, 2.5)) -> pd.DataFrame:
    """Stay quiet on the lowest q% by `score`, for each q. One curve per rule."""
    score = np.asarray(score, float)
    out = []
    for q in quiet_levels:
        cut = np.quantile(score, q / 100) if q > 0 else -np.inf
        out.append({"rule": name, "target quiet %": q, "cut": cut, **outcome(rows, score > cut if q > 0 else
                                                                                 np.ones(len(rows), bool))})
    return pd.DataFrame(out)


def grid(rows: pd.DataFrame, p: np.ndarray, net: np.ndarray,
         taus=np.round(np.arange(0.0, 0.81, 0.05), 2), mins=(-np.inf, 0, 0.25, 0.5, 1, 2, 3, 5, 10)) -> pd.DataFrame:
    """The combined rule at every (min_drop_chance, min_saving)."""
    p, net = np.asarray(p, float), np.asarray(net, float)
    return pd.DataFrame([{"rule": "chance + net $", "min_drop_chance": t, "min_saving": m,
                          **outcome(rows, (p >= t) & (net >= m))} for t in taus for m in mins])


def frontier(g: pd.DataFrame, by: str = "$ day N per decision") -> pd.DataFrame:
    """For plotting the combined rule as one curve: at each quiet level (2.5-point
    bins), the grid point with the most `by`."""
    g = g[g.shown > 0].assign(bucket=(g["quiet %"] / 2.5).round() * 2.5)
    return g.loc[g.groupby("bucket")[by].idxmax()].sort_values("quiet %").drop(columns="bucket")


def pick_operating_point(g: pd.DataFrame, min_precision: float = 50.0, min_shown: int = 200) -> pd.Series:
    """On validation: the most day-N dollars per decision among settings whose
    shown WAITs are right at least `min_precision`% of the time. The floor is the
    default from the #81 plan; the published curve shows what moving it costs."""
    ok = g[(g["precision %"] >= min_precision) & (g.shown >= min_shown) & np.isfinite(g.min_saving)]
    if not len(ok):
        raise ValueError(f"no setting reaches {min_precision}% precision on {min_shown}+ WAITs")
    best = ok["$ day N per decision"].max()
    # Ties (within a cent): the quieter setting.
    near = ok[ok["$ day N per decision"] >= best - 0.01]
    return near.sort_values(["quiet %", "min_drop_chance", "min_saving"], ascending=[False, False, False]).iloc[0]
