"""Shared setup for the R4 studies (s022, s023, s024; #81).

They read the scores `python -m ml.bench.r4` wrote for the test AND validation
rows of each walk-forward window, so no model is refitted here. Calibrators and
stay-quiet settings are fitted on validation; test only judges.

Which model leads: the first of PREFERRED found in the saved scores. Tuned
CatBoost is the model #79 serves; on main alone it isn't registered yet, so
LightGBM (tuned) leads there.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from ml.bench.r4 import data_tag, load_rows, run_tag, walk_forward_folds  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")
SOURCES = {
    "prices": os.path.join(HERE, "Prices.csv"),
    "apr2026_500": os.path.join(HERE, "snapshots", "apr2026_500", "Prices.csv.gz"),
}
# Best of R2 and R3 first (#79, #80), then what main registers today.
PREFERRED = ["CatBoost (tuned, 26 features)", "XGBoost (tuned, 26 features)", "LSTM (seed 0)",
             "LightGBM (tuned)", "XGBoost (clean recipe, retrained)", "Small neural net (tuned)"]
HORIZONS = (7, 14, 30)
THRESHOLDS = (0.05, 0.08, 0.10, 0.15)
KEYS = ["fold", "asin", "date"]

# Reference palette (dataviz skill), slots 1-3 validated together; raw scores in muted ink.
# Slot 3 is under 3:1 on the surface, so every series also gets a marker and a direct label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
})


def source(name: str) -> str:
    return SOURCES.get(name, name)


def tag(name: str, horizon: int | None = None, threshold: float | None = None) -> str:
    """Saved-file tag: the 14d/8% calibration run is `_r4`, sweep cells `_r4_h{H}_t{T:.2f}`."""
    run = "r4" if horizon is None else run_tag(horizon, threshold)
    return f"{data_tag(source(name))}_walk_{run}"


def models_in(df: pd.DataFrame) -> list[str]:
    """Probability models in a saved score table, best first."""
    return [m for m in PREFERRED if m in df.columns]


def primary(df: pd.DataFrame) -> str:
    found = models_in(df)
    if not found:
        raise SystemExit(f"none of {PREFERRED} in the saved scores")
    return found[0]


# Same segments as #79's ml.bench.breakdown.segments, built from what is known on the
# decision day. Cut-offs come from the window's train rows.
PRICE_BANDS = [0, 25, 75, 200, np.inf]
PRICE_LABELS = ["<$25", "$25-75", "$75-200", ">$200"]


def segments(rows: pd.DataFrame, train: pd.DataFrame, first_seen: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["price band"] = pd.cut(rows.price, PRICE_BANDS, labels=PRICE_LABELS, right=False).astype(str)
    edges = train.vol30.quantile([1 / 3, 2 / 3]).to_numpy()
    out["volatility"] = pd.cut(rows.vol30, [-np.inf, *edges, np.inf], labels=["calm", "medium", "twitchy"]).astype(str)
    out["product seen in training"] = np.where(rows.asin.isin(set(train.asin)), "seen", "unseen")
    days = (rows.date - rows.asin.map(first_seen)).dt.days
    out["history"] = pd.cut(days, [-1, 365, 730, np.inf], labels=["<1y", "1-2y", "2y+"]).astype(str)
    out["season"] = np.where(rows.date.dt.month.isin([11, 12]), "Nov-Dec", "rest of year")
    return out


def scores(name: str, horizon: int | None = None, threshold: float | None = None, with_val: bool = True):
    """(validation, test) score tables for one run; (None, test) without `with_val`."""
    t = tag(name, horizon, threshold)
    paths = [os.path.join(RESULTS, f"scores_val_{t}.csv.gz"), os.path.join(RESULTS, f"scores_{t}.csv.gz")]
    if not with_val:
        paths = paths[1:]
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found. Run python -m ml.bench.r4 for it first "
                             "(see FINDINGS.md, Running these).")
    out = tuple(pd.read_csv(p, parse_dates=["date"]) for p in paths)
    return out if with_val else (None, out[0])


def with_context(name: str, val: pd.DataFrame, test: pd.DataFrame, horizon: int = 14, threshold: float = 0.08):
    """Add segments and the features the stay-quiet rule reads (vol30, a
    recent-drops measure, drop_pct) to the saved rows, and return each fold's
    train rows too. Rebuilds the folds exactly as ml.bench.r4 did."""
    rows, panel = load_rows(source(name), horizon, threshold)
    folds = walk_forward_folds(rows, panel, horizon=horizon)
    first_seen = panel.groupby("asin").date.min()
    # past_drop_rate arrives with #79; until then jumpiness reads nchg30 (quiet.jumpiness).
    cols = [c for c in ("vol30", "past_drop_rate", "nchg30", "drop_pct") if c in rows.columns]
    seg_t, seg_v, trains = [], [], {}
    for f in folds:
        trains[f.name] = f.train
        s = segments(f.test, f.train, first_seen)
        seg_t.append(pd.concat([f.test[["asin", "date", *cols]], s], axis=1).assign(fold=f.name))
        sv = segments(f.val, f.train, first_seen)
        seg_v.append(pd.concat([f.val[["asin", "date", *cols]], sv], axis=1).assign(fold=f.name))
    test = test.merge(pd.concat(seg_t), on=KEYS, how="left", validate="one_to_one")
    val = val.merge(pd.concat(seg_v), on=KEYS, how="left", validate="one_to_one")
    assert test.vol30.notna().all() and val.vol30.notna().all(), "saved rows don't match the rebuilt folds"
    return val, test, trains


def save_fig(fig, name: str):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fmt_ci(est, lo, hi, d=3):
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def show(df: pd.DataFrame, title: str):
    pd.set_option("display.width", 220)
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


__all__ = ["np", "pd", "plt"]
