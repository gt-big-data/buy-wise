"""R4 runs (#81): fit any registered models for one horizon and drop threshold,
over four walk-forward windows, and save validation AND test scores.

Calibrators and stay-quiet settings are fitted on validation scores, and test
scores only judge them, so both are kept. The studies (studies/s022-s024)
read these files; nothing is refitted there.

    cd backend
    python -m ml.bench.r4 --models top90,rel90,lightgbm-tuned,xgb,small-nn-tuned --tag r4
    python -m ml.bench.r4 --models top90,rel90,lightgbm-tuned --horizon 30 --threshold 0.10

Writes studies/results/scores_<data>_walk_<tag>.csv.gz and scores_val_<...>.csv.gz.
The tag defaults to r4_h<H>_t<T>, the name the horizon study reads.

The walk-forward windows here are splits.walk_forward_folds (identical at 14
days) with the boundary gap following the horizon. If ml.evaluate learns
--horizon / --threshold and saves validation scores, this runner can go.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.bench.data import DEFAULT_SOURCE, STUDIES, _usable, load_prices
from ml.bench.interface import Fold, PriceHistory
from ml.bench.metrics import score_policies
from ml.bench.models import MODELS
from ml.bench.runner import hide_labels
from ml.features import DROP_THRESHOLD, HORIZON_DAYS, REQUIRED, add_features, add_labels, daily_panel

RESULTS = STUDIES / "results"
SAVED = ["asin", "date", "y", "price", "fwd_min", "fwd_last"]
KEYS = ["fold", "asin", "date"]


def label_rows(panel: pd.DataFrame, horizon: int = HORIZON_DAYS, threshold: float = DROP_THRESHOLD) -> pd.DataFrame:
    """Feature + label rows for this question from a daily panel.

    The features never change. The horizon sets the window (fwd_min, fwd_last);
    the threshold only re-reads drop_pct, which doesn't depend on it. At the
    defaults this is exactly what ml.bench.data.load_dataset builds."""
    rows = add_labels(add_features(panel), horizon).dropna(subset=REQUIRED).reset_index(drop=True)
    rows["y"] = (rows.drop_pct >= threshold).astype(int)
    return rows


def load_rows(source=DEFAULT_SOURCE, horizon: int = HORIZON_DAYS, threshold: float = DROP_THRESHOLD):
    """label_rows() for a Prices.csv-style file, and the daily prices behind them."""
    panel = daily_panel(_usable(load_prices(Path(source))))
    return label_rows(panel, horizon, threshold), panel


def walk_forward_folds(rows: pd.DataFrame, panel: pd.DataFrame, n: int = 4, test_days: int = 91,
                       val_days: int = 91, horizon: int = HORIZON_DAYS) -> list[Fold]:
    """`n` back-to-back test windows of `test_days`, ending at the last date (four
    91-day windows cover a year). Each fold trains on everything before its
    validation window. At both boundaries a row is kept only if its
    `horizon`-day answer finishes before the next period starts."""
    h = pd.Timedelta(days=horizon)
    end = rows.date.max() + pd.Timedelta(days=1)
    history = PriceHistory(panel)
    folds = []
    for i in range(n):
        t0 = end - pd.Timedelta(days=test_days * (n - i))
        t1 = t0 + pd.Timedelta(days=test_days)
        v0 = t0 - pd.Timedelta(days=val_days)
        folds.append(Fold(f"walk{i + 1}", rows[rows.date + h < v0], rows[(rows.date >= v0) & (rows.date + h < t0)],
                          rows[(rows.date >= t0) & (rows.date < t1)], history,
                          info={"protocol": f"walk-forward {n}x{test_days}d", "horizon_days": horizon,
                                "val_start": v0, "test_start": t0, "test_end": t1}))
    return folds


def _score(model, rows: pd.DataFrame, fold: Fold, what: str) -> np.ndarray:
    """One finite score per row, labels hidden, history cut at the rows' last day."""
    s = np.asarray(model.score(hide_labels(rows), fold.history.truncated(rows.date.max())), dtype=float).ravel()
    if len(s) != len(rows):
        raise ValueError(f"{model.name}: returned {len(s)} scores for {len(rows)} {what} rows")
    if not np.isfinite(s).all():
        raise ValueError(f"{model.name}: returned {int((~np.isfinite(s)).sum())} NaN/inf {what} scores")
    return s


def run(models, fold: Fold, verbose: bool = True):
    """Fit each model on train + val, then score validation and test.
    Returns ({name: val scores}, {name: test scores}, {name: seconds})."""
    val_scores, test_scores, seconds = {}, {}, {}
    fit_until = max(fold.train.date.max(), fold.val.date.max())
    for m in models:
        reason = m.available() or (m.valid_for(fold) if hasattr(m, "valid_for") else None)
        if reason:
            print(f"  skip {m.name}: {reason}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        m.fit(fold.train, fold.val, fold.history.truncated(fit_until))
        test_scores[m.name] = _score(m, fold.test, fold, "test")
        val_scores[m.name] = _score(m, fold.val, fold, "validation")
        seconds[m.name] = time.perf_counter() - t0
        if verbose:
            print(f"  {fold.name}: {m.name} ({seconds[m.name]:.0f}s)", file=sys.stderr)
    return val_scores, test_scores, seconds


def data_tag(source) -> str:
    """Same naming as #79's ml.evaluate.data_tag: the file's stem, prefixed with its
    folder when it isn't studies/ itself."""
    p = Path(source).resolve()
    stem = p.name.split(".")[0]
    return stem if p.parent == STUDIES.resolve() else f"{p.parent.name}_{stem}"


def run_tag(horizon: int, threshold: float, tag: str | None = None) -> str:
    return tag or f"r4_h{horizon}_t{threshold:.2f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ml.bench.r4", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_SOURCE))
    ap.add_argument("--models", default="top90,rel90,lightgbm-tuned,xgb,small-nn-tuned",
                    help=f"comma list of: {', '.join(MODELS)}")
    ap.add_argument("--horizon", type=int, default=HORIZON_DAYS, help="days a WAIT has for the drop to come")
    ap.add_argument("--threshold", type=float, default=DROP_THRESHOLD,
                    help="drop that makes a WAIT right, as a fraction of today's price")
    ap.add_argument("--tag", help="saved-file suffix (default r4_h<H>_t<T>)")
    args = ap.parse_args(argv)
    keys = args.models.split(",")
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}. Known: {', '.join(MODELS)}")

    rows, panel = load_rows(args.data, args.horizon, args.threshold)
    folds = walk_forward_folds(rows, panel, horizon=args.horizon)
    val_out, test_out, timing = [], [], []
    for fold in folds:
        print(fold.summary(), file=sys.stderr)
        models = [MODELS[k](args.data) for k in keys]
        v, t, sec = run(models, fold)
        val_out.append(fold.val[SAVED].assign(fold=fold.name, **v))
        test_out.append(fold.test[SAVED].assign(fold=fold.name, **t))
        timing += [{"fold": fold.name, "model": m, "seconds": s} for m, s in sec.items()]

    test = pd.concat(test_out, ignore_index=True)
    names = [c for c in test.columns if c not in SAVED + ["fold"]]
    print(f"\nall four windows, drop >= {args.threshold:.0%} within {args.horizon} days "
          f"(base rate {test.y.mean():.1%}):")
    print(score_policies(test, {n: test[n].to_numpy() for n in names})
          .round(3).to_string(index=False))

    tag = f"{data_tag(args.data)}_walk_{run_tag(args.horizon, args.threshold, args.tag)}"
    RESULTS.mkdir(exist_ok=True)
    test.to_csv(RESULTS / f"scores_{tag}.csv.gz", index=False)
    pd.concat(val_out, ignore_index=True).to_csv(RESULTS / f"scores_val_{tag}.csv.gz", index=False)
    pd.DataFrame(timing).to_csv(RESULTS / f"timing_{tag}.csv", index=False)
    print(f"\nsaved scores_{tag}.csv.gz and scores_val_{tag}.csv.gz to {RESULTS}", file=sys.stderr)


def merge_columns(target: Path, source: Path) -> None:
    """Add the model columns of another saved run (same data, folds, horizon and
    threshold) to `target`. Rows must match exactly; used to bring in models
    scored on other branches (e.g. #80's LSTM) before they are merged."""
    a, b = (pd.read_csv(p, parse_dates=["date"]) for p in (target, source))
    if len(a) != len(b) or not a[KEYS].reset_index(drop=True).equals(b[KEYS].reset_index(drop=True)):
        raise ValueError(f"{source.name}: rows don't match {target.name}")
    if not (a.y.to_numpy() == b.y.to_numpy()).all():
        raise ValueError(f"{source.name}: labels differ from {target.name}")
    new = [c for c in b.columns if c not in a.columns]
    a.assign(**{c: b[c].to_numpy() for c in new}).to_csv(target, index=False)


if __name__ == "__main__":
    main()
