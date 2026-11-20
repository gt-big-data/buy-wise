"""How rows are divided into train / validation / test periods.

The rule everywhere: a row may only sit in a period if its 14-day answer
finishes before the next period starts. Otherwise training would contain the
prices the model is later tested on.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.bench.interface import Fold, PriceHistory
from ml.features import HORIZON_DAYS


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(p: pd.DataFrame) -> Split:
    """60/20/20 by date. A row only lands in a period if its 14-day label
    finishes before the next period starts, so no answer leaks across."""
    dates = sorted(p.date.unique())
    c1 = pd.Timestamp(dates[int(len(dates) * 0.6)])
    c2 = pd.Timestamp(dates[int(len(dates) * 0.8)])
    h = pd.Timedelta(days=HORIZON_DAYS)
    return Split(
        train=p[p.date + h < c1],
        val=p[(p.date >= c1) & (p.date + h < c2)],
        test=p[p.date >= c2],
    )


def walk_forward_folds(rows: pd.DataFrame, panel: pd.DataFrame, n: int = 4,
                       test_days: int = 91, val_days: int = 91) -> list[Fold]:
    """`n` back-to-back test windows of `test_days`, ending at the last date.
    Four 91-day windows cover a full year, so every season is tested once and
    one lucky stretch can't decide the result.

    Each fold trains on everything before its validation window, so training
    grows fold by fold. At both boundaries a row is kept only if its 14-day
    answer finishes before the next period starts."""
    h = pd.Timedelta(days=HORIZON_DAYS)
    end = rows.date.max() + pd.Timedelta(days=1)
    history = PriceHistory(panel)
    folds = []
    for i in range(n):
        t0 = end - pd.Timedelta(days=test_days * (n - i))
        t1 = t0 + pd.Timedelta(days=test_days)
        v0 = t0 - pd.Timedelta(days=val_days)
        train = rows[rows.date + h < v0]
        val = rows[(rows.date >= v0) & (rows.date + h < t0)]
        test = rows[(rows.date >= t0) & (rows.date < t1)]
        folds.append(Fold(f"walk{i + 1}", train, val, test, history,
                          info={"protocol": f"walk-forward {n}x{test_days}d", "horizon_days": HORIZON_DAYS,
                                "val_start": v0, "test_start": t0, "test_end": t1}))
    return folds


def legacy_fold(rows: pd.DataFrame, panel: pd.DataFrame) -> Fold:
    """The single 60/20/20 split the spring leaderboard used. Kept so today's
    numbers can always be reproduced."""
    s = chronological_split(rows)
    return Fold("legacy", s.train, s.val, s.test, PriceHistory(panel),
                info={"protocol": "legacy 60/20/20", "horizon_days": HORIZON_DAYS})
