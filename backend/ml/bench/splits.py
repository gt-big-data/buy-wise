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


def legacy_fold(rows: pd.DataFrame, panel: pd.DataFrame) -> Fold:
    """The single 60/20/20 split the spring leaderboard used. Kept so today's
    numbers can always be reproduced."""
    s = chronological_split(rows)
    return Fold("legacy", s.train, s.val, s.test, PriceHistory(panel),
                info={"protocol": "legacy 60/20/20", "horizon_days": HORIZON_DAYS})
