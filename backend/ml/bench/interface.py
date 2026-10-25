"""What a model gets and what it must give back.

A model is any object with:

    name: str
    outputs_probability: bool       True if score() returns P(drop). Turns on Brier and calibration error.
    fit(train, val, history)        learn from the past. Rules and pre-trained models can do nothing.
    score(test, history) -> array   one number per test row. Higher = more reason to WAIT.

`train` and `val` are feature tables with labels. `test` has the label columns
removed before the model sees it, so a model can't read the answer by accident.

Sequence models (R3) that need raw prices instead of the feature table use
`history.windows(rows, length)`, which returns the last `length` daily prices
up to and including each row's decision day. Nothing after that day is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Forward-looking columns from features.add_labels. Never shown to a model at test time.
LABEL_COLS = ("y", "fwd_min", "fwd_last", "drop_pct")


class PriceHistory:
    """Raw daily prices (one row per ASIN per day, from features.daily_panel), no labels."""

    def __init__(self, panel: pd.DataFrame, until: pd.Timestamp | None = None):
        panel = panel[[c for c in panel.columns if c not in LABEL_COLS]]
        if until is not None:
            panel = panel[panel.date <= until]
        self.panel = panel.sort_values(["asin", "date"]).reset_index(drop=True)
        self.until = self.panel.date.max() if len(self.panel) else until
        self._groups = {a: g for a, g in self.panel.groupby("asin", sort=False)}

    def truncated(self, until: pd.Timestamp) -> "PriceHistory":
        """A copy that ends at `until`."""
        return PriceHistory(self.panel, until)

    def series(self, asin: str, until: pd.Timestamp, col: str = "price") -> pd.Series:
        """One product's daily `col`, indexed by date, up to and including `until`."""
        g = self._groups.get(asin)
        if g is None:
            return pd.Series(dtype=float, name=col)
        g = g[g.date <= pd.Timestamp(until)]
        return g.set_index("date")[col]

    def windows(self, rows: pd.DataFrame, length: int, col: str = "price") -> np.ndarray:
        """Array [len(rows), length]: the last `length` daily values of `col` ending
        on each row's date (column -1 is the decision day). Days before the
        product's first record are NaN."""
        out = np.full((len(rows), length), np.nan)
        pos = pd.RangeIndex(len(rows))
        for asin, idx in pd.Series(pos, index=rows.asin.values).groupby(level=0):
            g = self._groups.get(asin)
            if g is None:
                continue
            start = g.date.iloc[0]
            vals = np.concatenate([np.full(length - 1, np.nan), g[col].to_numpy(dtype=float)])
            win = sliding_window_view(vals, length)  # win[k] ends on day start + k
            offset = (pd.to_datetime(rows.date.values[idx.values]) - start).days.to_numpy()
            ok = (offset >= 0) & (offset < len(g))
            out[idx.values[ok]] = win[offset[ok]]
        return out


@dataclass
class Fold:
    """One train / validation / test period. Labels are present on all three;
    the runner strips them from `test` before a model scores it."""
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    history: PriceHistory
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        def span(d):
            return f"{d.date.min().date()} to {d.date.max().date()}" if len(d) else "empty"
        return (f"{self.name}: train {len(self.train):,} ({span(self.train)}) | "
                f"val {len(self.val):,} ({span(self.val)}) | test {len(self.test):,} ({span(self.test)})")


@runtime_checkable
class Model(Protocol):
    name: str
    outputs_probability: bool

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, history: PriceHistory) -> None: ...

    def score(self, test: pd.DataFrame, history: PriceHistory) -> np.ndarray: ...


class BaseModel:
    """Convenience base: no-op fit, not a probability, always available."""
    name = "unnamed"
    outputs_probability = False

    def available(self) -> str | None:
        """None if the model can run here, else the reason it can't (missing file, etc.)."""
        return None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, history: PriceHistory) -> None:
        pass

    def score(self, test: pd.DataFrame, history: PriceHistory) -> np.ndarray:
        raise NotImplementedError
