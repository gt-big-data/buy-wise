"""Fit each model on a fold and collect its test scores, with the guard rails:
labels are hidden at test time, history ends where each period ends, and a
model must return one finite number per test row."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.bench.interface import LABEL_COLS, BaseModel, Fold
from ml.bench.metrics import WAIT_RATES, score_policies


@dataclass
class FoldResult:
    fold: Fold
    scores: dict[str, np.ndarray] = field(default_factory=dict)   # model name -> one score per test row
    probabilistic: set[str] = field(default_factory=set)
    seconds: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)          # model name -> reason

    def table(self, wait_rates=WAIT_RATES) -> pd.DataFrame:
        r = score_policies(self.fold.test, self.scores, wait_rates, self.probabilistic)
        r.insert(0, "fold", self.fold.name)
        return r


def hide_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LABEL_COLS if c in df.columns])


def run_model(model: BaseModel, fold: Fold) -> np.ndarray:
    fit_until = max(fold.train.date.max(), fold.val.date.max()) if len(fold.val) else fold.train.date.max()
    model.fit(fold.train, fold.val, fold.history.truncated(fit_until))
    test = hide_labels(fold.test)
    s = np.asarray(model.score(test, fold.history.truncated(fold.test.date.max())), dtype=float).ravel()
    if len(s) != len(fold.test):
        raise ValueError(f"{model.name}: returned {len(s)} scores for {len(fold.test)} test rows")
    if not np.isfinite(s).all():
        raise ValueError(f"{model.name}: returned {int((~np.isfinite(s)).sum())} NaN/inf scores")
    return s


def run_fold(models: list[BaseModel], fold: Fold, verbose: bool = True) -> FoldResult:
    res = FoldResult(fold)
    for m in models:
        reason = m.available() or (m.valid_for(fold) if hasattr(m, "valid_for") else None)
        if reason:
            res.skipped[m.name] = reason
            if verbose:
                print(f"  skip {m.name}: {reason}")
            continue
        t0 = time.perf_counter()
        res.scores[m.name] = run_model(m, fold)
        res.seconds[m.name] = time.perf_counter() - t0
        if m.outputs_probability:
            res.probabilistic.add(m.name)
        if verbose:
            print(f"  ran  {m.name} ({res.seconds[m.name]:.1f}s)")
    return res
