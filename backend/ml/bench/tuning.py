"""Same tuning budget for every model (R2, #79).

A table model subclasses TabularModel and says two things: how to build the
estimator from a dict of settings (`make`), and which settings to search
(`SPACE`). Tuned(...) then gives it exactly the same search as every other model:

    budget    TUNE_BUDGET random draws from SPACE, seeded, so reruns match
    objective PR-AUC on the fold's validation rows
    stopping  early stopping on the last 15% of *train* dates (with the 14-day
              gap), so validation is used for one thing only: choosing settings

Test rows never reach this module. tune() only takes train and val.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.bench.interface import BaseModel
from ml.features import FEATURES, HORIZON_DAYS

TUNE_BUDGET = 40      # trials per model per fold. Same for everyone.
ES_FRACTION = 0.15    # share of train dates held back for early stopping


# ---- search spaces --------------------------------------------------------------

def uniform(lo, hi):
    return lambda rng: float(rng.uniform(lo, hi))


def log_uniform(lo, hi):
    return lambda rng: float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def integer(lo, hi):
    return lambda rng: int(rng.integers(lo, hi + 1))


def choice(*options):
    return lambda rng: options[int(rng.integers(len(options)))]


def sample(space: dict[str, Callable], rng: np.random.Generator) -> dict:
    return {k: draw(rng) for k, draw in space.items()}


# ---- the base class for table models ---------------------------------------------

def early_stopping_split(train: pd.DataFrame, fraction: float = ES_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold back the last `fraction` of train dates for early stopping. Rows whose
    14-day answer reaches into the held-back part are dropped from the fitting part."""
    dates = np.sort(train.date.unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1 - fraction))])
    return train[train.date + pd.Timedelta(days=HORIZON_DAYS) < cut], train[train.date >= cut]


class TabularModel(BaseModel):
    """A model that reads only the feature table. Subclasses set `key`, `DEFAULTS`,
    `SPACE` and implement `make(params)` and `_fit(est, X, y, X_es, y_es)`."""
    key = "tabular"
    outputs_probability = True
    DEFAULTS: dict = {}
    SPACE: dict[str, Callable] = {}

    def __init__(self, params: dict | None = None, feature_cols: list[str] | None = None, label: str = "default"):
        self.params = dict(self.DEFAULTS if params is None else params)
        self.feature_cols = list(feature_cols or FEATURES)
        extra = "" if self.feature_cols == FEATURES else f", {len(self.feature_cols)} features"
        self.name = f"{self.key} ({label}{extra})"

    def make(self, params: dict):
        raise NotImplementedError

    def _fit(self, est, X, y, X_es, y_es):
        est.fit(X, y)

    def fit(self, train, val, history=None):
        fit_rows, es_rows = early_stopping_split(train)
        cols = self.feature_cols
        self.est = self.make(self.params)
        self._fit(self.est, fit_rows[cols], fit_rows.y, es_rows[cols], es_rows.y)

    def score(self, test, history=None):
        return self.est.predict_proba(test[self.feature_cols])[:, 1]


def tune(model_cls: type[TabularModel], train: pd.DataFrame, val: pd.DataFrame,
         feature_cols: list[str] | None = None, budget: int | None = None,
         seed: int = 0) -> tuple[TabularModel, pd.DataFrame]:
    """Random search over model_cls.SPACE. The first trial is the model's defaults,
    so tuning can never end up worse on validation than not tuning.
    Returns the best fitted model and one row per trial."""
    budget = TUNE_BUDGET if budget is None else budget
    rng = np.random.default_rng(seed)
    best, best_score, trials = None, -np.inf, []
    for i in range(budget):
        params = dict(model_cls.DEFAULTS) if i == 0 else {**model_cls.DEFAULTS, **sample(model_cls.SPACE, rng)}
        m = model_cls(params, feature_cols)
        m.fit(train, val)
        s = average_precision_score(val.y, m.score(val))
        trials.append({"trial": i, "val PR-AUC": s, **params})
        if s > best_score:
            best, best_score = m, s
    return best, pd.DataFrame(trials)


class Tuned(BaseModel):
    """Wraps a TabularModel class: fit() runs tune() on this fold's train and val,
    then scores test with the winner."""
    outputs_probability = True

    def __init__(self, model_cls: type[TabularModel], feature_cols: list[str] | None = None,
                 budget: int | None = None):
        self.model_cls = model_cls
        self.feature_cols = list(feature_cols or FEATURES)
        self.budget = budget
        extra = "" if self.feature_cols == FEATURES else f", {len(self.feature_cols)} features"
        self.name = f"{model_cls.key} (tuned{extra})"
        self.trials: pd.DataFrame | None = None

    def fit(self, train, val, history=None):
        self.best, self.trials = tune(self.model_cls, train, val, self.feature_cols, self.budget)
        self.params = self.best.params

    def score(self, test, history=None):
        return self.best.score(test)
