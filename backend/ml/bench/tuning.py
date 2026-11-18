"""Same tuning budget for every model (R2, #79).

A table model subclasses TabularModel and says two things: how to build the
estimator from a dict of settings (`make`), and which settings to search
(`SPACE`). Tuned(...) then gives it exactly the same search as every other model:

    budget    TUNE_BUDGET random draws from SPACE, seeded, so reruns match
    objective PR-AUC on the fold's validation rows (price-weighted for the
              price-weighted models, so tuning and training agree on what counts)
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


# How much each decision counts. Unweighted, a $10 cable and a $400 TV count the
# same, so a model can rank well overall and still miss where the money is.
WEIGHTS = {None: None, "sqrt_price": np.sqrt, "price": lambda p: p}
WEIGHT_LABELS = {None: "", "sqrt_price": ", sqrt-price-weighted", "price": ", price-weighted"}


def row_weights(rows: pd.DataFrame, scheme: str | None) -> np.ndarray | None:
    """Per-row weights from today's price (known on the decision day), mean 1."""
    if WEIGHTS[scheme] is None:
        return None
    w = WEIGHTS[scheme](rows.price.to_numpy(float))
    return w / w.mean()


def objective(val: pd.DataFrame, scores: np.ndarray, scheme: str | None) -> float:
    """What tuning maximises: PR-AUC on validation, weighted the same way as training."""
    return average_precision_score(val.y, scores, sample_weight=row_weights(val, scheme))


def _suffix(feature_cols: list[str], weight: str | None) -> str:
    extra = "" if feature_cols == FEATURES else f", {len(feature_cols)} features"
    return f"{WEIGHT_LABELS[weight]}{extra}"


class TabularModel(BaseModel):
    """A model that reads only the feature table. Subclasses set `key`, `DEFAULTS`,
    `SPACE` and implement `make(params)` and `_fit(est, X, y, X_es, y_es, w, w_es)`
    (w and w_es are per-row weights, or None).

    `weight` (None, "sqrt_price" or "price") makes training count expensive
    decisions more. Weights come from each row's price and are only used to fit;
    scoring still sees nothing but the features."""
    key = "tabular"
    outputs_probability = True
    DEFAULTS: dict = {}
    SPACE: dict[str, Callable] = {}

    def __init__(self, params: dict | None = None, feature_cols: list[str] | None = None, label: str = "default",
                 weight: str | None = None):
        self.params = dict(self.DEFAULTS if params is None else params)
        self.feature_cols = list(feature_cols or FEATURES)
        self.weight = weight
        self.name = f"{self.key} ({label}{_suffix(self.feature_cols, weight)})"

    def make(self, params: dict):
        raise NotImplementedError

    def _fit(self, est, X, y, X_es, y_es, w=None, w_es=None):
        est.fit(X, y, sample_weight=w)

    def fit(self, train, val, history=None):
        fit_rows, es_rows = early_stopping_split(train)
        cols = self.feature_cols
        self.est = self.make(self.params)
        self._fit(self.est, fit_rows[cols], fit_rows.y, es_rows[cols], es_rows.y,
                  row_weights(fit_rows, self.weight), row_weights(es_rows, self.weight))

    def score(self, test, history=None):
        return self.est.predict_proba(test[self.feature_cols])[:, 1]


def tune(model_cls: type[TabularModel], train: pd.DataFrame, val: pd.DataFrame,
         feature_cols: list[str] | None = None, budget: int | None = None,
         seed: int = 0, weight: str | None = None) -> tuple[TabularModel, pd.DataFrame]:
    """Random search over model_cls.SPACE. The first trial is the model's defaults,
    so tuning can never end up worse on validation than not tuning.
    Returns the best fitted model and one row per trial."""
    budget = TUNE_BUDGET if budget is None else budget
    rng = np.random.default_rng(seed)
    best, best_score, trials = None, -np.inf, []
    for i in range(budget):
        params = dict(model_cls.DEFAULTS) if i == 0 else {**model_cls.DEFAULTS, **sample(model_cls.SPACE, rng)}
        m = model_cls(params, feature_cols, weight=weight)
        m.fit(train, val)
        scores = m.score(val)
        s = objective(val, scores, weight)
        trials.append({"trial": i, "val objective": s, "val PR-AUC": average_precision_score(val.y, scores),
                       "weight": weight or "none", **params})
        if s > best_score:
            best, best_score = m, s
    return best, pd.DataFrame(trials)


class Tuned(BaseModel):
    """Wraps a TabularModel class: fit() runs tune() on this fold's train and val,
    then scores test with the winner."""
    outputs_probability = True

    def __init__(self, model_cls: type[TabularModel], feature_cols: list[str] | None = None,
                 budget: int | None = None, weight: str | None = None):
        self.model_cls = model_cls
        self.feature_cols = list(feature_cols or FEATURES)
        self.budget = budget
        self.weight = weight
        self.name = f"{model_cls.key} (tuned{_suffix(self.feature_cols, weight)})"
        self.trials: pd.DataFrame | None = None

    def fit(self, train, val, history=None):
        self.best, self.trials = tune(self.model_cls, train, val, self.feature_cols, self.budget,
                                      weight=self.weight)
        self.params = self.best.params

    def score(self, test, history=None):
        return self.best.score(test)
