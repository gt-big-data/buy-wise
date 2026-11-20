"""Is "a 40% chance of a drop" really 40%? (R4, #81)

A calibrator maps a model's raw score to the chance the panel shows. Three are
compared, all fitted on a fold's *validation* rows and judged on its test rows:

  isotonic  a step function (today's served choice). Flexible, but many products
            end up showing the same number and the steps can overfit.
  platt     logistic regression on logit(score): two numbers, smooth.
  beta      logistic regression on log(score) and -log(1 - score): three numbers.
            Can bend differently at the low and high ends, which boosted trees
            often need (Kull et al., 2017).

Test rows never reach a fit. Error bars resample *products*, as in ci.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATORS = ("isotonic", "platt", "beta")
SIMPLEST = "platt"  # the tie-break when no calibrator is clearly better (#81 decision)
EPS = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def _logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    kind: str
    model: object
    keep: tuple = ()  # beta: which of its two inputs survived the monotonicity check

    def _x(self, p):
        p = _clip(p)
        if self.kind == "platt":
            return _logit(p)[:, None]
        if self.kind == "beta":
            cols = np.c_[np.log(p), -np.log(1 - p)]
            return cols[:, list(self.keep)]
        raise ValueError(self.kind)

    def predict(self, p) -> np.ndarray:
        if self.kind == "none":
            return np.asarray(p, float)
        if self.kind == "isotonic":
            return self.model.predict(np.asarray(p, float))
        return self.model.predict_proba(self._x(p))[:, 1]


def fit_calibrator(kind: str, p, y) -> Calibrator:
    """Fit `kind` on scores `p` and outcomes `y` (validation rows only)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    if kind == "none":
        return Calibrator("none", None)
    if kind == "isotonic":
        return Calibrator(kind, IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y))
    if kind == "platt":
        c = Calibrator(kind, None)
        c.model = LogisticRegression(C=1e6, max_iter=1000).fit(c._x(p), y)
        return c
    if kind == "beta":
        # Both coefficients must be >= 0 or the map can turn downwards. If one comes
        # out negative, drop that input and refit (as the betacal package does).
        keep = (0, 1)
        while True:
            c = Calibrator(kind, None, keep)
            c.model = LogisticRegression(C=1e6, max_iter=1000).fit(c._x(p), y)
            neg = [k for k, w in zip(keep, c.model.coef_[0]) if w < 0]
            if not neg or len(keep) == 1:
                return c
            keep = tuple(k for k in keep if k != neg[0])
    raise ValueError(f"unknown calibrator {kind!r}; known: none, {', '.join(CALIBRATORS)}")


def calibrate_folds(val: pd.DataFrame, test: pd.DataFrame, score_col: str, kind: str) -> np.ndarray:
    """Calibrated test scores: each fold's calibrator is fitted on that fold's
    validation rows only. Rows keep `test`'s order."""
    out = np.full(len(test), np.nan)
    pos = np.arange(len(test))
    for fold, t in test.groupby("fold", sort=False):
        v = val[val.fold == fold]
        cal = fit_calibrator(kind, v[score_col], v.y)
        out[pos[(test.fold == fold).to_numpy()]] = cal.predict(t[score_col])
    return out


def crossfit_val(val: pd.DataFrame, score_col: str, kind: str, k: int = 2, seed: int = 0) -> np.ndarray:
    """Calibrated *validation* scores without scoring a row with a calibrator
    fitted on it: products are split into `k` groups, and each group is
    calibrated by a fit on the others (same fold). Used to choose stay-quiet
    settings on validation without the in-sample optimism of isotonic."""
    out = np.full(len(val), np.nan)
    pos = np.arange(len(val))
    rng = np.random.default_rng(seed)
    for fold, v in val.groupby("fold", sort=False):
        asins = v.asin.unique()
        group = dict(zip(asins, rng.integers(0, k, len(asins))))
        g = v.asin.map(group).to_numpy()
        idx = pos[(val.fold == fold).to_numpy()]
        for i in range(k):
            cal = fit_calibrator(kind, v[score_col][g != i], v.y[g != i])
            out[idx[g == i]] = cal.predict(v[score_col][g == i])
    return out


# ---- how well it holds up ------------------------------------------------------

def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Said vs happened in `bins` equal-count bins of the shown chance."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    rows = []
    for i, chunk in enumerate(np.array_split(order, bins)):
        if len(chunk):
            rows.append({"bin": i, "rows": len(chunk), "said": p[chunk].mean(), "happened": y[chunk].mean(),
                         "said lo": p[chunk].min(), "said hi": p[chunk].max()})
    return pd.DataFrame(rows)


@dataclass
class Draws:
    """Product resamples. counts[0] is all ones (the estimate itself); counts[1:]
    draw products with replacement, so a product drawn twice counts twice.
    inv maps each row to its product. Every row of a product shares its count,
    so metrics are computed from per-product sums: no [draws x rows] matrix."""
    counts: np.ndarray  # [n_boot + 1, products]
    inv: np.ndarray     # [rows]


def product_draws(asins, n_boot: int = 1000, seed: int = 0) -> Draws:
    """Same seed and rows, same draws: build once and reuse it to pair calibrators."""
    products, inv = np.unique(np.asarray(asins), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(products), np.full(len(products), 1 / len(products)), size=n_boot).astype(float)
    return Draws(np.vstack([np.ones(len(products)), counts]), inv)


def _per_product(values: np.ndarray, inv: np.ndarray, groups: np.ndarray, n_groups: int, n_products: int):
    """[products, groups] sums of `values`."""
    return np.bincount(inv * n_groups + groups, weights=values,
                       minlength=n_products * n_groups).reshape(n_products, n_groups)


def boot_ece(y, p, draws: Draws, mask=None, bins: int = 10) -> np.ndarray:
    """ECE (metrics.ece: equal-count bins) for every draw, on the rows in `mask`.
    Bin edges come from the estimate's scores, so every draw is binned the same way."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    inv = draws.inv
    if mask is not None:
        mask = np.asarray(mask, bool)
        y, p, inv = y[mask], p[mask], inv[mask]
    order = np.argsort(p, kind="stable")
    b = np.empty(len(p), int)
    for i, chunk in enumerate(np.array_split(order, bins)):
        b[chunk] = i
    P = draws.counts.shape[1]
    n_b = draws.counts @ _per_product(np.ones(len(p)), inv, b, bins, P)
    gap = np.abs(draws.counts @ _per_product(p, inv, b, bins, P) - draws.counts @ _per_product(y, inv, b, bins, P))
    with np.errstate(invalid="ignore", divide="ignore"):
        return gap.sum(1) / n_b.sum(1)


def boot_mean(values, draws: Draws, mask=None) -> np.ndarray:
    """Mean of a per-row quantity for every draw (Brier: values = (p - y) ** 2)."""
    v = np.asarray(values, float)
    inv = draws.inv
    if mask is not None:
        mask = np.asarray(mask, bool)
        v, inv = v[mask], inv[mask]
    P = draws.counts.shape[1]
    zero = np.zeros(len(v), int)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((draws.counts @ _per_product(v, inv, zero, 1, P))[:, 0]
                / (draws.counts @ _per_product(np.ones(len(v)), inv, zero, 1, P))[:, 0])


def boot_brier(y, p, draws: Draws, mask=None) -> np.ndarray:
    return boot_mean((np.asarray(p, float) - np.asarray(y, float)) ** 2, draws, mask)


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """(estimate, lo, hi) from weighted_* output (row 0 = estimate)."""
    a = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(draws[1:], [a, 100 - a])
    return float(draws[0]), float(lo), float(hi)


def pick(ece_draws: dict[str, np.ndarray], level: float = 0.95) -> tuple[str, str]:
    """The decision rule for #81: lowest ECE wins, unless its paired difference with
    the simplest calibrator includes 0, in which case serve the simplest.
    `ece_draws`: calibrator -> weighted_ece output on the same weights.
    Returns (choice, reason)."""
    best = min(ece_draws, key=lambda k: ece_draws[k][0])
    if best == SIMPLEST or SIMPLEST not in ece_draws:
        return best, "lowest ECE"
    _, lo, hi = interval(ece_draws[SIMPLEST] - ece_draws[best], level)
    if lo > 0:
        return best, f"lowest ECE, and beats {SIMPLEST} by {ece_draws[SIMPLEST][0] - ece_draws[best][0]:.4f} " \
                     f"[{lo:.4f}, {hi:.4f}]"
    return SIMPLEST, f"{best} has the lowest ECE, but its paired gain over {SIMPLEST} [{lo:.4f}, {hi:.4f}] " \
                     f"includes 0, so the simplest wins"
