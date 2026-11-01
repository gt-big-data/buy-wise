"""Scoring. Every policy is compared at the SAME wait rate: each says WAIT on
exactly the top N% of decisions by its own score.

  precision   share of WAITs where the price really fell >= 8% within 14 days
  lift        precision / base rate (1.0 = no better than guessing)
  $ caught    average saving per decision if the shopper buys at the 14-day low
  $ no alert  average saving if they just buy 14 days later at whatever it costs
  AUC         ranking quality across all thresholds
  PR-AUC      average precision; more informative than AUC when drops are rare
  Brier       mean squared error of the score as a probability (probabilities only)
  ECE         calibration error: average gap between said and happened, 10 equal-size bins
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

WAIT_RATES = (0.10, 0.20, 0.30)


def _policy_row(te: pd.DataFrame, wait: np.ndarray) -> dict:
    y = te.y.values
    prec = y[wait].mean() if wait.any() else np.nan
    return {
        "wait %": wait.mean() * 100,
        "precision %": prec * 100,
        "lift": prec / y.mean(),
        "$ caught": te.price.mean() - np.where(wait, te.fwd_min, te.price).mean(),
        "$ no alert": te.price.mean() - np.where(wait, te.fwd_last, te.price).mean(),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p, kind="stable")
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return err / len(p)


def threshold_metrics(y: np.ndarray, s: np.ndarray, is_probability: bool) -> dict:
    """Metrics that don't depend on the wait rate."""
    ok = np.isfinite(s).all()
    out = {"AUC": roc_auc_score(y, s) if ok else np.nan,
           "PR-AUC": average_precision_score(y, s) if ok else np.nan,
           "Brier": np.nan, "ECE": np.nan}
    if ok and is_probability and s.min() >= 0 and s.max() <= 1:
        out["Brier"] = brier_score_loss(y, s)
        out["ECE"] = ece(y, s)
    return out


def score_policies(te: pd.DataFrame, scores: dict[str, np.ndarray],
                   wait_rates: tuple[float, ...] = WAIT_RATES,
                   probabilistic: frozenset[str] | set[str] = frozenset()) -> pd.DataFrame:
    """One row per (policy, wait rate). `probabilistic` names the policies whose
    scores are probabilities, which turns on Brier and ECE for them."""
    rows = []
    y = te.y.values
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        tm = threshold_metrics(y, s, name in probabilistic)
        for q in wait_rates:
            wait = s >= np.quantile(s, 1 - q)
            rows.append({"policy": name, "target wait %": int(round(q * 100)), "AUC": tm["AUC"],
                         **_policy_row(te, wait),
                         "PR-AUC": tm["PR-AUC"], "Brier": tm["Brier"], "ECE": tm["ECE"]})
    return pd.DataFrame(rows)


def score_decision(te: pd.DataFrame, name: str, wait: np.ndarray) -> pd.DataFrame:
    """Score a policy at its own operating point, not a matched wait rate."""
    return pd.DataFrame([{"policy": name, **_policy_row(te, np.asarray(wait, bool))}])


def oracle(te: pd.DataFrame) -> float:
    """Perfect foresight: always buy at the 14-day low."""
    return te.price.mean() - te.fwd_min.mean()
