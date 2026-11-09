"""
The benchmark's promises (#78):
  - today's leaderboard is reproduced exactly through the new interface
  - no row's 14-day answer reaches into a later period
  - a model can't see test labels or future prices
  - bad model output is refused, not silently scored

Run: pytest tests/test_bench.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.bench.data import DEFAULT_SOURCE
from ml.bench.interface import BaseModel, Fold, PriceHistory
from ml.bench.metrics import ece, score_policies
from ml.bench.runner import run_fold, run_model
from ml.bench.splits import chronological_split
from ml.features import HORIZON_DAYS, add_features, add_labels, daily_panel

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_leaderboard.csv"


def _prices(n_asins=4, days=500, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    for i in range(n_asins):
        price = 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
        keep = np.r_[True, rng.random(days - 1) < 0.25]  # Keepa-style: only change days
        out.append(pd.DataFrame({"asin": f"A{i}", "date": dates[keep], "price": price[keep].round(2)}))
    return pd.concat(out, ignore_index=True)


def _fold():
    panel = daily_panel(_prices())
    rows = add_labels(add_features(panel)).dropna(subset=["med90", "vol30", "pos90"]).reset_index(drop=True)
    s = chronological_split(rows)
    return Fold("synthetic", s.train, s.val, s.test, PriceHistory(panel))


# ---- reproduction ---------------------------------------------------------------

def test_legacy_reproduction():
    """Rules + served XGBoost + April model through the new interface give
    exactly the numbers the old evaluate.py printed (served AUC 0.802)."""
    from ml.bench.data import load_dataset
    from ml.bench.models import DEFAULT_MODELS, MODELS
    from ml.bench.splits import legacy_fold

    if not DEFAULT_SOURCE.exists():
        pytest.skip("studies/Prices.csv not present")
    ds = load_dataset(DEFAULT_SOURCE)
    res = run_fold([MODELS[k](ds.source) for k in DEFAULT_MODELS], legacy_fold(ds.rows, ds.panel), verbose=False)
    if res.skipped:
        pytest.skip(f"model files missing: {res.skipped}")
    got = res.table().drop(columns="fold")
    want = pd.read_csv(FIXTURE)
    cols = list(want.columns.drop(["policy", "target wait %"]))
    merged = want.merge(got[["policy", "target wait %", *cols]], on=["policy", "target wait %"],
                        suffixes=("_want", "_got"), how="left")
    assert len(merged) == len(want) and merged.notna().all().all()
    for col in cols:
        np.testing.assert_allclose(merged[f"{col}_got"], merged[f"{col}_want"], rtol=1e-6, err_msg=col)
    served = got[(got.policy == "clean model (served)") & (got["target wait %"] == 20)].iloc[0]
    assert round(served.AUC, 3) == 0.802


# ---- no leakage across periods -------------------------------------------------

def _assert_no_overlap(train, val, test):
    h = pd.Timedelta(days=HORIZON_DAYS)
    assert train.date.max() + h < val.date.min(), "a training answer reaches into validation"
    assert val.date.max() + h < test.date.min(), "a validation answer reaches into test"


def test_legacy_split_no_label_overlap_synthetic():
    f = _fold()
    _assert_no_overlap(f.train, f.val, f.test)


def test_legacy_split_no_label_overlap_real():
    from ml.bench.data import load_dataset

    if not DEFAULT_SOURCE.exists():
        pytest.skip("studies/Prices.csv not present")
    s = chronological_split(load_dataset(DEFAULT_SOURCE).rows)
    _assert_no_overlap(s.train, s.val, s.test)


# ---- what a model can and can't see --------------------------------------------

class _Peeker(BaseModel):
    name = "peeks at labels"

    def score(self, test, history):
        return test["y"].to_numpy(dtype=float)


class _Const(BaseModel):
    name = "constant"

    def __init__(self, n=None, value=0.0):
        self.n, self.value = n, value

    def score(self, test, history):
        return np.full(self.n if self.n is not None else len(test), self.value)


def test_test_labels_are_hidden():
    with pytest.raises(KeyError):
        run_model(_Peeker(), _fold())


def test_wrong_length_is_refused():
    with pytest.raises(ValueError, match="scores for"):
        run_model(_Const(n=3), _fold())


def test_nan_scores_are_refused():
    with pytest.raises(ValueError, match="NaN"):
        run_model(_Const(value=np.nan), _fold())


def test_history_windows_end_on_decision_day():
    f = _fold()
    rows = f.test.sample(50, random_state=0)
    w = f.history.windows(rows, length=30)
    assert w.shape == (50, 30)
    np.testing.assert_allclose(w[:, -1], rows.price.to_numpy())


def test_history_windows_ignore_the_future():
    prices = _prices()
    cutoff = pd.Timestamp("2025-01-01")
    tampered = prices.copy()
    tampered.loc[tampered.date > cutoff, "price"] *= 3
    rows = pd.DataFrame({"asin": ["A0", "A1", "A2"], "date": [cutoff] * 3})
    a = PriceHistory(daily_panel(prices)).windows(rows, 90)
    b = PriceHistory(daily_panel(tampered)).windows(rows, 90)
    np.testing.assert_array_equal(a, b)


def test_history_given_to_score_stops_at_test_end():
    seen = {}

    class Spy(_Const):
        def fit(self, train, val, history):
            seen["fit"] = history.until

        def score(self, test, history):
            seen["score"] = history.until
            return super().score(test, history)

    f = _fold()
    run_model(Spy(), f)
    assert seen["fit"] <= f.val.date.max()
    assert seen["score"] <= f.test.date.max()


# ---- metrics ---------------------------------------------------------------------

def test_random_model_has_lift_near_one():
    f = _fold()
    s = np.random.default_rng(1).random(len(f.test))
    r = score_policies(f.test, {"random": s}, (0.5,))
    assert 0.8 < r.lift.iloc[0] < 1.2


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert ece(y, p) < 0.01
    assert ece(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_old_evaluate_names_still_import():
    """train_clean.py depends on these."""
    from ml.evaluate import (baseline_scores, build_dataset, chronological_split, leaderboard,  # noqa: F401
                             load_prices, score_decision, score_policies)

    f = _fold()
    assert set(baseline_scores(f.test)) == {"rule: top of 90-day range", "rule: price vs 90-day median",
                                            "rule: wait in November"}
