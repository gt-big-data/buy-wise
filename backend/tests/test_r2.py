"""
R2 (#79) and the rest of R1 (#78): more test windows, error bars, tuning.

  - walk-forward folds never let a 14-day answer cross into a later period
  - table models are handed their feature columns and nothing else
  - tuning never sees test rows, and with shuffled labels it finds nothing
  - error bars come from resampling products, and match the plain scorer

Run: pytest tests/test_r2.py -q
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from ml.bench.ci import bootstrap, compare, summary, weighted_stats
from ml.bench.interface import BaseModel, Fold, PriceHistory
from ml.bench.metrics import score_policies
from ml.bench.models import XGB, CatBoost
from ml.bench.runner import FoldResult, run_fold, run_model
from ml.bench.splits import walk_forward_folds
from ml.bench.tuning import Tuned, early_stopping_split, row_weights, tune
from ml.features import ALL_FEATURES, FEATURES, HORIZON_DAYS, add_features, add_labels, daily_panel
from tests.test_bench import _assert_no_overlap, _fold, _prices


def _rows_panel(n_asins=6, days=900):
    panel = daily_panel(_prices(n_asins=n_asins, days=days))
    rows = add_labels(add_features(panel)).dropna(subset=["med90", "vol30", "pos90"]).reset_index(drop=True)
    return rows, panel


# ---- walk-forward folds -----------------------------------------------------------

def test_walk_forward_folds_never_overlap():
    rows, panel = _rows_panel()
    folds = walk_forward_folds(rows, panel, n=3, test_days=60, val_days=60)
    assert len(folds) == 3
    for f in folds:
        _assert_no_overlap(f.train, f.val, f.test)
    # back-to-back test windows, each training on more than the last
    for a, b in zip(folds, folds[1:]):
        assert a.test.date.max() < b.test.date.min()
        assert len(b.train) > len(a.train)
    assert folds[-1].test.date.max() == rows.date.max()


def test_walk_forward_folds_on_real_data():
    from ml.bench.data import DEFAULT_SOURCE, load_dataset

    if not DEFAULT_SOURCE.exists():
        pytest.skip("studies/Prices.csv not present")
    ds = load_dataset(DEFAULT_SOURCE)
    folds = walk_forward_folds(ds.rows, ds.panel)
    assert len(folds) == 4
    for f in folds:
        _assert_no_overlap(f.train, f.val, f.test)
        assert len(f.test) > 10_000


def test_early_stopping_rows_come_from_train_with_a_gap():
    f = _fold()
    fit_rows, es_rows = early_stopping_split(f.train)
    assert len(fit_rows) and len(es_rows)
    assert fit_rows.date.max() + pd.Timedelta(days=HORIZON_DAYS) < es_rows.date.min()
    assert es_rows.date.max() <= f.train.date.max()


# ---- table models only see their features -------------------------------------------

class _ReadsPrice(BaseModel):
    name = "reads the price column"
    feature_cols = FEATURES

    def score(self, test, history):
        return test.price.to_numpy()


class _ReadsHistory(BaseModel):
    name = "reads history"
    feature_cols = FEATURES

    def score(self, test, history):
        return np.array([history.series(a, "2099-01-01").iloc[-1] for a in test.asin])


def test_table_models_cant_read_price_or_history():
    for m in (_ReadsPrice(), _ReadsHistory()):
        with pytest.raises(AttributeError):
            run_model(m, _fold())


def test_table_model_gets_exactly_its_columns():
    seen = {}

    class Spy(BaseModel):
        name = "spy"
        feature_cols = ["rel90", "vol30"]

        def score(self, test, history):
            seen["cols"], seen["history"] = list(test.columns), history
            return np.zeros(len(test))

    run_model(Spy(), _fold())
    assert seen == {"cols": ["rel90", "vol30"], "history": None}


# ---- tuning ------------------------------------------------------------------------

def test_tune_cannot_be_given_test_rows():
    assert list(inspect.signature(tune).parameters)[:3] == ["model_cls", "train", "val"]
    assert "test" not in inspect.signature(tune).parameters


def test_first_trial_is_the_defaults_and_budget_is_respected():
    f = _fold()
    best, trials = tune(XGB, f.train, f.val, budget=3)
    assert len(trials) == 3
    for k, v in XGB.DEFAULTS.items():
        assert trials.iloc[0][k] == v
    assert trials["val PR-AUC"].max() == pytest.approx(
        __import__("sklearn.metrics").metrics.average_precision_score(f.val.y, best.score(f.val)))


def test_shuffled_labels_leave_nothing_to_find():
    """If tuning or fitting leaked the answer, a model trained on shuffled labels
    would still rank test rows well. It must be no better than chance."""
    f = _fold()
    rng = np.random.default_rng(0)
    train = f.train.assign(y=rng.permutation(f.train.y.values))
    val = f.val.assign(y=rng.permutation(f.val.y.values))
    m = Tuned(XGB, budget=3)
    s = run_model(m, Fold("shuffled", train, val, f.test, f.history))
    lift = score_policies(f.test, {"m": s}, (0.3,)).lift.iloc[0]
    assert 0.6 < lift < 1.4


def test_price_weights_follow_price_and_average_one():
    rows = pd.DataFrame({"price": [10.0, 40.0, 250.0]})
    assert row_weights(rows, None) is None
    for scheme in ("sqrt_price", "price"):
        w = row_weights(rows, scheme)
        assert w.mean() == pytest.approx(1) and (np.diff(w) > 0).all()


def test_weighted_tuning_scores_by_weighted_pr_auc():
    """A price-weighted model is chosen on price-weighted PR-AUC, and it really
    trains with weights (so its scores differ from the unweighted model's)."""
    from sklearn.metrics import average_precision_score

    f = _fold()
    best, trials = tune(XGB, f.train, f.val, budget=2, weight="price")
    assert (trials.weight == "price").all()
    assert trials["val objective"].max() == pytest.approx(
        average_precision_score(f.val.y, best.score(f.val), sample_weight=row_weights(f.val, "price")))
    plain = XGB()
    plain.fit(f.train, f.val)
    weighted = XGB(weight="price")
    weighted.fit(f.train, f.val)
    assert not np.allclose(plain.score(f.test), weighted.score(f.test))


@pytest.mark.parametrize("cls", [XGB, CatBoost])
def test_weighted_models_see_only_features_at_test_time(cls):
    if cls().available():
        pytest.skip(cls().available())
    s = run_model(cls(feature_cols=ALL_FEATURES, weight="sqrt_price"), _fold())
    assert ((s >= 0) & (s <= 1)).all()


@pytest.mark.parametrize("cls", [XGB, CatBoost])
def test_models_run_on_all_features(cls):
    if cls().available():
        pytest.skip(cls().available())
    s = run_model(cls(feature_cols=ALL_FEATURES), _fold())
    assert ((s >= 0) & (s <= 1)).all()


# ---- error bars --------------------------------------------------------------------

def _result():
    rows, panel = _rows_panel(n_asins=30, days=500)
    f = walk_forward_folds(rows, panel, n=1, test_days=120, val_days=60)[0]
    rng = np.random.default_rng(0)
    good = f.test.y.values + rng.normal(0, 0.7, len(f.test))
    return FoldResult(f, scores={"good": good, "random": rng.random(len(f.test))})


def test_unweighted_stats_match_the_plain_scorer():
    r = _result()
    te = r.fold.test.reset_index(drop=True)
    plain = score_policies(te, r.scores)
    for name, s in r.scores.items():
        w = weighted_stats(te, s, np.ones((1, len(te))))
        p = plain[plain.policy == name].set_index("target wait %")
        assert w["AUC"][0] == pytest.approx(p.AUC.iloc[0])
        assert w["PR-AUC"][0] == pytest.approx(p["PR-AUC"].iloc[0])
        for q in (10, 20, 30):
            assert w[f"$ caught@{q}"][0] == pytest.approx(p.loc[q, "$ caught"], rel=1e-3)
            assert w[f"precision %@{q}"][0] == pytest.approx(p.loc[q, "precision %"], rel=1e-3)


def test_intervals_cover_chance_for_random_and_separate_a_real_model():
    b = bootstrap([_result()], n_boot=300)
    t = summary(b).set_index(["policy", "target wait %"])
    rnd = t.loc[("random", 20)]
    assert rnd["lift lo"] < 1 < rnd["lift hi"]
    c = compare(b, "good", "random").set_index("metric")
    assert c.loc["PR-AUC", "lo"] > 0 and c.loc["PR-AUC", "p(a<=b)"] < 0.01
    same = compare(b, "good", "good").set_index("metric")
    assert (same["diff"] == 0).all()
