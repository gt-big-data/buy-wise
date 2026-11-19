import numpy as np
import pandas as pd
import pytest
from ml.bench.models import MODELS
from ml.bench.tabular import LightGBM, SmallNeuralNet, TUNING_BUDGET
from ml.features import FEATURES


def rows(seed, n=100):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    x['y'] = (x[FEATURES[0]] > 0).astype(int)
    x['used_gap'] = np.nan
    return x


@pytest.mark.parametrize('kind', [LightGBM, SmallNeuralNet])
def test_reproducible_probabilities(kind):
    a, b = kind(), kind()
    for m in (a, b):
        m.fit(rows(1), rows(2), None)
    scores = a.score(rows(3).drop(columns='y'), None)
    assert scores.shape == (100,)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    np.testing.assert_allclose(scores, b.score(rows(3).drop(columns='y'), None))


def test_train_only_preprocessing():
    train = rows(1)
    m = SmallNeuralNet()
    m.fit(train, rows(2).assign(rel30=1e9), None)
    imputer, scaler = m.clf.steps[0][1], m.clf.steps[1][1]
    assert imputer.statistics_[0] == pytest.approx(train.rel30.median())
    assert scaler.mean_[0] == pytest.approx(train.rel30.mean())
    before = scaler.mean_.copy()
    m.score(rows(3).assign(rel30=-1e9).drop(columns='y'), None)
    np.testing.assert_array_equal(before, scaler.mean_)


@pytest.mark.parametrize('kind', [LightGBM, SmallNeuralNet])
def test_equal_budget_and_validation_selection(kind):
    m = kind(tuned=True)
    assert len(m.candidates()) == TUNING_BUDGET
    m.fit(rows(1), rows(2), None)
    assert len(m.trials_) == TUNING_BUDGET
    assert m.validation_pr_auc_ == max(t['validation_pr_auc'] for t in m.trials_)
    assert np.isfinite(m.score(rows(3).drop(columns='y'), None)).all()


def test_registry():
    for k in ('lightgbm', 'lightgbm-tuned', 'small-nn', 'small-nn-tuned'):
        assert MODELS[k](None).outputs_probability
