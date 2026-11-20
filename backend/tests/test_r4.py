"""
R4's promises (#81):
  - horizon and threshold only change the label; the defaults change nothing
  - with a longer horizon, no answer reaches into a later period
  - validation scores are kept, and labels are hidden when scoring
  - calibrators are monotone, stay in [0, 1], and only learn from the rows they're given
  - the stay-quiet rule's arithmetic, and what it learns from (train only)

Run: pytest tests/test_r4.py -q
"""
import numpy as np
import pandas as pd
import pytest

from ml.bench import calibration as cal
from ml.bench import quiet
from ml.bench.interface import BaseModel
from ml.bench.metrics import ece
from ml.bench.r4 import label_rows, merge_columns, run, walk_forward_folds
from ml.features import DROP_THRESHOLD, HORIZON_DAYS, REQUIRED, add_features, add_labels, daily_panel

from tests.test_bench import _prices


def _rows(horizon=HORIZON_DAYS, threshold=DROP_THRESHOLD, **kw):
    panel = daily_panel(_prices(**kw))
    return label_rows(panel, horizon, threshold), panel


# ---- horizon and threshold --------------------------------------------------------

def test_label_defaults_unchanged():
    panel = daily_panel(_prices())
    want = add_labels(add_features(panel)).dropna(subset=REQUIRED).reset_index(drop=True)
    got = label_rows(panel)
    pd.testing.assert_frame_equal(got, want)


def test_lower_threshold_is_a_superset():
    panel = daily_panel(_prices())
    y5 = label_rows(panel, 14, 0.05).y
    y8 = label_rows(panel, 14, 0.08).y
    assert (y5 >= y8).all() and y5.sum() > y8.sum()


def test_longer_horizon_drops_more_trailing_rows():
    panel = daily_panel(_prices())
    h7, h30 = label_rows(panel, 7), label_rows(panel, 30)
    assert len(h30) < len(h7)
    # A longer window can only find an equal or lower minimum.
    m = h30.merge(h7, on=["asin", "date"], suffixes=("_30", "_7"))
    assert (m.fwd_min_30 <= m.fwd_min_7 + 1e-9).all()


@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_walk_forward_gap_follows_horizon(horizon):
    rows, panel = _rows(horizon, days=900)
    folds = walk_forward_folds(rows, panel, n=2, test_days=60, val_days=60, horizon=horizon)
    h = pd.Timedelta(days=horizon)
    for f in folds:
        assert f.info["horizon_days"] == horizon
        assert f.train.date.max() + h < f.val.date.min(), "a training answer reaches into validation"
        assert f.val.date.max() + h < f.test.date.min(), "a validation answer reaches into test"


# ---- validation scores ------------------------------------------------------------

class _Recorder(BaseModel):
    """Scores every row 0.5; remembers which periods it was asked to score."""
    name = "recorder"
    outputs_probability = True

    def __init__(self):
        self.seen = []

    def score(self, test, history):
        assert "y" not in test.columns and "fwd_min" not in test.columns
        assert history.until <= test.date.max()
        self.seen.append(len(test))
        return np.full(len(test), 0.5)


def test_run_keeps_validation_and_test_scores():
    rows, panel = _rows(days=900)
    f = walk_forward_folds(rows, panel, n=1, test_days=60, val_days=60)[0]
    m = _Recorder()
    v, t, _ = run([m], f, verbose=False)
    assert len(v[m.name]) == len(f.val) and len(t[m.name]) == len(f.test)


def test_merge_columns_needs_matching_rows(tmp_path):
    a = pd.DataFrame({"fold": "w", "asin": ["A", "B"], "date": pd.to_datetime(["2025-01-01"] * 2), "y": [0, 1],
                      "m1": [0.1, 0.2]})
    a.to_csv(tmp_path / "a.csv", index=False)
    a.drop(columns="m1").assign(m2=[0.3, 0.4]).to_csv(tmp_path / "b.csv", index=False)
    merge_columns(tmp_path / "a.csv", tmp_path / "b.csv")
    assert list(pd.read_csv(tmp_path / "a.csv").m2) == [0.3, 0.4]
    a.iloc[::-1].to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError):
        merge_columns(tmp_path / "a.csv", tmp_path / "c.csv")


# ---- calibrators ------------------------------------------------------------------

def _scores(n=4000, seed=0, bend=2.0):
    """Raw scores that are miscalibrated in a known way: true chance = p ** bend."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.random(n) < p ** bend).astype(int)
    return p, y


@pytest.mark.parametrize("kind", cal.CALIBRATORS)
def test_calibrator_monotone_and_bounded(kind):
    p, y = _scores()
    c = cal.fit_calibrator(kind, p, y)
    grid = np.linspace(0, 1, 501)
    out = c.predict(grid)
    assert np.all(out >= 0) and np.all(out <= 1)
    assert np.all(np.diff(out) >= -1e-9), "calibrated chance went down as the score went up"


@pytest.mark.parametrize("kind", cal.CALIBRATORS)
def test_calibrator_fixes_known_miscalibration(kind):
    p, y = _scores(seed=1)
    p2, y2 = _scores(seed=2)  # fresh rows from the same process
    c = cal.fit_calibrator(kind, p, y)
    assert ece(y2, c.predict(p2)) < ece(y2, p2) / 2


def test_isotonic_in_sample_is_near_perfect():
    p, y = _scores()
    c = cal.fit_calibrator("isotonic", p, y)
    assert ece(y, c.predict(p)) < 0.01


def test_calibrate_folds_never_reads_test_labels():
    p, y = _scores(n=2000)
    p_t, y_t = _scores(n=1000, seed=3)
    val = pd.DataFrame({"fold": "f1", "asin": "A", "s": p, "y": y})
    test = pd.DataFrame({"fold": "f1", "asin": "A", "s": p_t, "y": y_t})
    a = cal.calibrate_folds(val, test, "s", "beta")
    b = cal.calibrate_folds(val, test.assign(y=1 - test.y), "s", "beta")
    np.testing.assert_array_equal(a, b)


def test_calibrate_folds_fits_each_fold_on_its_own_validation():
    p, y = _scores(n=2000)
    val = pd.concat([pd.DataFrame({"fold": "f1", "s": p, "y": y}),
                     pd.DataFrame({"fold": "f2", "s": p, "y": np.zeros_like(y)})])
    test = pd.DataFrame({"fold": ["f1", "f2"], "s": [0.9, 0.9], "y": [0, 0]})
    out = cal.calibrate_folds(val, test, "s", "isotonic")
    assert out[0] > 0.5 and out[1] == 0.0


def test_crossfit_scores_every_row_out_of_sample():
    p, y = _scores(n=3000)
    val = pd.DataFrame({"fold": "f1", "asin": np.repeat([f"A{i}" for i in range(30)], 100), "s": p, "y": y})
    out = cal.crossfit_val(val, "s", "isotonic")
    assert np.isfinite(out).all()
    # Out of sample, isotonic is no longer near-perfect on these rows.
    assert ece(y, out) > ece(y, cal.fit_calibrator("isotonic", p, y).predict(p))


def test_boot_metrics_match_plain_ones():
    from sklearn.metrics import brier_score_loss

    p, y = _scores(n=1000)
    asins = np.repeat(np.arange(50), 20)
    d = cal.product_draws(asins, n_boot=20)
    assert d.counts.shape == (21, 50)
    assert cal.boot_ece(y, p, d)[0] == pytest.approx(ece(y, p))
    assert cal.boot_brier(y, p, d)[0] == pytest.approx(brier_score_loss(y, p))
    m = asins < 25
    assert cal.boot_ece(y, p, d, m)[0] == pytest.approx(ece(y[m], p[m]))


def test_boot_draw_equals_duplicated_rows():
    """A draw's ECE is the plain ECE of the rows it picked, duplicates included."""
    p, y = _scores(n=600)
    asins = np.repeat(np.arange(30), 20)
    d = cal.product_draws(asins, n_boot=3, seed=1)
    k = d.counts[1].astype(int)
    rows = np.concatenate([np.repeat(np.flatnonzero(asins == a), k[a]) for a in range(30)])
    want_brier = np.mean((p[rows] - y[rows]) ** 2)
    assert cal.boot_brier(y, p, d)[1] == pytest.approx(want_brier)


def test_pick_prefers_simplest_on_a_tie():
    rng = np.random.default_rng(0)
    same = np.r_[0.02, np.full(200, 0.02) + rng.normal(0, 0.005, 200)]
    worse = same + 0.03
    # Platt's point estimate is a hair worse, but the paired gap straddles 0.
    tie = cal.pick({"platt": same + np.r_[0.0001, rng.normal(0, 0.002, 200)], "isotonic": same})
    clear = cal.pick({"platt": worse, "isotonic": same})
    assert tie[0] == "platt"
    assert clear[0] == "isotonic"


# ---- stay quiet -------------------------------------------------------------------

def _decisions():
    """Six hand-built decisions, two per volatility band."""
    return pd.DataFrame({
        "price": [10.0, 10.0, 100.0, 100.0, 500.0, 500.0],
        "vol30": [0.001, 0.001, 0.01, 0.01, 0.1, 0.1],
        "past_drop_rate": [0.1, 0.1, 0.2, 0.2, 0.5, np.nan],
        "y": [1, 0, 1, 0, 1, 0],
        "fwd_min": [9.0, 10.0, 80.0, 100.0, 400.0, 500.0],
        "fwd_last": [9.5, 10.5, 90.0, 110.0, 450.0, 550.0],
        "drop_pct": [0.1, 0.0, 0.2, 0.0, 0.2, 0.0],
    })


def test_wait_economics_by_band():
    d = _decisions()
    edges = np.array([0.005, 0.05])
    econ = quiet.wait_economics(d, edges)
    assert list(econ.index) == list(quiet.BANDS)
    assert econ.loc["calm", "G_right"] == pytest.approx(0.05)     # 10 -> 9.5
    assert econ.loc["calm", "G_wrong"] == pytest.approx(-0.05)    # 10 -> 10.5
    assert econ.loc["twitchy", "G_right_low"] == pytest.approx(0.2)
    assert econ.loc["twitchy", "higher_share"] == 1.0
    assert econ.loc["medium", "higher_by"] == pytest.approx(0.1)


def test_net_saving_arithmetic():
    d = _decisions()
    edges = np.array([0.005, 0.05])
    econ = quiet.wait_economics(d, edges)
    p = np.full(len(d), 0.5)
    net = quiet.net_saving(p, d, econ, edges)
    # Twitchy, $500: 0.5 * 0.1 + 0.5 * (-0.1) = 0.
    assert net[4] == pytest.approx(0.0)
    p = np.full(len(d), 0.8)
    net = quiet.net_saving(p, d, econ, edges)
    assert net[4] == pytest.approx(500 * (0.8 * 0.1 + 0.2 * -0.1))


def test_economics_ignore_test_labels():
    rows, panel = _rows(days=900)
    folds = walk_forward_folds(rows, panel, n=1, test_days=90, val_days=90)
    f = folds[0]
    edges = quiet.band_edges(f.train)
    econ = quiet.wait_economics(f.train, edges)
    p = np.full(len(f.test), 0.4)
    a = quiet.net_saving(p, f.test, econ, edges)
    flipped = f.test.assign(y=1 - f.test.y, fwd_last=f.test.price * 2)
    b = quiet.net_saving(p, flipped, econ, edges)
    np.testing.assert_array_equal(a, b)


def test_outcome_limits():
    d = _decisions()
    everyone = quiet.outcome(d, np.ones(len(d), bool))
    nobody = quiet.outcome(d, np.zeros(len(d), bool))
    assert everyone["quiet %"] == 0 and everyone["precision %"] == pytest.approx(50)
    assert nobody["quiet %"] == 100 and nobody["$ day N per decision"] == 0
    # Wrong WAITs here always cost more on day N.
    assert everyone["wrong: paid more %"] == 100


def test_grid_and_pick():
    d = pd.concat([_decisions()] * 50, ignore_index=True)
    p = np.tile([0.9, 0.2, 0.7, 0.3, 0.6, 0.4], 50)
    net = d.price.to_numpy() * (p - 0.3)
    g = quiet.grid(d, p, net)
    loud = g[(g.min_drop_chance == 0) & ~np.isfinite(g.min_saving)].iloc[0]
    assert loud["quiet %"] == 0
    best = quiet.pick_operating_point(g, min_precision=99, min_shown=10)
    assert best["precision %"] == 100


def test_jumpiness_in_unit_range():
    d = _decisions()
    j = quiet.jumpiness(d, d)
    assert ((j >= 0) & (j <= 1)).all()
    assert j[4] > j[0]
