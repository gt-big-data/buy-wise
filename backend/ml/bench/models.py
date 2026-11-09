"""Every model and rule the benchmark knows about.

To add a model (R2, R3): write a class with `name`, `outputs_probability`,
`fit(train, val, history)` and `score(test, history)` (see interface.py), then
add it to MODELS below. Use only `train` and `val` to learn and tune.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ml.bench.data import DEFAULT_SOURCE
from ml.bench.interface import BaseModel, Fold, PriceHistory
from ml.features import FEATURES

ML_DIR = Path(__file__).resolve().parents[1]


# ---- the simple rules every model has to beat --------------------------------

class RuleTop90(BaseModel):
    name = "rule: top of 90-day range"

    def score(self, test, history):
        return test.pos90.to_numpy(dtype=float)


class RuleRel90(BaseModel):
    name = "rule: price vs 90-day median"

    def score(self, test, history):
        return test.rel90.to_numpy(dtype=float)


class RuleNovember(BaseModel):
    name = "rule: wait in November"

    def score(self, test, history):
        # Tiny random noise breaks the ties so exactly N% can be picked.
        return ((test.date.dt.month == 11).astype(float).to_numpy()
                + np.random.default_rng(0).random(len(test)) * 1e-6)


# ---- XGBoost ------------------------------------------------------------------

def xgb_classifier():
    """The recipe train_clean.py uses for the served model."""
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=2.0, eval_metric="aucpr",
                         early_stopping_rounds=50, n_jobs=-1, random_state=42)


class ServedXGB(BaseModel):
    """The frozen model in buywise_clean.joblib, exactly as the extension serves it
    (raw classifier probability, before the calibrator).

    It was trained on data before its own test period, so it can only be scored
    on folds that start after that. On earlier folds it would have seen the
    test prices. Use CleanXGB (same recipe, retrained per fold) there."""
    name = "clean model (served)"
    outputs_probability = True

    def __init__(self, path: Path | None = None):
        from ml.inference import MODEL_PATH

        self.path = path or MODEL_PATH
        self._bundle = None

    def available(self):
        return None if self.path.exists() else f"{self.path.name} not found (run ml.train_clean)"

    def valid_for(self, fold: Fold) -> str | None:
        meta = self.bundle.get("meta", {})
        start = meta.get("test_period", [None])[0]
        if start and fold.test.date.min() < pd.Timestamp(start):
            return f"trained on data up to {start}; this fold tests from {fold.test.date.min().date()}"
        return None

    @property
    def bundle(self):
        if self._bundle is None:
            import joblib

            self._bundle = joblib.load(self.path)
        return self._bundle

    def score(self, test, history):
        return self.bundle["classifier"].predict_proba(test[FEATURES])[:, 1]


class CleanXGB(BaseModel):
    """The served model's recipe, retrained from scratch on each fold's train
    data, with early stopping on that fold's validation data."""
    name = "XGBoost (clean recipe, retrained)"
    outputs_probability = True

    def fit(self, train, val, history):
        self.clf = xgb_classifier()
        self.clf.fit(train[FEATURES], train.y, eval_set=[(val[FEATURES], val.y)], verbose=False)

    def score(self, test, history):
        return self.clf.predict_proba(test[FEATURES])[:, 1]


class AprilModel(BaseModel):
    """The April 2026 model (xgb_cls_14d.joblib), rebuilt with its own leaky
    pipeline. Kept on the leaderboard as a warning, not a contender."""
    name = "April model (old)"
    outputs_probability = True

    def __init__(self, source: Path = DEFAULT_SOURCE):
        self.source = Path(source)

    def available(self):
        for f in ("xgb_cls_14d.joblib", "scaler.joblib"):
            if not (ML_DIR / f).exists():
                return f"{f} not found"
        if self.source.suffix != ".csv":
            return "needs a Prices.csv-style file"
        return None

    def score(self, test, history):
        return april_model_scores(test, self.source)


def april_model_scores(te: pd.DataFrame, prices_path: Path = DEFAULT_SOURCE) -> np.ndarray:
    """P(WAIT) from the April 2026 model. Rebuilds its 40 features with its own
    pipeline (which leaks; that is the point of scoring it)."""
    import joblib

    from ml.dataset import engineer_features, merge_product_features

    cls14 = joblib.load(ML_DIR / "xgb_cls_14d.joblib")
    scaler = joblib.load(ML_DIR / "scaler.joblib")
    old_feats = [
        "price", "new_price", "used_price", "list_price", "count_new", "count_used", "sales_score",
        "amazon_ma_7d", "amazon_ma_14d", "amazon_delta_7d", "amazon_pct_change_7d", "day_of_week", "month",
        "rsi_14d", "rsi_7d", "dist_from_30d_high", "dist_from_30d_low", "velocity_3d",
        "day_of_year_sin", "day_of_year_cos", "price_z_asin", "price_minmax_asin",
        "roll7_mean", "roll7_std", "roll7_min", "roll7_max", "price_vs_roll7_z", "price_lag1", "price_lag30",
        "pct_change_1d", "pct_change_14d", "week_of_year", "global_min", "global_max", "global_norm_sd",
        "log_mean_price", "global_mean", "price_vs_global_min", "price_vs_global_max", "price_vs_global_mean",
    ]
    px = pd.read_csv(prices_path, parse_dates=["datetime"])
    px.columns = px.columns.str.lower().str.strip()
    px = px.rename(columns={"datetime": "date", "amazon": "price", "new": "new_price",
                            "used": "used_price", "listprice": "list_price"})
    px["date"] = px["date"].dt.normalize()
    px = px[px.price.notna() & px.asin.isin(te.asin.unique())]
    out = []
    for a, g in px.sort_values("date").groupby("asin"):
        g = g.groupby("date", as_index=False).last().set_index("date")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D")).ffill()
        g["asin"] = a
        out.append(g.reset_index().rename(columns={"index": "date"}))
    p = pd.concat(out, ignore_index=True)
    p["day_of_week"] = p.date.dt.dayofweek
    p["week_of_year"] = p.date.dt.isocalendar().week.astype(int)
    p = engineer_features(p)
    pr = pd.read_csv(Path(prices_path).with_name("Product.csv"))
    pr.columns = pr.columns.str.lower().str.strip()
    mp = px.groupby("asin").price.mean()
    pr["global_norm_sd"] = pr.global_sd / pr.asin.map(mp)
    pr["log_mean_price"] = np.log1p(pr.asin.map(mp))
    p = merge_product_features(p, pr[["asin", "global_min", "global_max", "global_norm_sd", "log_mean_price"]])
    for f in old_feats:
        if f not in p.columns:
            p[f] = 0.0
    sf = list(scaler.feature_names_in_)
    p[sf] = p[sf].astype(float).fillna(p[sf].astype(float).median())
    p[sf] = scaler.transform(p[sf])
    p["old_proba"] = cls14.predict_proba(p[old_feats].astype(float).fillna(0).values)[:, 1]
    merged = te[["asin", "date"]].merge(p[["asin", "date", "old_proba"]], on=["asin", "date"], how="left")
    return merged.old_proba.fillna(0).values


# ---- registry -----------------------------------------------------------------

# key -> factory taking the data source. R2/R3: add your model here.
MODELS: dict[str, Callable[[Path], BaseModel]] = {
    "top90": lambda src: RuleTop90(),
    "rel90": lambda src: RuleRel90(),
    "november": lambda src: RuleNovember(),
    "served": lambda src: ServedXGB(),
    "xgb": lambda src: CleanXGB(),
    "april": lambda src: AprilModel(src),
}
RULES = ("top90", "rel90", "november")
# What `python -m ml.evaluate` shows by default: today's leaderboard.
DEFAULT_MODELS = (*RULES, "served", "april")
