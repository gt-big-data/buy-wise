"""Shared setup for the R2 studies (s003, s017, r2_charts).

They reuse the settings each model chose when it was tuned on each fold's
validation rows (`python -m ml.evaluate ... --save` writes them to
results/tuning_<data>_walk.csv), so nothing is re-tuned and nothing is chosen
on test.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from ml.bench.data import load_dataset  # noqa: E402
from ml.bench.models import XGB, CatBoost  # noqa: E402
from ml.bench.splits import walk_forward_folds  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SOURCES = {
    "prices": os.path.join(HERE, "Prices.csv"),
    "apr2026_500": os.path.join(HERE, "snapshots", "apr2026_500", "Prices.csv.gz"),
}
INT_PARAMS = {"n_estimators", "max_depth", "iterations", "depth", "border_count"}


def tag_for(source: str) -> str:
    return os.path.basename(source).split(".")[0] + "_walk"


def load(name: str):
    """Dataset and its four walk-forward folds."""
    src = SOURCES.get(name, name)
    ds = load_dataset(src)
    return ds, walk_forward_folds(ds.rows, ds.panel), tag_for(src)


def tuned_params(tag: str, model_name: str, fold: str) -> dict:
    """The settings `model_name` picked on `fold`'s validation rows."""
    t = pd.read_csv(os.path.join(RESULTS, f"tuning_{tag}.csv"))
    t = t[(t.model == model_name) & (t.fold == fold)]
    if t.empty:
        raise SystemExit(f"no tuning log for {model_name} / {fold}; run ml.evaluate --folds walk --save first")
    score = "val objective" if "val objective" in t and t["val objective"].notna().all() else "val PR-AUC"
    best = t.loc[t[score].idxmax()].drop(["trial", "val PR-AUC", "val objective", "weight", "model", "fold"],
                                         errors="ignore")
    out = {}
    for k, v in best.items():
        if isinstance(v, float) and math.isnan(v):
            continue
        out[k] = int(v) if k in INT_PARAMS else v
    return out


def fixed(model_cls, params, feature_cols, name):
    """A model with settings fixed in advance (from tuning), on any feature subset."""
    m = model_cls(params, feature_cols)
    m.name = name
    return m


def tuned_models(tag: str, fold: str, feature_cols, suffix=""):
    """XGBoost and CatBoost with the settings they were tuned to on this fold
    (the 26-feature runs), on `feature_cols`."""
    out = []
    for cls, key in ((XGB, "XGBoost"), (CatBoost, "CatBoost")):
        params = tuned_params(tag, f"{key} (tuned, 26 features)", fold)
        out.append(fixed(cls, params, feature_cols, f"{key}{suffix}"))
    return out

