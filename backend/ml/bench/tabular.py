"""R2 classifiers: identical features, train-only preprocessing, validation selection."""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.bench.interface import BaseModel
from ml.features import FEATURES

# Eight fitted candidates per tuned model, including its default configuration.
TUNING_BUDGET = 8


class TabularModel(BaseModel):
    outputs_probability = True
    family = "tabular"

    def __init__(self, tuned=False):
        self.tuned = tuned
        self.name = f"{self.family} ({'tuned' if tuned else 'default'})"

    def candidates(self):
        raise NotImplementedError

    def estimator(self, params):
        raise NotImplementedError

    @staticmethod
    def inputs(rows):
        return rows[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)

    def fit(self, train, val, history):
        if train.y.nunique() != 2:
            raise ValueError("training data must contain both label classes")
        if self.tuned and val.y.nunique() != 2:
            raise ValueError("tuning requires validation data with both label classes")
        configs = self.candidates() if self.tuned else [{}]
        self.trials_ = []
        self.validation_pr_auc_ = -np.inf
        # No refit on validation: its labels select candidates, never fit weights.
        for params in configs:
            clf = self.estimator(params)
            clf.fit(self.inputs(train), train.y)
            ap = (float(average_precision_score(val.y, clf.predict_proba(self.inputs(val))[:, 1]))
                  if len(val) else None)
            self.trials_.append({"params": params.copy(), "validation_pr_auc": ap})
            if not hasattr(self, "clf") or len(self.trials_) == 1 or ap > self.validation_pr_auc_:
                self.clf = clf
                self.best_params_ = params.copy()
                self.validation_pr_auc_ = ap

    def score(self, test, history):
        return self.clf.predict_proba(self.inputs(test))[:, 1]


class LightGBM(TabularModel):
    family = "LightGBM"

    def available(self):
        from importlib.util import find_spec
        return None if find_spec("lightgbm") else "install lightgbm from backend/requirements.txt"

    def candidates(self):
        return [{}, *[{"num_leaves": leaves, "learning_rate": rate, "reg_lambda": reg}
                     for leaves, rate, reg in product((15, 31), (0.03, 0.1), (0.0, 2.0))
                     if (leaves, rate, reg) != (31, 0.1, 0.0)]]

    def estimator(self, params):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **params)


class SmallNeuralNet(TabularModel):
    family = "Small neural net"

    def candidates(self):
        return [{}, *[{"hidden_layer_sizes": layers, "alpha": alpha, "learning_rate_init": rate}
                     for layers, alpha, rate in product(((32,), (32, 16)), (0.0001, 0.01), (0.001, 0.003))
                     if (layers, alpha, rate) != ((32, 16), 0.0001, 0.001)]]

    def estimator(self, params):
        config = dict(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42,
                      early_stopping=False)
        config.update(params)
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), MLPClassifier(**config))
