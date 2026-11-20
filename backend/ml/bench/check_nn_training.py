"""Compare neural-net training caps using validation only; never score test rows.

    python -m ml.bench.check_nn_training --epochs 100 400 --output training_check.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

from ml.bench.data import DEFAULT_SOURCE, load_dataset
from ml.bench.splits import legacy_fold
from ml.bench.tabular import SmallNeuralNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_SOURCE))
    parser.add_argument("--epochs", nargs="+", type=int, default=[100, 400])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(cap < 1 for cap in args.epochs):
        parser.error("epochs must be positive")
    ds = load_dataset(args.data)
    fold = legacy_fold(ds.rows, ds.panel)
    model = SmallNeuralNet()
    report = []
    for cap in args.epochs:
        clf = model.estimator({"max_iter": cap})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(model.inputs(fold.train), fold.train.y)
        net = clf.steps[-1][1]
        row = {
            "max_epochs": cap,
            "epochs_run": int(net.n_iter_),
            "training_loss": float(net.loss_),
            "validation_pr_auc": float(average_precision_score(
                fold.val.y, clf.predict_proba(model.inputs(fold.val))[:, 1])),
            "hit_cap": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        }
        report.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
