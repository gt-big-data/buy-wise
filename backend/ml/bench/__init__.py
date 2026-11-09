"""The R1 benchmark (#78): every model is scored the same way.

    interface  what a model must implement (Model), what it is given (Fold, PriceHistory)
    data       load prices and build the feature/label table
    splits     how rows are divided into train / validation / test periods
    metrics    matched-wait-rate scoring plus PR-AUC, Brier and calibration error
    models     the simple rules, the served XGBoost and the registry R2/R3 add to
    runner     fit each model on a fold and collect its scores

Run it with `python -m ml.evaluate` (see ml/evaluate.py).
"""
