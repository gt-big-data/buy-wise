"""The R1 benchmark (#78): every model is scored the same way.

    interface  what a model must implement (Model), what it is given (Fold, PriceHistory)
    data       load prices and build the feature/label table
    splits     how rows are divided into train / validation / test periods (legacy, walk-forward)
    metrics    matched-wait-rate scoring plus PR-AUC, Brier and calibration error
    models     the simple rules, the served XGBoost, R2's table models and the registry
    tuning     the same tuning budget for every table model, on validation only
    runner     fit each model on a fold and collect its scores
    ci         product-level error bars and paired "does A beat B?" comparisons
    breakdown  results by kind of product (price band, volatility, unseen products, ...)

The choices behind all this are in PROTOCOL.md.

Run it with `python -m ml.evaluate` (see ml/evaluate.py).
"""
