"""Train the leak-free buy/wait model that inference.py serves.

    cd backend && .venv/bin/python -m ml.train_clean

Three chronological periods (see evaluate.chronological_split):
  train       fits the models
  validation  early stopping and probability calibration
  test        touched once, at the end, for the report. Nothing is tuned on it.

WAIT is shown only when the calibrated chance of an 8%+ drop is at least
MIN_DROP_CHANCE. Fewer WAITs, but each one is worth acting on.

The model is saved only if it beats the best simple rule on validation at the
same wait rate. That gate is the reason evaluate.py exists.
"""
from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier, XGBRegressor

from ml.evaluate import (baseline_scores, build_dataset, chronological_split, leaderboard,
                         load_prices, score_decision, score_policies)
from ml.features import FEATURES
from ml.inference import MODEL_PATH

MIN_DROP_CHANCE = 0.40  # highest threshold where the model still beats the simple rules on validation


def main() -> None:
    data = build_dataset(load_prices())
    s = chronological_split(data)
    tr, va, te = s.train, s.val.copy(), s.test.copy()
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} rows, {data.asin.nunique()} products")

    clf = XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=2.0, eval_metric="aucpr",
                        early_stopping_rounds=50, n_jobs=-1, random_state=42)
    clf.fit(tr[FEATURES], tr.y, eval_set=[(va[FEATURES], va.y)], verbose=False)

    # Expected size of the 14-day low, as a fraction of today's price. Drives the
    # "predicted best price" the extension shows; the WAIT decision ignores it.
    reg = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=2.0, objective="reg:absoluteerror",
                       early_stopping_rounds=50, n_jobs=-1, random_state=42)
    reg.fit(tr[FEATURES], tr.drop_pct.clip(lower=0), eval_set=[(va[FEATURES], va.drop_pct.clip(lower=0))],
            verbose=False)

    va["proba"] = clf.predict_proba(va[FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(va.proba, va.y)
    wait_rate = float((calibrator.predict(va.proba) >= MIN_DROP_CHANCE).mean())

    # Gate: beat every rule on validation at the wait rate this threshold produces.
    gate = score_policies(va, {"model": va.proba.values, **baseline_scores(va)}, (wait_rate,))
    best_rule = gate[gate.policy != "model"]["$ caught"].max()
    model_val = gate[gate.policy == "model"]["$ caught"].iloc[0]
    print(f"\nvalidation gate @ {wait_rate:.0%} wait: model ${model_val:.2f} vs best rule ${best_rule:.2f}")
    if model_val <= best_rule:
        raise SystemExit("Model does not beat the best rule on validation. Not saving.")

    te["proba"] = clf.predict_proba(te[FEATURES])[:, 1]
    te["pred_drop"] = reg.predict(te[FEATURES]).clip(min=0)
    print("\n" + leaderboard(te, {"clean model": te.proba.values, **baseline_scores(te)}))

    te["p_drop"] = calibrator.predict(te.proba)
    served = te[te.p_drop >= MIN_DROP_CHANCE]
    own = score_decision(te, f"clean model, WAIT at {MIN_DROP_CHANCE:.0%}+", te.p_drop.values >= MIN_DROP_CHANCE)
    print(f"\n=== as served: WAIT when chance of a drop >= {MIN_DROP_CHANCE:.0%} ===")
    print(own.round(3).to_string(index=False))

    # What happens after a WAIT when the drop doesn't come. The panel quotes these.
    change = (served.fwd_last - served.price) / served.price
    wait_outcomes = {
        "higher_share": round(float((change > 0).mean()), 3),
        "about_same_share": round(float((change.abs() <= 0.02).mean()), 3),
        "avg_higher_pct": round(float(change[change > 0].mean()), 3),
    }
    print(f"after a WAIT, price on day 14: higher {wait_outcomes['higher_share']:.1%} of the time "
          f"(by {wait_outcomes['avg_higher_pct']:.1%} on average), within 2% {wait_outcomes['about_same_share']:.1%}")

    pred_low = te.price * (1 - te.pred_drop)
    mae_model = float(np.abs(pred_low - te.fwd_min).mean())
    mae_flat = float(np.abs(te.price - te.fwd_min).mean())
    print(f"\n14-day low forecast: MAE ${mae_model:.2f} vs 'price stays the same' ${mae_flat:.2f}")

    cal = te.groupby(np.digitize(calibrator.predict(te.proba), [0.2, 0.4, 0.6])).agg(
        n=("y", "size"), said=("proba", lambda x: calibrator.predict(x).mean()), happened=("y", "mean"))
    print("\ncalibration on test (shown confidence vs how often it happened):")
    print(cal.round(3).to_string())

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "products": int(data.asin.nunique()),
        "test_period": [str(te.date.min().date()), str(te.date.max().date())],
        "min_drop_chance": MIN_DROP_CHANCE,
        "wait_outcomes": wait_outcomes,
        "test": own.iloc[0].drop("policy").astype(float).round(4).to_dict(),
        "low_forecast_mae": {"model": round(mae_model, 2), "flat": round(mae_flat, 2)},
    }
    joblib.dump({"classifier": clf, "regressor": reg, "calibrator": calibrator,
                 "min_drop_chance": MIN_DROP_CHANCE, "wait_outcomes": wait_outcomes,
                 "features": FEATURES, "meta": meta}, MODEL_PATH)
    print(f"\nsaved {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
