"""The shared scoring tool. Any model or rule goes in, one standard report comes out.

Every policy is compared at the SAME wait rate: each one says WAIT on exactly the
top N% of decisions by its own score. Without that, a policy can look better just
by saying WAIT more often (the April model did exactly this).

    cd backend && .venv/bin/python -m ml.evaluate

Metrics, per policy and wait rate:
  precision   share of WAITs where the price really fell >= 8% within 14 days
  lift        precision / base rate (1.0 = no better than guessing)
  $ caught    average saving per decision if the shopper buys at the 14-day low
  $ no alert  average saving if they just buy 14 days later at whatever it costs
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.features import FEATURES, HORIZON_DAYS, REQUIRED, add_features, add_labels, daily_panel

STUDIES = Path(__file__).resolve().parents[2] / "studies"
WAIT_RATES = (0.10, 0.20, 0.30)
MIN_SPAN_DAYS = 270  # ASINs with less history can't fill the 180-day features


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_prices(path: Path = STUDIES / "Prices.csv") -> pd.DataFrame:
    """Read the Keepa export into the column names the backend DB uses."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"datetime": "date", "amazon": "price", "used": "used_price"})
    return df[["asin", "date", "price", "used_price", "count_new", "count_used"]]


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    span = prices.dropna(subset=["price"]).groupby("asin")["date"].agg(lambda s: (s.max() - s.min()).days)
    prices = prices[prices.asin.isin(span[span >= MIN_SPAN_DAYS].index)]
    p = add_labels(add_features(daily_panel(prices)))
    return p.dropna(subset=REQUIRED).reset_index(drop=True)


def chronological_split(p: pd.DataFrame) -> Split:
    """60/20/20 by date. A row only lands in a period if its 14-day label
    finishes before the next period starts, so no answer leaks across."""
    dates = sorted(p.date.unique())
    c1 = pd.Timestamp(dates[int(len(dates) * 0.6)])
    c2 = pd.Timestamp(dates[int(len(dates) * 0.8)])
    h = pd.Timedelta(days=HORIZON_DAYS)
    return Split(
        train=p[p.date + h < c1],
        val=p[(p.date >= c1) & (p.date + h < c2)],
        test=p[p.date >= c2],
    )


def baseline_scores(te: pd.DataFrame) -> dict[str, np.ndarray]:
    """The rules every model has to beat. Higher score = more reason to WAIT."""
    return {
        "rule: top of 90-day range": te.pos90.values,
        "rule: price vs 90-day median": te.rel90.values,
        "rule: wait in November": (te.date.dt.month == 11).astype(float).values
        + np.random.default_rng(0).random(len(te)) * 1e-6,  # break ties randomly
    }


def _policy_row(te: pd.DataFrame, wait: np.ndarray) -> dict:
    y = te.y.values
    prec = y[wait].mean() if wait.any() else np.nan
    return {
        "wait %": wait.mean() * 100,
        "precision %": prec * 100,
        "lift": prec / y.mean(),
        "$ caught": te.price.mean() - np.where(wait, te.fwd_min, te.price).mean(),
        "$ no alert": te.price.mean() - np.where(wait, te.fwd_last, te.price).mean(),
    }


def score_policies(te: pd.DataFrame, scores: dict[str, np.ndarray],
                   wait_rates: tuple[float, ...] = WAIT_RATES) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        auc = roc_auc_score(te.y, s) if np.isfinite(s).all() else np.nan
        for q in wait_rates:
            wait = s >= np.quantile(s, 1 - q)
            rows.append({"policy": name, "target wait %": int(q * 100), "AUC": auc, **_policy_row(te, wait)})
    return pd.DataFrame(rows)


def score_decision(te: pd.DataFrame, name: str, wait: np.ndarray) -> pd.DataFrame:
    """Score a policy at its own operating point, not a matched wait rate."""
    return pd.DataFrame([{"policy": name, **_policy_row(te, np.asarray(wait, bool))}])


def oracle(te: pd.DataFrame) -> float:
    return te.price.mean() - te.fwd_min.mean()


def deployed_april_model_scores(te: pd.DataFrame) -> np.ndarray | None:
    """P(WAIT) from the April 2026 model, for the leaderboard. Rebuilds its
    40 features with its own pipeline (which leaks; that is the point of scoring
    it). Returns None if the old model files are gone."""
    import joblib

    ml_dir = Path(__file__).parent
    try:
        cls14 = joblib.load(ml_dir / "xgb_cls_14d.joblib")
        scaler = joblib.load(ml_dir / "scaler.joblib")
    except FileNotFoundError:
        return None
    from ml.dataset import engineer_features, merge_product_features

    old_feats = [
        "price", "new_price", "used_price", "list_price", "count_new", "count_used", "sales_score",
        "amazon_ma_7d", "amazon_ma_14d", "amazon_delta_7d", "amazon_pct_change_7d", "day_of_week", "month",
        "rsi_14d", "rsi_7d", "dist_from_30d_high", "dist_from_30d_low", "velocity_3d",
        "day_of_year_sin", "day_of_year_cos", "price_z_asin", "price_minmax_asin",
        "roll7_mean", "roll7_std", "roll7_min", "roll7_max", "price_vs_roll7_z", "price_lag1", "price_lag30",
        "pct_change_1d", "pct_change_14d", "week_of_year", "global_min", "global_max", "global_norm_sd",
        "log_mean_price", "global_mean", "price_vs_global_min", "price_vs_global_max", "price_vs_global_mean",
    ]
    px = pd.read_csv(STUDIES / "Prices.csv", parse_dates=["datetime"])
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
    pr = pd.read_csv(STUDIES / "Product.csv")
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


def leaderboard(te: pd.DataFrame, scores: dict[str, np.ndarray]) -> str:
    r = score_policies(te, scores)
    lines = [
        f"test set: {len(te):,} decisions, {te.asin.nunique()} products, "
        f"{te.date.min().date()} to {te.date.max().date()}",
        f"base rate P(drop >= 8% in 14d) = {te.y.mean() * 100:.1f}%   "
        f"mean price ${te.price.mean():.2f}   perfect-foresight ceiling ${oracle(te):.2f}/decision",
    ]
    for q in sorted(r["target wait %"].unique()):
        block = r[r["target wait %"] == q].drop(columns="target wait %").sort_values("$ caught", ascending=False)
        lines += ["", f"=== everyone says WAIT on {q}% of decisions ===", block.round(3).to_string(index=False)]
    return "\n".join(lines)


def main() -> None:
    import joblib

    from ml.inference import MODEL_PATH

    data = build_dataset(load_prices())
    te = chronological_split(data).test.copy()
    scores = baseline_scores(te)
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        scores["clean model (served)"] = bundle["classifier"].predict_proba(te[FEATURES])[:, 1]
    old = deployed_april_model_scores(te)
    if old is not None:
        scores["April model (old)"] = old
    print(leaderboard(te, scores))


if __name__ == "__main__":
    main()
