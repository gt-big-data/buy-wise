"""Track record (#85): log predictions, check them against what happened, score them.

Three steps, each usable on its own:

  replay_predictions()  pretend it's each date in turn and ask the served model
                        (ml.inference.predict_for_asin) for a call, using only
                        prices up to that date. Produces a prediction log.
  resolve()             once a prediction's horizon has passed, look up what the
                        price actually did and mark the call right or wrong.
  score() / report()    the same measures as the offline benchmark
                        (ml.bench.metrics._policy_row), plus the misses.

Without Keepa access the log comes from replaying studies/Prices.csv. When fresh
prices exist (Keepa, or the extension's own page visits), the same resolve() and
score() run on a log written day by day, so live and replayed numbers line up.

What counts as right (HORIZON_DAYS = 14, DROP_THRESHOLD = 8%, from ml.features):
  WAIT is right if the price fell >= 8% below the predicted-on price at some point
       in the next 14 days. That's the drop a WAIT promised.
  BUY  is right if it did NOT. If an 8% drop came, the shopper would have done
       better waiting, so the BUY was wrong.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ml.bench.metrics import _policy_row
from ml.features import DROP_THRESHOLD, HORIZON_DAYS, add_labels, daily_panel

LOG_COLUMNS = ["asin", "predicted_on", "price_at_prediction", "p_drop", "recommendation",
               "horizon_days", "drop_threshold", "model_version"]

PredictFn = Callable[..., dict]  # predict_for_asin(records, today=...) -> {"recommendation", "p_drop", ...}


def _records(hist: pd.DataFrame) -> list[dict]:
    """Price rows in the shape the DB returns and predict_for_asin expects."""
    cols = [c for c in ("used_price", "count_new", "count_used") if c in hist.columns]
    out = []
    for r in hist.itertuples(index=False):
        rec = {"timestamp": r.date, "price": float(r.price)}
        for c in cols:
            v = getattr(r, c)
            rec[c] = None if pd.isna(v) else float(v)
        out.append(rec)
    return out


def replay_predictions(prices: pd.DataFrame, dates: Iterable, predict_fn: PredictFn,
                       model_version: str, horizon_days: int = HORIZON_DAYS,
                       drop_threshold: float = DROP_THRESHOLD) -> pd.DataFrame:
    """Predict every product on every date, seeing only prices up to that date.

    `prices` has asin, date, price (and optionally used_price, count_new,
    count_used), e.g. from ml.bench.data.load_prices(). Products the model can't
    score yet (too little history) are skipped for that date, as live would.
    """
    prices = prices.dropna(subset=["price"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    dates = sorted(pd.Timestamp(d).normalize() for d in dates)
    rows = []
    for asin, g in prices.sort_values("date").groupby("asin"):
        for d in dates:
            hist = g[g.date <= d]  # nothing after the day we're pretending it is
            if hist.empty:
                continue
            try:
                r = predict_fn(_records(hist), today=d)
            except RuntimeError:
                continue  # not enough history yet
            rows.append({
                "asin": asin,
                "predicted_on": d,
                # Keepa logs price changes, so the last price at or before d still holds on d
                "price_at_prediction": float(hist.price.iloc[-1]),
                "p_drop": float(r["p_drop"]),
                "recommendation": r["recommendation"],
                "horizon_days": horizon_days,
                "drop_threshold": drop_threshold,
                "model_version": model_version,
            })
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def resolve(log: pd.DataFrame, prices: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Add what actually happened to each logged prediction.

    `as_of` is the last day we have prices for (default: the latest date in
    `prices`). Adds:
      status            resolved | pending (horizon not over yet) | bad_data
      actual_min_price  lowest price in the horizon after predicted_on
      actual_end_price  price on the last day of the horizon
      drop_pct          how far below price_at_prediction it got
      dropped           drop_pct >= drop_threshold
      was_correct       WAIT and dropped, or BUY and not dropped

    Uses ml.features.add_labels, the same code that labels the offline benchmark,
    so "dropped" means exactly what it means there. bad_data is a +/-100% move,
    which add_labels treats as a Keepa error rather than a price.
    """
    out = log.copy()
    out["predicted_on"] = pd.to_datetime(out["predicted_on"]).dt.normalize()
    prices = prices.dropna(subset=["price"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else prices.date.max()
    prices = prices[prices.date <= as_of]

    resolved = []
    for h in out["horizon_days"].dropna().unique():
        h = int(h)
        lab = add_labels(daily_panel(prices[prices.asin.isin(out.asin.unique())], until=as_of), horizon=h)
        lab = lab[["asin", "date", "fwd_min", "fwd_last", "drop_pct"]].assign(horizon_days=h)
        resolved.append(lab.rename(columns={"date": "predicted_on", "fwd_min": "actual_min_price",
                                            "fwd_last": "actual_end_price"}))
    lab = pd.concat(resolved, ignore_index=True) if resolved else pd.DataFrame(
        columns=["asin", "predicted_on", "actual_min_price", "actual_end_price", "drop_pct", "horizon_days"])
    out = out.drop(columns=[c for c in ("actual_min_price", "actual_end_price", "drop_pct", "status",
                                        "dropped", "was_correct") if c in out.columns])
    out = out.merge(lab, on=["asin", "predicted_on", "horizon_days"], how="left")

    # Score against the price we logged, not the panel's, in case they differ.
    found = out.actual_min_price.notna()
    out.loc[found, "drop_pct"] = (out.price_at_prediction - out.actual_min_price) / out.price_at_prediction

    due = out.predicted_on + pd.to_timedelta(out.horizon_days, unit="D") <= as_of
    out["status"] = np.where(found, "resolved", np.where(due, "bad_data", "pending"))
    dropped = out.drop_pct >= out.drop_threshold
    out["dropped"] = dropped.where(found)
    out["was_correct"] = np.where(out.recommendation == "WAIT", dropped, ~dropped)
    out["was_correct"] = out["was_correct"].where(found)
    return out


def score(resolved: pd.DataFrame) -> dict:
    """Headline numbers over resolved predictions. Offline-comparable fields use
    ml.bench.metrics._policy_row, the function behind the benchmark's leaderboard."""
    r = resolved[resolved.status == "resolved"]
    counts = resolved.status.value_counts().to_dict()
    base = {"predictions": int(len(resolved)), "resolved": int(counts.get("resolved", 0)),
            "pending": int(counts.get("pending", 0)), "bad_data": int(counts.get("bad_data", 0))}
    if r.empty:
        return base

    wait = (r.recommendation == "WAIT").values
    dropped = r.dropped.astype(bool).values
    te = pd.DataFrame({"price": r.price_at_prediction.values, "fwd_min": r.actual_min_price.values,
                       "fwd_last": r.actual_end_price.values, "y": dropped.astype(int)})
    offline_style = _policy_row(te, wait)

    price, low, end = te.price.values, te.fwd_min.values, te.fwd_last.values
    right_wait, wrong_wait = wait & dropped, wait & ~dropped
    wrong_buy = ~wait & dropped
    return {
        **base,
        "products": int(r.asin.nunique()),
        "first_day": str(r.predicted_on.min().date()), "last_day": str(r.predicted_on.max().date()),
        "hit_rate %": float(r.was_correct.astype(bool).mean() * 100),
        "base_rate %": float(dropped.mean() * 100),
        "wait_calls": int(wait.sum()), "buy_calls": int((~wait).sum()),
        "wait_right %": float(dropped[wait].mean() * 100) if wait.any() else float("nan"),
        "buy_right %": float((~dropped[~wait]).mean() * 100) if (~wait).any() else float("nan"),
        # same definitions as the offline leaderboard
        "precision %": float(offline_style["precision %"]),
        "lift": float(offline_style["lift"]),
        "$ caught": float(offline_style["$ caught"]),
        "$ no alert": float(offline_style["$ no alert"]),
        # where the money went, in total dollars across all resolved calls
        "$ saved by right WAITs": float((price - low)[right_wait].sum()),
        "$ lost by wrong WAITs": float(np.clip(end - price, 0, None)[wrong_wait].sum()),
        "$ missed by wrong BUYs": float((price - low)[wrong_buy].sum()),
    }


def misses(resolved: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """The wrong calls that cost the most, so the report shows losses, not just wins."""
    r = resolved[(resolved.status == "resolved") & (resolved.was_correct == False)].copy()  # noqa: E712
    r["cost_usd"] = np.where(r.recommendation == "WAIT",
                           np.clip(r.actual_end_price - r.price_at_prediction, 0, None),  # paid more by waiting
                           r.price_at_prediction - r.actual_min_price)                     # missed the drop
    cols = ["asin", "predicted_on", "recommendation", "p_drop", "price_at_prediction",
            "actual_min_price", "actual_end_price", "cost_usd"]
    return r.sort_values("cost_usd", ascending=False)[cols].head(n)


def report(resolved: pd.DataFrame, offline: dict | None = None, title: str = "BuyWise track record",
           note: str = "") -> str:
    """Markdown report. `offline` is the model's own test-set numbers (bundle meta['test'])."""
    s = score(resolved)
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [f"{s['predictions']:,} predictions logged: {s['resolved']:,} resolved, "
              f"{s['pending']:,} still pending, {s['bad_data']:,} skipped for bad price data.", ""]
    if not s["resolved"]:
        return "\n".join(lines + ["Nothing has resolved yet."])

    pct = lambda k: "n/a" if np.isnan(s[k]) else f"{s[k]:.1f}%"  # noqa: E731
    lines += [
        f"{s['products']} products, predicted {s['first_day']} to {s['last_day']}.", "",
        "## How often it was right", "",
        "| | |", "|---|---|",
        f"| Overall hit rate | {pct('hit_rate %')} |",
        f"| WAIT calls right (price fell 8%+ within 14 days) | {pct('wait_right %')} of {s['wait_calls']:,} |",
        f"| BUY calls right (no 8% drop came) | {pct('buy_right %')} of {s['buy_calls']:,} |",
        f"| Base rate (how often an 8% drop came at all) | {pct('base_rate %')} |", "",
        "## Dollars", "",
        "| | |", "|---|---|",
        f"| Saved by right WAITs | ${s['$ saved by right WAITs']:,.2f} |",
        f"| Lost by wrong WAITs (price went up while waiting) | ${s['$ lost by wrong WAITs']:,.2f} |",
        f"| Missed by wrong BUYs (a drop came after we said buy) | ${s['$ missed by wrong BUYs']:,.2f} |", "",
    ]
    rows = [("precision %", "Precision"), ("lift", "Lift"), ("$ caught", "$ caught / decision"),
            ("$ no alert", "$ no alert / decision")]
    lines += ["## Compared with the offline benchmark", "",
              "Same definitions as `ml/bench/metrics.py`, at the model's served threshold.", "",
              "| Measure | This track record | Offline test set |", "|---|---|---|"]
    for k, label in rows:
        off = offline.get(k) if offline else None
        fmt = (lambda v: f"{v:.2f}") if k == "lift" else (lambda v: f"{v:.1f}%") if "%" in k else (lambda v: f"${v:.2f}")
        lines.append(f"| {label} | {fmt(s[k])} | {fmt(off) if off is not None else 'n/a'} |")
    m = misses(resolved)
    lines += ["", "## Biggest misses", ""]
    if m.empty:
        lines.append("None.")
    else:
        lines += ["| ASIN | Predicted on | Said | P(drop) | Price then | Low in 14d | Price day 14 | Cost |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in m.itertuples(index=False):
            lines.append(f"| {r.asin} | {pd.Timestamp(r.predicted_on).date()} | {r.recommendation} | "
                         f"{r.p_drop:.0%} | ${r.price_at_prediction:.2f} | ${r.actual_min_price:.2f} | "
                         f"${r.actual_end_price:.2f} | ${r.cost_usd:.2f} |")
    return "\n".join(lines) + "\n"
