"""
Track record (#85): how often BuyWise's served model was right, scored against what prices actually did.

Two commands:

  replay   Pretend it's each date in turn, ask the served model for BUY/WAIT using only
           prices up to that date, then check every call against the next 14 days.
           This is what the live job would have logged, run on studies/Prices.csv
           because we don't have Keepa access.

  score    Resolve and score an existing prediction log (a CSV with ml.track_record.LOG_COLUMNS),
           e.g. one written day by day once fresh prices exist.

Usage:
    cd backend
    .venv/bin/python scripts/track_record.py replay                    # weekly calls over the model's test period
    .venv/bin/python scripts/track_record.py replay --every 1          # daily calls (slower)
    .venv/bin/python scripts/track_record.py replay --data ../studies/snapshots/apr2026_500/Prices.csv.gz
    .venv/bin/python scripts/track_record.py score track_record_out/predictions.csv

Writes to --out (default track_record_out/, git-ignored):
    predictions.csv   every logged call
    resolved.csv      the same rows plus what happened and whether the call was right
    summary.json      headline numbers for the data viz dashboard
    report.md         hit rate, dollars saved and lost, live vs offline, biggest misses

Only replay dates in the model's test period. Before that, the model was trained on
those very prices, and its track record would look better than it is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.bench.data import DEFAULT_SOURCE, load_prices
from ml.features import HORIZON_DAYS
from ml.track_record import replay_predictions, report, resolve, score

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "track_record_out"


def _write(resolved: pd.DataFrame, out: Path, offline: dict | None, note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(out / "resolved.csv", index=False)
    (out / "summary.json").write_text(json.dumps(score(resolved), indent=2, default=float))
    (out / "report.md").write_text(report(resolved, offline=offline, note=note))
    print(report(resolved, offline=offline, note=note))
    print(f"\nwrote {out}/", file=sys.stderr)


def cmd_replay(args) -> None:
    from ml.inference import MODEL_VERSION, OFFLINE_TEST, TEST_PERIOD, predict_for_asin
    if MODEL_VERSION is None:
        raise SystemExit("model not loaded (ml/buywise_clean.joblib); see ml/train_clean.py")

    prices = load_prices(Path(args.data))
    start = pd.Timestamp(args.start or TEST_PERIOD[0])
    end = pd.Timestamp(args.end) if args.end else prices.date.max() - pd.Timedelta(days=HORIZON_DAYS)
    if Path(args.data).resolve() == DEFAULT_SOURCE.resolve() and start < pd.Timestamp(TEST_PERIOD[0]):
        print(f"WARNING: {start.date()} is before the model's test period ({TEST_PERIOD[0]}). "
              "It trained on those prices, so these results will look too good.", file=sys.stderr)
    dates = pd.date_range(start, end, freq=f"{args.every}D")
    print(f"replaying {MODEL_VERSION} on {prices.asin.nunique()} products x {len(dates)} dates "
          f"({start.date()} to {end.date()})", file=sys.stderr)

    log = replay_predictions(prices, dates, predict_for_asin, MODEL_VERSION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log.to_csv(out / "predictions.csv", index=False)
    note = (f"Replayed on `{Path(args.data).name}`: the served model ({MODEL_VERSION}) predicted every "
            f"product every {args.every} day(s) using only earlier prices, then each call was checked "
            f"against the next {HORIZON_DAYS} days. No live Keepa data this semester, so this is a "
            "replay, not calls shoppers saw.")
    _write(resolve(log, prices), out, OFFLINE_TEST, note)


def cmd_score(args) -> None:
    log = pd.read_csv(args.log, parse_dates=["predicted_on"])
    prices = load_prices(Path(args.data))
    try:
        from ml.inference import OFFLINE_TEST
    except Exception:  # scoring alone doesn't need the model
        OFFLINE_TEST = None
    _write(resolve(log, prices, as_of=args.as_of), Path(args.out), OFFLINE_TEST,
           f"Scored `{Path(args.log).name}` against `{Path(args.data).name}`.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="replay the served model over past prices, then score it")
    r.add_argument("--data", default=str(DEFAULT_SOURCE), help="Prices.csv-style file (default studies/Prices.csv)")
    r.add_argument("--start", help="first prediction date (default: start of the model's test period)")
    r.add_argument("--end", help=f"last prediction date (default: {HORIZON_DAYS} days before the data ends)")
    r.add_argument("--every", type=int, default=7, help="days between prediction dates (default 7)")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.set_defaults(func=cmd_replay)

    s = sub.add_parser("score", help="resolve and score an existing prediction log")
    s.add_argument("log", help="CSV with asin, predicted_on, price_at_prediction, p_drop, recommendation, "
                               "horizon_days, drop_threshold, model_version")
    s.add_argument("--data", default=str(DEFAULT_SOURCE), help="prices to check the calls against")
    s.add_argument("--as-of", help="last day of known prices (default: end of --data)")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
