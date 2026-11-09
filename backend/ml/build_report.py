"""Dataset snapshot builder, funnel reporter, and experiment logger.

Usage:
    python -m ml.build_report build --source ../studies/Prices.csv
    python -m ml.build_report evaluate --snapshot <snapshot_id> --name "baseline_run" [--allow-dirty]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
import subprocess
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from ml.evaluate import (
    STUDIES,
    MIN_SPAN_DAYS,
    build_dataset,
    chronological_split,
    load_prices,
    score_policies,
    baseline_scores,
)

SNAPSHOTS_DIR = STUDIES / "snapshots"
LOG_PATH = STUDIES / "experiment_log.csv"
WAIT_RATE = 20    # as a percentage. need to keep consistent for cross-experiment comparisons


def _compute_hash(path: Path) -> str:
    # Hash the file chunk by chunk to get a stable ID
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:8]


def _check_git_clean(allow_dirty: bool = False) -> str:
    # Ensure there are no uncommitted changes so experiments are reproducible
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    if res.stdout.strip():
        print("\n[WARNING] Git working directory has uncommitted changes.")
        if allow_dirty:
            print("[INFO] Proceeding anyway due to --allow-dirty flag.")
        else:
            choice = input("Do you want to proceed and log anyway for testing? (y/N): ").strip().lower()
            if choice != 'y':
                raise RuntimeError(
                    "Evaluation aborted due to uncommitted changes."
                )
            print("[INFO] Proceeding with uncommitted changes...")

    commit_res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return commit_res.stdout.strip()


def build_snapshot(source_path: Path) -> str:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    snap_id = _compute_hash(source_path)
    snap_dir = SNAPSHOTS_DIR / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building snapshot [{snap_id}] from {source_path}...")

    # 1. Load raw data
    raw_df = pd.read_csv(source_path, parse_dates=["datetime"])
    raw_df.columns = raw_df.columns.str.lower().str.strip()
    total_raw_rows = len(raw_df)
    total_raw_asins = raw_df["asin"].nunique()

    # 2. Drop rows with missing prices
    price_col = "amazon" if "amazon" in raw_df.columns else "price"
    valid_price_df = raw_df.dropna(subset=[price_col])
    valid_rows = len(valid_price_df)
    valid_asins = valid_price_df["asin"].nunique()

    # 3. Filter out products with insufficient history span
    span = valid_price_df.groupby("asin")["datetime"].agg(lambda s: (s.max() - s.min()).days)
    passing_span_asins = span[span >= MIN_SPAN_DAYS].index
    span_filtered_df = valid_price_df[valid_price_df["asin"].isin(passing_span_asins)]
    span_asins_count = len(passing_span_asins)

    # 4. Run pipeline to get final modeling dataset count
    processed_df = build_dataset(load_prices(source_path))
    final_rows = len(processed_df)
    final_asins = processed_df["asin"].nunique()

    missing_pct = (valid_price_df.isna().mean() * 100).to_dict()
    date_min = valid_price_df["datetime"].min()
    date_max = valid_price_df["datetime"].max()
    date_span_days = (date_max - date_min).days

    # Save raw snapshot copy for downstream evaluation loader
    snapshot_csv = snap_dir / "Prices.csv"
    shutil.copyfile(source_path, snapshot_csv)

    # Save final processed dataset and compute true output hash
    processed_csv = snap_dir / "processed_dataset.csv"
    processed_df.to_csv(processed_csv, index=False)
    output_hash = _compute_hash(processed_csv)

    # Generate markdown report
    current_time_utc = datetime.now(timezone.utc).isoformat()
    report_content = f"""# Data Snapshot Report: `{snap_id}`

- **Created At:** {current_time_utc} UTC
- **Output Hash:** `{output_hash}`
- **Source File:** `{source_path.name}`
- **Date Range:** {date_min.date()} to {date_max.date()} ({date_span_days} days)

## Filtering Funnel

| Stage | Unique ASINs | Total Rows | Drop-off / Notes |
| :--- | :--- | :--- | :--- |
| **1. Raw Input** | {total_raw_asins:,} | {total_raw_rows:,} | Initial load from source file |
| **2. Valid Prices** | {valid_asins:,} | {valid_rows:,} | Removed rows with missing price values |
| **3. History Span ($\ge$ {MIN_SPAN_DAYS} days)** | {span_asins_count:,} | {len(span_filtered_df):,} | Filtered out products with insufficient history |
| **4. Final Modeling Dataset** | {final_asins:,} | {final_rows:,} | Dropped rows missing required rolling feature windows |

## Columns Missing (%)
"""
    for col, pct in missing_pct.items():
        report_content += f"- **{col}**: {pct:.2f}%\n"

    report_path = snap_dir / "report.md"
    report_path.write_text(report_content)
    print(f"Snapshot created: {snap_dir}")
    print(f"Output Hash: {output_hash}")
    print(f"Report written: {report_path}")
    return snap_id


def evaluate_and_log(snapshot_id: str, settings_name: str, allow_dirty: bool = False) -> None:
    # Verify git status and grab current commit hash
    git_commit = _check_git_clean(allow_dirty=allow_dirty)
    snap_dir = SNAPSHOTS_DIR / snapshot_id
    prices_path = snap_dir / "Prices.csv"

    if not prices_path.exists():
        raise FileNotFoundError(f"Snapshot ID {snapshot_id} not found at {snap_dir}")

    print(f"Evaluating snapshot {snapshot_id} (Commit: {git_commit[:7]})...")

    # Load data from the snapshot
    prices = load_prices(prices_path)
    data = build_dataset(prices)
    te = chronological_split(data).test.copy()

    # Get baseline and served model scores
    scores = baseline_scores(te)
    from ml.inference import MODEL_PATH
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and "classifier" in bundle:
            feats = bundle.get("features", list(scores.keys()))
            scores["clean model (served)"] = bundle["classifier"].predict_proba(te[feats])[:, 1]

    # Score policies and extract metrics at 20% wait rate
    r = score_policies(te, scores)
    sub = r[r["target wait %"] == WAIT_RATE]

    log_rows = []
    user = os.getenv("USER", "teammate")
    timestamp = datetime.now(timezone.utc).isoformat()

    for _, row in sub.iterrows():
        log_rows.append({
            "timestamp": timestamp,
            "user": user,
            "git_commit": git_commit,
            "snapshot_id": snapshot_id,
            "settings": settings_name,
            "policy": row["policy"],
            "wait_pct": row["wait %"],
            "precision_pct": row["precision %"],
            "lift": row["lift"],
            "dollar_caught": row["$ caught"],
        })

    log_df = pd.DataFrame(log_rows)
    header = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LOG_PATH, mode="a", header=header, index=False)

    print(f"Results appended to {LOG_PATH}")
    print(r.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build snapshots and log evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a snapshot from prices CSV.")
    build_parser.add_argument("--source", type=Path, default=STUDIES / "Prices.csv")

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation and append to log.")
    eval_parser.add_argument("--snapshot", type=str, required=True)
    eval_parser.add_argument("--name", type=str, default="default_run")
    eval_parser.add_argument("--allow-dirty", action="store_true", help="Bypass dirty git check without prompting.")

    args = parser.parse_args()

    if args.command == "build":
        build_snapshot(args.source)
    elif args.command == "evaluate":
        evaluate_and_log(args.snapshot, args.name, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()