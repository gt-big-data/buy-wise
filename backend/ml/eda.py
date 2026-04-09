"""
backend/ml/eda.py
─────────────────
Exploratory Data Analysis for BuyWise price-history datasets.

Run after dataset.py has generated the CSVs:

    python backend/ml/eda.py

Saves all figures to backend/ml/data/eda/
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_HERE    = Path(__file__).parent
DATA_DIR = _HERE / "data"
EDA_DIR  = DATA_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Consistent style ──────────────────────────────────────────────────────────
PALETTE = {
    "buy":    "#4C9BE8",
    "wait":   "#E87B4C",
    "train":  "#5AB88C",
    "test":   "#B85A8C",
    "neutral":"#8C8C8C",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#EEEEEE",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})

def _save(fig: plt.Figure, name: str) -> None:
    path = EDA_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved  %s", path)


# ── 1. Class balance ──────────────────────────────────────────────────────────

def plot_class_balance(train7: pd.DataFrame, train14: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Balance (Train Set)", fontsize=14, fontweight="bold")

    for ax, df, horizon in zip(axes, [train7, train14], [7, 14]):
        col   = f"label_{horizon}d"
        counts = df[col].value_counts().sort_index()
        labels = ["BUY (0)", "WAIT (1)"]
        colors = [PALETTE["buy"], PALETTE["wait"]]
        bars   = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")

        for bar, count in zip(bars, counts.values):
            pct = count / counts.sum() * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.02,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10,
            )

        ax.set_title(f"{horizon}-Day Forecast")
        ax.set_ylabel("Row count")
        ax.set_ylim(0, counts.max() * 1.2)
        ax.grid(axis="x", visible=False)

    plt.tight_layout()
    _save(fig, "01_class_balance")


# ── 2. Train / test timeline ──────────────────────────────────────────────────

def plot_train_test_split(train7: pd.DataFrame, test7: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    for df, label, color in [
        (train7, "Train", PALETTE["train"]),
        (test7,  "Test",  PALETTE["test"]),
    ]:
        daily = df.groupby("date").size().reset_index(name="products")
        ax.fill_between(
            pd.to_datetime(daily["date"]),
            daily["products"],
            alpha=0.4, color=color, label=label,
        )
        ax.plot(pd.to_datetime(daily["date"]), daily["products"], color=color, linewidth=1)

    ax.axvline(
        pd.Timestamp("2024-12-01"),
        color="black", linestyle="--", linewidth=1.5, label="Cutoff (Dec 1 2024)",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.set_title("Products with Price Data per Day")
    ax.set_ylabel("# products")
    ax.legend()
    plt.tight_layout()
    _save(fig, "02_train_test_split")


# ── 3. Price distribution ─────────────────────────────────────────────────────

def plot_price_distribution(train7: pd.DataFrame) -> None:
    """Plot KDE of price_lag1 (≈ current price) split by label."""
    col   = "price_lag1"
    if col not in train7.columns:
        log.warning("price_lag1 not in columns — skipping price distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for label_val, color, name in [
        (0, PALETTE["buy"],  "BUY"),
        (1, PALETTE["wait"], "WAIT"),
    ]:
        prices = train7.loc[train7["label_7d"] == label_val, col].dropna()
        prices = prices[prices < prices.quantile(0.995)]   # clip extreme outliers

        kde = gaussian_kde(prices, bw_method=0.2)
        x   = np.linspace(prices.min(), prices.max(), 300)
        ax.fill_between(x, kde(x), alpha=0.35, color=color, label=name)
        ax.plot(x, kde(x), color=color, linewidth=1.5)

    ax.set_title("Price Distribution by Label (7-Day, Train)")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    _save(fig, "03_price_distribution")


# ── 4. Rolling-mean feature ───────────────────────────────────────────────────

def plot_rolling_means(train7: pd.DataFrame) -> None:
    """Scatter of roll7_mean vs roll30_mean, coloured by label."""
    for col_x, col_y in [("roll7_mean", "roll30_mean")]:
        if col_x not in train7.columns or col_y not in train7.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        sample = train7.sample(min(5_000, len(train7)), random_state=0)

        for label_val, color, name in [
            (0, PALETTE["buy"],  "BUY"),
            (1, PALETTE["wait"], "WAIT"),
        ]:
            sub = sample[sample["label_7d"] == label_val]
            ax.scatter(sub[col_x], sub[col_y], c=color, alpha=0.3, s=8, label=name)

        ax.set_title("7-Day vs 30-Day Rolling Mean (normalised)")
        ax.set_xlabel("roll7_mean (normalised)")
        ax.set_ylabel("roll30_mean (normalised)")
        ax.legend(markerscale=3)
        plt.tight_layout()
        _save(fig, "04_rolling_mean_scatter")


# ── 5. Seasonal patterns ──────────────────────────────────────────────────────

def plot_seasonal_wait_rate(train7: pd.DataFrame) -> None:
    """Mean WAIT rate by month and by day-of-week."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Seasonal WAIT Rate (Train, 7-Day Label)", fontsize=13, fontweight="bold")

    # By month
    month_rate = train7.groupby("month")["label_7d"].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    axes[0].bar(
        month_rate.index, month_rate.values * 100,
        color=PALETTE["wait"], alpha=0.75, edgecolor="white",
    )
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names, rotation=45, ha="right")
    axes[0].set_title("By Month")
    axes[0].set_ylabel("WAIT rate (%)")

    # By day of week
    dow_rate  = train7.groupby("day_of_week")["label_7d"].mean()
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    axes[1].bar(
        dow_rate.index, dow_rate.values * 100,
        color=PALETTE["neutral"], alpha=0.75, edgecolor="white",
    )
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_names)
    axes[1].set_title("By Day of Week")
    axes[1].set_ylabel("WAIT rate (%)")

    plt.tight_layout()
    _save(fig, "05_seasonal_wait_rate")


# ── 6. Feature correlation heatmap ────────────────────────────────────────────

def plot_feature_correlation(train7: pd.DataFrame) -> None:
    from matplotlib.colors import TwoSlopeNorm

    num_cols = [
        "price_lag1", "price_lag7", "price_lag14",
        "roll7_mean", "roll14_mean", "roll30_mean",
        "pct_change_1d", "pct_change_7d", "price_vs_30d_mean",
        "days_to_prime_day", "days_to_black_friday",
        "label_7d",
    ]
    num_cols = [c for c in num_cols if c in train7.columns]
    corr     = train7[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    norm    = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im      = ax.imshow(corr.values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < 0.6 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature Correlation Matrix (Train 7-Day)", pad=12)
    plt.tight_layout()
    _save(fig, "06_feature_correlation")


# ── 7. Deal-flag impact ───────────────────────────────────────────────────────

def plot_deal_flag_impact(train7: pd.DataFrame) -> None:
    if "deal_flag" not in train7.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    groups = train7.groupby("deal_flag")["label_7d"].mean() * 100
    labels = ["No deal", "Deal active"]
    colors = [PALETTE["buy"], PALETTE["wait"]]

    bars = ax.bar(
        labels,
        [groups.get(False, 0), groups.get(True, 0)],
        color=colors, width=0.45, edgecolor="white",
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_title("WAIT Rate by Deal Flag (Train, 7-Day)")
    ax.set_ylabel("WAIT rate (%)")
    ax.set_ylim(0, max(groups.values) * 1.25)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    _save(fig, "07_deal_flag_impact")


# ── 8. Sample product price trace ─────────────────────────────────────────────

def plot_sample_product(train7: pd.DataFrame, test7: pd.DataFrame) -> None:
    """Show price history + WAIT labels for a single product."""
    pid = train7["product_id"].value_counts().idxmax()

    tr = train7[train7["product_id"] == pid].sort_values("date")
    te = test7[test7["product_id"] == pid].sort_values("date")

    if "price_lag1" not in tr.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    for df, color, label in [(tr, PALETTE["train"], "Train"), (te, PALETTE["test"], "Test")]:
        ax.plot(pd.to_datetime(df["date"]), df["price_lag1"],
                color=color, linewidth=1.5, label=label)
        wait = df[df["label_7d"] == 1]
        ax.scatter(pd.to_datetime(wait["date"]), wait["price_lag1"],
                   color=PALETTE["wait"], s=18, zorder=5, label="WAIT label" if label == "Train" else "")

    ax.axvline(pd.Timestamp("2024-12-01"), color="black",
               linestyle="--", linewidth=1.2, label="Cutoff")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.set_title(f"Price History with WAIT Labels — product_id={pid}")
    ax.set_ylabel("Price ($, normalised)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "08_sample_product_trace")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eda() -> None:
    log.info("Loading datasets from %s …", DATA_DIR)

    required = ["train_7day.csv", "test_7day.csv", "train_14day.csv"]
    for name in required:
        if not (DATA_DIR / name).exists():
            log.error(
                "Missing %s — run `python -m backend.ml.dataset --mock` first.", name
            )
            return

    train7  = pd.read_csv(DATA_DIR / "train_7day.csv",  parse_dates=["date"])
    test7   = pd.read_csv(DATA_DIR / "test_7day.csv",   parse_dates=["date"])
    train14 = pd.read_csv(DATA_DIR / "train_14day.csv", parse_dates=["date"])

    log.info("Running EDA plots …")
    plot_class_balance(train7, train14)
    plot_train_test_split(train7, test7)
    plot_price_distribution(train7)
    plot_rolling_means(train7)
    plot_seasonal_wait_rate(train7)
    plot_feature_correlation(train7)
    plot_deal_flag_impact(train7)
    plot_sample_product(train7, test7)

    log.info("All plots saved to %s", EDA_DIR)


if __name__ == "__main__":
    run_eda()
