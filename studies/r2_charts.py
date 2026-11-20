"""Charts for the paper, regenerated from the saved result tables (never by hand).

Needs the tables from:
    python -m ml.evaluate --models <...> --folds walk --ci --save   (leaderboard, tuning log)
    python s003_signal_inventory.py                                 (feature groups)
    python s017_error_breakdown.py                                  (segments)

    cd studies && python r2_charts.py [prices|apr2026_500]

Writes results/figures/<tag>_fig*.png.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from r2_common import RESULTS, SOURCES, tag_for  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-2, validated together;
# text and rules in neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})


def _ci(ax, y, est, lo, hi, color, label=None, size=40):
    ax.hlines(y, lo, hi, color=color, linewidth=2, zorder=2)
    ax.scatter(est, y, s=size, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)


def _is_rule(name):
    return name.startswith("rule:")


def fig_leaderboard(tag, out):
    lb = pd.read_csv(os.path.join(RESULTS, f"leaderboard_{tag}.csv"))
    order = (lb[lb["target wait %"] == 20].sort_values("$ caught").policy.tolist())
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.32 * len(order) + 1.4), sharey=True)
    for ax, q in zip(axes, (10, 20, 30)):
        d = lb[lb["target wait %"] == q].set_index("policy").loc[order]
        for i, (name, r) in enumerate(d.iterrows()):
            _ci(ax, i, r["$ caught"], r["$ caught lo"], r["$ caught hi"], MUTED if _is_rule(name) else BLUE)
        ax.set_title(f"WAIT on {q}% of decisions")
        ax.set_xlabel("$ caught per decision (95% interval)")
    folds = lb.groupby("policy").folds.first()
    labels = [f"{p}  [{folds[p]} of {folds.max()} windows]" if folds[p] < folds.max() else p for p in order]
    axes[0].set_yticks(range(len(order)), labels)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Every model at the same wait rates, test windows pooled  (gray = simple rules)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_tuning(tag, out):
    """Tuned minus default, paired on the same product resamples."""
    path = os.path.join(RESULTS, f"paired_{tag}.csv")
    if not os.path.exists(path):
        return
    p = pd.read_csv(path)
    p = p[p.a.str.contains("(tuned", regex=False) & (p.b == p.a.str.replace("(tuned", "(default", regex=False))]
    if p.empty:
        return
    names = list(dict.fromkeys(p.a))
    metrics = [("PR-AUC", "PR-AUC"), ("$ caught@10", "$ caught, 10% wait"),
               ("$ caught@20", "$ caught, 20% wait"), ("$ caught@30", "$ caught, 30% wait")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 0.5 * len(names) + 1.3), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        d = p[p.metric == m].set_index("a").loc[names]
        for i, (name, r) in enumerate(d.iterrows()):
            real = r.lo > 0 or r.hi < 0
            _ci(ax, i, r["diff"], r.lo, r.hi, BLUE if real else MUTED)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)), [n.replace(" (tuned", " (").replace("(, ", "(").replace(" ()", "")
                                           for n in names])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("tuned minus default (95%)")
    fig.suptitle("What tuning buys: same products, same windows  (blue = interval excludes 0)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_groups(tag, out):
    path = os.path.join(RESULTS, f"s003_groups_{tag}.csv")
    if not os.path.exists(path):
        return
    g = pd.read_csv(path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
    for row, key in enumerate(("XGBoost", "CatBoost")):
        d = g[g.model == key]
        for col, (prefix, title, color) in enumerate([
                ("without ", "PR-AUC lost when the group is removed (vs all 26)", BLUE),
                ("price position + ", "PR-AUC added on top of price position alone", ORANGE)]):
            ax = axes[row, col]
            v = d[d.variant.str.startswith(prefix)].copy()
            v["group"] = v.variant.str.replace(prefix, "", regex=False)
            if prefix == "without ":  # show "lost" as a positive bar
                v["est"], v["lo"], v["hi"] = -v["d PR-AUC"], -v["d PR-AUC hi"], -v["d PR-AUC lo"]
            else:
                v["est"], v["lo"], v["hi"] = v["d PR-AUC"], v["d PR-AUC lo"], v["d PR-AUC hi"]
            v = v.sort_values("est")
            y = range(len(v))
            ax.barh(y, v.est, color=color, height=0.6, zorder=2)
            ax.hlines(y, v.lo, v.hi, color=INK2, linewidth=1.2, zorder=3)
            ax.axvline(0, color=INK2, linewidth=0.8)
            ax.set_yticks(list(y), v.group)
            ax.set_title(f"{key}: {title}", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("change in test PR-AUC (95% interval)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_segments(tag, out, policy="XGBoost (tuned, 26)"):
    path = os.path.join(RESULTS, f"s017_breakdown_{tag}.csv")
    if not os.path.exists(path):
        return
    b = pd.read_csv(path)
    b = b[(b.policy == policy) & (b.by != "brand")]
    groups = list(dict.fromkeys(b.by))
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.5, 0.3 * len(b) + 0.8 * len(groups) + 0.6),
                             sharex=True, gridspec_kw={"height_ratios": [b.by.eq(g).sum() for g in groups]})
    axes = [axes] if len(groups) == 1 else axes
    order = ["<$25", "$25-75", "$75-200", ">$200", "calm", "medium", "twitchy", "seen", "unseen",
             "<1y", "1-2y", "2y+", "Nov-Dec", "rest of year"]
    for ax, by in zip(axes, groups):
        d = b[b.by == by].copy()
        d["k"] = d.segment.map({s: i for i, s in enumerate(order)})
        d = d.sort_values("k").reset_index(drop=True)
        for i, r in d.iterrows():
            _ci(ax, i, r["PR-AUC"], r["PR-AUC lo"], r["PR-AUC hi"], BLUE, "PR-AUC" if i == 0 and by == groups[0] else None)
            ax.scatter(r["base rate %"] / 100, i, marker="|", s=120, color=INK2, linewidth=2, zorder=3,
                       label="base rate (PR-AUC of guessing)" if i == 0 and by == groups[0] else None)
        ax.set_yticks(range(len(d)), d.segment)
        ax.set_title(by, fontsize=9)
        ax.invert_yaxis()
    axes[-1].set_xlabel("test PR-AUC within the segment (95% interval)")
    fig.suptitle(f"Where {policy} predicts well and badly", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2,
               labelcolor=INK2, bbox_to_anchor=(0.99, 0.975))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(data="prices"):
    tag = tag_for(SOURCES.get(data, data))
    fig_dir = os.path.join(RESULTS, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for n, fn in enumerate((fig_leaderboard, fig_tuning, fig_groups, fig_segments), 1):
        out = os.path.join(fig_dir, f"{tag}_fig{n}_{fn.__name__[4:]}.png")
        fn(tag, out)
        if os.path.exists(out):
            print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
