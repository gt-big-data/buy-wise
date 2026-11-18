"""Study 023 (closes Study 002b): which horizon and drop threshold? (R4, #81)

Horizons of 7, 14 and 30 days by drop thresholds of 5, 8, 10 and 15%: twelve
versions of the question "will it drop?". The model is retrained on each (the
leading model in r4_common.PREFERRED that the runs scored, plus the price
-position rule as a reference), over four walk-forward windows.

The measure decided on #81 is **day-N dollars**: the shopper waits and buys on
the last day of the window, with no alert catching the low. Dollars don't
depend on the threshold, only on prices, so all twelve are in the same units.

The rule: 14d/8% stays unless another version beats it at a 20% wait rate with
a paired 95% interval above 0. Among versions that do, take the best, unless a
shorter wait is statistically tied with it.

Paired: every version is scored on the same decisions, the rows all twelve
share (a 30-day answer needs 30 days of later prices, so 30-day versions lose
rows at the end of each window), and products are resampled with the same
draws. Each version's WAITs are its own top 20% in each window.

Needs (from backend/), for H in 7 14 30 and T in 0.05 0.08 0.10 0.15:
    python -m ml.bench.r4 --models top90,rel90,lightgbm-tuned --horizon H --threshold T

    cd studies && python s023_horizon_threshold.py [prices|apr2026_500]
"""
import os
import sys

from r4_common import (HORIZONS, INK, INK2, KEYS, RESULTS, SURFACE, THRESHOLDS, fmt_ci, models_in, np, pd, plt,
                       primary, save_fig, scores, show, tag)

from ml.bench import calibration as cal  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

RULE = "rule: top of 90-day range"
BASE = (14, 0.08)
RATE = 0.20
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # palette ramp, 100-700


def label(h, t):
    return f"{h}d {t:.0%}"


def wait_set(df, s, rate=RATE):
    """Each window's own top `rate` by score (metrics.score_policies' cut)."""
    wait = np.zeros(len(df), bool)
    for _, idx in df.groupby("fold").indices.items():
        wait[idx] = s[idx] >= np.quantile(s[idx], 1 - rate)
    return wait


def pr_auc(df, s):
    """Per-window PR-AUC, pooled weighted by decisions."""
    parts = [(len(idx), average_precision_score(df.y.to_numpy()[idx], s[idx]))
             for _, idx in df.groupby("fold").indices.items()]
    return sum(n * a for n, a in parts) / sum(n for n, _ in parts)


def main(data="prices", n_boot=1000):
    cells = {(h, t): scores(data, h, t, with_val=False)[1] for h in HORIZONS for t in THRESHOLDS}
    common = None
    for test in cells.values():
        common = test[KEYS] if common is None else common.merge(test[KEYS], on=KEYS)
    common = common.sort_values(KEYS).reset_index(drop=True)
    print(f"{len(common):,} decisions shared by all 12 versions "
          f"(each has {min(map(len, cells.values())):,} to {max(map(len, cells.values())):,})", file=sys.stderr)
    cells = {k: common.merge(v, on=KEYS, how="left", validate="one_to_one") for k, v in cells.items()}
    lead = primary(cells[BASE])
    models = [m for m in models_in(cells[BASE]) if all(m in c for c in cells.values())] + [RULE]
    draws = cal.product_draws(common.asin, n_boot)  # one set of draws for every version and model: paired

    rows, d_dayn = [], {}
    for (h, t), df in cells.items():
        g_day = (df.price - df.fwd_last).to_numpy(float)
        g_low = (df.price - df.fwd_min).to_numpy(float)
        y = df.y.to_numpy(float)
        for m in models:
            s = df[m].to_numpy(float)
            w = wait_set(df, s).astype(float)
            shown = cal.boot_mean(w, draws)
            day = cal.boot_mean(w * g_day, draws)
            prec = cal.boot_mean(w * y, draws) / shown * 100
            low = cal.boot_mean(w * g_low, draws)
            d_dayn[(m, h, t)] = day
            higher = (w > 0) & (df.fwd_last > df.price).to_numpy()
            rows.append({
                "model": m, "horizon": h, "threshold": t, "version": label(h, t),
                "base rate %": y.mean() * 100, "PR-AUC": pr_auc(df, s),
                **dict(zip(["precision %", "precision lo", "precision hi"], cal.interval(prec))),
                **dict(zip(["$ at low", "$ at low lo", "$ at low hi"], cal.interval(low))),
                **dict(zip(["$ day N", "$ day N lo", "$ day N hi"], cal.interval(day))),
                "ceiling day N $": np.clip(g_day, 0, None).mean(),
                "$ day N per day waited": day[0] / h,
                "WAITs: paid more on day N %": higher.sum() / w.sum() * 100,
                "WAITs: extra paid when more $": (df.fwd_last - df.price)[higher].mean(),
            })
    tab = pd.DataFrame(rows)
    paired = []
    for m in models:
        for (h, t) in cells:
            if (h, t) != BASE:
                diff = d_dayn[(m, h, t)] - d_dayn[(m, *BASE)]
                est, lo, hi = cal.interval(diff)
                paired.append({"model": m, "version": label(h, t), "horizon": h, "threshold": t,
                               "day N $ vs 14d 8%": est, "lo": lo, "hi": hi})
    paired = pd.DataFrame(paired)

    tg = tag(data)
    tab.to_csv(os.path.join(RESULTS, f"s023_versions_{tg}.csv"), index=False)
    paired.to_csv(os.path.join(RESULTS, f"s023_paired_{tg}.csv"), index=False)
    for m in models:
        t_m = tab[tab.model == m].assign(
            **{"$ day N [95%]": lambda d: [fmt_ci(r["$ day N"], r["$ day N lo"], r["$ day N hi"]) for _, r in d.iterrows()]})
        p_m = paired[paired.model == m].set_index("version")
        t_m["vs 14d 8% [95%]"] = [fmt_ci(p_m.loc[v, "day N $ vs 14d 8%"], p_m.loc[v, "lo"], p_m.loc[v, "hi"])
                                  if v in p_m.index else "today" for v in t_m.version]
        show(t_m[["version", "base rate %", "PR-AUC", "precision %", "$ at low", "$ day N [95%]", "vs 14d 8% [95%]",
                  "$ day N per day waited", "WAITs: paid more on day N %", "WAITs: extra paid when more $"]].round(3),
             f"{m}: WAIT on {RATE:.0%} of decisions, dollars per decision")

    choice, why = decide(paired[paired.model == lead], d_dayn, lead)
    print(f"\nRULE ({lead}): serve {choice}. {why}")
    pd.DataFrame([{"model": lead, "serve": choice, "why": why}]).to_csv(
        os.path.join(RESULTS, f"s023_pick_{tg}.csv"), index=False)
    plot(tab[tab.model == lead], paired[paired.model == lead], lead, tg)


def decide(paired, d_dayn, lead):
    """14d/8% unless clearly beaten on day-N dollars; then the best, unless a
    shorter horizon ties with it."""
    beats = paired[paired.lo > 0]
    if beats.empty:
        return label(*BASE), (f"no version beats {label(*BASE)} on day-N dollars at a {RATE:.0%} wait with an "
                              "interval above 0, so it stays")
    best = beats.loc[beats["day N $ vs 14d 8%"].idxmax()]
    for _, c in beats[beats.horizon < best.horizon].sort_values("horizon").iterrows():
        est, lo, hi = cal.interval(d_dayn[(lead, best.horizon, best.threshold)] - d_dayn[(lead, c.horizon, c.threshold)])
        if lo <= 0:
            return c.version, (f"beats {label(*BASE)} and ties {best.version} ({est:+.2f} [{lo:+.2f}, {hi:+.2f}]), "
                               "and it is the shorter wait")
    return best.version, (f"beats {label(*BASE)} by ${best['day N $ vs 14d 8%']:.2f} [{best.lo:.2f}, {best.hi:.2f}] "
                          "per decision, the most of any version")


def plot(tab, paired, lead, tg):
    """Heatmap: day-N dollars per decision at a 20% wait, with each version's
    paired difference from today's 14d/8% underneath."""
    grid = tab.pivot(index="horizon", columns="threshold", values="$ day N").loc[list(HORIZONS), list(THRESHOLDS)]
    d = paired.set_index("version")
    steps = np.linspace(np.nanmin(grid.values), np.nanmax(grid.values), len(BLUES))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.grid(False)
    for i, h in enumerate(HORIZONS):
        for j, th in enumerate(THRESHOLDS):
            v = grid.loc[h, th]
            k = int(np.argmin(np.abs(steps - v)))
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=BLUES[k], linewidth=0))
            ink = SURFACE if k >= 3 else INK
            name = label(h, th)
            sub = "today" if (h, th) == BASE else (
                f"{d.loc[name, 'day N $ vs 14d 8%']:+.2f} [{d.loc[name, 'lo']:+.2f}, {d.loc[name, 'hi']:+.2f}]")
            # y grows downwards here (row 0 = 7 days at the top).
            ax.text(j + 0.48, i + 0.38, f"${v:.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color=ink)
            ax.text(j + 0.48, i + 0.68, sub, ha="center", va="center", fontsize=7, color=ink)
            if (h, th) == BASE:
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xlim(0, len(THRESHOLDS))
    ax.set_ylim(len(HORIZONS), 0)
    ax.set_xticks(np.arange(len(THRESHOLDS)) + 0.48, [f"{x:.0%} drop" for x in THRESHOLDS])
    ax.set_yticks(np.arange(len(HORIZONS)) + 0.46, [f"{h} days" for h in HORIZONS])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Day-N dollars per decision, {lead} WAIT on {RATE:.0%}\n"
                 "under each: paired difference vs 14 days / 8% (95% interval)", color=INK)
    ax.set_xlabel("what counts as a drop (the model's training label)", color=INK2)
    save_fig(fig, f"s023_horizon_threshold_{tg}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
