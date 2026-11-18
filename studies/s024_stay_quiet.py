"""Study 024: when should the panel stay quiet? (R4, #81)

The rule decided on #81: show WAIT only when the calibrated chance of a drop is
at least `min_drop_chance` AND the expected net saving is at least `min_saving`
dollars. Net saving counts what a wrong WAIT costs (ml/bench/quiet.py):

    net = price * (p * G_right + (1 - p) * G_wrong)

G_right / G_wrong are the average day-N saving when the drop did / didn't come,
per volatility band, measured on each window's training rows.

Everything is chosen on validation and judged once on test:
  - the calibrator is Study 022's pick, fitted on validation (cross-fitted by
    product when scoring validation itself, so isotonic can't flatter itself)
  - (min_drop_chance, min_saving) is the validation setting with the most
    day-N dollars per decision among those right at least 50% of the time

The curves compare four ways to stay quiet at every level of quietness:
chance alone, net dollars alone, "how jumpy is this product" alone, and the
combined rule (its best setting per level, chosen on validation).

Needs the sweep run for the chosen horizon/threshold (`python -m ml.bench.r4
--horizon H --threshold T`, which keeps validation scores), and s022's pick.

    cd studies && python s024_stay_quiet.py [prices|apr2026_500] [horizon] [threshold]
"""
import json
import os
import sys

from r4_common import (AQUA, BLUE, INK, INK2, MUTED, ORANGE, RESULTS, fmt_ci, np, pd, plt, primary, save_fig, scores,
                       show, tag, with_context)

from ml.bench import calibration as cal  # noqa: E402
from ml.bench import quiet  # noqa: E402

YELLOW = "#eda100"  # palette slot 4, validated with slots 1-3
TODAY = 0.40         # MIN_DROP_CHANCE served today
MIN_PRECISION = 50.0
RULE_COLORS = {"chance + net $": BLUE, "chance alone": ORANGE, "net $ alone": AQUA, "jumpiness alone": YELLOW}
RULE_MARKERS = {"chance + net $": "o", "chance alone": "s", "net $ alone": "^", "jumpiness alone": "D"}


def chosen(data, what, default, model=None):
    """s022 / s023's pick from its saved CSV, else `default`."""
    t = tag(data)
    path = os.path.join(RESULTS, f"s02{'2' if what == 'calibrator' else '3'}_pick_{t}.csv")
    if not os.path.exists(path):
        print(f"({os.path.basename(path)} not found; using {default})", file=sys.stderr)
        return default
    p = pd.read_csv(path)
    if what == "calibrator":
        return p.set_index("model").serve.get(model, default)
    h, th = p.serve.iloc[0].split("d ")
    return int(h), float(th.rstrip("%")) / 100


def per_fold(val, test, trains, kind, model):
    """Calibrated chance, net saving and jumpiness for validation and test rows,
    each fold using only its own validation (calibrator) and train (economics)."""
    val = val.assign(p=cal.crossfit_val(val, model, kind))
    test = test.assign(p=cal.calibrate_folds(val, test, model, kind))
    econ_rows = []
    for name, part in (("val", val), ("test", test)):
        net, jump = np.full(len(part), np.nan), np.full(len(part), np.nan)
        for fold, idx in part.groupby("fold").indices.items():
            tr = trains[fold]
            edges = quiet.band_edges(tr)
            econ = quiet.wait_economics(tr, edges)
            rows = part.iloc[idx]
            net[idx] = quiet.net_saving(rows.p, rows, econ, edges)
            jump[idx] = quiet.jumpiness(rows, tr)
            if name == "test":
                econ_rows.append(econ.reset_index().assign(fold=fold, **{"vol30 cut 1": edges[0],
                                                                         "vol30 cut 2": edges[1]}))
        part["net"], part["jump"] = net, jump
    return val, test, pd.concat(econ_rows, ignore_index=True)


def curves(rows):
    return pd.concat([quiet.single_knob_curve(rows, rows.p, "chance alone"),
                      quiet.single_knob_curve(rows, rows.net, "net $ alone"),
                      quiet.single_knob_curve(rows, rows.jump, "jumpiness alone")], ignore_index=True)


def judged_on_test(test, settings):
    """Score validation-chosen (min_drop_chance, min_saving) settings on test."""
    out = []
    for _, s in settings.iterrows():
        speak = (test.p >= s.min_drop_chance) & (test.net >= s.min_saving)
        out.append({"rule": "chance + net $", "min_drop_chance": s.min_drop_chance, "min_saving": s.min_saving,
                    **quiet.outcome(test, speak.to_numpy())})
    return pd.DataFrame(out)


def with_intervals(test, rules: dict, draws):
    """Precision, day-N $ per decision and per shown WAIT, with product-level
    95% intervals; paired difference of each rule against today's 40% rule."""
    g = (test.price - test.fwd_last).to_numpy(float)
    y = test.y.to_numpy(float)
    stats = {}
    for name, speak in rules.items():
        s = speak.astype(float)
        shown = cal.boot_mean(s, draws)
        stats[name] = {"quiet %": (1 - shown) * 100,
                       "precision %": cal.boot_mean(s * y, draws) / shown * 100,
                       "$ day N per decision": cal.boot_mean(s * g, draws),
                       "$ day N per WAIT": cal.boot_mean(s * g, draws) / shown}
    rows = []
    for name, st in stats.items():
        row = {"rule": name}
        for m, d in st.items():
            row[m] = fmt_ci(*cal.interval(d), 3 if "$" in m else 1)
            if name != "today: chance >= 40%":
                row[f"{m} vs today"] = fmt_ci(*cal.interval(d - stats["today: chance >= 40%"][m]),
                                              3 if "$" in m else 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main(data="prices", horizon=None, threshold=None, n_boot=1000):
    h, th = (int(horizon), float(threshold)) if horizon else chosen(data, "version", (14, 0.08))
    val, test = scores(data, h, th)
    model = primary(test)
    kind = chosen(data, "calibrator", cal.SIMPLEST, model)
    print(f"{model}, calibrator {kind}, horizon {h} days, drop {th:.0%}", file=sys.stderr)
    val, test, trains = with_context(data, val, test, h, th)
    val, test, econ = per_fold(val, test, trains, kind, model)

    # 1. The four rules on validation, and the combined rule's choice.
    g_val = quiet.grid(val, val.p, val.net)
    pick = quiet.pick_operating_point(g_val, MIN_PRECISION)
    front_val = quiet.frontier(g_val[np.isfinite(g_val.min_saving)])

    # 2. Judged once on test.
    c_test = curves(test)
    front_test = judged_on_test(test, front_val)
    speak = ((test.p >= pick.min_drop_chance) & (test.net >= pick.min_saving)).to_numpy()
    rules = {"R4: chance + net $": speak,
             "today: chance >= 40%": (test.p >= TODAY).to_numpy(),
             "T5 placeholder: chance >= 40% and price >= $25": ((test.p >= TODAY) & (test.price >= 25)).to_numpy()}
    draws = cal.product_draws(test.asin, n_boot)
    ci_tab = with_intervals(test, rules, draws)

    by = []
    for seg in ("price band", "volatility", "product seen in training"):
        for value, part in test.groupby(seg):
            m = (test[seg] == value).to_numpy()
            for name, sp in rules.items():
                by.append({"by": seg, "segment": value, "rule": name, "decisions": int(m.sum()),
                           **quiet.outcome(part, sp[m])})
    by = pd.DataFrame(by)

    # 3. What a wrong WAIT costs, by band: train estimate vs what happened on test.
    shown = test[speak]
    wrong = shown[shown.y == 0]
    happened = wrong.groupby("volatility").apply(lambda w: pd.Series({
        "wrong WAITs": len(w), "paid more on day N %": (w.fwd_last > w.price).mean() * 100,
        "mean extra paid $": (w.fwd_last - w.price).clip(lower=0).mean(),
        "mean day-N change %": ((w.fwd_last - w.price) / w.price).mean() * 100}), include_groups=False)
    expected = econ.groupby("band")[["G_right", "G_right_low", "G_wrong", "higher_share", "higher_by"]].mean()

    t = tag(data)
    sfx = f"{t}_h{h}_t{th}"
    for name, df in (("curves_test", pd.concat([c_test, front_test], ignore_index=True)), ("grid_val", g_val),
                     ("frontier_val", front_val), ("rules_test", ci_tab), ("segments_test", by),
                     ("economics", econ)):
        df.to_csv(os.path.join(RESULTS, f"s024_{name}_{sfx}.csv"), index=False)

    show(pd.DataFrame([pick]).round(3), f"chosen on validation (precision >= {MIN_PRECISION:.0f}%, most day-N $)")
    show(ci_tab, "judged on test, 95% intervals (paired vs today's rule)")
    show(by[by.by == "price band"].round(2), "by price band, on test")
    show(expected.round(4).reset_index(), "what a WAIT is worth, train rows (fraction of price; mean over windows)")
    show(happened.round(2).reset_index(), "wrong WAITs the R4 rule showed on test, by volatility")

    # 4. The hand-off to T5 (#87): the `quiet` block of analysis/results/v<N>.json.
    last = econ[econ.fold == econ.fold.max()].set_index("band")  # newest window: the most training data
    rule = {
        "source": f"R4 #81, Study 024 ({t}): chosen on validation, judged on test",
        "model": model, "horizon_days": h, "drop_threshold": th, "calibrator": kind,
        "min_drop_chance": float(pick.min_drop_chance), "min_saving": float(pick.min_saving),
        "min_price_for_wait": 0.0,
        "net_saving": "price * (p * G_right[band] + (1 - p) * G_wrong[band]); band from vol30",
        "vol30_cuts": [float(last["vol30 cut 1"].iloc[0]), float(last["vol30 cut 2"].iloc[0])],
        "bands": {b: {"G_right": round(float(last.loc[b, "G_right"]), 4),
                      "G_wrong": round(float(last.loc[b, "G_wrong"]), 4)} for b in quiet.BANDS},
        "test": {k: round(float(v), 3) for k, v in quiet.outcome(test, speak).items()},
    }
    path = os.path.join(RESULTS, f"r4_quiet_rule_{t}.json")
    with open(path, "w") as f:
        json.dump(rule, f, indent=2)
    print(f"wrote {path}")

    plot(c_test, front_test, test, speak, sfx)


def plot(c_test, front_test, test, speak, sfx):
    """Three panels sharing the x-axis (how often we stay quiet), one measure
    each: no dual axes."""
    today = quiet.outcome(test, (test.p >= TODAY).to_numpy())
    ours = quiet.outcome(test, speak)
    panels = [("precision %", "right when we say WAIT (%)"),  # "\$": a literal dollar, not matplotlib mathtext
              ("$ day N per WAIT", r"\$ per shown WAIT (buys on day N)"),
              ("$ day N per decision", r"\$ per decision (quiet ones count \$0)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (m, title) in zip(axes, panels):
        for rule, color in RULE_COLORS.items():
            rule_label = rule.replace("$", r"\$")
            d = front_test if rule == "chance + net $" else c_test[c_test.rule == rule]
            d = d.sort_values("quiet %")
            ax.plot(d["quiet %"], d[m], color=color, linewidth=2, marker=RULE_MARKERS[rule], markersize=4,
                    markevery=max(1, len(d) // 10), label=rule_label)
        ax.scatter([today["quiet %"]], [today[m]], s=70, color=MUTED, edgecolor=INK, zorder=4,
                   label="today (chance >= 40%)")
        ax.scatter([ours["quiet %"]], [ours[m]], s=90, marker="*", color=BLUE, edgecolor=INK, zorder=5,
                   label="R4 pick (chosen on validation)")
        ax.set_title(title)
        ax.set_xlabel("stay quiet on this % of decisions")
        ax.set_xlim(0, 100)
        ax.axhline(0, color=INK2, linewidth=0.8)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("What staying quiet buys, on test (settings chosen on validation)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    save_fig(fig, f"s024_stay_quiet_{sfx}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
