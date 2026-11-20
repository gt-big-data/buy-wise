"""Study 022: is the "chance of a drop" honest? (R4, #81)

The best models in the saved run (r4_common.PREFERRED: tuned CatBoost and XGBoost
on 26 features and the LSTM when #79/#80 scored them; LightGBM, the clean XGBoost
recipe and the small neural net, which main registers), each with four ways to turn its score into the shown chance: none (raw score),
isotonic (served today), Platt, and beta. Every calibrator is fitted on a
fold's validation rows and judged on that fold's test rows. Results are pooled
over the four walk-forward windows, with product-level 95% intervals.

The rule decided on #81: serve the calibrator with the lowest test ECE on
products the model never saw in training, unless its paired gain over Platt
(the simplest) has an interval that includes 0. Then serve Platt.

Needs (from backend/):
    python -m ml.bench.r4 --models top90,rel90,lightgbm-tuned,xgb,small-nn-tuned --tag r4

    cd studies && python s022_calibration.py [prices|apr2026_500]
"""
import os
import sys

from r4_common import (AQUA, BLUE, INK2, MUTED, ORANGE, RESULTS, fmt_ci, models_in, np, pd, plt, primary, save_fig,
                       scores, show, tag, with_context)

from ml.bench import calibration as cal  # noqa: E402

KINDS = ("none", *cal.CALIBRATORS)
COLORS = {"none": MUTED, "isotonic": BLUE, "platt": ORANGE, "beta": AQUA}
MARKERS = {"none": "x", "isotonic": "o", "platt": "s", "beta": "^"}
SEGMENTS = ["product seen in training", "price band", "volatility", "history", "season"]
SHOWN = 0.40  # today's MIN_DROP_CHANCE: "when we say 40%+, does it happen?"


def main(data="prices", n_boot=1000):
    val, test = scores(data)
    val, test, _ = with_context(data, val, test)
    models = [m for m in models_in(test) if m in val.columns][:3]
    lead = primary(test)
    print(f"models: {models}", file=sys.stderr)
    y = test.y.to_numpy(float)
    draws = cal.product_draws(test.asin, n_boot)  # same draws for every model and calibrator: paired
    unseen = (test["product seen in training"] == "unseen").to_numpy()

    summary, segs, rel, picks, at_shown = [], [], [], [], []
    calibrated = {}
    for model in models:
        ece_unseen = {}
        for kind in KINDS:
            p = cal.calibrate_folds(val, test, model, kind)
            calibrated[(model, kind)] = p
            for scope, mask in (("all products", None), ("unseen products", unseen)):
                e, b = cal.boot_ece(y, p, draws, mask), cal.boot_brier(y, p, draws, mask)
                summary.append({"model": model, "calibrator": kind, "scope": scope,
                                "rows": int(len(y) if mask is None else mask.sum()),
                                **dict(zip(["ECE", "ECE lo", "ECE hi"], cal.interval(e))),
                                **dict(zip(["Brier", "Brier lo", "Brier hi"], cal.interval(b)))})
                if scope == "unseen products" and kind != "none":
                    ece_unseen[kind] = e
            for by in SEGMENTS:
                for value in sorted(test[by].unique()):
                    m = (test[by] == value).to_numpy()
                    if m.sum() < 200 or test.y[m].nunique() < 2:
                        continue
                    e = cal.boot_ece(y, p, draws, m)
                    segs.append({"model": model, "calibrator": kind, "by": by, "segment": value, "rows": int(m.sum()),
                                 "base rate %": y[m].mean() * 100, "mean said %": p[m].mean() * 100,
                                 **dict(zip(["ECE", "ECE lo", "ECE hi"], cal.interval(e)))})
            for scope, mask in (("all products", np.ones(len(y), bool)), ("unseen products", unseen)):
                r = cal.reliability(y[mask], p[mask]).assign(model=model, calibrator=kind, scope=scope)
                rel.append(r)
            # The panel's question: of the times we'd show 40%+, how often did it drop?
            s = p >= SHOWN
            at_shown.append({"model": model, "calibrator": kind, "shown 40%+ rows": int(s.sum()),
                             "share of decisions %": s.mean() * 100,
                             "said (mean) %": p[s].mean() * 100 if s.any() else np.nan,
                             "happened %": y[s].mean() * 100 if s.any() else np.nan})
        choice, reason = cal.pick(ece_unseen)
        picks.append({"model": model, "serve": choice, "why": reason})

    summary, segs = pd.DataFrame(summary), pd.DataFrame(segs)
    rel, at_shown, picks = pd.concat(rel, ignore_index=True), pd.DataFrame(at_shown), pd.DataFrame(picks)

    t = tag(data)
    for name, df in (("summary", summary), ("segments", segs), ("reliability", rel), ("at40", at_shown),
                     ("pick", picks)):
        df.to_csv(os.path.join(RESULTS, f"s022_{name}_{t}.csv"), index=False)

    out = summary.assign(ECE=[fmt_ci(r.ECE, r["ECE lo"], r["ECE hi"], 4) for _, r in summary.iterrows()],
                         Brier=[fmt_ci(r.Brier, r["Brier lo"], r["Brier hi"], 4) for _, r in summary.iterrows()])
    show(out[["model", "scope", "calibrator", "rows", "ECE", "Brier"]], f"calibration on test, {t} (95% intervals)")
    show(at_shown.round(1), "when the panel would show 40%+ (today's threshold)")
    show(segs[segs.model == lead].pivot_table(index=["by", "segment"], columns="calibrator", values="ECE")
         .round(4).reset_index(), f"ECE by kind of product, {lead}")
    show(picks, "the rule's pick (lowest unseen-product ECE; Platt unless clearly beaten)")

    plot_reliability(rel, models, t)


def plot_reliability(rel, models, t):
    # The curves converge at the top, so end labels would collide: identity comes from
    # the legend plus a distinct marker per calibrator (and the CSV behind the figure).
    top = min(1.0, np.ceil((rel[["said", "happened"]].max().max() + 0.05) * 10) / 10)
    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 7.2), squeeze=False)
    for col, model in enumerate(models):
        for row, scope in enumerate(("all products", "unseen products")):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (2, 2)), zorder=1)
            for kind in KINDS:
                r = rel[(rel.model == model) & (rel.calibrator == kind) & (rel.scope == scope)]
                ax.plot(r.said, r.happened, color=COLORS[kind], linewidth=2, marker=MARKERS[kind], markersize=5,
                        markeredgecolor="#fcfcfb", label=kind, zorder=2)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal")
            ax.set_title(f"{model.replace(', 26 features', '')}\n{scope}", fontsize=9)
            ax.set_xlabel("said (mean shown chance per tenth)")
            if col == 0:
                ax.set_ylabel("happened (share that dropped)")
    axes[0, 0].legend(loc="upper left", fontsize=8, title="calibrator", title_fontsize=8)
    fig.suptitle("Said vs happened on test, calibrators fitted on validation (dashed = perfect)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"s022_reliability_{t}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
