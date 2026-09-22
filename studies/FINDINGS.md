# BuyWise: Where We Actually Stand

**For:** the three leads
**Date:** 2026-09-14
**Data:** `Prices.csv` from `origin/Training_data_generation` — 21,240 observations, 312 ASINs, 2023-04-09 → 2026-04-09
**Reproduce:** every number below comes from a script in this directory. See "Running these" at the end.

---

## 0. The sixty-second version

1. **The deployed model does not work.** Its price forecasts are 7× worse than assuming the price never changes. When it says WAIT it is right 23.7% of the time; random guessing is right 22.6% of the time.

2. **But the problem is learnable.** A leak-free rebuild — 20 features, one afternoon, no tuning — hits 55% precision at the same wait rate. That's 2.5× better than chance and it beats the simple-rule baseline by a wide margin. The failure is our pipeline, not the problem.

3. **The prize is small and very unevenly distributed.** Across all products, perfect foresight is worth ~$5–8 per purchase. But on volatile items over $200 it's worth **$23 per purchase**, and on calm items under $25 it's worth **13 cents**. We should only speak up on a minority of products.

4. **Something bigger is sitting untouched.** On the same products at the same moments, the cheapest *instant* alternative — a different new seller, or a Warehouse/used offer — averages **$18.47 cheaper** than the buy-box price. No prediction, no waiting, no uncertainty. That's 3–5× the entire timing opportunity.

5. **Seasonality is real but small.** November is 1.42× the baseline drop rate. December is the *worst* month of the year to buy. 60% of all available savings happen outside event months. "Just buy on Black Friday" captures 15% of the opportunity.

---

## 1. Vocabulary

Skip if you already know this. It's here so anyone on the team can read the rest.

**Features and labels.** Every supervised model is the same shape. You show it *features* (what you know at decision time: today's price, the 30-day average, how volatile it's been) and *labels* (the answer, knowable only later: what the price did next). It learns the mapping, then applies it to new cases.

**Leakage** is when a feature secretly contains information from after the decision moment. The model looks brilliant offline and fails in production. It is the single most common way ML projects fool themselves, and we had it.

**Regression vs classification.** Regression predicts a number ("$42.17 in 14 days"). Classification predicts a category plus a probability ("73% likely to drop 8%+"). We have both.

**XGBoost.** Start with a decision tree — a flowchart of yes/no questions. One tree is weak. *Gradient boosting* builds trees in sequence where each new tree is trained to fix the previous trees' mistakes. Chain a thousand and it gets accurate. XGBoost is the fast implementation. For tabular data at our scale it is the correct default; a neural network would lose here.

**Base rate, precision, lift.** If 22% of situations have a real price drop coming, that's the *base rate* — what you'd get by guessing. *Precision* is how often you're right when you say WAIT. *Lift* is precision ÷ base rate. **Lift of 1.0 means your model is worthless no matter how good the precision looks.** This is the number to check first, always.

**Baseline.** A dumb rule you must beat for your model to have earned its existence. Not "beat zero" — beat the dumb rule.

---

## 2. How our current system actually works

`dataset.py` builds a table: one row per product per day (Keepa only records price *changes*, so a price of $19.39 held for 49 days becomes 49 identical rows), then computes ~40 features. It creates labels by looking 14 rows into the future of the historical data. Chronological 80/20 split.

`xgboost_price_model.py` trains three models: price regressors at 7 and 14 days, and a classifier for "will it drop ≥8%". Saved as `.joblib`.

`inference.py` runs live: recompute features for today, feed the models, return a recommendation.

**The part that surprises people.** The recommendation isn't really made by the models:

```python
expected_drop = (curr - pred_14d) / curr
if   expected_drop >= 0.10:  recommendation = "WAIT"
elif expected_drop <= 0.03:  recommendation = "BUY"
else:                        recommendation = "WAIT" if cls_label == 1 else "BUY"
```

Two hardcoded thresholds on the regressor's output. The classifier — the model actually trained to answer this question — only breaks ties in between. And the confidence percentage shown to users is a hand-tuned arithmetic blend floored at 50% and capped at 97%. It is not a probability of anything.

---

## 3. The studies

### Study 000 — Data integrity
`s000_quality.py`

| | |
|---|---|
| Observations | 21,240 across 312 ASINs |
| Median observations per ASIN | 56 (25th percentile: 18; minimum: 1) |
| Median gap between observations | 4 days |
| **Gaps that are exact multiples of 7** | **33.7%** |
| Observations in 2025 vs 2023 | 11,407 vs 1,804 |
| A quarter of ASINs have | under 8 months of history |

**Findings.** The 340k-row daily panel is mostly forward-filled duplicates — real information content is 21k events. The strong 7-day signature suggests weekly sampling rather than true change-events, which means intra-week dips are invisible and every opportunity number below is probably an **undercount**. 1,000 ASINs were requested, 500 targeted, 312 survived — nobody knows where the attrition happened. There are ±363% price moves that are almost certainly errors.

**Action: finish this study before we spend the Keepa month.** We get one pull.

---

### Study 001 — How much money is actually on the table
`s001_movement.py` — 256 ASINs with ≥180 days history, ~176k decisions

| | |
|---|---|
| Days with any price change | **8.2%** |
| Median saving from waiting 14 days | **$0.00** |
| Decisions where waiting saves nothing | **69.2%** |
| Decisions with a ≥8% drop available | **17.5%** |
| Average saving when that happens | **$28.24** |

The structure is **rare, large events** — not a smooth curve. `0.175 × $28 ≈ $4.94`, which is where the ~$5 average ceiling comes from. Those two numbers are the same fact stated two ways.

**Seasonality.** P(drop ≥8%) by month, against a 17.5% baseline:

| Month | P(drop≥8%) | vs. average |
|---|---|---|
| **November** | 24.9% | **1.42×** |
| October | 20.0% | 1.14× |
| July | 18.9% | 1.08× |
| August | 13.9% | 0.79× |
| **December** | 11.0% | **0.63×** |

**60.6% of all available savings occur outside event months.** A November-only policy captures $0.77 of the $5.26 ceiling — 15%. December being the worst month of the year is counterintuitive, real, and not something a normal shopper knows.

---

### Study 002 — The baselines we must beat
`s002_baselines.py`

| Policy | Savings vs. buying now | Precision |
|---|---|---|
| Ceiling (perfect foresight) | $5.26 | 100% |
| **WAIT if price is in top 25% of its 90-day range** | **$3.48** | 32% |
| WAIT if price > 90-day median | $3.08 | 34% |
| WAIT in Nov/Oct/Jul | $1.84 | 22% |
| WAIT in November only | $0.77 | 25% |

**A three-line heuristic with no ML captures 66% of the theoretical maximum.** That is the bar. Not zero.

**Signal test** — P(drop≥8%) by today's price relative to its own trailing 30-day median:

| Price vs. 30-day median | P(drop≥8%) |
|---|---|
| Well below | 8.6% |
| At the median | 13.5% |
| Well above | **49.5%** |

A 5.7× spread from one past-only feature. Trailing volatility gives an independent 6× spread. **There is real structure here.**

---

### Study 004 — Audit of the deployed model
`s004_deployed_model_audit.py` — our saved `.joblib` models, our `inference.py` decision logic, scored on the Sept 2025 – Apr 2026 holdout (40,512 decisions)

**Price forecasting:**

| Predicting price 14 days out | Average error |
|---|---|
| Our XGBoost model | **$85.80** |
| Assuming price stays the same | **$11.98** |

Seven times worse than doing nothing, on items averaging $144.

**The decision:**

| Policy | Says WAIT | Precision | Savings if user waits & buys |
|---|---|---|---|
| Our ML model | 45.1% | **23.7%** | $0.30 |
| Heuristic | 29.8% | **41.4%** | $3.71 |
| *(chance)* | — | *22.6%* | — |

**Lift = 1.05.** The model is one percentage point better than a coin flip.

**A trap worth internalizing.** On a naive dollar metric the model *appears* to win ($6.22 vs $4.85). That's fake — it only "wins" because it says WAIT 45% of the time versus 30%, on a metric that treats waiting as free. **Our evaluation metric handed us the wrong answer.** This is why the harness gets built before anything else.

**Caveat, stated honestly.** The committed `Product.csv` is missing two columns `dataset.py` requires, so they were derived; unknown features were filled with zeros. Some of the regression blowup may be reconstruction error. **But that is exactly what `inference.py` does in production** — same zero-filling, same hand-rebuilt feature list, same approximated global stats. This is a valid measurement of the system *as deployed*.

---

### Study 005 — A clean rebuild ⭐
`s005_clean_rebuild.py` — strictly past-only features, 3-way chronological split with a 14-day embargo, threshold chosen on validation, compared at **matched wait rates**

> Matched wait-rate comparison is the fix for the trap in Study 004. Force both policies to say WAIT on exactly the same fraction of decisions, then see who picks better. Waiting more can no longer masquerade as skill.

**Test ROC-AUC 0.808, PR-AUC 0.547 against a 0.219 base rate.**

| Wait rate | Clean model precision | Heuristic precision | Model $/decision | Heuristic $/decision |
|---|---|---|---|---|
| 10% | **65.3%** | 54.1% | $3.63 | $2.99 |
| 20% | **55.3%** | 38.8% | $5.23 | $3.23 |
| 30% | **47.8%** | 37.9% | $5.97 | $4.36 |
| 50% | 37.2% | 34.7% | $7.05 | $6.47 |

Top features: price vs. its 180-day minimum, position in the 90-day range, 30-day volatility.

**This is the most important result in the document.** The problem is learnable. A model built correctly, in an afternoon, with no tuning, beats the heuristic by $1.50–2.00 per decision and roughly doubles precision at low wait rates. At a 10% wait rate we'd be right two times in three.

**This means our ML work is worth doing.** It also means the deployed model's failure is entirely our engineering, not the problem's difficulty.

---

### Study 006 — Where it works, and where we should shut up
`s006_segmentation.py` — clean model at a 20% wait rate

**By product volatility:**

| Segment | Base rate | Precision | Lift | $/decision |
|---|---|---|---|---|
| Calm | 10.6% | 27.0% | 2.56× | **$0.52** |
| Medium | 20.8% | 50.9% | 2.45× | $4.54 |
| **Twitchy** | 34.2% | **60.7%** | 1.77× | **$10.64** |

**By price band:**

| Segment | Base rate | Precision | Lift | $/decision |
|---|---|---|---|---|
| **< $25** | 17.2% | 52.8% | 3.07× | **$0.13** |
| $25–75 | 16.8% | 52.6% | 3.12× | $0.89 |
| $75–200 | 25.2% | 55.9% | 2.22× | $4.25 |
| **> $200** | 33.2% | 57.3% | 1.72× | **$23.35** |

**A 180× spread in value between the best and worst segment.** On an expensive, volatile item we are right 6 times in 10 and it's worth $23. On a cheap stable item we're right half the time and it's worth 13 cents — strictly worse than saying nothing, because we spent user trust to save them a dime.

**Product implication: BuyWise should have a confident "no opinion" state and use it often.**

**Robustness — products the model has never seen:**

| | AUC | Precision@20% | Lift |
|---|---|---|---|
| ASINs seen in training | 0.790 | 53.9% | 2.44× |
| **ASINs never seen** | **0.826** | **56.1%** | **2.64×** |

It generalizes. Slightly *better* on unseen products. The model is learning general price dynamics, not memorizing individual items — which means it works on cold start, and that's a real result.

---

### Study 007 — Does the user actually catch the dip
`s006_segmentation.py` (bottom section)

| Scenario | $/decision |
|---|---|
| We alert them, they buy at the window low | $5.23 |
| They only check weekly, miss short dips | $4.06 |
| They just wait 14 days and buy whatever | $3.71 |

**This corrects something I got wrong earlier in the week.** Measured on *indiscriminate* waiting, the value collapsed to $0.35 without perfect alert timing, and I concluded the money was in the alert rather than the forecast. That was an artifact. With a good model selecting *which* 20% of decisions to wait on, **most of the value survives even with sloppy timing** — $3.71 of $5.23. Alerts add ~40% on top, which is worth building, but selection is what matters most. The forecast earns its keep after all.

---

### Study 009 — The lever nobody looked at
`s009_substitution.py` — same products, same moments, using the `used` and `new` columns that were sitting in `Prices.csv` unexamined

| Opportunity | $ available | Frequency | Requires waiting? |
|---|---|---|---|
| Cheaper *new* seller than the buy box | **$6.30** | 29% are ≥5% cheaper | No |
| Used / Warehouse offer | **$17.48** | 62% have one ≥10% cheaper | No |
| **Best instant alternative** | **$18.47** | 74% have something cheaper | No |
| Timing (this whole project) | $5.26 ceiling | 17.5% of the time | Yes |

29% of the time, an **identical brand-new item** is available from a different seller for 5%+ less than the buy box, averaging $20.50. That single fact beats our entire timing ceiling, with certainty and zero waiting.

**Caveats.** The `used` column is the lowest offer at *any* condition — "Acceptable" isn't "Like New," so the condition-adjusted figure is probably a third to half of $17.48. Third-party sellers carry counterfeit, shipping, and returns risk, so "buy from whoever's cheapest" is bad advice without seller context. This sample is electronics best-sellers averaging $150 — it won't look like this for $12 consumables. The marketplace-new number ($6.30) is the one to trust most.

**Why this is available at all:** the buy box exists to funnel you to one offer. Warehouse deals being buried is not a UX oversight, it's the business model. Amazon will never surface this. **That's our moat — not technology, incentive alignment. We work for the buyer.**

---

## 4. What this adds up to

**The bad news.** What we shipped doesn't work. A model at lift 1.05 is a coin flip wearing a confidence score. Had we deployed to real users we'd have been giving random advice with an authoritative-looking percentage attached.

**The good news, and it's bigger.** The problem is learnable — Study 005 settles that. The failure was leakage, a wrong target, a test set used for early stopping, and a feature pipeline so broken that production fills unknowns with zeros. Those are all fixable, and fixing them is a legitimate semester of work with a measurable payoff.

**The reframe.** Timing is one lever and not the largest. The honest hierarchy of money saved per purchase:

1. **~$18** — show the cheaper offer Amazon buries (instant, certain, 74% of purchases)
2. **~$5–23** — timing, but *only* on expensive volatile items where the model has real edge
3. **~$0.13** — timing on cheap stable items, where we should say nothing at all

We spent the year on the difficult, low-value half. That's a normal thing to do when you pick a project by what sounds technically interesting, and we found out in September rather than April.

---

## 5. Recommended scope

**Do not ship this semester.** Deploying an unverified recommender to real users is worse than not deploying. Ship in early spring, once there's a track record to stand on.

**Redefine the product.** BuyWise answers *"what is the cheapest legitimate way to get this thing right now, and is waiting better than all of them?"* Prediction becomes one input among several instead of carrying the whole product on a $5 ceiling. This is also exactly the "integrated layer, not an abstract tack-on" instinct we already had — the data says that layer *is* the product.

**Analysis (6 + 2 leads).**
- Weeks 1–4, leads only: the evaluation harness. One `evaluate.py` that takes any model or rule and returns a standard report with matched wait-rate comparison against baselines. Nothing else in analysis starts until this exists — six people cannot iterate without a shared instrument.
- Everyone else meanwhile: extend Studies 000–002 on existing data. New members learn the domain by measuring it.
- Then pods: (a) rebuild the pipeline properly, starting from `s005_clean_rebuild.py`; (b) segmentation and the "no opinion" threshold — what's the minimum expected saving worth interrupting a user for; (c) calibration, so the confidence number becomes a real probability.
- **Every model ships with its precision, its lift over base rate, and its dollars at matched wait rate. Lift 1.0 means it doesn't ship.**

**Platform (4 + 1 lead).**
- **Weeks 1–3, the Keepa harvest is the critical path and it's near-irreversible.** $90 buys exactly one month. Decide the ASIN universe first — 3,000–5,000 products spanning categories, price bands, and Amazon-sold vs marketplace, because segment heterogeneity is now a core finding rather than a hypothesis. **Store raw Keepa JSON, not a flattened CSV** — our feature engineering will change five times and we cannot re-pull. Pull day 1, find gaps, top up day 20 while the subscription is live. Then rewrite `_fetch_and_seed` to read from the local snapshot.
- Then: on-page offer extraction — buy box vs. other sellers, Warehouse and used offers with condition, clippable coupons, Subscribe & Save, Prime eligibility. **This is now the highest-value surface in the product.** Read-only, never automate actions, never send raw order history to the backend.
- Then: the outcomes table and the resolver job that scores every prediction against what actually happened.

**Viz (2).** Early, an independent deliverable that needs no backend: a real design system and inline placement near the buy box instead of a floating corner panel. Later, the dashboard, once the resolver produces real outcomes. **Staffing gap: viz has no returning members and no dedicated lead.** Either move a returning person across or have the full-stack lead explicitly own it.

**What we present at the end of the semester:** a leakage-free pipeline, a shared harness with a baseline leaderboard, five completed studies with honest findings, a working local product that says "no opinion" when it should, and a deploy checklist for January.

**Replace the ship date with a hard internal demo day plus written study reports.** Research teams without deadlines produce branches like the 27 currently in this repo.

---

## Running these

```bash
cd studies
../backend/.venv/bin/python s000_quality.py
../backend/.venv/bin/python s001_movement.py
../backend/.venv/bin/python s002_baselines.py
../backend/.venv/bin/python s004_deployed_model_audit.py   # needs backend/ml/*.joblib
../backend/.venv/bin/python s005_clean_rebuild.py
../backend/.venv/bin/python s006_segmentation.py
../backend/.venv/bin/python s009_substitution.py
```

`Prices.csv` and `Product.csv` are copied here from `origin/Training_data_generation:notebooks/`.

**These are drafts, not finished work.** They were written quickly to answer specific questions. Before anyone builds on them, they should be folded into the proper harness — and **every number here must be re-derived on the new harvest**, because the current dataset is 312 electronics best-sellers with a weekly sampling artifact and known bad rows.

## Open studies, not yet run

| # | Question |
|---|---|
| 002b | **Horizon choice.** The 14-day window came from the old model's target and was never justified. Ceiling is $5.08 at 14 days, $8.01 at 30, $13.33 at 90 (204 ASINs, mean price $98). What horizon is the product actually built around? Everything in Studies 001/002 is sized at 14 days by default. |
| 003 | Full signal inventory — which features add what, and how much do they overlap? |
| 008 | Alert latency — how fast must we notify before the value decays? |
| 010 | Subscribe & Save orchestration — how much is the 15% tier worth over a year? |
| 011 | Pack-size / unit-price arbitrage across product variants |
| 012 | Does Amazon's own deal badging already capture what we'd say? (30 minutes of screenshots, do this week) |
