# Benchmark protocol

How every model in the paper is scored. The code is in this folder; the one command is

```
cd backend && python -m ml.evaluate --models r2 --folds walk --ci --save
```

## The decision
On each product-day the model gives a score. Higher means more reason to WAIT. A WAIT is right if the price falls 8% or more within the next 14 days (`features.add_labels`).

## Matched wait rate
Each policy says WAIT on exactly its top 10%, 20% and 30% of decisions, and we always report all three. Without this, a model can look better just by saying WAIT more often (Study 004). Metrics per rate:
- precision and lift
- $ caught: buying at the 14-day low after a WAIT
- $ no alert: buying on day 14

Rate-free metrics are AUC and PR-AUC, plus Brier and ECE for probabilities.

## Time windows
`walk_forward_folds` has four back-to-back 91-day test windows ending at the last date, which covers a year, so every season is tested once. Each fold has:
- validation: the 91 days before the test window
- train: everything before validation, so training grows fold by fold

A row is kept only if its 14-day answer ends before the next period starts. `legacy_fold` (the spring 60/20/20 split) is kept so the old numbers always reproduce.

## What a model can see
- `fit` gets train and validation with labels, and price history up to the end of validation.
- `score` gets test rows with the label columns removed.
- Table models (`feature_cols` set) get *only* their feature columns: no price, date, product or history. A model can't read a later row's price even by accident.
- Sequence models get `history.windows()`, which ends on each row's decision day.
- Every feature uses data up to day t-1, plus day t's own price, list price, offer counts and sales rank. `tests/test_clean_model.py` wrecks everything after a cutoff and checks that no feature up to the cutoff moves.

## Tuning
- Every table model gets the same budget: 40 seeded random draws from its search space, and the first draw is always its defaults.
- The objective is PR-AUC on validation. It is rank-based and stable. The alternative, dollars at a fixed rate, is dominated by the few $200+ products (Study 006), so tuning on it fits a handful of items.
- Early stopping uses the last 15% of *train* dates, with the 14-day gap, so validation does one job: choosing settings.
- Test rows never reach `tuning.py`.
- Default and tuned results are both reported.

## Error bars
Rows from one product are correlated, so we resample *products*: 1000 draws with replacement, the same draws for every model and fold.
- A metric is pooled over folds as the per-fold value weighted by decisions.
- Intervals are 95% percentile intervals.
- "A beats B" is a *paired* difference on the same draws. It's real if the interval excludes 0. We also report the share of draws where A is no better.

## By kind of product
`--by` breaks results down by price band, volatility tercile (cut-offs from train), product seen in training, history length and season, plus category or brand when the data has them. Each segment shows how well the model ranks inside the segment (PR-AUC), and what its overall 20% policy does there (precision, $ caught, the segment's ceiling).

## Known limits
- `data._usable` keeps products whose whole history spans 270+ days. That choice uses the future, so test products are ones that stayed listed. It applies to every model equally.
- Prices.csv looks weekly-sampled (Study 000), so dollar amounts are probably undercounts.
- Prices.csv and the April snapshot are one category, so category results wait for the Keepa pull (T1, #83).
