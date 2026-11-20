# BuyWise track record

Replayed on `Prices.csv`: the served model (clean@2026-11-06T17:36:46+00:00) predicted every product every 7 day(s) using only earlier prices, then each call was checked against the next 14 days. No live Keepa data this semester, so this is a replay, not calls shoppers saw.

8,516 predictions logged: 8,515 resolved, 0 still pending, 1 skipped for bad price data.

293 products, predicted 2025-08-28 to 2026-03-26.

## How often it was right

| | |
|---|---|
| Overall hit rate | 84.1% |
| WAIT calls right (price fell 8%+ within 14 days) | 65.2% of 598 |
| BUY calls right (no 8% drop came) | 85.5% of 7,917 |
| Base rate (how often an 8% drop came at all) | 18.1% |

## Dollars

| | |
|---|---|
| Saved by right WAITs | $19,358.94 |
| Lost by wrong WAITs (price went up while waiting) | $265.63 |
| Missed by wrong BUYs (a drop came after we said buy) | $31,265.16 |

## Compared with the offline benchmark

Same definitions as `ml/bench/metrics.py`, at the model's served threshold.

| Measure | This track record | Offline test set |
|---|---|---|
| Precision | 65.2% | 60.0% |
| Lift | 3.61 | 2.75 |
| $ caught / decision | $2.31 | $4.50 |
| $ no alert / decision | $1.65 | $3.40 |

## Biggest misses

| ASIN | Predicted on | Said | P(drop) | Price then | Low in 14d | Price day 14 | Cost |
|---|---|---|---|---|---|---|---|
| B0D2PNH494 | 2025-11-06 | BUY | 32% | $797.99 | $174.00 | $697.16 | $623.99 |
| B0D2PNH494 | 2025-11-13 | BUY | 32% | $797.99 | $174.00 | $697.16 | $623.99 |
| B0FQPJXQB3 | 2026-02-12 | BUY | 6% | $2072.40 | $1656.99 | $1893.98 | $415.41 |
| B0FD49H754 | 2026-01-15 | BUY | 33% | $1695.98 | $1345.98 | $1345.98 | $350.00 |
| B0FQPJXQB3 | 2026-01-08 | BUY | 37% | $2266.75 | $1937.56 | $1937.56 | $329.19 |
| B0FD49H754 | 2025-09-25 | BUY | 6% | $1775.98 | $1450.98 | $1450.98 | $325.00 |
| B0FD49H754 | 2025-10-02 | BUY | 12% | $1775.98 | $1450.98 | $1453.98 | $325.00 |
| B0CG19FGQ5 | 2025-11-13 | BUY | 33% | $979.00 | $669.00 | $669.00 | $310.00 |
| B0D2PNH494 | 2025-10-02 | BUY | 37% | $999.99 | $695.99 | $695.99 | $304.00 |
| B0FQPJXQB3 | 2026-02-05 | BUY | 5% | $2078.41 | $1777.51 | $1786.91 | $300.90 |
| B0FD49H754 | 2025-12-11 | BUY | 33% | $1645.98 | $1395.98 | $1395.98 | $250.00 |
| B0FQPJXQB3 | 2025-10-30 | BUY | 6% | $2193.99 | $1946.94 | $2093.98 | $247.05 |
| B0FQPJXQB3 | 2025-10-23 | BUY | 6% | $2193.99 | $1946.94 | $2093.60 | $247.05 |
| B0FD49H754 | 2025-12-18 | BUY | 32% | $1635.98 | $1395.98 | $1575.98 | $240.00 |
| B0CG19FGQ5 | 2025-10-16 | BUY | 22% | $979.00 | $739.99 | $739.99 | $239.01 |
