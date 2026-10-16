# BuyWise Fall 2026 Roadmap

## What we're building

An Amazon product page often lists many sellers but has one Add to Cart button. Amazon chooses which seller gets that button, and cheaper offers from other sellers, used copies and other versions of the product sit on a separate list most shoppers never open. BuyWise reads every offer on the page, decides whether a cheaper one is worth it for this shopper (given price, condition, shipping speed, returns and seller), and shows a one-line recommendation with the reason. When Amazon's default is already the best option, BuyWise says so and shows nothing else.

When waiting is the better move, the shopper can watch the product. They set a target price, or ask to be notified when it's a good time to buy, and BuyWise sends an alert.

BuyWise keeps a small profile of each shopper: three questions asked once at setup, plus what it can infer from the page (such as Prime membership). It only asks a question at the moment of purchase when the answer would change the recommendation.

The price-drop model is one input to these recommendations, not the core of the product.

**At demo day:** a working extension and dashboard that do all of the above, and a measured hit rate and dollars saved for BuyWise's past recommendations.

## How we work

- **Every deliverable says how it was checked.** "The scraper works" isn't enough. "The offer reader is correct on these 100 pages, and here's the list I verified by hand" is.
- **Teams.** Platform and data viz work in pods that each own a feature: UI, Offers and Recommendations. Analysis works in Team R (benchmark, models, paper) and Team T (data, track record, seller ranking), planned in [#88](https://github.com/gt-big-data/buy-wise/issues/88).
- **Everyone picks their own issue.** Each issue has one owner, assigned on GitHub, who answers for it on Friday. The rest of the pod or team helps however they agree.
- **Your issue is your plan.** It says what to build and what to show each Friday until demo day. This page only gives the big picture.
- **Owner and reviewer are different people.**
- **Main always builds.** Every PR passes the automatic checks before merging.
- **Models must beat the baselines.** A model or score goes live only if it beats the simple rules in `backend/ml/evaluate.py`.
- **Shopper data stays on the shopper's device.** The profile and purchase history are stored locally. The backend never receives raw order history.

## Every Friday

Weekly meetings are on Fridays. Each one is where we show what's running, not talk about it. Details for each line are in the linked issue.

| Friday | What we show |
|---|---|
| **Oct 23** | Checks on every PR and one-command setup ([#72](https://github.com/gt-big-data/buy-wise/issues/72)) · Keepa plan signed off and access bought ([#70](https://github.com/gt-big-data/buy-wise/issues/70)) · daily price job running ([#90](https://github.com/gt-big-data/buy-wise/issues/90)) · seller list read on a live page ([#69](https://github.com/gt-big-data/buy-wise/issues/69)) · three panel states with fake data ([#71](https://github.com/gt-big-data/buy-wise/issues/71)) · first-run profile ([#77](https://github.com/gt-big-data/buy-wise/issues/77)) · list of pages where the extension breaks, with fixes ([#73](https://github.com/gt-big-data/buy-wise/issues/73)) · Analysis: shared data snapshot ([#83](https://github.com/gt-big-data/buy-wise/issues/83)), benchmark interface ([#78](https://github.com/gt-big-data/buy-wise/issues/78)) |
| **Oct 30** | Offer reader correct across product types ([#69](https://github.com/gt-big-data/buy-wise/issues/69)) · Keepa data stored and daily prediction logging running ([#70](https://github.com/gt-big-data/buy-wise/issues/70), [#85](https://github.com/gt-big-data/buy-wise/issues/85)) · decision endpoint format agreed, with placeholder rules ([#92](https://github.com/gt-big-data/buy-wise/issues/92)) · panel in its final spot, with offer comparison ([#71](https://github.com/gt-big-data/buy-wise/issues/71)) · target prices and watch status ([#74](https://github.com/gt-big-data/buy-wise/issues/74)) · watch flow and dashboard designs ([#93](https://github.com/gt-big-data/buy-wise/issues/93)) · Analysis: first model leaderboard ([#78](https://github.com/gt-big-data/buy-wise/issues/78)–[#80](https://github.com/gt-big-data/buy-wise/issues/80)), first offer-ranking rule on real offers ([#86](https://github.com/gt-big-data/buy-wise/issues/86)) |
| **Nov 6** | Live recommendations with reasons on real pages ([#92](https://github.com/gt-big-data/buy-wise/issues/92), [#71](https://github.com/gt-big-data/buy-wise/issues/71)) · alerts firing ([#74](https://github.com/gt-big-data/buy-wise/issues/74)) · product versions read ([#69](https://github.com/gt-big-data/buy-wise/issues/69)) · purchase-time question ([#77](https://github.com/gt-big-data/buy-wise/issues/77)) · watch flow on real data ([#93](https://github.com/gt-big-data/buy-wise/issues/93)) · demo script draft ([#94](https://github.com/gt-big-data/buy-wise/issues/94)) · Analysis: models re-run on Keepa data, first predictions checked against real prices ([#85](https://github.com/gt-big-data/buy-wise/issues/85)) |
| **Nov 13: freeze** | Final rules in the decision endpoint ([#92](https://github.com/gt-big-data/buy-wise/issues/92)) · dashboard with the real track record ([#93](https://github.com/gt-big-data/buy-wise/issues/93), [#85](https://github.com/gt-big-data/buy-wise/issues/85)) · usability test results ([#71](https://github.com/gt-big-data/buy-wise/issues/71)) · charts, website and a full demo run-through ([#94](https://github.com/gt-big-data/buy-wise/issues/94)). After this, only bug fixes. Every number shown at the demo is final. |
| **Nov 20** | Demo day |

If something slips, we cut scope, not move dates.

## Analysis

The full analysis plan is in [#88](https://github.com/gt-big-data/buy-wise/issues/88). In short:

- **Team R:** a benchmark every model plugs into ([#78](https://github.com/gt-big-data/buy-wise/issues/78)), tabular and deep/pre-trained model comparisons ([#79](https://github.com/gt-big-data/buy-wise/issues/79), [#80](https://github.com/gt-big-data/buy-wise/issues/80)), calibration and when to say nothing ([#81](https://github.com/gt-big-data/buy-wise/issues/81)), and a workshop paper ([#82](https://github.com/gt-big-data/buy-wise/issues/82)).
- **Team T:** shared data snapshots and one results log ([#83](https://github.com/gt-big-data/buy-wise/issues/83)), whether reviews show the cheaper version is worse ([#84](https://github.com/gt-big-data/buy-wise/issues/84)), the track record ([#85](https://github.com/gt-big-data/buy-wise/issues/85)), ranking the offers on a page ([#86](https://github.com/gt-big-data/buy-wise/issues/86)), and handing results to the decision endpoint ([#87](https://github.com/gt-big-data/buy-wise/issues/87)).

## How the teams connect

- Platform's offer reader produces the data analysis needs for seller, condition and version work.
- Analysis outputs become rules inside the decision endpoint ([#92](https://github.com/gt-big-data/buy-wise/issues/92)). The stability score decides when a WAIT is shown and when an alert fires.
- One daily price job ([#90](https://github.com/gt-big-data/buy-wise/issues/90)) feeds both the track record ([#85](https://github.com/gt-big-data/buy-wise/issues/85)) and price alerts ([#74](https://github.com/gt-big-data/buy-wise/issues/74)).
- Agree on handoff formats early, with a fake example, so both sides can build in parallel: Keepa files ([#70](https://github.com/gt-big-data/buy-wise/issues/70) → [#83](https://github.com/gt-big-data/buy-wise/issues/83)), offer lists ([#69](https://github.com/gt-big-data/buy-wise/issues/69) → [#86](https://github.com/gt-big-data/buy-wise/issues/86)), analysis results ([#87](https://github.com/gt-big-data/buy-wise/issues/87) → decision endpoint), and track-record numbers ([#85](https://github.com/gt-big-data/buy-wise/issues/85) → UI).
- The UI pod builds against fake data until the decision endpoint is live (Nov 6), then switches to real answers.
- The UI pod builds the profile ([#77](https://github.com/gt-big-data/buy-wise/issues/77)) and the decision endpoint uses it. Analysis tells us which questions actually change outcomes, so the profile stays at three questions.

## After demo day

- Decide where fresh prices come from once Keepa access ends (around Nov 22): reading the page when a shopper visits, a scheduled check, or a cheaper paid tracking service.
