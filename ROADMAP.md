# BuyWise Fall 2026 Roadmap

This is the plan for the semester: what we're building, who owns what, how we work, and what gets shown each Friday until demo day on **Nov 20**. Each issue has the details for its piece.

## What we're building

An Amazon product page often lists many sellers but has one Add to Cart button. Amazon chooses which seller gets that button, and cheaper offers from other sellers, used copies and other versions of the product sit on a separate list most shoppers never open. BuyWise reads every offer on the page, decides whether a cheaper one is worth it for this shopper (given price, condition, shipping speed, returns and seller), and shows a one-line recommendation with the reason. When Amazon's default is already the best option, BuyWise says so and shows nothing else.

When waiting is the better move, the shopper can watch the product. They set a target price, or ask to be notified when it's a good time to buy, and BuyWise sends an alert.

BuyWise keeps a small profile of each shopper: three questions asked once at setup, plus what it can infer from the page (such as Prime membership). It only asks a question at the moment of purchase when the answer would change the recommendation.

The price-drop model is one input to these recommendations, not the core of the product.

**At demo day:** a working extension and dashboard that do all of the above, and a measured hit rate and dollars saved for BuyWise's past recommendations.

## Teams

Leads: Aashish (platform and data viz), Rithwik and Vansh (analysis).

| Team | People | Owns by demo day | Issues |
|---|---|---|---|
| **UI** | Data viz | Everything the shopper sees: styling and the panel, the first-run profile, the watch flow and the dashboard | [#71](https://github.com/gt-big-data/buy-wise/issues/71), [#77](https://github.com/gt-big-data/buy-wise/issues/77), [#93](https://github.com/gt-big-data/buy-wise/issues/93) |
| **Offers** | Platform | Reading every seller and product version off the page, and price alerts | [#69](https://github.com/gt-big-data/buy-wise/issues/69), [#74](https://github.com/gt-big-data/buy-wise/issues/74) |
| **Engine** | Platform | The Keepa data, the daily price job, and the decision endpoint that turns everything into one answer per page | [#70](https://github.com/gt-big-data/buy-wise/issues/70), [#90](https://github.com/gt-big-data/buy-wise/issues/90), [#92](https://github.com/gt-big-data/buy-wise/issues/92) |
| **Research** | Risheet, Miguel, Siya | A benchmark every model plugs into, model comparisons, calibration and when to say nothing, and a workshop paper | [#78](https://github.com/gt-big-data/buy-wise/issues/78)–[#82](https://github.com/gt-big-data/buy-wise/issues/82) |
| **Tooling** | Miles, Himank, Nirv | Shared data, whether cheaper versions are actually worse, the track record, seller ranking, and handing results to the Engine | [#83](https://github.com/gt-big-data/buy-wise/issues/83)–[#87](https://github.com/gt-big-data/buy-wise/issues/87) |

Anyone can take [#72](https://github.com/gt-big-data/buy-wise/issues/72) (checks and setup) or [#73](https://github.com/gt-big-data/buy-wise/issues/73) (pages where the extension breaks). Leads own [#94](https://github.com/gt-big-data/buy-wise/issues/94) (demo day), with volunteers. People can move between teams. Who is working on what is whoever is assigned on GitHub. The analysis plan in more detail is in [#88](https://github.com/gt-big-data/buy-wise/issues/88).

## How we work

- **Pick your own issue.** Each issue has one owner, assigned on GitHub, who answers for it on Friday. The rest of the team helps however they agree.
- **Your issue is your plan.** It says what to build and what to show each Friday until demo day.
- **Meetings are Fridays, 30 minutes,** and they're working time. Each team shows what's running, not slides. Analysis also meets separately during the week and posts a short update on each issue before the Friday meeting: what's done, the current number, what's blocking.
- **Branches and PRs.** Name each branch `<issue number>-short-name`, push it, and open a PR into main. Someone other than the author reviews it. For analysis work, the reviewer re-runs it and gets the same numbers. Squash and merge.
- **Main always builds.** A PR can't merge if the checks fail ([#72](https://github.com/gt-big-data/buy-wise/issues/72)).
- **Every deliverable says how it was checked.** "The scraper works" isn't enough. "It's correct on these 30 pages, and here's how I checked" is.
- **Use AI tools, but write your own issue and PR descriptions,** and be able to explain what you submitted.
- **Numbers come from the benchmark.** Any number at the demo or in the paper comes from [#78](https://github.com/gt-big-data/buy-wise/issues/78), compared at the same wait rate, with an error bar. A model goes live only if it beats the simple rules. Tune on validation data, never on test data.
- **Shopper data stays on the shopper's device.** The backend never stores the profile or order history, and no shopper data goes into any dataset or the paper.
- **Have an idea?** Open an issue with a short description, or DM a lead.

## Every Friday

| Friday | What we show |
|---|---|
| **Oct 23** | Checks on every PR and one-command setup ([#72](https://github.com/gt-big-data/buy-wise/issues/72)) · Keepa plan signed off and access bought ([#70](https://github.com/gt-big-data/buy-wise/issues/70)) · daily price job running ([#90](https://github.com/gt-big-data/buy-wise/issues/90)) · seller list read on a live page ([#69](https://github.com/gt-big-data/buy-wise/issues/69)) · three panel states with fake data ([#71](https://github.com/gt-big-data/buy-wise/issues/71)) · first-run profile ([#77](https://github.com/gt-big-data/buy-wise/issues/77)) · list of pages where the extension breaks, with fixes ([#73](https://github.com/gt-big-data/buy-wise/issues/73)) · Tooling: shared data snapshot ([#83](https://github.com/gt-big-data/buy-wise/issues/83)) · Research: benchmark interface ([#78](https://github.com/gt-big-data/buy-wise/issues/78)) |
| **Oct 30** | Offer reader correct across product types ([#69](https://github.com/gt-big-data/buy-wise/issues/69)) · Keepa data stored and daily prediction logging running ([#70](https://github.com/gt-big-data/buy-wise/issues/70), [#85](https://github.com/gt-big-data/buy-wise/issues/85)) · decision endpoint format agreed, with placeholder rules ([#92](https://github.com/gt-big-data/buy-wise/issues/92)) · panel in its final spot, with offer comparison ([#71](https://github.com/gt-big-data/buy-wise/issues/71)) · target prices and watch status ([#74](https://github.com/gt-big-data/buy-wise/issues/74)) · watch flow and dashboard designs ([#93](https://github.com/gt-big-data/buy-wise/issues/93)) · Research: first model leaderboard ([#78](https://github.com/gt-big-data/buy-wise/issues/78)–[#80](https://github.com/gt-big-data/buy-wise/issues/80)) · Tooling: first offer-ranking rule on real offers ([#86](https://github.com/gt-big-data/buy-wise/issues/86)) · advisor check-in 1 |
| **Nov 6** | Live recommendations with reasons on real pages ([#92](https://github.com/gt-big-data/buy-wise/issues/92), [#71](https://github.com/gt-big-data/buy-wise/issues/71)) · alerts firing ([#74](https://github.com/gt-big-data/buy-wise/issues/74)) · product versions read ([#69](https://github.com/gt-big-data/buy-wise/issues/69)) · purchase-time question ([#77](https://github.com/gt-big-data/buy-wise/issues/77)) · watch flow on real data ([#93](https://github.com/gt-big-data/buy-wise/issues/93)) · demo script draft ([#94](https://github.com/gt-big-data/buy-wise/issues/94)) · Research: models re-run on Keepa data · Tooling: first predictions checked against real prices ([#85](https://github.com/gt-big-data/buy-wise/issues/85)) |
| **Nov 13: freeze** | Final rules in the decision endpoint ([#92](https://github.com/gt-big-data/buy-wise/issues/92)) · dashboard with the real track record ([#93](https://github.com/gt-big-data/buy-wise/issues/93), [#85](https://github.com/gt-big-data/buy-wise/issues/85)) · usability test results ([#71](https://github.com/gt-big-data/buy-wise/issues/71)) · charts, website and a full demo run-through ([#94](https://github.com/gt-big-data/buy-wise/issues/94)) · Research: paper draft to the advisor before check-in 2 ([#82](https://github.com/gt-big-data/buy-wise/issues/82)). After this, only bug fixes, and every number shown at the demo is final. The paper keeps going. |
| **Nov 20** | Demo day |

If something slips, we cut scope, not move dates.

## How the teams connect

Agree on the format of every handoff early, with a fake example, so both sides can build at the same time.

- **Offers → Tooling:** real offer lists from the page ([#69](https://github.com/gt-big-data/buy-wise/issues/69) → [#86](https://github.com/gt-big-data/buy-wise/issues/86)).
- **Engine → Tooling:** the raw Keepa files ([#70](https://github.com/gt-big-data/buy-wise/issues/70) → [#83](https://github.com/gt-big-data/buy-wise/issues/83)).
- **Engine → Tooling and Offers:** one daily price job ([#90](https://github.com/gt-big-data/buy-wise/issues/90)) feeds both the track record ([#85](https://github.com/gt-big-data/buy-wise/issues/85)) and price alerts ([#74](https://github.com/gt-big-data/buy-wise/issues/74)).
- **Tooling and Research → Engine:** seller ranking ([#86](https://github.com/gt-big-data/buy-wise/issues/86)), when to say nothing ([#81](https://github.com/gt-big-data/buy-wise/issues/81)) and version verdicts ([#84](https://github.com/gt-big-data/buy-wise/issues/84)) go to the Engine through [#87](https://github.com/gt-big-data/buy-wise/issues/87). The Engine combines them in the decision endpoint ([#92](https://github.com/gt-big-data/buy-wise/issues/92)).
- **Engine → UI:** the decision endpoint's answers ([#92](https://github.com/gt-big-data/buy-wise/issues/92) → [#71](https://github.com/gt-big-data/buy-wise/issues/71)). UI builds on fake data until it's live.
- **Tooling → UI:** track-record numbers ([#85](https://github.com/gt-big-data/buy-wise/issues/85) → [#93](https://github.com/gt-big-data/buy-wise/issues/93)) and version verdicts ([#84](https://github.com/gt-big-data/buy-wise/issues/84) → [#71](https://github.com/gt-big-data/buy-wise/issues/71)).
- **UI ↔ Tooling:** which profile questions change the recommendation ([#77](https://github.com/gt-big-data/buy-wise/issues/77), [#86](https://github.com/gt-big-data/buy-wise/issues/86)).

## Not this semester

- A full seller-trust model. Seller ranking uses simple signals from the page.
- Scraping Amazon reviews. We use a public reviews dataset.
- Inferring repeat purchases. It would need order history.
- Real user accounts. Each install gets its own ID.

## After demo day

- Decide where fresh prices come from once Keepa access ends (around Nov 22): reading the page when a shopper visits, a scheduled check, or a cheaper paid tracking service.
