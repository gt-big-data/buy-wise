# BuyWise Fall 2026 Roadmap

## What we're building

An Amazon product page often lists many sellers but has one Add to Cart button. Amazon chooses which seller gets that button, and cheaper offers from other sellers, used copies and other versions of the product sit on a separate list most shoppers never open. BuyWise reads every offer on the page, decides whether a cheaper one is worth it for this shopper (given price, condition, shipping speed, returns and seller), and shows a one-line recommendation with the reason. When Amazon's default is already the best option, BuyWise says so and shows nothing else.

When waiting is the better move, the shopper can watch the product. They set a target price, or ask to be notified when it's a good time to buy, and BuyWise sends an alert.

BuyWise keeps a small profile of each shopper: three questions asked once at setup, plus what it can infer from the page (such as Prime membership). It only asks a question at the moment of purchase when the answer would change the recommendation.

The price-drop model is one input to these recommendations, not the core of the product.

**At demo day:** a working extension and dashboard that do all of the above, and a measured hit rate and dollars saved for BuyWise's past recommendations.

## How we work

- **Every deliverable says how it was checked.** "The scraper works" isn't enough. "The offer reader is correct on these 100 pages, and here's the list I verified by hand" is.
- **Teams.** Platform and data viz work in pods that each own a feature: UI, Offers and Recommendations. Analysis works in Team R (benchmark, models, paper) and Team T (data, track record, seller ranking), planned in #88.
- **Everyone picks their own issue.** Each issue has one owner, assigned on GitHub, who answers for it on Friday. The rest of the pod or team helps however they agree.
- **Owner and reviewer are different people.**
- **Main always builds.** Every PR passes the automatic checks before merging.
- **Models must beat the baselines.** A model or score goes live only if it beats the simple rules in `backend/ml/evaluate.py`.
- **Shopper data stays on the shopper's device.** The profile and purchase history are stored locally. The backend never receives raw order history.

## Timeline

Each phase starts at a Friday meeting and ends the Thursday night before the next one. At each meeting, every pod shows its checkpoint running.

| Phase | Dates | Result |
|---|---|---|
| 1. Setup and design | Oct 16 – Oct 22 | One-command setup, CI, approved panel designs |
| 2. Offer reader and data | Oct 23 – Oct 29 | Every offer read correctly; Keepa data stored |
| 3. Recommendations and alerts | Oct 30 – Nov 5 | Live recommendations and working alerts |
| 4. Track record and demo | Nov 6 – Nov 12 | Scored results on the dashboard; demo path built |
| Freeze and cleanup | Nov 13 – Nov 19 | Bug fixes and demo rehearsal. No new features, and every number shown at the demo is final. |
| Demo day | Nov 20 | |

Features run across phases; the phases are checkpoints. If a checkpoint slips, we cut scope from later phases rather than move demo day.

## Phase 1: Setup and design (Oct 16 – 22)

**Platform**
- Automatic checks on every PR: build the extension and run the backend tests.
- One-command local setup that starts the database, loads both seed files and runs the backend.
- Plan the Keepa data pull. Pick products across categories and price bands, store the raw responses, and include which seller held the Add to Cart button. Buy the month of access on Oct 23 so it lasts through demo day. The budget has to cover daily re-fetches for the track record and alerts, not just the one-time pull (#70).
- Local storage for the shopper profile.

**Data visualization**
- Turn the website's look into the extension's design system: colors, type, spacing and base components. Start from the tokens in `website/app/globals.css`. `website/app/preview` shows the target look.
- Rebuild the current panel in the new style using mock data.
- Design the "no better option" state. Shoppers will see it more than any other state.
- Design the first-run profile: three questions, skippable, and editable later.

**Done when:**
- [ ] A new member goes from clone to a running extension with one command.
- [ ] A PR that breaks the build or tests can't merge.
- [ ] The leads have approved the new panel and first-run designs.

## Phase 2: Offer reader and data (Oct 23 – 29)

**Platform**
- The offer reader pulls every offer off a product page: seller, price, condition, shipping, returns, and prices hidden until checkout.
- The offer reader also captures the product's other versions (sizes, colors, generations, pack counts) with their prices and per-unit cost.
- Save real Amazon pages as test fixtures. When Amazon changes its layout, the reader reports an error instead of silently returning nothing.
- Run the Keepa pull, store the raw responses, and point the backend at the local copy so development doesn't depend on the live API.
- Host one daily price job, running from Oct 23, that both the track record and price alerts read from (#90).
- Infer what we can without asking, such as Prime membership from the page and repeat purchases.

**Data visualization**
- An offer comparison component: Amazon's default next to the best alternative, with the reasons in plain words.
- A version comparison view for products with meaningful variants.
- Move the panel next to the Add to Cart button instead of floating in a corner.

**Done when:**
- [ ] The offer reader is correct on a hand-checked set of pages across several categories.
- [ ] The Keepa data is stored and validated.

## Phase 3: Recommendations and alerts (Oct 30 – Nov 5)

**Platform**
- One decision endpoint. Given a page's offers, the product's price history and the shopper's profile, it returns the best option, the reason, and whether the result is worth showing.
- The purchase-time question. When the cheaper option is slower or used, ask once whether the shopper needs it soon. Skip the question when the answer wouldn't change the recommendation.
- Watching and alerts. A shopper can watch any product with a target price, or ask to be told when it's a good time to buy. Alerts show the new price and the reason to buy now.
- Log every recommendation the extension shows so it can be scored in Phase 4.

**Data visualization**
- Connect the new panel to the decision endpoint, with all three outcomes designed: better offer, wait, and no better option.
- The watch flow: tapping Watch, setting a target, and receiving an alert.
- A usability test with people outside the team. They should be able to say what the panel recommends and why, and shouldn't find the questions annoying.

**Done when:**
- [ ] The panel runs on live product pages and shows a reason with every recommendation.
- [ ] It shows a recommendation only on pages where the saving clears the threshold from the analysis work, and shows "no better option" everywhere else.
- [ ] A watched product sends a correct alert.

## Phase 4: Track record and demo (Nov 6 – 12, finished during freeze week)

**Platform**
- The track record comes from analysis (#85). Starting Oct 23, every Keepa product gets a prediction each day, and each prediction is checked against the real price once its 14 days are up. Predictions from the first week resolve before demo day. Platform keeps the daily job (#90) running.

**Data visualization**
- The full dashboard: watched products and their status, recent alerts, the profile, and the track record with hit rate and dollars saved, including the misses.
- Demo day materials: charts of our findings, a scripted live demo, and the website updated with the team and results.

**Done when:**
- [ ] The dashboard shows a hit rate and dollars saved computed by the scoring job.
- [ ] The live demo has been run end to end at least once before demo day.

**After demo day:** decide where fresh prices for watched products come from once Keepa access ends (reading the page when a shopper visits, a scheduled check, or a cheaper paid tracking service).

## Analysis

The full analysis plan is in #88. In short:

- **Team R:** a benchmark every model plugs into (#78), tabular and deep/pre-trained model comparisons (#79, #80), calibration and when to say nothing (#81), and a workshop paper (#82).
- **Team T:** shared data snapshots and one results log (#83), whether reviews show the cheaper version is worse (#84), the track record (#85), ranking the offers on a page (#86), and handing results to the decision endpoint (#87).

## How the teams connect

- Platform's offer reader produces the data analysis needs for seller, condition and version work.
- Analysis outputs become rules inside the decision endpoint. The stability score decides when a WAIT is shown and when an alert fires.
- One daily price job (#90) feeds both the track record (#85) and price alerts (#74).
- Agree on handoff formats early, with a fake example, so both sides can build in parallel: Keepa files (#70 → #83), offer lists (#69 → #86), analysis results (#87 → decision endpoint), and track-record numbers (#85 → UI).
- Data visualization builds against mock data until the decision endpoint exists in Phase 3, then switches to real data.
- Data visualization designs the profile, platform stores it, and the decision endpoint uses it. Analysis tells us which questions actually change outcomes, so the profile stays at three questions.
