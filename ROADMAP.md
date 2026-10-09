# BuyWise Fall 2026 Roadmap

## Where we're going

An Amazon product page lists many sellers but has only one Add to Cart button. Amazon picks the seller behind that button for a typical shopper and for its own business, and the cheaper offers sit a click away where almost nobody looks. BuyWise reads every offer on the page, weighs what makes a cheaper one worth it or not *for this shopper*, and says so in one line. When Amazon's pick is already the best sensible deal, it says that instead and gets out of the way.

When waiting is the better move, BuyWise watches the product so the shopper doesn't have to. They pick a price they'd pay, or ask to be told when it's a good time to buy, and BuyWise notifies them.

Every shopper is different, so BuyWise learns a few things about them with as little friction as possible. It asks three questions once, infers what it can, and only asks something in the moment when the answer would change the recommendation.

The price-drop model is one input to these decisions. It doesn't carry the whole product.

By demo day we will have a working extension and dashboard that do all of this, along with a real track record of how often BuyWise was right.

## How we work

- **Every deliverable is a claim someone can check.** "The scraper works" can't be checked. "The offer reader works on these 100 pages, and here's the list I verified by hand" can.
- **The person who builds something is not the person who signs off on it.** Every piece of work has an owner and a different reviewer.
- **Main always builds.** Every PR passes the automatic checks before merging.
- **Models ship only if they win.** A model or score goes live only if it beats the simple rules in `backend/ml/evaluate.py`.
- **Shopper data stays on the shopper's device.** The profile and purchase history are kept locally. The backend never receives raw order history.

## Phase 1: Solid ground

*Everyone can run the project, nothing breaks without anyone noticing, and the product's look and first-run experience are defined.*

**Platform**
- Automatic checks on every PR: build the extension and run the backend tests.
- One-command local setup that starts the database, loads both seed files and runs the backend.
- Plan the Keepa data pull. Pick products across categories and price bands, store the raw responses, and include which seller held the Add to Cart button. Test-pull a small batch and check it by hand before spending the month of access.
- Decide where fresh prices for watched products come from once the Keepa month ends: page reads when a shopper visits, a scheduled check, or a lighter paid tracking option. Alerts in Phase 3 depend on this decision.
- Local storage for the shopper profile.

**Data visualization**
- Turn the website's look into the extension's design system: colors, type, spacing and base components. The tokens in `website/app/globals.css` are the starting point, and `website/app/preview` shows where the product is headed.
- Rebuild the current panel in the new style using mock data.
- Design the "checked, you're good" state. It is what users will see most often, so it gets the same care as everything else.
- Design the first-run profile: three questions, skippable, and changeable later.

**Phase 1 is done when:** a new member goes from clone to a running extension with one command, a broken PR can't merge, and the new panel and first-run designs are approved.

## Phase 2: See every offer

*The extension can read everything on a product page, and we have the real data.*

**Platform**
- The offer reader pulls every offer off a product page: seller, price, condition, shipping, returns, and prices hidden until checkout.
- The offer reader also captures the product's other versions (sizes, colors, generations, pack counts) along with their prices and per-unit cost.
- Real Amazon pages are saved as test fixtures. When Amazon changes its layout, the reader reports that it broke instead of silently returning nothing.
- Run the Keepa pull, store everything raw, and point the backend at the local copy so development doesn't depend on the live API.
- Infer what we can without asking, such as Prime membership from the page and repeat purchases.

**Data visualization**
- An offer comparison component: Amazon's pick next to the best alternative, with the reasons in plain words.
- A version comparison view for products with meaningful variants.
- Move the panel next to the Add to Cart button instead of floating in a corner.

**Phase 2 is done when:** the offer reader is correct on a hand-checked set of pages across several categories, and the Keepa data is stored and validated.

## Phase 3: Make the call

*The extension tells each shopper something true and useful, or stays quiet, and keeps watching when waiting is the right move.*

**Platform**
- One decision endpoint. Given a page's offers, the product's price history and the shopper's profile, it returns the best sensible option, the reason, and whether the result is worth showing.
- The moment-of-purchase check. When the cheaper option is slower or used, ask once whether the shopper needs it soon. Skip the question when the answer wouldn't change anything.
- Watching and alerts. A shopper can watch any product with a target price, or ask to be told when it's a good time to buy. Notifications show the new price and why now.
- Every recommendation the extension shows is logged so it can be scored later.

**Data visualization**
- The new panel connected to the decision endpoint, with all three outcomes designed: a better offer, wait, and you're good.
- The watch flow, from tapping Watch to setting a target to receiving a notification.
- A usability check with people outside the team. They should be able to tell what the panel is saying and why, without feeling interrogated.

**Phase 3 is done when:** the panel runs on live product pages, shows a reason for every recommendation, stays quiet most of the time, and a watched product sends a correct alert.

## Phase 4: Prove it and show it

*A working extension and dashboard, and a real track record at demo day.*

**Platform**
- A job that scores past recommendations against what actually happened: how much money following BuyWise saved or cost.

**Data visualization**
- The dashboard in full: watched products and their status, recent alerts, the profile, and the track record with hit rate and dollars saved, including the misses.
- Demo day materials: charts of our findings, a live demo path, and the website updated with the team and results.

**Phase 4 is done when:** we can state our real hit rate out loud and show it running live.

## Analysis

Analysis drives the research questions the product depends on. The analysis leads set the specific projects within these areas:

- **How often the button already points to the cheapest sensible seller.** The new data answers this, and the answer sets how large our opportunity really is.
- **What condition is worth.** For example, how much cheaper a used "Very Good" item should be than new. There's no published answer to this.
- **Seller trust.** Which sellers are safe to recommend. This is the highest-stakes question in the project, because a bad seller recommendation can hurt someone.
- **How risky a wait is.** A price drop on a steadily declining product is very different from a brief dip on a volatile one. A score that captures stability, using measures like variance and drawdowns, tells the shopper how much to trust a WAIT and when an alert is worth sending.
- **What reviews say across versions.** When versions differ in price, find out whether reviews show a real difference in quality, so the version comparison can say whether the cheaper one is worth it.
- **When to stay quiet.** The smallest saving worth interrupting a shopper for, and how that changes with their profile.
- **Model upkeep.** Retrain the price-drop model on the new data and score it with `evaluate.py`. The model is one input, and it gets a bounded amount of effort.

## How the teams connect

- Platform's offer reader produces the data analysis needs for seller, condition and version work.
- Analysis's answers become the rules inside the decision endpoint, and the risk score decides when a WAIT is shown and when an alert fires.
- Data visualization builds against mock data throughout and switches to real data once the decision endpoint exists.
- The profile is designed by data visualization, stored by platform, and used by the decision endpoint. Analysis tells us which questions actually change outcomes, so the profile stays at three questions.
