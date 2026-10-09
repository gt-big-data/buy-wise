# BuyWise Fall 2026 Roadmap

## Where we're going

Amazon product pages have many sellers but only one Add to Cart button. Amazon picks the seller behind that button for a typical shopper and for its own business, and the cheaper offers sit a click away where almost nobody looks. BuyWise reads every offer on the page, weighs what makes a cheaper one worth it or not, and tells the shopper in one line. When Amazon's pick is already the best sensible deal, it says so and stays quiet.

The price-drop model stays as one input to that decision. It no longer carries the whole product.

By demo day we will have a working extension that does this, along with a real track record of how often it was right.

## How we work

- **Every deliverable is a claim someone can check.** "The scraper works" can't be checked. "The offer reader works on these 100 pages, and here's the list I verified by hand" can.
- **The person who builds something is not the person who signs off on it.** Every piece of work has an owner and a different reviewer.
- **Main always builds.** Every PR passes the automatic checks before merging.
- **Models ship only if they win.** A model goes live only if it beats the simple rules in `backend/ml/evaluate.py`.

## Phase 1: Solid ground

*Everyone can run the project, nothing breaks without anyone noticing, and the new look is defined.*

**Platform**
- Automatic checks on every PR: build the extension and run the backend tests.
- One-command local setup that starts the database, loads both seed files and runs the backend.
- Plan the Keepa data pull. Pick products across categories and price bands, store the raw responses, and include which seller held the Add to Cart button. Test-pull a small batch and check it by hand before spending the month of access.

**Data visualization**
- Turn the website's look into the extension's design system: colors, type, spacing and base components. The tokens in `website/app/globals.css` are the starting point.
- Rebuild the current panel in the new style using mock data.
- Design the "checked, you're good" state. It is what users will see most often, so it gets the same care as everything else.

**Phase 1 is done when:** a new member goes from clone to a running extension with one command, a broken PR can't merge, and the new panel design is approved.

## Phase 2: See every offer

*The extension can read every seller on a page, and we have the real data.*

**Platform**
- The offer reader pulls every offer off a product page: seller, price, condition, shipping, returns, and prices hidden until checkout.
- Real Amazon pages are saved as test fixtures. When Amazon changes its layout, the reader reports that it broke instead of silently returning nothing.
- Run the Keepa pull, store everything raw, and point the backend at the local copy so development doesn't depend on the live API.

**Data visualization**
- An offer comparison component: Amazon's pick next to the best alternative, with the reasons in plain words.
- Move the panel next to the Add to Cart button instead of floating in a corner.

**Phase 2 is done when:** the offer reader is correct on a hand-checked set of pages across several categories, and the Keepa data is stored and validated.

## Phase 3: Make the call

*The extension tells someone something true and useful, or stays quiet.*

**Platform**
- One decision endpoint. Given a page's offers and the product's price history, it returns the best sensible option, the reason, and whether the result is worth showing.
- Every recommendation the extension shows is logged so it can be scored later.

**Data visualization**
- The new panel connected to the decision endpoint, with all three outcomes designed: a better offer, wait, and you're good.
- A usability check with people outside the team. They should be able to tell what the panel is saying and why.

**Phase 3 is done when:** the panel runs on live product pages, shows a reason for every recommendation, and stays quiet most of the time.

## Phase 4: Prove it and show it

*A working extension and a real track record at demo day.*

**Platform**
- A job that scores past recommendations against what actually happened: how much money following BuyWise saved or cost.

**Data visualization**
- A track-record dashboard in the popup showing hit rate and dollars saved, including the misses.
- Demo day materials: charts of our findings, a live demo path, and the website updated with the team and results.

**Phase 4 is done when:** we can state our real hit rate out loud and show it running live.

## Analysis

Analysis drives the research questions the product depends on. The analysis leads set the specific projects within these areas:

- **How often the button already points to the cheapest sensible seller.** The new data answers this, and the answer sets how large our opportunity really is.
- **What condition is worth.** For example, how much cheaper a used "Very Good" item should be than new. There's no published answer to this.
- **Seller trust.** Which sellers are safe to recommend. This is the highest-stakes question in the project, because a bad seller recommendation can hurt someone.
- **When to stay quiet.** The smallest saving worth interrupting a shopper for.
- **Model upkeep.** Retrain the price-drop model on the new data and score it with `evaluate.py`. The model is one input, and it gets a bounded amount of effort.

## How the teams connect

- Platform's offer reader produces the data analysis needs for seller and condition work.
- Analysis's answers become the rules inside the decision endpoint.
- Data visualization builds against mock data throughout and switches to real data once the decision endpoint exists.
