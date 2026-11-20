# Demo day guide (#94)

Demo day is an open exploration: visitors try BuyWise themselves. This is what to set up, what to point them at, and how to answer questions.

The story, in three parts:

1. **Live seller comparison (the headline).** Amazon's Add to Cart button goes to one seller. BuyWise reads every offer on the page and names a trusted seller with the same item for less, or says Amazon's pick is already the best. On the 49 pages the team hand-checked, it found a better offer on 9, worth $258.07 together, the same picks the team made by hand.
2. **Wait, told through the past.** Our price history ends in April 2026, so the live panel rarely says Wait: whenever today's page price is more than 10% away from our last price, Wait is off. The Wait story is the dashboard instead: **How BuyWise has done** (65% of Wait calls dropped vs 18% at random, 293 products) and **Replay a past call**, which shows the panel the model would have shown on a real past day, the real prices around it, and what happened.
3. **The data.** One month of Keepa built a 3-year research dataset (649 products, Apr 2023 – Apr 2026). Every live visit records the real page price (#123). Live forecasting at scale needs a price data subscription.

## Pages for visitors

Amazon changes offers daily. Open each the night before and write down what the panel shows.

| Station | ASIN | What it is | Hand check (Nov 18) |
|---|---|---|---|
| Better offer | `B0D6RQGFGG` | Bose | Add to Cart seller Adorama ($299.00) fails the 90% feedback bar. OlympicWireless has it Used - Like New for $209.99. Needs "Like new or better". |
| Better offer, Amazon Resale | `B0F8YDRXNY` | Roku | Amazon Resale, Used - Like New, $116.99 vs $189.88. Shows "Sold by Amazon" instead of ratings. Needs "Like new or better". |
| Better offer, any profile | `B004R4N2B4` | Kodak | Neptune Photo Inc., New, $30.57 vs Amazon's $34.99. |
| Nothing better | `B0BS1T9J4Y` | Garmin Forerunner 265 | Amazon at $299.99 with 13+ offers, none better. |
| Nothing better, spare | `B0D9WWB3WX` | Oura | Amazon at $249.00. The only cheaper offer fails the feedback bar. |

Notes are in `studies/hand_check/pages.csv`. Profile for the demo: **Like new or better**, **Can wait a week or two**, alerts on. If a visitor picks "New only", the Kodak page still shows a better offer.

## The night before

| # | Do this | Check |
|---|---|---|
| 1 | `git pull` on `main`, then `docker compose up` from the repo root. | `curl localhost:8000/health` returns `ok` |
| 2 | `curl localhost:8000/predict/<ASIN>` for each ASIN above. | JSON, not `not_tracked` |
| 3 | `chrome://extensions` → Developer mode → Load unpacked → `extension/dist` (the build is committed). | Note the extension ID |
| 4 | Bookmark `chrome-extension://<ID>/profile.html?welcome` and `chrome-extension://<ID>/dashboard.html`. | Both open |
| 5 | Open the dashboard. | **How BuyWise has done** shows 293 products and 65% vs 18%. **Replay a past call** lists 7 calls; the first is a Hisense TV, Wait on Sep 25, 2025, dropped to $696.99 on Oct 6, 2025. |
| 6 | Open every page above and write down the panel's answer. | Each station has a page that works. Replace any that don't. |
| 7 | Record backups: `01-profile.png`, `02-better-offer.png`, `03-nothing-better.png`, `04-dashboard.png`, `05-replay.png`, and a short screen recording. | On the laptop and a USB stick |

## Stations

| Station | Let them do | Say |
|---|---|---|
| Profile | Open the first-run bookmark and answer the three questions. | "Three questions, stored only in this browser." |
| Better offer | Open a better-offer page. If the panel says "Based on the offers on this page", click **Compare all offers**. | "Amazon's button goes to one seller. This one is trusted and cheaper, and here's why." |
| Nothing better | Open the Garmin page. | "When Amazon's pick is already best, BuyWise says so and nothing more." |
| Replay | Dashboard → **Replay a past call**. Let them click through the calls, including the misses. | "This is the panel the model would have shown that day, with the real prices before and after. Here's one it got right, and here's one it got wrong." |
| Track record | Dashboard → **How BuyWise has done**. | "Every weekly call on 293 products from Aug 2025 to Mar 2026, each checked against the next 14 days. We show the misses too." |

Test mode (`?bw_state=wait` on any product URL) shows the Wait panel with made-up numbers. Use it only to show what the Wait panel and **Watch this price** look like, and say the numbers are made up.

## If something breaks

| Problem | Do this |
|---|---|
| Backend won't start | Show the backups |
| "Couldn't reach BuyWise" | `docker compose ps`, restart with `docker compose up` |
| "Doesn't have price history" | `curl localhost:8000/predict/<ASIN>`, reload, or use a spare |
| Captcha or a different page | Screenshots for that station |
| Panel doesn't appear | Click the toolbar icon, or reload the extension and the tab |
| Dashboard says "No track record yet" or no replay | Old checkout: pull `main` and restart the backend |

## Judge questions

Only quote numbers from this table. Each one is in the repo.

| Question | Answer |
|---|---|
| Where does the price data come from? | One month of Keepa access built a research dataset: `studies/Prices.csv` and `studies/snapshots/apr2026_500/`, 649 products, Apr 2023 – Apr 2026. Dates are real; nothing is shifted to today. Every live visit also records the real page price (#123). |
| Is the Wait about today's prices? | Mostly no, and the panel doesn't pretend. If today's page price is more than 10% from our last April price, Wait is off. The Wait evidence is the replay of past calls on real prices. Live forecasting at scale needs a data subscription. |
| How is Wait decided? | A CatBoost model gives the chance of an 8%+ drop in the next 14 days, calibrated (Study 022). We show Wait only if that chance is at least 15% **and** the expected net saving is at least $2, counting what a wrong Wait costs on items this volatile (Study 024, `backend/analysis/results/v2.json`). |
| How good is it? | Replay of the served model on past prices (`backend/analysis/results/track_record/`): 65% of Wait calls dropped 8%+ within 14 days, vs 18% for any call at random, over 293 products. Right Waits saved $19,358.94; wrong Waits cost $265.63. |
| Wrong Buys missed $31k. Isn't that bad? | That's the trade-off we chose: say Wait only when fairly sure. Missed drops mean the shopper paid what they'd have paid anyway; sending people waiting for nothing costs them. |
| How does Replay pick its calls? | A fixed rule, shown on the dashboard: the 3 biggest savings from right Waits, the biggest overpay from a wrong Wait, the biggest drop missed after a Buy, and the 2 biggest rises after a right Buy, one call per product. |
| How are sellers ranked? | T4's ranker: total price with shipping, used items discounted by condition (Like New 13%, Very Good 30%), a saving of at least $2 or 3%. Sellers need 4.0+ stars, 100+ ratings and 90%+ positive. Amazon's own offers count as trusted. |
| Does it save money? | On 49 hand-checked pages, a better offer on 9, $258.07 together, matching the team's own picks. On the rest, Amazon's pick was already best and the panel says so. |
| What about privacy? | The profile stays in the browser. The backend uses it for one answer and never stores it. No accounts. |
| What's next? | A price data subscription for live forecasting, and a track record of calls shoppers actually saw. |

Don't describe the model's accuracy in any way the table above doesn't. If a judge asks for a number that isn't here, say you'll follow up.
