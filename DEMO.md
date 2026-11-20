# Demo day script (#94)

A five-minute live run of BuyWise, with a backup for every step.

The story: Amazon picks one seller for the Add to Cart button. That seller isn't always the best deal for you. BuyWise reads every offer on the page and says one of three things: a trusted seller has it cheaper, wait because a drop is likely, or Amazon's pick is already the best. When waiting wins, you watch the price and get a Chrome alert.

Pages marked **CHECK** are live Amazon pages nobody has verified for the demo yet. Amazon changes offers and prices daily, so open each one the night before and write down what the panel shows.

## Products

| Step | ASIN | What it is | Why this one | Status |
|---|---|---|---|---|
| Better offer | `B0D6RQGFGG` | Bose | Hand check on Nov 18: Add to Cart seller (Adorama, $299.00) fails the 90% feedback bar. OlympicWireless has it Used - Like New for $209.99 (93% positive, 21,295 ratings). Needs the "Like new or better" profile answer. | **CHECK** |
| Better offer, spare | `B004R4N2B4` | Kodak | Hand check on Nov 18: Neptune Photo Inc. sells it New for $30.57 vs Amazon's $34.99 (95% positive, 812 ratings). Works with any profile answer. | **CHECK** |
| Nothing better | `B0BS1T9J4Y` | Garmin Forerunner 265 (seeded) | Hand check on Nov 18: Amazon at $299.99, its all-time low, with 13+ offers and none better. | **CHECK**: the model could show WAIT here instead |
| Nothing better, spare | `B0D9WWB3WX` | Oura | Hand check on Nov 18: Amazon at $249.00. The only cheaper offer fails the feedback bar (69%). | **CHECK** |
| Wait | `B00378KMFU` | Polk Audio | Got a WAIT in a recent check (75% drop chance at $349). Its dataset price is $349.00 and it sat at $279 two weeks before the data ends. | **CHECK** |
| Wait, spares | `B000AJIF4E`, `B0093162RM` | Sony ($113.00 in the data), Philips ($108.99 in the data) | Also got a WAIT in the same check. | **CHECK** |
| Pre-seeded watch | `B000AJIF4E` | Sony | Watched the night before, so the dashboard is never empty. | Set up the night before |

The hand-check notes are in `studies/hand_check/pages.csv`. Dataset prices are the last rows of `studies/snapshots/apr2026_500/Prices.csv.gz`.

Two things can change what a live page shows:

- The extension records the live Amazon price each time a product page loads, and the decision uses that page price (#135). So a live page can give a different answer than the dataset price would.
- A trusted cheaper seller beats WAIT unless waiting is worth more. If the WAIT page shows "better offer", use a spare.

## Profile answers for the demo

Pick these on the first-run screen. "Need it soon" turns WAIT off, so never pick it.

| Question | Answer |
|---|---|
| What condition would you buy? | Like new or better |
| How patient are you? | Can wait a week or two |
| Price alerts | On |

If "Like new or better" turns the WAIT page into a better offer, switch to "New only" and use the Kodak page for the better offer step. Decide this the night before.

## The night before

Use the laptop and the Chrome profile you will demo on.

| # | Do this | Check |
|---|---|---|
| 1 | `git pull` on `main`. Then `docker compose up` from the repo root. | `curl localhost:8000/health` returns `ok` |
| 2 | Load every demo product into the database: `curl localhost:8000/predict/<ASIN>` for each ASIN in the table above. | Each returns JSON, not `not_tracked`. Dataset products load from the CSVs on first sight. |
| 3 | Nothing to build: the track record ships with the repo (`backend/analysis/results/track_record/`, #138). | The dashboard shows "How BuyWise has done": 293 products, 65% of Wait calls dropped vs 18% at random. |
| 4 | `chrome://extensions` → Developer mode → Load unpacked → `extension/dist`. The build is committed. Run `cd extension && npm run build` only if extension code changed. | Note the extension ID shown on its card. |
| 5 | Bookmark the first-run screen: `chrome-extension://<extension ID>/profile.html?welcome`. Bookmark the dashboard: `chrome-extension://<extension ID>/dashboard.html`. | Both open |
| 6 | Allow Chrome notifications: macOS System Settings → Notifications → Google Chrome → Allow. Turn off Focus and Do Not Disturb. | A test alert shows up (step 9) |
| 7 | Open every **CHECK** page above. Write down what the panel says and the price. If the panel shows "Based on the offers on this page", click **Compare all offers** and wait for it to update. | Each step has one page that shows the right answer. Replace any that don't. |
| 8 | Pre-seed the watch: open `https://www.amazon.com/dp/B000AJIF4E`. On a WAIT, click **Watch this price**. Otherwise open `https://www.amazon.com/dp/B000AJIF4E?bw_state=wait` and watch from the test panel. | It shows on the dashboard under Watching |
| 9 | Rehearse the alert once on a spare WAIT ASIN (see step 4 of the run). | A Chrome notification appears |
| 10 | Record the backups (list below). | Files are on the demo laptop and on a USB stick or shared drive |
| 11 | Check the website (see step 6 of the run). | Team, numbers and demo date are current |

**After rehearsing**, the extension sends one alert per ASIN and target. At the demo, use a target you haven't used before on that ASIN, or the notification won't fire again.

### Backups to record

| File | What it shows |
|---|---|
| `01-profile.png` | First-run screen with the three answers picked |
| `02-better-offer.png` | Panel on the better-offer page, with seller, price and reason |
| `03-nothing-better.png` | Panel saying "This is the best offer for you." |
| `04-wait.png` | WAIT panel with drop chance, price now and expected low |
| `05-watching.png` | "Watching at $X." after clicking Watch |
| `06-alert.png` | The Chrome notification on screen |
| `07-dashboard.png` | Dashboard: Watching, Recent alerts and How BuyWise has done |
| `08-website.png` | Website home page |
| `demo-run.mp4` | The whole five minutes, recorded during the final rehearsal |

## 10 minutes before

| # | Do this |
|---|---|
| 1 | Docker Desktop running. `docker compose up`. `curl localhost:8000/health` returns `ok`. |
| 2 | Wi-Fi works. Open one Amazon page to be sure it isn't showing a captcha. |
| 3 | Open tabs in order: profile bookmark, better-offer page, nothing-better page, WAIT page, dashboard bookmark, website. Keep the fake-mode URLs in a text file. |
| 4 | Open a terminal at the repo root with the alert command typed but not run. |
| 5 | Notifications allowed. Do Not Disturb off. Close Slack and email so nothing else pops up. |
| 6 | Zoom the browser to 125% so the panel is readable from the back. |
| 7 | Screenshots and the recording open in a Finder window, one click away. |

## The run (about 5 minutes)

### 0. Hook (0:00 – 0:20)

| | |
|---|---|
| Show | A title slide, or any Amazon product page with several sellers |
| Say | "Amazon shows one Add to Cart button, but there are often a dozen sellers behind it. BuyWise reads all of them and tells you the best move." |
| Backup | None needed |

### 1. First-run profile (0:20 – 0:50)

| | |
|---|---|
| Show | Profile bookmark: `chrome-extension://<extension ID>/profile.html?welcome`. Pick the three answers above. Click **Done**. |
| Say | "On install we ask three questions: what condition you'd accept, how patient you are, and whether you want alerts. It's stored on this device only." |
| Backup | `01-profile.png` |

### 2. Better offer (0:50 – 1:50)

| | |
|---|---|
| Show | `https://www.amazon.com/dp/B0D6RQGFGG`. The panel loads top right. If it says "Based on the offers on this page", click **Compare all offers**. Point at the two prices and the seller line. |
| Say | "Amazon's button goes to one seller. BuyWise found a trusted seller with the same item for less, and says why: rating, ratings count, percent positive." |
| Backup 1 | Spare page `https://www.amazon.com/dp/B004R4N2B4` |
| Backup 2 | Test mode: `https://www.amazon.com/dp/B0D6RQGFGG?bw_state=better_offer`. The seller and prices are made up. Say so: "This is our test mode, same screen." |
| Backup 3 | `02-better-offer.png` |

Optional, 15 seconds: open `https://www.amazon.com/dp/B0BS1T9J4Y`. Say: "When Amazon's pick is already best, BuyWise says so and nothing more." Backups: `https://www.amazon.com/dp/B0D9WWB3WX`, then `?bw_state=nothing_better`, then `03-nothing-better.png`.

### 3. Wait and watch (1:50 – 2:50)

| | |
|---|---|
| Show | `https://www.amazon.com/dp/B00378KMFU`. Read the headline: "Wait. N% chance this drops 8% or more in the next 14 days." Click **Watch this price**. Set a target below the price you haven't used before (the box starts at the expected low). Click **Watch**. It says "Watching at $X." **Close this tab.** |
| Say | "Here no seller is cheaper, but the price history says a drop is likely. We only say wait when the expected saving is worth it. I'll set a target and BuyWise will tell me." |
| Backup 1 | Spare WAIT pages `B000AJIF4E`, `B0093162RM` |
| Backup 2 | Test mode: `https://www.amazon.com/dp/B00378KMFU?bw_state=wait`. The numbers are made up, but **Watch** still saves a real watch, because step 2 of the night before loaded this ASIN. |
| Backup 3 | `04-wait.png`, `05-watching.png` |

Close the Amazon tab before step 4. A reload records the real Amazon price again and can overwrite the demo price.

### 4. The alert fires (2:50 – 3:30)

| | |
|---|---|
| Show | In the terminal, with a price at or below your target: `python3 backend/scripts/demo_price_drop.py B00378KMFU <price>`. Open the dashboard bookmark. Click **Check prices now**. The Chrome notification appears. |
| Say | "To skip two weeks of waiting, I'm inserting the lower price the backend would see. The extension checks every hour, and on demand. There's the alert." |
| Backup 1 | Run the same command on the pre-seeded watch: `python3 backend/scripts/demo_price_drop.py B000AJIF4E <price below its target>` |
| Backup 2 | `06-alert.png` |

If the notification doesn't show, the dashboard's **Recent alerts** list still updates. Point at that.

### 5. Dashboard and track record (3:30 – 4:30)

| | |
|---|---|
| Show | Dashboard. **Watching**: both products, one "At your target". **Recent alerts**: the alert just sent. Scroll to **How BuyWise has done**: read the WAIT hit rate next to the base rate, dollars saved, and dollars lost. Then **Your profile**. |
| Say | "Every WAIT the model would have made, replayed against what prices did over the next 14 days. We show the misses too, not only the wins." |
| Backup | `07-dashboard.png` |

Read the numbers off the screen. Don't quote them from memory: they come from the replay run the night before.

### 6. Website (4:30 – 5:00)

| | |
|---|---|
| Show | The project site. Deployed URL: _fill in_. Or locally: `cd website && npm install && npm run dev`, then `http://localhost:3000`. The extension and dashboard preview with synthetic data is at `/preview`. |
| Say | "The site has the research behind this and the team. Questions?" |
| Backup | `08-website.png` |

Before showing it, check `website/content/site.ts`. As of this PR it still has placeholder team cards, says "Keepa price history for 300+ products" and "See it at demo day in December". Fix those or skip the site.

## If everything breaks

| Problem | Do this |
|---|---|
| Backend won't start | Play `demo-run.mp4` and narrate over it |
| Panel shows "Couldn't reach BuyWise" | `docker compose ps`. Restart with `docker compose up`. Then switch to test mode (`?bw_state=...`) for the panel steps. |
| Panel shows "doesn't have price history" | Run `curl localhost:8000/predict/<ASIN>`, then reload. Or switch to a spare. |
| Amazon shows a captcha or a different page | Use the screenshots for that step |
| Panel doesn't appear | Click the BuyWise toolbar icon to reopen it. Then reload the extension at `chrome://extensions` and the tab. |
| No notification | Check Do Not Disturb. Use a new target. Point at Recent alerts on the dashboard. |
| Track record says "No track record yet" | The backend is on an old checkout. Pull main and restart it, or use `07-dashboard.png`. |

## Judge questions

Only quote numbers from this table. Each one is in the repo.

| Question | Answer |
|---|---|
| Where does the price data come from? | Two price-history datasets in the repo: `studies/Prices.csv` and `studies/snapshots/apr2026_500/`, 649 products together. Both end in April 2026. For the demo, each history is shifted so it ends today. The gaps and drops are unchanged. The live page price is also recorded on each visit. |
| So the WAIT is about real prices today? | The history is real but from 2023–2026, shifted to today. The live system needs a fresh price feed. That's the next step after demo day. |
| How is WAIT decided? | A CatBoost model gives the chance of an 8%+ drop in the next 14 days. The chance is calibrated: after calibration, on calls where it says 40% or more, it says 56% on average and 56% happen (Study 022). We show WAIT only if that chance is at least 15% **and** the expected net saving is at least $2. The saving counts what a wrong WAIT costs on items as volatile as this one (Study 024, `backend/analysis/results/v2.json`). |
| How good is it? | On the test period, with thresholds picked on separate validation data: it stays quiet on 81% of decisions. When it says WAIT it's right 52% of the time. Over 14 days, 8%+ drops happen 21% of the time (Study 023). A WAIT saves $19.60 on average if you buy on day 14 (`v2.json`). This rule beats a plain 40% drop-probability cut-off by $0.55 per decision, interval [$0.29, $0.88] (Study 024). |
| What happens when it's wrong? | 48% of WAITs are wrong. In 20% of those wrong calls the price went up instead, and `v2.json` puts the extra paid at $3.58. The dashboard shows misses next to wins. |
| How are sellers ranked? | T4's ranker. It compares total price with shipping. It discounts used items by condition (Like New counts as 13% less valuable, Very Good 30%). It needs a saving of at least $2 or 3%, whichever is larger. The seller needs 4.0+ stars, 100+ ratings and 90%+ positive feedback. |
| How do you know the seller ranking is right? | The team hand-checked 50 live product pages and picked the best offer on each. The ranker agreed on 50 of 50, and on 42 of 42 pages that had at least two offers (PR #132). The condition discounts and minimum saving were tuned on the first 25 of those pages. |
| Why focus on other sellers and not only waiting? | On the same products and moments, the cheapest other offer averaged $18.47 below the buy-box price (Study 001). Waiting pays mostly on expensive, volatile items. |
| What was tested? | Backend tests and builds of the extension and website run on every PR. `main` only accepts changes that pass. Seller ranking was hand-checked on 50 pages. The WAIT rule was picked on validation data and judged once on test data. |
| What about my privacy? | The profile stays in the browser. The backend uses it for one answer and never stores it. Each install gets a random ID. There are no accounts. |
| Why "Used - Like New"? | Because the shopper said they'd take like-new. A "New only" shopper never sees used offers. The panel also warns that the manufacturer warranty may not apply. |
| What's next? | A fresh price source to replace the 2026 datasets, and a longer track record of calls shoppers actually saw. |

Don't describe the model's accuracy in any way the table above doesn't. If a judge asks for a number that isn't here, say you'll follow up.
