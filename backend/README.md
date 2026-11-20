# BuyWise Backend

## Overview

The backend is a FastAPI server backed by MySQL. It has three layers:

| Layer | Location | Purpose |
|---|---|---|
| API server | `main.py` | Chrome extension calls these endpoints |
| DB layer | `db/connection.py` | All MySQL access goes through here |
| Price data | `jobs/csv_history.py` | Loads a product's price history from the datasets in `studies/` (649 products) and writes it to the DB |

---

## Setup

From the repo root, with [Docker Desktop](https://www.docker.com/products/docker-desktop) running:

```bash
docker compose up
```

This starts MySQL, loads `db/schema.sql`, `db/seed.sql` and `db/seed_real.sql`, and runs the server on http://localhost:8000 (interactive docs at `/docs`). `backend/` is mounted into the container, so edits reload without a rebuild. After changing `requirements.txt`, run `docker compose up --build`.

The SQL loads only into an empty database. `docker compose down -v` wipes it so the next `up` starts fresh.

`studies/` is mounted read-only at `/studies`, so the backend can load any of the 649 products in the price datasets. No Keepa key is needed: BuyWise no longer uses Keepa.

### Running tests

Tests run on your machine against the MySQL from `docker compose up`:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults point at the compose database
python -m pytest tests test_asin.py -q
```

The same tests run on every pull request (`.github/workflows/ci.yml`).

### Running the server outside Docker

Keep `docker compose up mysql` running for the database, then from `backend/` with the venv active: `uvicorn main:app --reload`. Stop the `backend` container first so port 8000 is free.

---

## API Endpoints (`main.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/predict/{asin}` | Recommendation, confidence, predicted price, potential savings, horizon |
| GET | `/price-history/{asin}` | Weekly-bucketed price history + 30-day forecast points for the chart |
| GET | `/product-info/{asin}` | Product metadata (title, brand, category, current price) |
| POST | `/activity` | Log a user action on a recommendation |
| POST | `/watchlist` | Add a product to a user's watchlist, records recommendation at time of adding |
| DELETE | `/watchlist/{user_id}/{product_id}` | Remove a product from a user's watchlist |
| GET | `/watchlist/{user_id}` | Retrieve full watchlist with current recommendations and `recommendation_changed` flag |

> **Note:** The `users` table has not yet been implemented. `user_id` is accepted as a plain integer with no foreign key constraint until the user authentication feature is complete.


All responses use Pydantic models — see `/docs` for full schemas.

### Fetch-on-demand behavior

The first time `/predict/{asin}` or `/decision` sees an ASIN that isn't in the database, the server:

1. Loads its price history from `studies/snapshots/apr2026_500/Prices.csv.gz`, else `studies/Prices.csv` (`jobs/csv_history.py`). Both end in April 2026, so each history is shifted to end today; gaps and drops are unchanged.
2. Scores it with the served model (`ml/buywise_clean.joblib`) and stores the prediction. With too little history (under ~30 days) it stores nothing rather than a made-up forecast.

Every one of the 649 products answers. A product outside the datasets still gets its sellers compared; it just never gets a WAIT.

### Where the decision's rules come from

`/decision` reads its thresholds from `analysis/results/v<N>.json` (the newest), built by `scripts/build_analysis_results.py`:

- **When to show WAIT:** R4 (#81, Study 024), the 14-day rule: calibrated drop chance at least 15% and expected net saving at least $2, where the saving counts what a wrong WAIT costs on products as volatile as this one.
- **Which other seller to recommend:** T4 (#86, `ml/offer_ranking.py`): condition discounts (a "Very Good" copy is worth 30% less than new), a minimum saving of max($2, 3%), and T4's score, which agreed with the hand checks on 50 of 50 pages. Sellers must also show at least 4.0 stars, 100 ratings and 90% positive (#129).
- **Version verdicts:** T2 (#84).

---

## Database Layer (`db/`)

**`schema.sql`** — Four tables: `products`, `prices`, `predictions`, `watchlist`. Run this from scratch to initialize. Safe to re-run (uses `DROP IF EXISTS`).

**`migrations/`** — incremental changes for databases that already exist. `schema.sql` already includes them, so you only need these if you set up before the change landed. Run in order:

```bash
docker compose exec -T mysql mysql -uroot -proot buywise < backend/db/migrations/001_prices_keepa_columns.sql
docker compose exec -T mysql mysql -uroot -proot buywise < backend/db/migrations/002_watchlist_user_id_string.sql
```

**`seed.sql`** — 5 fake products with price history and predictions. Useful for local development without real Keepa data.

**`connection.py`** — The only file the rest of the backend should import for DB access. Uses a connection pool (size 5).

| Function | Description |
|---|---|
| `get_product(asin)` | Returns product row dict or None |
| `get_product_by_id(product_id)` | Returns product row dict by product_id or None |
| `insert_product(asin, title, brand, category)` | Upserts a product |
| `insert_price(product_id, price, availability, deal_flag)` | Inserts a price record |
| `get_price_history(product_id, limit=100)` | Returns price rows ordered by timestamp DESC |
| `insert_prediction(product_id, pred_7d, pred_14d, pred_30d, recommendation, confidence)` | Inserts a prediction |
| `get_latest_prediction(product_id)` | Returns the most recent prediction row or None |
| `add_to_watchlist(user_id, product_id, recommendation_at_add, target_price)` | Adds a product to a user's watchlist |
| `remove_from_watchlist(user_id, product_id)` | Removes a product from a user's watchlist |
| `get_watchlist(user_id)` | Returns all watchlist items with product info, latest prediction, and `recommendation_changed` boolean |

---

## Keepa Integration (`jobs/keepa_fetch.py`)

Fetches Amazon price history for a given ASIN from the Keepa API and writes it to the DB.

**Usage:**

```python
from jobs.keepa_fetch import fetch_price_history

records = fetch_price_history("B08N5WRWNW")
# records is also written to the DB automatically
```

Each record in the returned list has:

```python
{
    "asin": "B08N5WRWNW",
    "name": "Product Title",
    "timestamp": 1700000000,   # Unix seconds
    "price": 29.99             # USD
}
```

**Rate limiting:** Keepa uses a token bucket. When the bucket is empty the API returns a `refillIn` field (milliseconds until refill). `fetch_price_history` will sleep and retry up to 3 times before raising a `RuntimeError`. The paid key refills fast enough that retries should rarely trigger under normal load. If they do, the caller should back off and reschedule rather than retrying in a tight loop.

**Timestamp encoding:** Keepa stores timestamps as minutes since 2011-01-01 UTC. The module converts these to Unix seconds transparently — all records returned have standard Unix timestamps.

### Live price observations and alert demo (#74)

`POST /prices` accepts `{"asin":"B08N5WRWNW","price":99.99}` and returns
201 after saving a timestamped price row. The product must already be tracked
(404 otherwise). ASINs must be ten uppercase letters/digits; USD prices must be
positive and finite. Database write failures return an error rather than success.

The extension reports `extractOffers().oneTimePrice` through its background
worker when an Amazon.com product page opens, including delayed buy boxes and
variant changes. It retries briefly when prices or backend data aren't ready.
After a successful write it checks alert targets immediately. Hourly checks,
startup checks, and the dashboard's **Check prices now** also read the latest
stored price. This does not scrape Amazon in the background: products that no
one visits still depend on the existing data jobs for fresh prices.

To demo against your local database:

1. Start `docker compose up` and load the built extension in Chrome. Allow Chrome
   notifications in your operating system.
2. Visit a tracked product, set an alert target below its current price, and open
   the extension dashboard. Close that Amazon tab so it won't report a real price
   over the demo value.
3. From the repository root run (substitute the same ASIN and a price below the target):
   `python3 backend/scripts/demo_price_drop.py B08N5WRWNW 49.99`.
4. Click **Check prices now** in the dashboard. Its current price updates and a
   Chrome notification appears via the normal alert checker. The script itself
   only inserts the price; Chrome sends the notification.
5. Repeat checks to confirm there is no duplicate notification. For another run,
   choose a different target: the extension remembers one notification per ASIN
   and target. Revisit the real product to record its actual price again. Demo
   rows remain in history, so use a local demo database.
