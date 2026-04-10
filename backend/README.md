# BuyWise Backend

## Overview

The backend is a FastAPI server backed by MySQL. It has three layers:

| Layer | Location | Purpose |
|---|---|---|
| API server | `main.py` | Chrome extension calls these endpoints |
| DB layer | `db/connection.py` | All MySQL access goes through here |
| Data jobs | `jobs/keepa_fetch.py` | Fetches price history from Keepa and writes to DB |

---

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Database Setup: Docker

Run MySQL locally using Docker to avoid installation issues and ensure a consistent environment across all teammates.

1. **Install Docker:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)

2. **Start MySQL container:**
```bash
docker run --name buywise-mysql \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=buywise \
  -p 3306:3306 \
  -d mysql:8
```

3. **Verify container is running:**
```bash
docker ps
```
*(You should see a container named `buywise-mysql`)*

4. **Initialize the database:**
```bash
docker exec -i buywise-mysql mysql -uroot -proot buywise < db/schema.sql
docker exec -i buywise-mysql mysql -uroot -proot buywise < db/seed.sql
```

5. **Update `.env`:**
Copy `.env.example` to `.env` and use these local Docker credentials:
```text
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=root
DB_NAME=buywise
KEEPA_API_KEY=your_keepa_key
```

*Optional: Stop / restart container*
```bash
docker stop buywise-mysql
docker start buywise-mysql
```

3. Copy `.env.example` to `.env` and fill in your values:

```
DB_HOST=localhost
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=buywise
KEEPA_API_KEY=your_keepa_key
```

4. Initialize the database:

```bash
mysql -u root -p < db/schema.sql
mysql -u root -p buywise < db/seed.sql
```

5. Start the server:

```bash
uvicorn main:app --reload
```

Interactive docs available at http://127.0.0.1:8000/docs

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

If `/predict/{asin}` or `/price-history/{asin}` is called for an ASIN not yet in the database, the server automatically:

1. Calls `keepa_fetch(asin)` to pull real price history from Keepa and write it to the DB
2. Runs a stub trend-based prediction (`_fetch_and_seed` in `main.py`) and inserts it

This means the extension works on any Amazon product page without pre-seeding. The DB is populated on first hit.

### Stub prediction (temporary)

The real ML model (XGBoost/LightGBM) has not been built yet. Until the Analysis team delivers it, `_fetch_and_seed` generates predictions using a simple heuristic:

- Compute `diff = (current_price − 30-day average) / 30-day average`
- If `diff > 0.05` (current price is 5%+ above its own history): **WAIT**, predicted prices trend back toward average over 7/14/30 days
- Otherwise: **BUY**, predicted prices drift slightly upward (2% / 4% / 6%)
- Confidence = `min(90%, 55% + |diff|%)`

When the real model is ready, it should call `insert_prediction(...)` with its own values — the latest prediction row is always what the extension sees, so the stub row gets naturally superseded.

---

## Database Layer (`db/`)

**`schema.sql`** — Four tables: `products`, `prices`, `predictions`, `watchlist`. Run this from scratch to initialize. Safe to re-run (uses `DROP IF EXISTS`).

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
