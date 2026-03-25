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
| GET | `/product-info/{asin}` | Product metadata (title, brand, category, price, rating) |
| POST | `/activity` | Log a user action on a recommendation |

All responses use Pydantic models — see `/docs` for full schemas.

---

## Database Layer (`db/`)

**`schema.sql`** — Three tables: `products`, `prices`, `predictions`. Run this from scratch to initialize. Safe to re-run (uses `DROP IF EXISTS`).

**`seed.sql`** — 5 fake products with price history and predictions. Useful for local development without real Keepa data.

**`connection.py`** — The only file the rest of the backend should import for DB access. Uses a connection pool (size 5).

| Function | Description |
|---|---|
| `get_product(asin)` | Returns product row dict or None |
| `insert_product(asin, title, brand, category)` | Upserts a product |
| `insert_price(product_id, price, availability, deal_flag)` | Inserts a price record |
| `get_price_history(product_id, limit=100)` | Returns price rows ordered by timestamp DESC |
| `insert_prediction(product_id, pred_7d, pred_14d, pred_30d, recommendation, confidence)` | Inserts a prediction |
| `get_latest_prediction(product_id)` | Returns the most recent prediction row or None |

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
