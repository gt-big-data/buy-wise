# BuyWise Backend Mock API

This backend provides a FastAPI skeleton with mocked data endpoints for the Chrome extension integration.

## Endpoints

- `GET /health` - basic liveness check
- `GET /predict/{asin}` - returns mock recommendation data (recommendation, confidence, predicted_price, potential_savings, horizon_days)
- `GET /product-info/{asin}` - returns mock product metadata (title, brand, category, price, rating...)
- `POST /activity` - accepts user's action on recommendation, returns acknowledgment

## Run

1. Create and activate Python venv (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run server:

```bash
uvicorn main:app --reload
```

4. Open docs:

- Interactive docs: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

## Testing Quickcalls

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/predict/B08N5WRWNW
curl http://127.0.0.1:8000/product-info/B08N5WRWNW
curl -X POST http://127.0.0.1:8000/activity -H "Content-Type: application/json" -d '{"asin":"B08N5WRWNW","action":"clicked","user_id":"u1"}'
```
