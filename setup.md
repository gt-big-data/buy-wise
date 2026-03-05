# BuyWise: Technical Architecture Guide

## Project Summary

**What**: Chrome extension that predicts Amazon price drops using ML time-series forecasting  
**Why**: Help consumers avoid impulsive purchases by predicting if waiting will save money  
**How**: Historical price data → ML model → BUY/WAIT recommendation shown at checkout

**Core Value Proposition**: Proactive predictions (when prices *will* drop) vs reactive tracking (when prices *did* drop)

---

## System Architecture
```
┌─────────────────────────────────────────────────┐
│           CHROME EXTENSION (React)              │
│  - Detects Amazon product pages (/dp/[ASIN])    │
│  - Calls backend API with ASIN                  │
│  - Renders prediction overlay on page           │
└────────────────────┬────────────────────────────┘
                     │
                     ↓ HTTPS
┌─────────────────────────────────────────────────┐
│           FASTAPI BACKEND (Python)              │
│  - /predict/{asin} - main prediction endpoint   │
│  - /product-info/{asin} - metadata              │
│  - /activity - logs user actions                │
│  - Orchestrates cache/DB/ML pipeline            │
└───────┬─────────────────┬───────────────────────┘
        │                 │
        ↓                 ↓
┌──────────────┐   ┌─────────────────────────────┐
│ REDIS CACHE  │   │     POSTGRESQL DB           │
│ - 1hr TTL    │   │  - products (metadata)      │
│ - Popular    │   │  - prices (time-series)     │
│   predictions│   │  - predictions (forecasts)  │
└──────────────┘   └─────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ↓                         ↓
            ┌──────────────┐         ┌───────────────┐
            │   KEEPA API  │         │   ML MODELS   │
            │ - Historical │         │  - XGBoost    │
            │   price data │         │  - Feature    │
            │ - Daily sync │         │    engineering│
            └──────────────┘         └───────────────┘
```

---

## Data Flow

### User Interaction Flow
1. User visits `amazon.com/product-name/dp/B08N5WRWNW/...`
2. Extension extracts ASIN (`B08N5WRWNW`) from URL
3. Extension calls `GET /predict/B08N5WRWNW`
4. Backend checks Redis cache for recent prediction
5. If cache miss: query DB → run ML model → cache result
6. Return JSON: `{recommendation: "WAIT", confidence: 0.87, predicted_savings: 23.50, ...}`
7. Extension injects overlay above "Add to Cart" button

### Data Collection Flow (Background Job)
1. Daily cron job queries Keepa API for 500 product ASINs
2. Parse response to extract: `{asin, timestamp, price, availability}`
3. Insert into `prices` table as new time-series rows
4. Trigger model retraining weekly on new data

---

## Component Responsibilities

### Extension (Visualization Team)
**Build**: Chrome extension using Manifest V3, React for UI  
**Responsibilities**:
- Detect Amazon product pages via URL pattern matching
- Extract ASIN from URL path
- Call backend API (handle loading states, errors)
- Render prediction overlay with recommendation
- Provide "Add to watchlist" and "Dismiss" actions
- Display historical price graph (Chart.js)

**Key Challenges**:
- Keep overlay responsive (< 500ms display time)
- Handle API failures gracefully
- Ensure compatibility across different Amazon page layouts

### Backend (Platform Team)
**Build**: FastAPI application with PostgreSQL and Redis  
**Responsibilities**:
- Expose REST API for predictions and product info
- Implement caching strategy (Redis with 1hr TTL)
- Manage database connections and queries
- Orchestrate ML model inference
- Collect data from Keepa API (rate limit: 1000 requests/day)
- Log user actions for model validation

**Key Challenges**:
- Minimize latency (target: 200-400ms per prediction)
- Handle concurrent requests efficiently
- Manage Keepa API rate limits
- Ensure database queries scale to 1000+ products

### ML Pipeline (Analysis Team)
**Build**: Time-series forecasting models + classification layer  
**Responsibilities**:
- Engineer features from historical prices (rolling averages, lags, volatility)
- Train XGBoost/LightGBM regressor to predict prices at t+7, t+14 days
- Build classification layer to convert predictions into BUY/WAIT recommendations
- Generate confidence scores based on prediction variance
- Validate predictions against actual outcomes (backtest weekly)

**Key Challenges**:
- Handle products with sparse/irregular price history
- Account for seasonal patterns (Prime Day, Black Friday, holidays)
- Balance precision (don't false alarm) vs recall (catch real deals)
- Set appropriate thresholds (e.g., only recommend WAIT if >8% savings expected)

---

## Database Schema

### Products Table
```sql
CREATE TABLE products (
    asin VARCHAR(10) PRIMARY KEY,
    title TEXT NOT NULL,
    category VARCHAR(100),
    brand VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Prices Table (Time-Series)
```sql
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(10) REFERENCES products(asin),
    price DECIMAL(10, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    availability BOOLEAN DEFAULT TRUE,
    INDEX idx_asin_timestamp (asin, timestamp)
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(10) REFERENCES products(asin),
    prediction_date TIMESTAMP DEFAULT NOW(),
    current_price DECIMAL(10, 2),
    predicted_price_7d DECIMAL(10, 2),
    predicted_price_14d DECIMAL(10, 2),
    confidence DECIMAL(3, 2),
    recommendation VARCHAR(10),  -- 'BUY' or 'WAIT'
    INDEX idx_asin_date (asin, prediction_date)
);
```

**Indexing Strategy**: Compound index on (asin, timestamp) for fast time-series queries

---

## Machine Learning Architecture

### Feature Engineering
Transform raw price history into ML-ready features:
```
Raw prices: [120, 125, 119, 130, 128, ...]
                    ↓
Features for model:
- current_price: 128
- price_7d_avg: 124.3
- price_14d_avg: 122.8
- price_30d_avg: 121.5
- volatility_7d: 4.2
- rate_of_change: +0.06
- lag_1: 130
- lag_7: 125
- days_since_last_drop: 12
- month: 11 (November - holiday season)
- day_of_week: 2 (Tuesday)
- days_until_black_friday: 8
```

### Model Pipeline

**Step 1: Regression (Price Forecasting)**
- Model: XGBoost regressor or LightGBM
- Target: Predict price at t+7 and t+14 days
- Training: Rolling window approach (use past to predict future, never train on future data)
- Output: `predicted_price_7d, predicted_price_14d, prediction_variance`

**Step 2: Classification (BUY/WAIT Decision)**
- Calculate: `potential_savings = current_price - predicted_price`
- Calculate: `savings_percentage = potential_savings / current_price`
- Apply threshold: If `savings_percentage > 0.08` AND `confidence > 0.75` → WAIT, else BUY
- Confidence score: Based on prediction variance and historical model accuracy

**Step 3: Validation (Backtesting)**
- Every week: Compare old predictions vs actual outcomes
- Track metrics: Precision (when we said WAIT, did price drop?), False positive rate
- Adjust thresholds to maintain >80% accuracy

---

## API Specification

### GET /predict/{asin}
**Purpose**: Get BUY/WAIT recommendation for a product

**Response**:
```json
{
  "asin": "B08N5WRWNW",
  "recommendation": "WAIT",
  "confidence": 0.87,
  "current_price": 149.99,
  "predicted_price": 125.00,
  "predicted_savings": 24.99,
  "days_to_wait": 14,
  "explanation": "Based on 2 years of data, this product typically drops 18% in mid-November"
}
```

### GET /product-info/{asin}
**Purpose**: Get product metadata

**Response**:
```json
{
  "asin": "B08N5WRWNW",
  "title": "Sony WH-1000XM4 Wireless Headphones",
  "category": "Electronics",
  "brand": "Sony"
}
```

### POST /activity
**Purpose**: Log user actions for model validation

**Request**:
```json
{
  "asin": "B08N5WRWNW",
  "action": "added_to_watchlist",  // or "dismissed", "purchased"
  "recommendation_shown": "WAIT",
  "timestamp": "2025-11-15T14:30:00Z"
}
```

---

## Caching Strategy

### Why Cache?
- ML inference takes 200-400ms per product
- Popular products (AirPods, PS5) get requested frequently
- Goal: Serve 90% of requests from cache

### Implementation
```python
# Redis key structure: "prediction:{asin}"
cache_key = f"prediction:{asin}"
cached = redis.get(cache_key)

if cached:
    return json.loads(cached)  # Cache hit - instant response
else:
    prediction = run_ml_model(asin)  # Cache miss - run model
    redis.setex(cache_key, 3600, json.dumps(prediction))  # Cache for 1 hour
    return prediction
```

### Cache Invalidation
- TTL: 1 hour (prices don't change rapidly enough to need real-time updates)
- Manual: Clear cache when new price data arrives (daily batch job)

---

## External API Integration

### Keepa API
**Purpose**: Retrieve historical Amazon price data

**Key Endpoints**:
- `GET /product?key={API_KEY}&domain=1&asin={ASIN}` - Get product data
- Returns: Historical prices as CSV array, current price, sales rank

**Rate Limits**:
- Free tier: 60 requests/month
- Paid tier: Starting at $20/month for 1000 requests/day
- Strategy: Use paid tier, collect data for 500 products daily

**Data Format**:
```python
{
  "asin": "B08N5WRWNW",
  "title": "Sony WH-1000XM4...",
  "csv": [  # Time-series data
    1699920000, 14999,  # [timestamp_minutes, price_cents]
    1700006400, 14999,
    1700092800, 12999,  # Price dropped!
    ...
  ]
}
```

### Amazon Product Advertising API (Stretch Goal)
**Purpose**: Get availability, reviews, deal status  
**Use Case**: Improve predictions by knowing when items are low stock or on limited-time deals

---

## Deployment Strategy

### Development Environment
- Backend: Local FastAPI server (`uvicorn main:app --reload`)
- Database: Local PostgreSQL or Supabase free tier
- Cache: Local Redis instance
- Extension: Load unpacked in Chrome developer mode

### Production Environment (Future)
- Backend: Deploy to AWS Lambda or Google Cloud Run
- Database: Managed PostgreSQL (Supabase, AWS RDS)
- Cache: Redis Cloud or AWS ElastiCache
- Extension: Publish to Chrome Web Store

---

## Success Metrics

### Technical Metrics
- **Latency**: 95th percentile response time < 500ms
- **Accuracy**: >80% of WAIT recommendations result in actual price drops
- **Cache Hit Rate**: >85% of requests served from cache
- **Uptime**: 99.5% API availability

### Product Metrics
- **User Savings**: Track total $ saved by users following recommendations
- **Engagement**: % of users who return after first use
- **Conversion**: % of WAIT recommendations where users add to watchlist

---

## MVP Scope (4 months)

### Must Have
- Extension detects Amazon product pages and displays predictions
- Backend API serves predictions with <500ms latency
- ML model trained on 500 electronics products with >75% accuracy
- Caching implemented for popular products
- Database stores 6+ months of historical price data

### Nice to Have
- Dashboard showing user's saved money and past recommendations
- Price history graph visualization
- Email/browser notifications when watchlist prices drop

### Deferred (Future)
- Multi-product optimization (knapsack problem)
- SHAP explanations for feature importance
- Expansion to categories beyond electronics
- Cross-retailer support (Walmart, Target, Best Buy)

---

## Key Technical Decisions

### Why XGBoost/LightGBM over Neural Networks?
- Smaller data requirements (500 products, not millions)
- Faster inference (critical for <500ms latency)
- Easier interpretability for confidence scores
- Better handling of tabular time-series features

### Why PostgreSQL over NoSQL?
- Need structured relationships (products → prices)
- Time-series queries benefit from SQL indexing
- ACID guarantees for prediction logging
- Easier to enforce data integrity

### Why Redis over In-Memory?
- Shared cache across multiple backend instances (horizontal scaling)
- Persistent cache survives server restarts
- Built-in TTL expiration
- Production-ready monitoring tools

### Why Keepa API over Web Scraping?
- Legal compliance (avoids ToS violations)
- Reliability (no bot detection to bypass)
- Historical data access (4+ years available immediately)
- Maintained by dedicated team

---

## Implementation Priority

### Phase 1: Foundation (Month 1)
- Set up PostgreSQL database with schema
- Build FastAPI endpoints with mock data
- Create Chrome extension that detects product pages
- Integrate Keepa API and collect first batch of data

### Phase 2: ML Pipeline (Month 2)
- Implement feature engineering from raw price data
- Train baseline XGBoost model on collected data
- Build classification layer with conservative thresholds
- Connect model to backend API (replace mock data)

### Phase 3: Optimization (Month 3)
- Add Redis caching layer
- Optimize database queries with proper indexing
- Improve model with more sophisticated features
- Implement backtesting workflow

### Phase 4: Polish (Month 4)
- Build user dashboard showing savings
- Add price history visualization
- Implement watchlist functionality
- Prepare Chrome Web Store submission

---

## Risk Mitigation

### Data Risk: Keepa API costs exceed budget
- **Mitigation**: Start with 500 products, use free trial initially
- **Fallback**: Reduce product count or switch to CamelCamelCamel's free data

### Model Risk: Predictions are inaccurate
- **Mitigation**: Set high confidence thresholds (only show when >75% confident)
- **Validation**: Weekly backtesting to catch accuracy drift early

### Technical Risk: Extension doesn't work on all Amazon pages
- **Mitigation**: Test on multiple product categories (electronics, books, home goods)
- **Fallback**: Graceful degradation - show generic message if prediction fails

### Legal Risk: Amazon blocks extension
- **Mitigation**: Extension only reads public data, doesn't modify functionality
- **Compliance**: Follow Chrome Web Store policies, don't scrape Amazon directly

---

## Team Structure

### Platform Team (3-4 people)
- Backend API development (FastAPI)
- Database design and optimization
- Keepa API integration
- Caching implementation
- DevOps and deployment

### Analysis Team (3-4 people)
- Feature engineering
- Model training and evaluation
- Backtesting infrastructure
- Threshold tuning
- Performance monitoring

### Visualization Team (3 people)
- Chrome extension development
- UI/UX design
- Frontend API integration
- User testing and feedback

---

## Testing Strategy

### Backend Testing
- Unit tests for API endpoints (pytest)
- Integration tests for DB + cache + ML pipeline
- Load testing (can handle 100 concurrent requests?)

### ML Testing
- Validation set performance (never train on this data)
- Backtesting (did past predictions match reality?)
- Edge case testing (new products, price spikes, stockouts)

### Extension Testing
- Manual testing on different Amazon page layouts
- Cross-browser compatibility (Chrome, Edge, Brave)
- Performance testing (does overlay slow down page load?)

---

## Documentation Requirements

### For Team
- API documentation (automatically generated by FastAPI)
- Database schema diagram
- Model training notebooks with explanations
- Deployment runbook

### For Users
- Extension FAQ ("How does BuyWise work?")
- Privacy policy (what data we collect)
- Accuracy disclaimer (predictions aren't guarantees)

---

## Questions to Answer During Build

1. **How granular should price timestamps be?** Daily? Hourly? (affects DB size)
2. **What confidence threshold minimizes false positives?** 75%? 80%? 85%?
3. **Should we use separate models per category?** Or one universal model?
4. **How long should cache TTL be?** 1 hour? 6 hours? 24 hours?
5. **What's the minimum historical data needed?** 3 months? 6 months? 1 year?

---

## Expected Challenges

### Technical
- Keepa API rate limits force careful request batching
- Database queries slow down with millions of price records
- ML model struggles with new products (no historical data)
- Extension breaks when Amazon redesigns product pages

### Organizational
- Analysis team blocked if Platform hasn't finished data collection
- Different skill levels across team members
- Scope creep (adding features instead of finishing MVP)

### Product
- Users don't trust AI recommendations initially
- Hard to prove value before accumulating real user savings
- Difficult to balance false positives (say WAIT when shouldn't) vs false negatives (say BUY when should wait)

---

## Success Looks Like

**After 4 months (MVP launch)**:
- Extension available on Chrome Web Store
- 500 products tracked with 6+ months of data
- Model achieves >75% accuracy on validation set
- 10-20 beta users testing and providing feedback

**After 8 months (Public launch)**:
- 1000+ products across multiple categories
- Model achieves >80% accuracy
- Caching reduces 90% of latency
- 100+ active users, $5000+ in tracked savings

**After 12 months (Scale)**:
- 5000+ products including non-electronics
- Advanced features (watchlist optimization, SHAP explanations)
- 1000+ active users, $20,000+ in tracked savings
- Revenue model (freemium, API access for developers)

---

## Tech Stack Summary

| Component | Technology | Why |
|-----------|-----------|-----|
| Extension | React + Manifest V3 | Standard for modern Chrome extensions |
| Backend | FastAPI + Python | Fast async framework, great ML ecosystem |
| Database | PostgreSQL | Relational time-series data, ACID guarantees |
| Cache | Redis | Fast in-memory KV store, built-in TTL |
| ML Framework | XGBoost/LightGBM | Excellent for tabular time-series |
| Data Source | Keepa API | Legal, reliable, comprehensive |
| Hosting | Local → Cloud (future) | Dev locally, scale when needed |
