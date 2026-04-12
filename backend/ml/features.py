# features.py

# Columns to drop entirely — metadata, not features
DROP_COLS = [
    "date", "asin", "brand", "keepa_minutes",
    "new_price", "used_price", "list_price",   # redundant with price_vs_* ratios
    "label_7d", "label_14d",                   # targets
]

# Short-term volatility signal
VOLATILITY_FEATURES = [
    "roll7_std", "roll14_std", "roll30_std",
    "roll7_min", "roll7_max",
    "roll14_min", "roll14_max",
    "roll30_min", "roll30_max",
    "pct_change_1d", "pct_change_14d",
    "amazon_pct_change_7d",
    "amazon_delta_7d",
]

# Lag and trend
LAG_FEATURES = [
    "price", "price_lag1", "price_lag30",
    "amazon_lag_7d", "amazon_lag_14d",
    "amazon_ma_7d", "amazon_ma_14d",
    "roll14_mean", "roll30_mean",
]

# Price position — where current price sits relative to history
POSITION_FEATURES = [
    "price_vs_7d_mean", "price_vs_14d_mean", "price_vs_30d_mean",
    "price_vs_global_min", "price_vs_global_max", "price_vs_global_mean",
    "price_vs_new", "price_vs_used", "price_vs_list",
    "global_norm_sd", "log_mean_price", "global_min", "global_max", "global_mean",
]

# Long-term seasonal patterns
SEASONAL_FEATURES = [
    "day_of_week", "month", "week_of_year", "is_weekend",
]

# Demand / inventory context
DEMAND_FEATURES = [
    "sales_rank", "sales_score",
    "count_new", "count_used",
]

# Combine for model input
FEATURE_COLS = (
    VOLATILITY_FEATURES
    + LAG_FEATURES
    + POSITION_FEATURES
    + SEASONAL_FEATURES
    + DEMAND_FEATURES
)

TARGET_7D  = "label_7d"
TARGET_14D = "label_14d"
