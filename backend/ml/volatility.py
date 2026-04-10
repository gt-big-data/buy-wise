import numpy as np
from dataclasses import dataclass

@dataclass
class VolatilityResult:
    value: float
    std_dollars: float
    window: int
    sufficient_data: bool

def compute_volatility(
    price_history: list[dict],
    window_days: int = 365
) -> VolatilityResult:
    """
    Computes relative price volatility from get_price_history() output.



Expected input:
    price_history = [
        {
            "price": 149.99,
            "availability": True,
            "timestamp": datetime(...)
        },
        ...
    ]

Important assumptions:
    - input rows are newest first
    - timestamps are valid datetime objects
    - only available prices are used
    - 365-day window is recommended for seasonality

Returns:
    VolatilityResult:
        value:
            relative volatility ratio
            (std(price) / mean(price))

        std_dollars:
            raw standard deviation in dollar units

        window:
            number of observations used

        sufficient_data:
            True if >= 365 historical rows exist
"""
    
    if not price_history or len(price_history) < 2:
        return VolatilityResult(
            value=0.05,
            std_dollars=0.0,
            window=0,
            sufficient_data=False
        )

    available = [
        row for row in price_history
        if row["availability"]
    ]

    if len(available) < 2:
        return VolatilityResult(
            value=0.05,
            std_dollars=0.0,
            window=0,
            sufficient_data=False
        )

    sorted_history = sorted(available, key=lambda x: x["timestamp"])

    recent = sorted_history[-window_days:]

    prices_arr = np.array([row["price"] for row in recent])
    mean_price = np.mean(prices_arr)
    std_dollars = float(np.std(prices_arr, ddof=1))
    relative_vol = std_dollars / mean_price if mean_price > 0 else 0.05

    return VolatilityResult(
        value=round(relative_vol, 4),
        std_dollars=round(std_dollars, 2),
        window=len(recent),
        sufficient_data=len(sorted_history) >= window_days
    )