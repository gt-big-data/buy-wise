import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY = os.getenv("KEEPA_API_KEY")


def fetch_price_history(asin):
    url = "https://api.keepa.com/product"

    params = {
        "key": API_KEY,
        "domain": 1,
        "asin": asin,
        "history": 1,
        "days": 30
    }

    while True:
        res = requests.get(url, params=params)
        data = res.json()

        if "products" not in data:
            if "refillIn" in data:
                time.sleep(data["refillIn"] / 1000)
                continue
            else:
                raise Exception(data)

        product = data["products"][0]
        break

    csv_data = product.get("csv", [])
    if len(csv_data) > 0:
        new_prices = csv_data[0]
    else:
        new_prices = []
    
    name = product.get("title", "Unknown Product")
    records = []

    for i in range(0, len(new_prices), 2):
        t = new_prices[i]
        price = new_prices[i + 1]

        if price == -1:
            continue

        records.append({
            "asin": asin,
            "name": name,
            "timestamp": (t + 21564000) * 60,
            "price": price / 100.0
        })

    return records

# Next step is to write these records into the database