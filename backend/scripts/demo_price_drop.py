"""Write a demo price through the real API; no DB credentials or Keepa key needed."""
import argparse
import json
import math
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asin", help="An already tracked ASIN")
    parser.add_argument("price", type=float, help="Demo USD price, at or below your alert target")
    parser.add_argument("--backend", default="http://localhost:8000")
    args = parser.parse_args()
    if not math.isfinite(args.price) or not 0 < args.price <= 99999999.99:
        parser.error("price must be positive, finite, and fit in the database")
    request = Request(
        f"{args.backend.rstrip('/')}/prices",
        data=json.dumps({"asin": args.asin.upper(), "price": args.price}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        print(response.read().decode())
    print("In the BuyWise extension dashboard, click 'Check prices now' to send the Chrome alert.")


if __name__ == "__main__":
    main()
