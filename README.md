# BuyWise

A Chrome extension that finds the cheapest sensible way to buy what's on an Amazon product page.

Many Amazon listings have several sellers but only one Add to Cart button, and Amazon decides which seller gets it. Cheaper offers from other sellers, used copies and other versions of the product sit on a separate list most shoppers never open. BuyWise reads every offer on the page and decides whether a cheaper one is worth it for this shopper, weighing price, condition, shipping speed, returns and the seller. It then shows a one-line recommendation with the reason. When Amazon's default is already the best option, BuyWise says so and shows nothing else.

When waiting is the better move, the shopper can watch the product. They set a target price, or ask to be told when it's a good time to buy, and BuyWise sends an alert.

Built by the Georgia Tech Big Data Club, fall 2026. Demo day is Nov 20.

## How it fits together

| Part | Folder | What it does |
|---|---|---|
| Extension | [`extension/`](extension) | Reads the offers on Amazon product pages and shows the recommendation panel (Manifest V3, TypeScript) |
| Backend | [`backend/`](backend) | FastAPI service: Keepa price data, the daily price job, and the decision endpoint that turns everything into one answer per page |
| Website | [`website/`](website) | The project site and a product preview (Next.js) |
| Studies | [`studies/`](studies) | The data studies behind the product direction, with their scripts |

Shopper data stays on the shopper's device. The backend never stores the shopper profile or order history.

## Run it locally

You need [Docker Desktop](https://www.docker.com/products/docker-desktop) and Node.

```bash
# backend + database (MySQL, schema and seed data) on localhost:8000
docker compose up

# extension
cd extension && npm install && npm run build
```

Then open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, and pick `extension/dist`.

[ONBOARDING.md](ONBOARDING.md) walks through the full first-week setup.

## Working on it

- [ROADMAP.md](ROADMAP.md) is the semester plan: teams, issues and what each Friday shows.
- [CONTRIBUTING.md](CONTRIBUTING.md) covers branches, pull requests, reviews and merging.
- The [project board](https://github.com/users/AashishAnanth/projects/1) tracks every issue.

Every PR runs the backend tests and builds the extension and website, and `main` only accepts changes that pass.

[setup.md](setup.md) describes the original price-prediction architecture. Parts of it are out of date. Where it disagrees with the roadmap, the roadmap is right.
