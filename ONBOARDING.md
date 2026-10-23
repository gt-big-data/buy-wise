# Week one: get it running

You don't need a Keepa API key. Leave it blank.

## 1. Backend

Detail in [`backend/README.md`](backend/README.md). You need [Docker Desktop](https://www.docker.com/products/docker-desktop) running.

From the repo root:

```bash
docker compose up
```

That starts MySQL, loads the schema and both seed files, and runs the backend on port 8000. Code changes in `backend/` reload on their own. Stop it with Ctrl+C; your data stays for next time. `docker compose down -v` wipes the database so the next `up` reloads it from scratch.

If you set up before this existed, run `docker stop buywise-mysql` first so the old container isn't holding port 3306.

Check it:

```bash
curl localhost:8000/health
curl localhost:8000/predict/B0CCZ26B5V
```

The second should return a recommendation, the chance of a price drop, and a predicted price.

**Use only the seeded ASINs.** They're the only products that work without a Keepa key.

These have real price history, and the model scores them live:

```
B0CCZ26B5V   Bose QuietComfort Headphones      B0DDV3FRHR   Sony WH-1000XM5
B0CRGJC5ZD   Samsung Odyssey G55C monitor      B0D8WJYSF9   Google Streamer 4K
B0BS1T9J4Y   Garmin Forerunner 265             B0C8PR4W22   Beats Studio Pro
```

These have sample data with hand-written predictions, useful for UI work:

```
B0GR6BVYS5   B08N5WRWNW   B07FZ8S74R   B07PXGQC1Q   B08L5TNJHG   B09G9HD6PD
```

## 2. Extension

Detail in [`extension/README.md`](extension/README.md).

```bash
cd extension
npm install
npm run build
```

`chrome://extensions` → Developer mode → Load unpacked → `extension/dist`.

Open `amazon.com/dp/B0CCZ26B5V` with the backend running. Panel appears top-right.

## 3. DM your lead

- A screenshot of the panel on the Amazon page
- One well thought out feature you'd like to build in the app this semester
