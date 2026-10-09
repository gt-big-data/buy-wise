# Week one: get it running

You don't need a Keepa API key. Leave it blank.

## 1. Backend

Detail in [`backend/README.md`](backend/README.md).

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults work as-is

docker run --name buywise-mysql \
  -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=buywise \
  -p 3306:3306 -d mysql:8

docker exec -i buywise-mysql mysql -uroot -proot buywise < db/schema.sql
docker exec -i buywise-mysql mysql -uroot -proot buywise < db/seed.sql
docker exec -i buywise-mysql mysql -uroot -proot buywise < db/seed_real.sql

uvicorn main:app --reload
```

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
