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

uvicorn main:app --reload
```

Check it:

```bash
curl localhost:8000/health
curl localhost:8000/predict/B08N5WRWNW
```

The second should return a recommendation, a confidence score and a predicted price.

**Use only these ASINs.** They're the seeded ones and the only products that work without a Keepa key. Anything else fails and falls back to a heuristic.

```
B08N5WRWNW   B07FZ8S74R   B07PXGQC1Q   B08L5TNJHG   B09G9HD6PD
```

## 2. Extension

Detail in [`extension/README.md`](extension/README.md).

```bash
cd extension
npm install
npm run build
```

`chrome://extensions` → Developer mode → Load unpacked → `extension/dist`.

Open `amazon.com/dp/B08N5WRWNW` with the backend running. Panel appears top-right.

## 3. DM your lead

- A screenshot of the panel on the Amazon page
- Think of the last thing you bought on Amazon. One thing you'd have wanted the panel to tell you about it.
