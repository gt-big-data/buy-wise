# BuyWise website

Next.js + Tailwind landing page. Brand tokens (colors, fonts, shadows) live in
`app/globals.css`; text, stats and the team roster live in `content/site.ts`.

```bash
cd website
npm install
npm run dev      # http://localhost:3000
```

## Deploying

Vercel project root directory: `website`. Framework preset: Next.js. No env vars.
Pushes to `main` that touch `website/` redeploy it automatically once the
project is connected to this repo.
