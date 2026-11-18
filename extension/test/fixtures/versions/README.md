# Version fixtures

Each `<name>.html` is a product page's version rows (sizes, colors, pack counts),
and `<name>.expected.json` is what `readVersions` must return for it. Every pair is
tested automatically (`npm test`).

The `synthetic-*` files were written by hand from live pages on Nov 12, 2026
(Duracell AA B0035LCFNQ and Hanes undershirts B00D1ARZMC). Real captures should
join them.

## Adding a real page

1. Open the product page and wait for the swatches to show prices. Amazon fills
   them in after the page loads; the raw HTML has none.
2. Paste `scripts/capture-versions.js` into the DevTools console and save the
   clipboard as `<ASIN>.html`.
3. Write `<ASIN>.expected.json` from the swatches **on screen**, not from the
   reader's output: ASIN (from the swatch), name, price, Amazon's per-unit price.
4. Run `npm test`. A mismatch means the reader or your expected file is wrong.
   Check the screen again before changing either.
