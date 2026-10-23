# Offer list fixtures

Each `<name>.html` is Amazon's "Other sellers" panel, and `<name>.expected.json` is
what `parseOfferList` must return for it. Every pair is tested automatically
(`npm test`).

The `synthetic-*` files were written by hand from the structure seen on live pages
(Oct 16, 2026). Real captures should replace them.

## Adding a real page

1. Open the product page and click "Other sellers on Amazon" yourself.
2. Paste `scripts/capture-aod.js` into the DevTools console and save the clipboard
   as `<ASIN>.html`.
3. Write `<ASIN>.expected.json` by reading each offer **off the panel on screen**,
   not off the parser's output: seller, price, condition, shipping, delivery date.
4. Run `npm test`. A mismatch means either the parser or your expected file is
   wrong. Check the screen again before changing either.
