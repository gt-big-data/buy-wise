// Paste into the DevTools console on an Amazon product page AFTER opening the
// "Other sellers on Amazon" panel yourself. Scroll the panel first if you want
// more than the first 10 offers in the fixture.
//
// Copies the panel's HTML to the clipboard. Save it as
// extension/test/fixtures/aod/<ASIN>.html, then write <ASIN>.expected.json by
// checking each offer against the panel on screen (see test/fixtures/aod/README.md).
(() => {
  const root = document.querySelector("#aod-container, #all-offers-display");
  if (!root) return console.warn("No offers panel found. Open 'Other sellers on Amazon' first.");
  const asin = location.pathname.match(/\/(?:dp|gp\/product)\/([A-Z0-9]{10})/i)?.[1] ?? "unknown";
  const rows = root.querySelectorAll("#aod-offer").length;
  const header = `<!-- ${asin} captured ${new Date().toISOString().slice(0, 10)}, ${rows} rows loaded -->\n`;
  copy(header + root.outerHTML); // DevTools console helper
  console.log(`Copied ${rows} offer rows. Save as test/fixtures/aod/${asin}.html`);
})();
