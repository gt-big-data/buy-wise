// Paste into the DevTools console on an Amazon product page that has sizes,
// colors or pack counts. Wait until the swatches show their prices first:
// Amazon fills them in a second or two after the page loads.
//
// Copies the product title, ASIN and version rows to the clipboard, with scripts
// and URLs removed. Save it as extension/test/fixtures/versions/<ASIN>.html, then
// write <ASIN>.expected.json from the swatches on screen (see the README there).
(() => {
  const rows = [...document.querySelectorAll('[id^="inline-twister-row-"]')];
  if (!rows.length) return console.warn("No version rows found on this page.");
  const asin = document.querySelector("input#ASIN")?.value ?? "unknown";
  const titles = rows.map((r) => document.getElementById(r.id.replace("-row-", "-dim-title-")));
  const html = [document.querySelector("#productTitle"), document.querySelector("input#ASIN"), ...titles, ...rows]
    .filter(Boolean)
    .map((el) => {
      const c = el.cloneNode(true);
      c.querySelectorAll("script, style, noscript, svg").forEach((s) => s.remove());
      [c, ...c.querySelectorAll("*")].forEach((e) =>
        [...e.attributes].forEach((a) => {
          if (/^(href|src|srcset|action)$|url|^on/i.test(a.name)) e.removeAttribute(a.name);
        })
      );
      return c.outerHTML;
    })
    .join("\n");
  const header = `<!-- ${asin} captured ${new Date().toISOString().slice(0, 10)}, ${rows.length} version rows -->\n`;
  copy(header + html); // DevTools console helper
  console.log(`Copied. Save as test/fixtures/versions/${asin}.html`);
})();
