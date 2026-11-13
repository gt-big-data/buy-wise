import { readdirSync, readFileSync, existsSync } from "fs";
import { join } from "path";
import { describe, expect, it, vi } from "vitest";
import { parseOfferList } from "../src/content/offerList";

const FIXTURES = join(__dirname, "fixtures/aod");

function load(name: string): Document {
  return new DOMParser().parseFromString(readFileSync(join(FIXTURES, name), "utf8"), "text/html");
}

vi.spyOn(console, "warn").mockImplementation(() => {});

// Every <name>.html with a hand-checked <name>.expected.json next to it
const fixtures = readdirSync(FIXTURES)
  .filter((f) => f.endsWith(".html") && existsSync(join(FIXTURES, f.replace(/\.html$/, ".expected.json"))));

describe("parseOfferList fixtures", () => {
  it.each(fixtures)("%s matches its expected offers", (name) => {
    const expected = JSON.parse(readFileSync(join(FIXTURES, name.replace(/\.html$/, ".expected.json")), "utf8"));
    expect(parseOfferList(load(name))).toEqual(expected);
  });
});

describe("parseOfferList layout changes", () => {
  it("reports unreadable when an offer row loses its price", () => {
    const doc = load("synthetic-mixed.html");
    doc.querySelectorAll("#aod-offer [data-csa-c-price-to-pay]")[1].removeAttribute("data-csa-c-price-to-pay");
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "row 2: no price", offers: [] });
  });

  it("reports unreadable when the seller markup is renamed", () => {
    const doc = load("synthetic-mixed.html");
    doc.querySelectorAll("#aod-offer-soldBy").forEach((el) => (el.id = "aod-offer-seller"));
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "pinned offer: no seller" });
  });

  it("reports unreadable when rows are renamed but the count says there are offers", () => {
    const doc = load("synthetic-mixed.html");
    doc.querySelectorAll("#aod-offer").forEach((el) => (el.id = "aod-offer-card"));
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "no offer rows, count says 12" });
  });

  it("skips a pinned block that is only the product header", () => {
    const doc = load("synthetic-mixed.html");
    const pinned = doc.querySelector("#aod-pinned-offer")!;
    pinned.innerHTML = '<div id="aod-asin-title"><h5 id="aod-asin-title-text">Some Book</h5></div><div id="aod-asin-reviews">4.8 out of 5 stars</div>';
    const result = parseOfferList(doc);
    expect(result.status).toBe("ok");
    expect(result.offers.map((o) => o.seller)).toEqual(["Amazon Resale", "Goodwillswpa", "MobileMonster"]);
    expect(result.offers.every((o) => !o.pinned)).toBe(true);
  });

  it("reports unreadable for markup it doesn't recognise at all", () => {
    const doc = new DOMParser().parseFromString("<div id='aod-container'><p>new layout</p></div>", "text/html");
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "no offer rows and no offer count" });
  });
});

// A real offer list (Bose QuietComfort, B0CCZ26B5V, Nov 13). The expected numbers were
// read from the page's text separately, not from this parser's output.
describe("seller ratings on a real offer list", () => {
  const html = readFileSync(join(__dirname, "fixtures/seller-ratings/B0CCZ26B5V.html"), "utf8");
  const result = parseOfferList(new DOMParser().parseFromString(html, "text/html"));
  const offers = result.offers.filter((o) => !o.pinned);

  it("reads prices when Amazon's hidden price text is blank", () => {
    expect(result.status).toBe("ok");
    expect(result.offers[0]).toMatchObject({ pinned: true, seller: "Frontrow Tech LLC", price: 359 });
    expect(offers.slice(0, 3).map((o) => o.price)).toEqual([178.99, 178.99, 194.9]);
  });

  it("reads stars, rating count and % positive for every seller", () => {
    expect(offers.map((o) => [o.seller, o.stars, o.ratingCount, o.positivePercent])).toEqual([
      ["ADV Sellers", 4.5, 390, 94],
      ["Win-Plus shop", 4.5, 28, 88],
      ["Silmarils (Serial Number Recorded)", 2.5, 2255, 34],
      ["Officeworld Stores", 5, 5990, 97],
      ["FDBV Corporate Group", 4, 662, 73],
      ["Beach Camera Same Day Shipping", 4.5, 353956, 88],
      ["Toms jeans and ties", 4, 143, 78],
      ["Tech Geeks", 4.5, 315, 86],
      ["Mostly Pre-Owned Audio", 5, 1001, 100],
      ["wilog2001", 4, 10, 70],
    ]);
  });
});
