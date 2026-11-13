import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it, vi } from "vitest";
import { buildDecisionRequest, readBuyBoxOffer, toIsoDate } from "../src/content/decisionRequest";
import { toPanelData } from "../src/content/decision";
import { FAKE } from "../src/content/fakeData";
import { parseOfferList } from "../src/content/offerList";
import type { PageOffers } from "../src/content/offers";

vi.spyOn(console, "warn").mockImplementation(() => {});

const TODAY = new Date(2026, 10, 12); // Thu Nov 12, 2026
const PROFILE = { condition: "new_only", patience: "week_or_two", alerts: "on" };

const PAGE: PageOffers = {
  oneTimePrice: 19.79,
  snsPrice: null,
  snsTierText: null,
  otherOffersLabel: "New & Used",
  otherOffersCount: 9,
  otherOffersLowestPrice: 15.5,
  usedPrice: 17.1,
  usedCondition: "Save with Used - Very Good",
  usedSeller: "Warehouse Deals",
  usedShipping: 0,
};

const BUY_BOX = `
  <div id="merchantInfoFeature_feature_div"><span class="offer-display-feature-text-message">Amazon.com</span></div>
  <div id="fulfillerInfoFeature_feature_div"><span class="offer-display-feature-text-message">Amazon.com</span></div>
  <div id="mir-layout-DELIVERY_BLOCK"><span data-csa-c-delivery-price="FREE" data-csa-c-delivery-time="Saturday, November 14">x</span></div>`;

const doc = (html: string) => new DOMParser().parseFromString(html, "text/html");

describe("toIsoDate", () => {
  it.each([
    ["Saturday, November 14", "2026-11-14"],
    ["Nov 16 - 17", "2026-11-17"],
    ["October 30 - November 2", "2026-11-02"],
    ["Tomorrow, Nov 13", "2026-11-13"],
    ["Today", "2026-11-12"],
    ["Tomorrow", "2026-11-13"],
    ["January 4", "2027-01-04"],
    ["November 3", "2026-11-03"],
    ["FREE delivery", null],
    ["February 30", null],
    [null, null],
  ])("%s -> %s", (text, iso) => {
    expect(toIsoDate(text, TODAY)).toBe(iso);
  });
});

describe("readBuyBoxOffer", () => {
  it("reads the Add to Cart offer as the pinned offer", () => {
    expect(readBuyBoxOffer(PAGE, doc(BUY_BOX), TODAY)).toEqual({
      seller: "Amazon.com",
      sellerId: null,
      price: 19.79,
      condition: "New",
      shippingCost: 0,
      deliveryDate: "2026-11-14",
      fastestDelivery: null,
      shipsFrom: "Amazon.com",
      shipsFromAmazon: true,
      pinned: true,
    });
  });

  it("is null when the page shows no one-time price", () => {
    expect(readBuyBoxOffer({ ...PAGE, oneTimePrice: null }, doc(BUY_BOX), TODAY)).toBeNull();
  });
});

describe("buildDecisionRequest", () => {
  it("sends the buy box and used offer before the full list is read", () => {
    const r = buildDecisionRequest({ asin: "B0035LCFNQ", profile: PROFILE, page: PAGE, root: doc(BUY_BOX), today: TODAY });
    expect(r.page_price).toBe(19.79);
    expect(r.offers_source).toBe("page");
    expect(r.offers.map((o) => [o.seller, o.price, o.condition, o.pinned])).toEqual([
      ["Amazon.com", 19.79, "New", true],
      ["Warehouse Deals", 17.1, "Used - Very Good", false],
    ]);
    expect(r.other_offers).toEqual({ count: 9, lowest_price: 15.5, label: "New & Used" });
    expect(r.versions.status).toBe("none");
    expect(r.profile).toEqual(PROFILE);
  });

  it("sends every offer from Amazon's full list once it's read, with ISO delivery dates", () => {
    const aod = doc(readFileSync(join(__dirname, "fixtures/aod/synthetic-mixed.html"), "utf8"));
    const list = parseOfferList(aod);
    expect(list.status).toBe("ok");
    const r = buildDecisionRequest({ asin: "B0035LCFNQ", profile: PROFILE, page: PAGE, list, root: doc(BUY_BOX), today: TODAY });
    expect(r.offers_source).toBe("full");
    expect(r.offers).toHaveLength(list.offers.length + (list.offers.some((o) => o.pinned) ? 0 : 1));
    expect(r.offers.filter((o) => o.pinned)).toHaveLength(1);
    for (const o of r.offers) expect(o.deliveryDate === null || /^\d{4}-\d{2}-\d{2}$/.test(o.deliveryDate)).toBe(true);
  });

  it("falls back to the buy box when the full list couldn't be read", () => {
    const list = { status: "unreadable" as const, reason: "x", totalCount: null, offers: [] };
    const r = buildDecisionRequest({ asin: "B0035LCFNQ", profile: PROFILE, page: PAGE, list, root: doc(BUY_BOX), today: TODAY });
    expect(r.offers_source).toBe("page");
    expect(r.offers[0].pinned).toBe(true);
  });

  it("sends the product's versions", () => {
    const html = readFileSync(join(__dirname, "fixtures/versions/synthetic-pack-counts.html"), "utf8");
    const r = buildDecisionRequest({ asin: "B0035LCFNQ", profile: PROFILE, page: PAGE, root: doc(BUY_BOX + html), today: TODAY });
    expect(r.versions.status).toBe("ok");
    expect(r.versions.versions.find((v) => v.value === "8 Count")?.pricePerItem).toBe(1.15);
  });
});

describe("toPanelData with the page price", () => {
  it("shows the page's price in the wait state, not the stored one (#104)", () => {
    const p = toPanelData(FAKE.wait, TODAY, 329.99);
    expect(p).toMatchObject({ state: "wait", price: 329.99, expectedLow: 285 });
  });

  it("keeps the endpoint's price when the page has none", () => {
    expect(toPanelData(FAKE.wait, TODAY, null)).toMatchObject({ state: "wait", price: 349 });
  });
});
