// @vitest-environment jsdom
import { expect, test } from "vitest";
import { parseFetchedOfferList } from "../src/content/offerPanel";

const parse = (html: string) => parseFetchedOfferList(new DOMParser().parseFromString(html, "text/html"));

test("reads offers from a fetched offer list", () => {
  const r = parse(`<div id="aod-container"><input id="aod-total-offer-count" value="1">
    <div id="aod-offer"><span data-csa-c-price-to-pay="118.87"></span>
    <div id="aod-offer-heading">Used - Like New</div>
    <div id="aod-offer-soldBy"><a href="/gp/aag/main?seller=A1">Amazon Resale</a></div></div></div>`);
  expect(r?.status).toBe("ok");
  expect(r?.offers[0]).toMatchObject({ price: 118.87, condition: "Used - Like New", seller: "Amazon Resale" });
});

test("returns null for a page that is not an offer list", () => {
  expect(parse("<html><body>Sorry, we just need to make sure you're not a robot.</body></html>")).toBeNull();
});
