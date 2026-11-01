import { afterEach, describe, expect, it } from "vitest";
import { extractASIN, extractASINFromUrl, isAmazonProductPage } from "../src/content/amazon";

// Every shape here opened a real product page in the #73 survey
const PRODUCT_URLS: [string, string][] = [
  ["search result", "https://www.amazon.com/Anker-Charging-MacBook-Galaxy-Charger/dp/B088NRLMPV/ref=sr_1_3?dib=abc&keywords=usb+c+cable"],
  ["sponsored result", "https://www.amazon.com/Car-CarPlay-Cable-iPhone-Pro/dp/B0DNFHM4TW/ref=sr_1_2_sspa?sp_csd=d2lkZ2V0&psc=1"],
  ["related carousel", "https://www.amazon.com/dp/B0FHWGB7FW/ref=sspa_dk_detail_0?psc=1"],
  ["deal", "https://www.amazon.com/Premier-Protein-Shake-Caramel-count/dp/B07MJL8TPK?ref=dlx_deals_dg_dcl"],
  ["bare /dp/ with variant params", "https://www.amazon.com/dp/B088NRLMPV?th=1&psc=1"],
  ["Spanish", "https://www.amazon.com/-/es/dp/B088NRLMPV"],
  ["/gp/product/", "https://www.amazon.com/gp/product/B088NRLMPV"],
  ["/gp/aw/d/ (#67)", "https://www.amazon.com/gp/aw/d/B088NRLMPV"],
  ["/gp/product/glance/", "https://www.amazon.com/gp/product/glance/B088NRLMPV"],
  ["/exec/obidos/ASIN/", "https://www.amazon.com/exec/obidos/ASIN/B088NRLMPV"],
  ["/o/ASIN/", "https://www.amazon.com/o/ASIN/B088NRLMPV"],
  ["lowercase", "https://www.amazon.com/dp/b088nrlmpv"],
];

describe("extractASINFromUrl", () => {
  it.each(PRODUCT_URLS)("reads the ASIN from a %s link", (_, url) => {
    expect(extractASINFromUrl(url)).toMatch(/^[A-Z0-9]{10}$/);
  });

  it.each([
    ["search page", "https://www.amazon.com/s?k=usb+c+cable"],
    ["reviews page", "https://www.amazon.com/product-reviews/B088NRLMPV"],
    ["11-character id", "https://www.amazon.com/dp/B088NRLMPVX"],
    ["look-alike host", "https://amazon.com.example.net/dp/B088NRLMPV"],
    ["not a URL", "dp/B088NRLMPV"],
  ])("ignores a %s", (_, url) => {
    expect(extractASINFromUrl(url)).toBeNull();
  });
});

describe("reading the ASIN from the page", () => {
  afterEach(() => {
    document.body.innerHTML = "";
    window.history.replaceState(null, "", "/");
  });

  const productPage = (asin: string) => {
    document.body.innerHTML = `<span id="productTitle">Cable</span><input type="hidden" id="ASIN" value="${asin}">`;
  };

  it("detects a product page whose URL shape it doesn't know", () => {
    window.history.replaceState(null, "", "/some/new/shape/B088NRLMPV");
    productPage("B088NRLMPV");
    expect(isAmazonProductPage()).toBe(true);
    expect(extractASIN()).toBe("B088NRLMPV");
  });

  it("follows the variant on screen when it differs from the URL", () => {
    window.history.replaceState(null, "", "/dp/B088NRLMPV");
    productPage("B0CG1LGWR6");
    expect(extractASIN()).toBe("B0CG1LGWR6");
  });

  it("doesn't treat a page with a stray ASIN input as a product page", () => {
    window.history.replaceState(null, "", "/s?k=cable");
    document.body.innerHTML = `<input type="hidden" id="ASIN" value="B088NRLMPV">`;
    expect(isAmazonProductPage()).toBe(false);
  });
});
