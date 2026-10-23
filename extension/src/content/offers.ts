export interface PageOffers {
  oneTimePrice: number | null;
  snsPrice: number | null;
  snsTierText: string | null;
  // "Other sellers on Amazon" box
  otherOffersLabel: string | null; // "New & Used", "New", or "Used"
  otherOffersCount: number | null;
  otherOffersLowestPrice: number | null;
  // "Used" option in the buy box
  usedPrice: number | null;
  usedCondition: string | null;
  usedSeller: string | null;
  usedShipping: number | null; // 0 means free
}

const EMPTY_OFFERS: PageOffers = {
  oneTimePrice: null, snsPrice: null, snsTierText: null,
  otherOffersLabel: null, otherOffersCount: null, otherOffersLowestPrice: null,
  usedPrice: null, usedCondition: null, usedSeller: null, usedShipping: null,
};

function readOptionPrices(): Record<string, number> {
  const prices: Record<string, number> = {};
  document.querySelectorAll<HTMLElement>("[data-csa-c-price-to-pay]").forEach((el) => {
    const type = el.dataset.csaCBuyingOptionType;
    const price = Number(el.dataset.csaCPriceToPay);
    if (type && !(type in prices) && !Number.isNaN(price)) prices[type] = price;
  });
  return prices;
}

// "$1,160.58" -> 1160.58; anything without digits -> null
export function parsePrice(text: string | null | undefined): number | null {
  const cleaned = text?.replace(/[^0-9.]/g, "");
  if (!cleaned) return null;
  const n = Number(cleaned);
  return Number.isNaN(n) ? null : n;
}

function readOtherSellers() {
  const box = document.querySelector("#dynamic-aod-ingress-box");
  const label = box?.querySelector("#aod-ingress-link .a-color-base")?.textContent?.trim() ?? "";
  // "New & Used (9) from" -> label "New & Used", count "9"
  const match = label.match(/^(.*?)\s*\((\d+)\)/);
  return {
    otherOffersLabel: match?.[1] || null,
    otherOffersCount: match ? Number(match[2]) : null,
    otherOffersLowestPrice: parsePrice(box?.querySelector(".a-price .a-offscreen")?.textContent),
  };
}

function readUsedOption() {
  const row = document.querySelector("#usedAccordionRow");
  let seller: string | null = null;
  row?.querySelectorAll("#sfsb_accordion_head .a-row").forEach((line) => {
    const spans = line.querySelectorAll("span");
    if (spans[0]?.textContent?.includes("Sold by")) seller = spans[1]?.textContent?.trim() || null;
  });
  const shipping = row?.querySelector<HTMLElement>("[data-csa-c-shipping-charge]")?.dataset.csaCShippingCharge;
  return {
    usedCondition: row?.querySelector(".accordion-caption")?.textContent?.trim() || null,
    usedSeller: seller,
    usedShipping: parsePrice(shipping),
  };
}

export function extractOffers(): PageOffers {
  try {
    const prices = readOptionPrices();
    const tier = document.querySelector("#sns-tiered-price .apex-pricetopay-accessibility-label");
    return {
      // One-time option is typed "NEW" on most pages, "BUNDLE" on some
      oneTimePrice: prices.NEW ?? prices.BUNDLE ?? null,
      snsPrice: prices.SNS ?? null,
      snsTierText: tier?.textContent?.trim() || null,
      ...readOtherSellers(),
      usedPrice: prices.USED ?? null,
      ...readUsedOption(),
    };
  } catch {
    return EMPTY_OFFERS;
  }
}
