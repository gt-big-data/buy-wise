import { extractOffers, PageOffers, parsePrice } from "./offers";
import { Offer, OfferListResult } from "./offerList";
import { readVersions, VersionsResult } from "./versions";

// Everything the decision endpoint (#92) gets about one product page: the price on
// the page, every offer we could read, the product's other versions, and the
// shopper's profile. The profile is sent with each request and never stored.

export type ProfileAnswers = { condition: string; patience: string; alerts: string };

// One offer, in the field names the endpoint accepts (its aliases).
// deliveryDate is ISO (YYYY-MM-DD); the endpoint ignores anything else.
export type RequestOffer = {
  seller: string;
  sellerId: string | null;
  price: number;
  condition: string | null;
  shippingCost: number | null;
  deliveryDate: string | null;
  fastestDelivery: string | null;
  shipsFrom: string | null;
  shipsFromAmazon: boolean;
  pinned: boolean;
  // Seller rating from the offer list; null when it isn't shown (always for the buy box read)
  stars: number | null;
  ratingCount: number | null;
  positivePercent: number | null;
};

export type DecisionRequest = {
  asin: string;
  page_price: number | null; // the one-time price on the page, same read as price alerts (#74)
  offers: RequestOffer[];
  // full: Amazon's whole offer list was read. page: only the buy box, so there may be more.
  offers_source: "full" | "page";
  other_offers: { count: number | null; lowest_price: number | null; label: string | null };
  versions: VersionsResult;
  profile: ProfileAnswers;
};

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

// Amazon's delivery text -> YYYY-MM-DD, or null when it isn't a date we can place.
// "Saturday, November 14" / "Nov 16 - 17" / "October 21 - 23" / "Tomorrow, Nov 13".
// A range gives its last day: the shopper can't count on the earlier one. Dates
// with no year are this year, or next year when that month has already passed.
export function toIsoDate(text: string | null | undefined, today = new Date()): string | null {
  if (!text) return null;
  const t = text.toLowerCase();
  const iso = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const start = new Date(today.getFullYear(), today.getMonth(), today.getDate());

  const m = t.match(/\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:\s*[-–]\s*(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(\d{1,2}))?/);
  if (m) {
    const month = MONTHS.indexOf(m[3] ?? m[1]);
    const day = Number(m[4] ?? m[2]);
    let d = new Date(start.getFullYear(), month, day);
    // More than a month in the past means it's next year's date (a December page showing January)
    if (d.getTime() < start.getTime() - 31 * 86_400_000) d = new Date(start.getFullYear() + 1, month, day);
    return d.getMonth() === month ? iso(d) : null;
  }
  if (/\btoday\b/.test(t)) return iso(start);
  if (/\btomorrow\b/.test(t)) return iso(new Date(start.getFullYear(), start.getMonth(), start.getDate() + 1));
  return null;
}

function toRequestOffer(o: Offer, today: Date): RequestOffer {
  return {
    seller: o.seller,
    sellerId: o.sellerId,
    price: o.price,
    condition: o.condition,
    shippingCost: o.shippingCost,
    deliveryDate: toIsoDate(o.deliveryDate, today),
    fastestDelivery: toIsoDate(o.fastestDelivery, today),
    shipsFrom: o.shipsFrom,
    shipsFromAmazon: o.shipsFromAmazon,
    pinned: o.pinned,
    stars: o.stars,
    ratingCount: o.ratingCount,
    positivePercent: o.positivePercent,
  };
}

function text(el: Element | null): string | null {
  return el?.textContent?.replace(/\s+/g, " ").trim() || null;
}

// The offer behind Add to Cart, read from the buy box. Null when the page shows
// no one-time price (unavailable, or a layout we don't read).
export function readBuyBoxOffer(page: PageOffers, root: ParentNode = document, today = new Date()): RequestOffer | null {
  if (page.oneTimePrice === null) return null;
  const seller =
    text(root.querySelector("#merchantInfoFeature_feature_div .offer-display-feature-text-message")) ??
    text(root.querySelector("#sellerProfileTriggerId")) ??
    "Amazon's pick";
  const shipsFrom = text(root.querySelector("#fulfillerInfoFeature_feature_div .offer-display-feature-text-message"));
  const delivery = root.querySelector<HTMLElement>(
    "#mir-layout-DELIVERY_BLOCK [data-csa-c-delivery-time], #deliveryBlockMessage [data-csa-c-delivery-time]"
  );
  const charge = delivery?.dataset.csaCDeliveryPrice?.trim() ?? "";
  return {
    seller,
    sellerId: null,
    price: page.oneTimePrice,
    condition: "New",
    shippingCost: /^free$/i.test(charge) ? 0 : parsePrice(charge),
    deliveryDate: toIsoDate(delivery?.dataset.csaCDeliveryTime, today),
    fastestDelivery: null,
    shipsFrom,
    shipsFromAmazon: /^amazon(\.com)?$/i.test(shipsFrom ?? seller),
    pinned: true,
    stars: null,
    ratingCount: null,
    positivePercent: null,
  };
}

// The buy box's "Used" option, when there is one
function readUsedOffer(page: PageOffers): RequestOffer | null {
  if (page.usedPrice === null) return null;
  return {
    seller: page.usedSeller ?? "Used seller",
    sellerId: null,
    price: page.usedPrice,
    condition: page.usedCondition?.replace(/^save with\s+/i, "") ?? "Used",
    shippingCost: page.usedShipping,
    deliveryDate: null,
    fastestDelivery: null,
    shipsFrom: null,
    shipsFromAmazon: false,
    pinned: false,
    stars: null,
    ratingCount: null,
    positivePercent: null,
  };
}

export function buildDecisionRequest(args: {
  asin: string;
  profile: ProfileAnswers;
  list?: OfferListResult | null; // Amazon's full offer list, once the shopper opened it
  page?: PageOffers;
  versions?: VersionsResult;
  root?: ParentNode;
  today?: Date;
}): DecisionRequest {
  const today = args.today ?? new Date();
  const page = args.page ?? extractOffers();
  const full = args.list?.status === "ok" ? args.list.offers.map((o) => toRequestOffer(o, today)) : null;
  // The full list's pinned row is the buy box; use the page's one when the list doesn't pin one
  const buyBox = readBuyBoxOffer(page, args.root, today);
  const offers = full
    ? full.some((o) => o.pinned) || !buyBox ? full : [buyBox, ...full]
    : [buyBox, readUsedOffer(page)].filter((o): o is RequestOffer => o !== null);

  return {
    asin: args.asin,
    page_price: page.oneTimePrice,
    offers,
    offers_source: full ? "full" : "page",
    other_offers: { count: page.otherOffersCount, lowest_price: page.otherOffersLowestPrice, label: page.otherOffersLabel },
    versions: args.versions ?? readVersions(args.root),
    profile: args.profile,
  };
}
