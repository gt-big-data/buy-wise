import { parsePrice } from "./offers";

// Reads the offer list inside Amazon's "Other sellers" panel (AOD). The panel
// only exists after the shopper opens it; this never fetches anything itself.

export interface Offer {
  seller: string;
  sellerId: string | null;
  price: number;
  condition: string | null; // "New", "Used - Very Good", ...
  shippingCost: number | null; // 0 means free
  deliveryDate: string | null; // "Sunday, October 18" or "October 21 - 23"
  fastestDelivery: string | null;
  shipsFrom: string | null;
  shipsFromAmazon: boolean;
  returnPolicy: null; // not shown in the offer list
  pinned: boolean; // the offer behind the Add to Cart button
}

export interface OfferListResult {
  status: "ok" | "no_offers" | "unreadable";
  reason?: string;
  totalCount: number | null; // Amazon's count, can exceed offers.length until the shopper scrolls
  offers: Offer[];
}

// Offer rows embed <style> blocks, so read text with those removed
function text(el: Element | null | undefined): string | null {
  if (!el) return null;
  const copy = el.cloneNode(true) as Element;
  copy.querySelectorAll("style, script").forEach((s) => s.remove());
  return copy.textContent?.replace(/\s+/g, " ").trim() || null;
}

function readSellerId(href: string | null | undefined): string | null {
  return href?.match(/[?&]seller=([A-Z0-9]+)/i)?.[1] ?? null;
}

function readRow(row: Element, pinned: boolean): Partial<Offer> {
  const sellerLink = row.querySelector("#aod-offer-soldBy a");
  const seller = text(sellerLink) ?? text(row.querySelector("#aod-offer-soldBy .a-col-right .a-size-small"));
  const priceAttr = row.querySelector<HTMLElement>("[data-csa-c-price-to-pay]")?.dataset.csaCPriceToPay;
  const price = parsePrice(priceAttr) ?? parsePrice(row.querySelector(".a-price .a-offscreen")?.textContent);
  const shipsFrom = text(row.querySelector("#aod-offer-shipsFrom .a-col-right"));

  // Each delivery option carries a price ("FREE", "$3.99") or the word "fastest"
  let shippingCost: number | null = null;
  let deliveryDate: string | null = null;
  let fastestDelivery: string | null = null;
  row.querySelectorAll<HTMLElement>("[data-csa-c-delivery-time]").forEach((el) => {
    const charge = el.dataset.csaCDeliveryPrice?.trim() ?? "";
    const time = el.dataset.csaCDeliveryTime?.trim() || null;
    if (/^fastest$/i.test(charge)) {
      fastestDelivery ??= time;
    } else if (deliveryDate === null) {
      deliveryDate = time;
      shippingCost = /^free$/i.test(charge) ? 0 : parsePrice(charge);
    }
  });

  return {
    seller: seller ?? undefined,
    sellerId: readSellerId(sellerLink?.getAttribute("href")),
    price: price ?? undefined,
    condition: text(row.querySelector("#aod-offer-heading")),
    shippingCost,
    deliveryDate,
    fastestDelivery,
    shipsFrom,
    shipsFromAmazon: shipsFrom !== null && /^amazon(\.com)?$/i.test(shipsFrom),
    returnPolicy: null,
    pinned,
  };
}

export function parseOfferList(root: ParentNode): OfferListResult {
  const countAttr = root.querySelector<HTMLInputElement>("#aod-total-offer-count")?.value;
  const totalCount = countAttr != null && countAttr !== "" ? Number(countAttr) : null;

  const rows: [Element, boolean][] = [];
  const pinnedRow = root.querySelector("#aod-pinned-offer");
  if (pinnedRow) rows.push([pinnedRow, true]);
  root.querySelectorAll("#aod-offer").forEach((row) => rows.push([row, false]));

  // The pinned row is the buy box; without other rows there is nothing to compare
  if (rows.every(([, pinned]) => pinned)) {
    if (totalCount === 0) return { status: "no_offers", totalCount, offers: [] };
    return unreadable(totalCount === null ? "no offer rows and no offer count" : `no offer rows, count says ${totalCount}`, totalCount);
  }

  const offers: Offer[] = [];
  for (const [i, [row, pinned]] of rows.entries()) {
    const offer = readRow(row, pinned);
    const label = pinned ? "pinned offer" : `row ${i + (pinnedRow ? 0 : 1)}`;
    if (offer.price === undefined) return unreadable(`${label}: no price`, totalCount);
    if (offer.seller === undefined) return unreadable(`${label}: no seller`, totalCount);
    offers.push(offer as Offer);
  }
  return { status: "ok", totalCount, offers };
}

function unreadable(reason: string, totalCount: number | null): OfferListResult {
  console.warn("BuyWise offers:", reason);
  return { status: "unreadable", reason, totalCount, offers: [] };
}
