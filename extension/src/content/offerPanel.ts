import { OfferListResult, parseOfferList } from "./offerList";

const PANEL_ROOT = "#aod-container, #all-offers-display";
const LOAD_TIMEOUT_MS = 8000;
const FETCH_TIMEOUT_MS = 3000;

// Loads the same offer list Amazon shows under "Other sellers on Amazon", without
// opening the panel, so the first answer already covers every seller. Same-origin
// request from the product page. Resolves null when it fails or takes too long;
// the panel then answers from the page and re-asks if the shopper opens the list.
export async function fetchOfferList(asin: string): Promise<OfferListResult | null> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const url = `/gp/product/ajax/aodAjaxMain/?asin=${encodeURIComponent(asin)}&pc=dp&experienceId=aodAjaxMain`;
    const res = await fetch(url, { credentials: "include", signal: controller.signal });
    if (!res.ok) return null;
    const doc = new DOMParser().parseFromString(await res.text(), "text/html");
    return parseFetchedOfferList(doc);
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

export function parseFetchedOfferList(doc: Document): OfferListResult | null {
  const root = doc.querySelector(PANEL_ROOT);
  if (!root || !root.querySelector("#aod-offer, #aod-pinned-offer, #aod-total-offer-count")) return null;
  return parseOfferList(root);
}

// Opens Amazon's own offers panel, the same as the shopper clicking
// "Other sellers on Amazon". Only call this from a shopper's click.
export function openOffersPanel(): boolean {
  const link = document.querySelector<HTMLElement>(
    "#aod-ingress-link, #buybox-see-all-buying-choices a, #buybox-see-all-buying-choices"
  );
  link?.click();
  return Boolean(link);
}

// Reports the parsed list each time the panel's rows change (first load, and
// more rows as the shopper scrolls). Returns a function that stops watching.
export function watchOfferList(onChange: (result: OfferListResult) => void): () => void {
  let lastKey = "";
  let debounce: number | undefined;
  let loaded = false;

  const check = () => {
    const root = document.querySelector(PANEL_ROOT);
    if (!root || !root.querySelector("#aod-offer, #aod-total-offer-count")) return;
    loaded = true;
    const result = parseOfferList(root);
    const key = `${result.status}:${result.offers.length}:${result.totalCount}`;
    if (key !== lastKey) {
      lastKey = key;
      onChange(result);
    }
  };

  const observer = new MutationObserver(() => {
    window.clearTimeout(debounce);
    debounce = window.setTimeout(check, 150);
  });
  observer.observe(document.body, { childList: true, subtree: true });
  check();

  const timeout = window.setTimeout(() => {
    if (!loaded) onChange({ status: "unreadable", reason: "panel did not load", totalCount: null, offers: [] });
  }, LOAD_TIMEOUT_MS);

  return () => {
    observer.disconnect();
    window.clearTimeout(debounce);
    window.clearTimeout(timeout);
  };
}
