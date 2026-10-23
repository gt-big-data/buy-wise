import { OfferListResult, parseOfferList } from "./offerList";

const PANEL_ROOT = "#aod-container, #all-offers-display";
const LOAD_TIMEOUT_MS = 8000;

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
