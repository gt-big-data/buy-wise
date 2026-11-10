import { extractASIN } from "./amazon";
import { extractOffers } from "./offers";

// Independent of the panel: also works when it is closed or showing demo data.
// Retry briefly for delayed buy-box rendering and first-time backend seeding.
export function followLivePrices(): void {
  let observedAsin: string | null = null;
  let attempts = 0;
  let saved = false;
  let pending = false;
  window.setInterval(async () => {
    const asin = extractASIN();
    if (asin !== observedAsin) {
      observedAsin = asin;
      attempts = 0;
      saved = false;
    }
    if (!asin || saved || pending || attempts >= 20) return;
    attempts += 1;
    const price = extractOffers().oneTimePrice;
    if (price === null || !Number.isFinite(price) || price <= 0) return;
    pending = true;
    try {
      const result = await chrome.runtime.sendMessage({ type: "BUYWISE_RECORD_PRICE", asin, price });
      if (observedAsin === asin && result?.ok) saved = true;
    } catch {
      // Backend or extension may be temporarily unavailable; retry on next tick.
    } finally {
      pending = false;
    }
  }, 1000);
}

