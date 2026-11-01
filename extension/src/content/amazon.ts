// Every path shape Amazon serves a product page on. Keep in sync with
// backend/asin.py. Ones missing here are why /gp/aw/d/ links showed no panel (#67).
const PRODUCT_PATH =
  /\/(?:dp|gp\/product(?:\/glance)?|gp\/aw\/d|exec\/obidos\/ASIN|o\/ASIN)\/([A-Z0-9]{10})(?:[/?#]|$)/i;

export function isAmazonProductPage(): boolean {
  return isAmazonProductPageUrl(window.location.href) || readPageASIN() !== null;
}

export function isAmazonProductPageUrl(url: string): boolean {
  return extractASINFromUrl(url) !== null;
}

// The ASIN of the product on screen. The page's own value wins over the URL:
// it covers URL shapes we haven't seen, and it follows color/size changes.
export function extractASIN(): string | null {
  return readPageASIN() ?? extractASINFromUrl(window.location.href);
}

export function extractASINFromUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (!/(^|\.)amazon\.com$/i.test(parsed.hostname)) return null;
    return parsed.pathname.match(PRODUCT_PATH)?.[1]?.toUpperCase() ?? null;
  } catch {
    return null;
  }
}

// Product pages carry the ASIN of the product showing in a hidden form input
function readPageASIN(): string | null {
  if (!document.querySelector("#productTitle")) return null;
  const value = document.querySelector<HTMLInputElement>("input#ASIN")?.value?.trim().toUpperCase();
  return value && /^[A-Z0-9]{10}$/.test(value) ? value : null;
}

export function extractProductImage(): string | null {
  try {
    const img = document.querySelector<HTMLImageElement>(
      "#landingImage, #imgBlkFront, #imgTagWrapperId img"
    );
    return img ? img.src : null;
  } catch {
    return null;
  }
}
