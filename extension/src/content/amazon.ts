export function isAmazonProductPage(): boolean {
  return isAmazonProductPageUrl(window.location.href);
}

export function isAmazonProductPageUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname;
    const path = parsed.pathname;

    const isAmazon = host.includes("amazon.com");
    const isProductPath = /\/(dp|gp\/product)\//.test(path);

    return isAmazon && isProductPath;
  } catch {
    return false;
  }
}

export function extractASIN(): string | null {
  return extractASINFromUrl(window.location.href);
}

export function extractASINFromUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    const match = parsed.pathname.match(/\/(?:dp|gp\/product)\/([A-Z0-9]{10})/i);

    if (match?.[1]) {
      return match[1].toUpperCase();
    }

    return null;
  } catch {
    return null;
  }
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
