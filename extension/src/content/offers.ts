export interface PageOffers {
  oneTimePrice: number | null;
  snsPrice: number | null;
  snsTierText: string | null;
}

function readOptionPrices(): Record<string, number> {
  const prices: Record<string, number> = {};
  document.querySelectorAll<HTMLElement>("[data-csa-c-price-to-pay]").forEach((el) => {
    const type = el.dataset.csaCBuyingOptionType;
    const price = Number(el.dataset.csaCPriceToPay);
    if (type && !(type in prices) && !Number.isNaN(price)) prices[type] = price;
  });
  return prices;
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
    };
  } catch {
    return { oneTimePrice: null, snsPrice: null, snsTierText: null };
  }
}