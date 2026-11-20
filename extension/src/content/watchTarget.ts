// The watch target the panel suggests. It must sit below today's price, or the
// watch would alert the moment it's saved. Starts from the expected low; when that's
// within 1% of the price, suggests about 5% below instead. Whole dollars from $20 up.
export function defaultTarget(price: number, expectedLow: number): number {
  const low = Math.floor(expectedLow * 100) / 100;
  const raw = low > 0 && low < price * 0.99 ? low : price * 0.95;
  const rounded = price >= 20 ? Math.floor(raw) : Math.floor(raw * 100) / 100;
  return rounded > 0 && rounded < price ? rounded : Math.floor(price * 95) / 100;
}

export const targetText = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(2));
