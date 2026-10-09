export type Recommendation = "BUY" | "WAIT";

export type PricePoint = {
  label: string;
  actual?: number;
  predicted?: number;
};

export type BuyWiseData = {
  asin: string;
  productTitle: string;
  imageUrl?: string;
  currentPrice: number;
  predictedBestPrice: number;
  expectedSavings: number;
  dropChance: number; // 0–100: chance the price falls 8%+ in the next 14 days
  higherAfterWait?: number; // 0–100: share of past WAITs where the price ended higher; WAIT only
  recommendation: Recommendation;
  why: string;
  chartTitle: string;
  points: PricePoint[];
  isWatched?: boolean;
};
