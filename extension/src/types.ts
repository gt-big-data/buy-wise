export type Recommendation = "BUY" | "WAIT";

export type PricePoint = {
  label: string;
  actual?: number;
  predicted?: number;
};

export type BuyWiseData = {
  asin: string;
  productTitle: string;
  currentPrice: number;
  predictedBestPrice: number;
  expectedSavings: number;
  confidence: number;
  recommendation: Recommendation;
  why: string;
  chartTitle: string;
  points: PricePoint[];
};
