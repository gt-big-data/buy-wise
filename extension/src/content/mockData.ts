import { BuyWiseData } from "../types";

export function getMockBuyWiseData(asin: string): BuyWiseData {
  return {
    asin,
    productTitle: document.title.replace("Amazon.com: ", "").trim(),
    currentPrice: 59.97,
    predictedBestPrice: 49.99,
    expectedSavings: 10.0,
    confidence: 91,
    recommendation: "WAIT",
    why: "Recent price movement suggests this item may dip over the next 7 days. Similar grocery and household listings have shown short-term promotional price drops, so waiting could save you around $10.",
    chartTitle: "Predicted Price Graph (next 21 days)",
    points: [
      { label: "-3w", actual: 54 },
      { label: "-2w", actual: 61 },
      { label: "-1w", actual: 58 },
      { label: "Today", actual: 79, predicted: 79 },
      { label: "1w", predicted: 60 },
      { label: "2w", predicted: 67 },
      { label: "3w", predicted: 75 }
    ]
  };
}
