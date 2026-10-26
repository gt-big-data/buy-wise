export type PanelData =
  | {
      state: "better_offer";
      offersChecked: number;
      amazon: { price: number };
      best: { seller: string; price: number; stars: number; years: number; freeReturns: boolean; fullWarranty: boolean };
    }
  | {
      state: "wait";
      offersChecked: number;
      price: number;
      expectedLow: number;
      dropChance: number; 
      drops: number;
      months: number;
    }
  | {
      state: "nothing_better";
      offersChecked: number;
      nextBest: { price: number; daysLater: number } | null;
    };

export const FAKE: Record<PanelData["state"], PanelData> = {
  better_offer: {
    state: "better_offer",
    offersChecked: 28,
    amazon: { price: 449.99 },
    best: { seller: "TrailTech Outfitters", price: 369.0, stars: 4.8, years: 7, freeReturns: true, fullWarranty: false },
  },
  wait: {
    state: "wait",
    offersChecked: 11,
    price: 349.0,
    expectedLow: 285.0,
    dropChance: 0.74,
    drops: 4,
    months: 4,
  },
  nothing_better: {
    state: "nothing_better",
    offersChecked: 14,
    nextBest: { price: 254.0, daysLater: 9 },
  },
};

export function pickFakeData(url: string): PanelData {
  const state = new URL(url).searchParams.get("bw_state");
  return state && state in FAKE ? FAKE[state as PanelData["state"]] : FAKE.nothing_better;
}