import { Decision, toPanelData } from "./decision";

type Offer = { seller: string; condition: string; price: number; arrives: string; daysOut: number };

// What the panel draws. Built from the decision endpoint's answer by toPanelData.
export type PanelData =
  | {
      state: "better_offer";
      offersChecked: number;
      reason: string;
      amazon: Offer;
      best: Offer & { stars: number; years: number; freeReturns: boolean; fullWarranty: boolean };
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

function daysFromNow(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() + n);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}

// Fake answers in the decision endpoint's format (#92), until it's live.
export const FAKE: Record<Decision["status"], Decision> = {
  better_offer: {
    show: true,
    status: "better_offer",
    offers_checked: 28,
    amazon: { seller: "Amazon.com", condition: "New", price: 449.99, arrives: daysFromNow(2) },
    best: {
      seller: "TrailTech Outfitters",
      condition: "New",
      price: 369.0,
      arrives: daysFromNow(4),
      stars: 4.8,
      years: 7,
      free_returns: true,
      full_warranty: false,
    },
    reason: "New, $80.99 less. TrailTech Outfitters has 4.8 stars over 7 years and free returns. Arrives 2 days later.",
    wait: null,
    next_best: null,
  },
  wait: {
    show: true,
    status: "wait",
    offers_checked: 11,
    amazon: null,
    best: null,
    reason: "Dropped 4 times in the last 4 months.",
    wait: { price: 349.0, expected_low: 285.0, drop_chance: 0.74, drops: 4, months: 4 },
    next_best: null,
  },
  nothing_better: {
    show: true,
    status: "nothing_better",
    offers_checked: 14,
    amazon: null,
    best: null,
    reason: "Amazon's pick is the best offer for you.",
    wait: null,
    next_best: { price: 254.0, days_later: 9 },
  },
};

// Developer mode: ?bw_state=better_offer, wait or nothing_better. Returns null when the URL has none.
export function pickFakeData(url: string): PanelData | null {
  const state = new URL(url).searchParams.get("bw_state");
  if (!state || !(state in FAKE)) return null;
  return toPanelData(FAKE[state as Decision["status"]]);
}