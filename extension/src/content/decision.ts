import type { PanelData } from "./fakeData";

// What the decision endpoint (#92) returns for one product page. Proposed format,
// not final yet. If the backend changes a field, only toPanelData needs to change.

type ApiOffer = {
  seller: string;
  condition: string;
  price: number;
  arrives: string; // YYYY-MM-DD
};

export type Decision = {
  show: boolean;
  status: "better_offer" | "wait" | "nothing_better";
  offers_checked: number;
  amazon: ApiOffer | null;
  best: (ApiOffer & { stars: number; years: number; free_returns: boolean; full_warranty: boolean }) | null;
  reason: string;
  wait: { price: number; expected_low: number; drop_chance: number; drops: number; months: number } | null;
  next_best: { price: number; days_later: number } | null;
};

const DAY_MS = 24 * 60 * 60 * 1000;

function parseDay(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

function toOffer(o: ApiOffer, today: Date) {
  const day = parseDay(o.arrives);
  const start = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  return {
    seller: o.seller,
    condition: o.condition,
    price: o.price,
    arrives: day.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }),
    daysOut: Math.round((day.getTime() - start.getTime()) / DAY_MS),
  };
}

// Turns the endpoint's answer into what the panel draws. Returns null when the
// panel should stay hidden, or when the answer is missing what its status needs.
export function toPanelData(d: Decision, today = new Date()): PanelData | null {
  if (!d.show) return null;

  if (d.status === "better_offer") {
    if (!d.amazon || !d.best) return null;
    return {
      state: "better_offer",
      offersChecked: d.offers_checked,
      reason: d.reason,
      amazon: toOffer(d.amazon, today),
      best: {
        ...toOffer(d.best, today),
        stars: d.best.stars,
        years: d.best.years,
        freeReturns: d.best.free_returns,
        fullWarranty: d.best.full_warranty,
      },
    };
  }

  if (d.status === "wait") {
    if (!d.wait) return null;
    return {
      state: "wait",
      offersChecked: d.offers_checked,
      price: d.wait.price,
      expectedLow: d.wait.expected_low,
      dropChance: d.wait.drop_chance,
      drops: d.wait.drops,
      months: d.wait.months,
    };
  }

  return {
    state: "nothing_better",
    offersChecked: d.offers_checked,
    nextBest: d.next_best ? { price: d.next_best.price, daysLater: d.next_best.days_later } : null,
  };
}