import type { PanelData } from "../content/fakeData";

// A past call from the track record replay, as /track-record/replay returns it.
export type ReplayCall = {
  asin: string;
  brand: string | null;
  predicted_on: string;
  said: "WAIT" | "BUY";
  kind: "wait_right" | "wait_wrong" | "buy_right" | "buy_wrong";
  right: boolean;
  price: number;
  drop_chance: number;
  low: number;
  end: number;
  amount: number;
};

// One call with the real daily prices around it, from /track-record/replay/{asin}/{day}.
export type ReplayDetail = ReplayCall & {
  horizon_days: number;
  expected_low: number | null;
  drops: number;
  months: number;
  low_on: string | null;
  end_on: string;
  prices: [string, number][];
  model_version: string;
};

const money = (n: number) => `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const replayDay = (iso: string) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

export function outcome(c: ReplayDetail): string {
  const on = (iso: string | null) => (iso ? ` on ${replayDay(iso)}` : "");
  switch (c.kind) {
    case "wait_right":
      return `Dropped to ${money(c.low)}${on(c.low_on)}. ${money(c.amount)} saved by waiting.`;
    case "wait_wrong":
      return c.amount > 0
        ? `Didn't drop 8%. It was ${money(c.end)}${on(c.end_on)}, ${money(c.amount)} more than buying that day.`
        : "Didn't drop 8%. Buying that day would have been just as good.";
    case "buy_right":
      return c.amount > 0
        ? `Didn't drop. Buying that day was right: it was ${money(c.end)}${on(c.end_on)}, ${money(c.amount)} higher.`
        : "Didn't drop. Buying that day was right.";
    case "buy_wrong":
      return `A miss: it dropped to ${money(c.low)}${on(c.low_on)}. Waiting would have saved ${money(c.amount)}.`;
  }
}

// The panel as it would have looked that day: Wait, or no reason to wait.
export function replayPanel(c: ReplayDetail): PanelData {
  if (c.said === "WAIT") {
    return {
      state: "wait",
      offersChecked: 0,
      price: c.price,
      expectedLow: c.expected_low,
      dropChance: c.drop_chance,
      drops: c.drops,
      months: c.months,
    };
  }
  return { state: "nothing_better", offersChecked: 0, nextBest: null };
}
