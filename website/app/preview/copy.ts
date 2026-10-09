// Every piece of text the extension shows comes from these templates, so the same
// fact always reads the same way on every screen. Add a template here before adding
// a one-off sentence to a component.

export const money = (n: number) => `$${n.toFixed(2)}`;
export const pct = (p: number) => `${Math.round(p * 100)}%`;

const dateFmt = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
export const date = (iso: string) => dateFmt.format(new Date(`${iso}T12:00:00Z`));
export const dateRange = (a: string, b?: string) => (b ? `${date(a)} – ${date(b)}` : date(a));
export const daysBetween = (a: string, b: string) =>
  Math.round((new Date(`${b}T12:00:00Z`).getTime() - new Date(`${a}T12:00:00Z`).getTime()) / 86_400_000);

// One status vocabulary for the panel, the offer list, the dashboard and alerts.
export type Status = "better_offer" | "wait" | "buy_here" | "buy_now";
export const STATUS: Record<Status, string> = {
  better_offer: "Better offer",
  wait: "Wait",
  buy_here: "Buy here",
  buy_now: "Buy now",
};

// Row labels. Panels pick from this list; they never invent their own.
export const LABEL = {
  arrives: "Arrives",
  returns: "Returns",
  warranty: "Warranty",
  dropChance: "Chance of a drop",
  expectedLow: "Expected 14-day low",
  low90: "90-day low",
  pattern: "Pattern",
  bestOther: "Best other offer",
  target: "Your target",
  price: "Price",
  sinceWatching: "Since you started watching",
  ifNoDrop: "If it doesn't drop",
} as const;

export type Warranty = "full" | "may_not_apply";
export const WARRANTY: Record<Warranty, string> = {
  full: "Full manufacturer warranty",
  may_not_apply: "Warranty may not apply",
};

export type SkipReason = "condition_profile" | "low_ratings" | "slow_shipping";
export const SKIP: Record<SkipReason, string> = {
  condition_profile: "Condition doesn't match your profile",
  low_ratings: "Too few seller ratings",
  slow_shipping: "Arrives more than a week later",
};

export const checked = (n: number) => `Checked ${n} offers`;
export const dropChanceLine = (p: number) => `${pct(p)} chance of an 8%+ drop in 14 days`;
export const patternLine = (drops: number, months: number, minDays: number, maxDays: number) =>
  `${drops} drops in ${months} months, each lasting ${minDays}–${maxDays} days`;
export const MIN_WAIT_CHANCE = 0.4;
export const ifNoDropLine = (higherShare: number) => `Usually about the same price. ${pct(higherShare)} of past waits ended higher`;
export const bestOtherLine = (o: { price: number; condition: string; arrives: string } | null) =>
  o ? `${money(o.price)}, ${o.condition.toLowerCase()}, arrives ${date(o.arrives)}` : "None cheaper";
export const deliveryDelta = (featured: string, other: string) => {
  const d = daysBetween(featured, other);
  return d === 0 ? "Same day" : d > 0 ? `${d} day${d > 1 ? "s" : ""} later` : `${-d} day${d < -1 ? "s" : ""} sooner`;
};
export const savingsVs = (base: number, price: number) =>
  price < base ? `${money(base - price)} less` : price > base ? `${money(price - base)} more` : "Same price";

// Headlines: one sentence per status, always built from numbers.
export const headline = {
  better_offer: (savings: number, condition: string) => `Same item, ${condition.toLowerCase()}, ${money(savings)} less.`,
  wait: (p: number) => `${pct(p)} chance it drops 8% or more in the next 14 days.`,
  buy_here: () => "This is the best offer for you.",
  buy_now_target: (price: number, target: number) => `${money(price)}, below your ${money(target)} target.`,
  buy_now_low: (price: number) => `${money(price)}, its lowest in 90 days.`,
};

export const profileSummary = (p: { condition: string; patience: string; prime: boolean }) =>
  [p.condition, p.patience, p.prime ? "Prime" : null].filter(Boolean).join(" · ");

export type OfferTag = "best" | "featured" | "eligible" | "skipped";
export const OFFER_TAG: Record<OfferTag, string> = {
  best: "Best for you",
  featured: "Amazon's pick",
  eligible: "Eligible",
  skipped: "Skipped",
};

export const ratingLine = (r: { stars: number; count: number }) => `${r.stars.toFixed(1)} stars · ${r.count.toLocaleString("en-US")} ratings`;
export const sellerLine = (r?: { stars: number; count: number; years: number }) =>
  r ? `${ratingLine(r)} · ${r.years} yr${r.years === 1 ? "" : "s"} selling` : "Sold by Amazon";
export const fulfillmentLine = (o: { fulfilledByAmazon: boolean; freeReturns: boolean }) =>
  `${o.fulfilledByAmazon ? "Fulfilled by Amazon" : "Ships from seller"} · ${o.freeReturns ? "Free returns" : "Buyer pays return shipping"}`;

export const WATCH_LABEL = { target: "Target", low90: "90-day low", drop: "Drop chance" } as const;
export const watchFacts = (w: { target: number | null; low90: number; dropChance: number }) =>
  [
    w.target ? `${WATCH_LABEL.target} ${money(w.target)}` : "No target",
    `${WATCH_LABEL.low90} ${money(w.low90)}`,
    `${WATCH_LABEL.drop} ${pct(w.dropChance)}`,
  ].join(" · ");
export const watchingConfirm = (target: number | null) =>
  target ? `Watching. We'll notify you at ${money(target)} or lower.` : "Watching. We'll notify you when it's a good time to buy.";

export type Miss = { kind: "no_drop"; change: number } | { kind: "dropped"; amount: number; days: number };
export const missLine = (said: Status, m: Miss) =>
  `Said ${STATUS[said]}. ` +
  (m.kind === "no_drop"
    ? `No 8% drop within 14 days; price ${m.change >= 0 ? "rose" : "fell"} ${money(Math.abs(m.change))}.`
    : `Dropped ${money(m.amount)} ${m.days} days later.`);

export const PROFILE_LABEL = {
  condition: "Condition",
  patience: "Patience",
  alerts: "Price alerts",
  prime: "Prime",
  minSaving: "Minimum saving shown",
} as const;
