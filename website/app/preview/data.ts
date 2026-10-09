// Synthetic data shaped like what the extension and backend should produce by the
// end of the semester. Garmin and Bose prices come from our April 2026 data;
// sellers, ratings and track-record numbers are made up. No sentences live here:
// all text is generated from these fields by copy.ts.

import type { SkipReason, Status, Warranty } from "./copy";

export const TODAY = "2026-10-14";

// The one shopper profile every screen reads from. Option lists are what the
// first-run screen offers; the selected values are what the panel and dashboard show.
export const PROFILE_OPTIONS = {
  condition: ["Like new or better", "Any good condition", "New only"],
  patience: ["Need it soon", "Can wait a week or two", "Will wait for a deal"],
  alerts: ["On", "Only when I ask", "Off"],
} as const;

export const profile = {
  condition: "New only" as (typeof PROFILE_OPTIONS.condition)[number],
  patience: "Can wait a week or two" as (typeof PROFILE_OPTIONS.patience)[number],
  alerts: "On" as (typeof PROFILE_OPTIONS.alerts)[number],
  prime: true,
  minSaving: { dollars: 10, share: 0.05 },
};

export type Offer = {
  seller: string;
  condition: "New" | "Used, Like New";
  price: number;
  arrives: string;
  arrivesBy?: string;
  fulfilledByAmazon: boolean;
  freeReturns: boolean;
  rating?: { stars: number; count: number; years: number };
  warranty: Warranty;
  featured?: boolean;
  skip?: SkipReason;
};

export const garmin = {
  title: "Garmin Forerunner 265 Running Smartwatch, 46mm, AMOLED Display",
  rating: { stars: 4.6, count: 9812 },
  offersChecked: 28,
  offers: [
    {
      seller: "TrailTech Outfitters",
      condition: "New",
      price: 369.0,
      arrives: "2026-10-18",
      fulfilledByAmazon: true,
      freeReturns: true,
      rating: { stars: 4.8, count: 12410, years: 7 },
      warranty: "may_not_apply",
    },
    {
      seller: "Runner's Depot",
      condition: "New",
      price: 429.95,
      arrives: "2026-10-19",
      fulfilledByAmazon: false,
      freeReturns: true,
      rating: { stars: 4.9, count: 3201, years: 11 },
      warranty: "full",
    },
    {
      seller: "Amazon.com",
      condition: "New",
      price: 449.99,
      arrives: "2026-10-16",
      fulfilledByAmazon: true,
      freeReturns: true,
      warranty: "full",
      featured: true,
    },
    {
      seller: "Amazon Resale",
      condition: "Used, Like New",
      price: 322.4,
      arrives: "2026-10-17",
      fulfilledByAmazon: true,
      freeReturns: true,
      warranty: "may_not_apply",
      skip: "condition_profile",
    },
    {
      seller: "PeakRun Supply",
      condition: "New",
      price: 355.0,
      arrives: "2026-10-29",
      arrivesBy: "2026-11-06",
      fulfilledByAmazon: false,
      freeReturns: false,
      rating: { stars: 3.9, count: 86, years: 1 },
      warranty: "may_not_apply",
      skip: "low_ratings",
    },
  ] as Offer[],
  versions: [
    { name: "Forerunner 265, 46mm", price: 449.99, current: true, differs: [] as string[] },
    { name: "Forerunner 265S, 42mm", price: 449.99, differs: ["Smaller case", "About 2 days less battery"] },
    { name: "Forerunner 255", price: 249.99, differs: ["Previous generation", "No AMOLED screen"] },
  ],
};

// 120 days of daily prices for the Bose, ending today.
export const boseHistory = [
  349, 349, 349, 349, 349, 349, 349, 329, 299, 279, 279, 279, 279, 279, 299, 349, 349, 349, 349, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 329, 289, 279, 279, 279, 279, 289, 349, 349, 349, 349, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 319, 289, 279, 279, 279, 279, 279, 279, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 329, 299, 279, 279, 279, 289,
  329, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 349, 349,
];

export const bose = {
  title: "Bose QuietComfort Wireless Noise Cancelling Headphones",
  short: "Bose QuietComfort Headphones",
  rating: { stars: 4.5, count: 14388 },
  price: 349.0,
  arrives: "2026-10-15",
  offersChecked: 11,
  dropChance: 0.74,
  expectedLow: 285.0,
  higherAfterWait: 0.137,
  pattern: { drops: 4, months: 4, minDays: 5, maxDays: 8 },
};

export const sony = {
  title: "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
  rating: { stars: 4.4, count: 22105 },
  price: 248.0,
  arrives: "2026-10-14",
  offersChecked: 14,
  dropChance: 0.04,
  bestOther: { price: 254.0, condition: "New", arrives: "2026-10-23" },
  skippedCheaper: { price: 211.0, condition: "Used, Very Good", reason: "condition_profile" as SkipReason },
};

export type Watched = {
  title: string;
  price: number;
  startPrice: number;
  target: number | null;
  dropChance: number;
  low90: number;
  status: Extract<Status, "wait" | "buy_now">;
  history: number[];
};

export const watching: Watched[] = [
  {
    title: bose.short,
    price: 279.0,
    startPrice: 349.0,
    target: 290,
    dropChance: 0.09,
    low90: 279.0,
    status: "buy_now",
    history: boseHistory.slice(-57).concat([329, 299, 279]),
  },
  {
    title: "Google Streamer 4K",
    price: 79.99,
    startPrice: 99.99,
    target: null,
    dropChance: 0.06,
    low90: 79.99,
    status: "buy_now",
    history: [99, 99, 99, 99, 99, 99, 99, 89, 89, 99, 99, 99, 99, 99, 99, 99, 79, 79, 79, 79],
  },
  {
    title: "Samsung 32\" Odyssey G55C Monitor",
    price: 349.99,
    startPrice: 349.99,
    target: null,
    dropChance: 0.81,
    low90: 299.99,
    status: "wait",
    history: [329, 329, 349, 349, 349, 349, 309, 299, 299, 329, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349],
  },
  {
    title: "Garmin Forerunner 265",
    price: 449.99,
    startPrice: 449.99,
    target: 400,
    dropChance: 0.42,
    low90: 399.99,
    status: "wait",
    history: [449, 449, 435, 435, 449, 449, 449, 399, 399, 449, 449, 449, 449, 449, 431, 431, 449, 449, 449, 449],
  },
];

export const trackRecord = {
  since: "2026-09-14",
  shown: 31,
  correct: 22,
  saved: 486.0,
  quietShare: 0.68,
  misses: [
    { title: "Anker 737 Power Bank", said: "wait" as Status, outcome: { kind: "no_drop" as const, change: 6.0 } },
    { title: "Beats Studio Pro", said: "buy_here" as Status, outcome: { kind: "dropped" as const, amount: 30.0, days: 4 } },
  ],
};
