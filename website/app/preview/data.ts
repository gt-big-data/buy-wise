// Synthetic data shaped like what the extension and backend should produce by the
// end of the semester. Prices for the Garmin and Bose come from our April 2026 data;
// sellers, ratings and track-record numbers are made up.

export type Offer = {
  seller: string;
  condition: string;
  price: number;
  arrives: string;
  fulfilled: "Amazon" | "Seller";
  returns: string;
  rating?: string;
  years?: number;
  verdict: "best" | "featured" | "skipped" | "ok";
  note?: string;
};

export const garmin = {
  title: "Garmin Forerunner 265 Running Smartwatch, 46mm, AMOLED Display",
  brand: "Garmin",
  rating: "4.6 out of 5 · 9,812 ratings",
  featured: { seller: "Amazon.com", price: 449.99, arrives: "Friday, Oct 16" },
  offers: [
    {
      seller: "TrailTech Outfitters",
      condition: "New",
      price: 369.0,
      arrives: "Sun, Oct 18",
      fulfilled: "Amazon",
      returns: "Free 30-day returns",
      rating: "4.8 · 12,410 ratings",
      years: 7,
      verdict: "best",
      note: "Not on Garmin's authorized-dealer list. Warranty may not apply.",
    },
    {
      seller: "Amazon.com",
      condition: "New",
      price: 449.99,
      arrives: "Fri, Oct 16",
      fulfilled: "Amazon",
      returns: "Free 30-day returns",
      verdict: "featured",
      note: "Amazon's pick. Full Garmin warranty.",
    },
    {
      seller: "Amazon Resale",
      condition: "Used, Like New",
      price: 322.4,
      arrives: "Sat, Oct 17",
      fulfilled: "Amazon",
      returns: "Free 30-day returns",
      verdict: "skipped",
      note: "Skipped: your profile says new only.",
    },
    {
      seller: "PeakRun Supply",
      condition: "New",
      price: 355.0,
      arrives: "Oct 29 – Nov 6",
      fulfilled: "Seller",
      returns: "Returns to seller, buyer pays shipping",
      rating: "3.9 · 86 ratings",
      years: 1,
      verdict: "skipped",
      note: "Skipped: few ratings, ships from overseas.",
    },
    {
      seller: "Runner's Depot",
      condition: "New",
      price: 429.95,
      arrives: "Mon, Oct 19",
      fulfilled: "Seller",
      returns: "30-day returns",
      rating: "4.9 · 3,201 ratings",
      years: 11,
      verdict: "ok",
      note: "Authorized Garmin dealer. $20 less than Amazon, arrives Monday.",
    },
  ] as Offer[],
  versions: [
    { name: "Forerunner 265, 46mm", price: 449.99, current: true, reviews: "" },
    { name: "Forerunner 265S, 42mm", price: 449.99, reviews: "Same features, smaller case. Reviews note slightly shorter battery." },
    {
      name: "Forerunner 255 (previous gen)",
      price: 249.99,
      reviews: "No AMOLED screen. Reviewers who run outdoors rarely mention missing it.",
    },
  ],
};

// 120 days of daily prices, ending today.
export const boseHistory = [
  349, 349, 349, 349, 349, 349, 349, 329, 329, 299, 279, 279, 279, 299, 329, 349, 349, 349, 349, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 329, 299, 279, 279, 279, 279, 299, 329, 349, 349, 349, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 329, 299, 279, 279, 279, 279, 279, 299, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 329, 299, 279, 279, 279, 299,
  329, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349,
  349, 349, 349, 349, 349, 349, 349, 349, 349, 349,
];

export const watching = [
  {
    title: "Bose QuietComfort Headphones",
    price: 279.0,
    was: 349.0,
    target: 290,
    status: "target" as const,
    history: boseHistory.slice(-60).concat([329, 299, 279]),
  },
  {
    title: "Samsung 32\" Odyssey G55C Monitor",
    price: 349.99,
    was: 349.99,
    target: null,
    status: "waiting" as const,
    note: "81% chance of a drop in 14 days",
    history: [329, 329, 349, 349, 349, 349, 309, 299, 299, 329, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349],
  },
  {
    title: "Garmin Forerunner 265",
    price: 449.99,
    was: 449.99,
    target: 400,
    status: "waiting" as const,
    note: "Target $400. Lowest in 90 days: $399.99",
    history: [449, 449, 435, 435, 449, 449, 449, 399, 399, 449, 449, 449, 449, 449, 431, 431, 449, 449, 449, 449],
  },
  {
    title: "Google Streamer 4K",
    price: 79.99,
    was: 99.99,
    target: null,
    status: "buy" as const,
    note: "Good time to buy. Drops past this are rare.",
    history: [99, 99, 99, 99, 99, 99, 99, 89, 89, 99, 99, 99, 99, 99, 99, 99, 79, 79, 79, 79],
  },
];

export const trackRecord = {
  since: "Sep 14",
  shown: 31,
  played: 22,
  saved: 486,
  quietPct: 68,
  misses: [
    {
      title: "Anker 737 Power Bank",
      what: "We said wait. It didn't drop for three weeks, then went up $6.",
    },
    {
      title: "Beats Studio Pro",
      what: "We said buy. It dropped $30 four days later during an unannounced sale.",
    },
  ],
};
