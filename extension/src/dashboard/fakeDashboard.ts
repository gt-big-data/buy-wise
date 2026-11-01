export type Watched = {
  title: string;
  price: number;
  startPrice: number;
  target: number;
  dropChance: number; 
};

export type Alert = { when: string; text: string };

export type Call = {
  title: string;
  said: "Wait" | "Buy here" | "Better offer";
  happened: string;
  right: boolean;
  amount: number; 
};

export const WATCHING: Watched[] = [
  { title: "Bose QuietComfort Headphones", price: 279.0, startPrice: 349.0, target: 285, dropChance: 0.09 },
  { title: 'Samsung 32" Odyssey G55C Monitor', price: 349.99, startPrice: 349.99, target: 300, dropChance: 0.81 },
  { title: "Garmin Forerunner 265", price: 449.99, startPrice: 449.99, target: 400, dropChance: 0.42 },
];

export const ALERTS: Alert[] = [
  { when: "Today, 9:12 AM", text: "Bose QuietComfort Headphones dropped to $279.00, below your $285.00 target." },
  { when: "Oct 24", text: "Google Streamer 4K dropped to $79.99, its lowest in 90 days." },
];

export const TRACK_RECORD = {
  since: "Sep 14",
  calls: 31,
  right: 22,
  saved: 486.0,
  lost: 54.0,
  recent: [
    { title: "Bose QuietComfort Headphones", said: "Wait", happened: "Dropped $70.00 six days later", right: true, amount: 70 },
    { title: "Anker 737 Power Bank", said: "Wait", happened: "Didn't drop. Went up $6.00", right: false, amount: 6 },
    { title: "Beats Studio Pro", said: "Buy here", happened: "Dropped $30.00 four days later", right: false, amount: 30 },
    { title: "Google Streamer 4K", said: "Wait", happened: "Dropped $20.00 nine days later", right: true, amount: 20 },
    { title: "Kindle Paperwhite", said: "Better offer", happened: "Other seller was $18.00 less and arrived on time", right: true, amount: 18 },
  ] as Call[],
};