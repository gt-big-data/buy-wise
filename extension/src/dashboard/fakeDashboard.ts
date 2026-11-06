// Fake track record for the dashboard (#93) until #85 is live. Watching and
// alerts are real now.

export type Call = {
  title: string;
  said: "Wait" | "Buy here" | "Better offer";
  happened: string;
  right: boolean;
  amount: number; // saved when right, lost when wrong
};

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