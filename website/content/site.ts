// Everything on the page that people will want to edit lives here.

export const GITHUB_URL = "https://github.com/gt-big-data/buy-wise";

// Measured in studies/ (FINDINGS.md, s009) and on the live Garmin listing, Oct 2026.
export const stats = [
  { value: "28", label: "offers on one product page we checked" },
  { value: "~$80", label: "gap between sellers, same new watch" },
  { value: "3 in 10", label: "products had a cheaper new seller" },
  { value: "~$20", label: "average gap when they did" },
];

export const built = [
  { title: "Chrome extension", detail: "React panel injected on Amazon pages" },
  { title: "Backend API", detail: "FastAPI + MySQL, watchlist & activity" },
  { title: "Price data pipeline", detail: "Keepa price history for 300+ products" },
  { title: "Price-drop model", detail: "Calibrated chance of a drop in 14 days" },
];

export const done = [
  {
    title: "Shipped the spring prototype",
    detail: "Extension, backend, data pipeline and model, working together",
  },
  {
    title: "Rebuilt the forecasting model",
    detail: "Right 55% of the time when it says wait, against a 22% base rate, on months it never trained on",
  },
  {
    title: "Measured where shoppers lose money",
    detail: "Found that price gaps between sellers outweigh waiting for a price drop",
  },
];

export const building = [
  { title: "A bigger, multi-category dataset", detail: "Thousands of products across categories and price ranges" },
  { title: "An offer reader", detail: "Every seller, condition and price on the page, read reliably" },
  { title: "A decision layer", detail: "Seller trust, warranty, shipping and condition, weighed for the buyer" },
  { title: "A redesigned extension", detail: "A cleaner panel on this design system, with a real track record" },
];

// Fill in names as the roster is confirmed. Leave linkedin/github empty to hide them.
export type Member = { name: string; role: string; linkedin?: string; github?: string; photo?: string };

const placeholder = (role: string, n: number): Member[] =>
  Array.from({ length: n }, () => ({ name: "Name Surname", role }));

export const team: { group: string; members: Member[] }[] = [
  { group: "Project leads", members: placeholder("Project Lead", 3) },
  { group: "Analysis", members: placeholder("Analysis", 4) },
  { group: "Platform", members: placeholder("Platform", 4) },
  { group: "Data visualization", members: placeholder("Data Viz", 2) },
];
