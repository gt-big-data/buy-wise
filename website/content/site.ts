// Everything on the page that people will want to edit lives here.

export const GITHUB_URL = "https://github.com/gt-big-data/buy-wise";

export const steps: { title: string; body: string; factors?: string }[] = [
  {
    title: "Amazon is a marketplace",
    body: "Amazon sells on the product page, and so do outside businesses: authorized dealers, liquidators and resellers. Each one sets its own price, and many reprice automatically several times a day.",
  },
  {
    title: "Amazon picks one seller for the button",
    body: "The featured offer is chosen for a typical shopper and for Amazon's business. It isn't always the best deal for you, and some of the lowest prices stay hidden until checkout.",
    factors: "Weighed by: price with shipping, delivery speed, seller track record, stock.",
  },
  {
    title: "BuyWise reads the rest and works for you",
    body: "We look at every offer, weigh what makes a cheaper one worth it or not, and tell you plainly with the reason: seller trust, warranty, shipping, returns and condition.",
  },
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

// Fill in each person as the roster is confirmed. Every field shows on their card;
// placeholders render in grey until replaced.
export type Member = {
  name: string;
  role: string;
  major: string; // short form, e.g. "CS"
  gradYear: string; // two digits, e.g. "28"
  interests: [string, string, string];
  linkedin: string; // full URL
  github: string; // full URL
  photo?: string; // path under public/, e.g. "/team/jane.jpg"
};

export const PLACEHOLDER = "Name Surname";

const placeholder = (role: string, n: number): Member[] =>
  Array.from({ length: n }, () => ({
    name: PLACEHOLDER,
    role,
    major: "Major",
    gradYear: "YY",
    interests: ["Interest", "Interest", "Interest"],
    linkedin: "",
    github: "",
  }));

export const team: { group: string; members: Member[] }[] = [
  { group: "Project leads", members: placeholder("Project Lead", 3) },
  {
    group: "Members",
    members: [...placeholder("Analysis", 4), ...placeholder("Platform", 4), ...placeholder("Data Viz", 2)],
  },
];
