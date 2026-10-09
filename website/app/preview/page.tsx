import type { Metadata } from "next";
import Preview from "./Preview";

export const metadata: Metadata = {
  title: "BuyWise: product preview",
  description: "Where the BuyWise extension and dashboard are headed by the end of fall 2026. Synthetic data.",
  robots: { index: false },
};

export default function Page() {
  return <Preview />;
}
