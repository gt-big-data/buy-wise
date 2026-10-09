import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: "BuyWise — Buy smarter on Amazon",
  description:
    "BuyWise is a Chrome extension that reads every offer on an Amazon product page and tells you, in one plain line, when there's a better way to buy, and why. A GT Big Data project at Georgia Tech.",
  openGraph: {
    title: "BuyWise — Buy smarter on Amazon",
    description: "Every offer on the page, weighed for you. A GT Big Data project.",
    images: ["/mark.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
