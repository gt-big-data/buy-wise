// Price alerts for watched products (#74). The backend keeps the watchlist and
// the prices; this checks each watched product against its target and sends a
// Chrome notification the first time it's at or below it.

import { BACKEND_URL } from "../config";
import { getUserId } from "../userId";

export type WatchedItem = {
  asin: string;
  title: string;
  target: number | null;
  price: number | null; // null when the backend has no price for it
  recommendation: string | null;
};

export type SentAlert = { asin: string; title: string; price: number; target: number; at: string };

const ALERTS_KEY = "buywise_alerts"; // newest first, shown on the dashboard
const ALERTED_KEY = "buywise_alerted"; // asin -> target it already alerted for

async function currentPrice(asin: string): Promise<number | null> {
  const r = await fetch(`${BACKEND_URL}/price-history/${asin}`);
  if (!r.ok) return null;
  const body = await r.json();
  return typeof body.current_price === "number" ? body.current_price : null;
}

export async function loadWatchlist(): Promise<WatchedItem[]> {
  const userId = await getUserId();
  const r = await fetch(`${BACKEND_URL}/watchlist/${userId}`);
  if (!r.ok) throw new Error(`watchlist: ${r.status}`);
  const body = await r.json();

  const items: WatchedItem[] = [];
  for (const row of body.watchlist ?? []) {
    items.push({
      asin: row.asin,
      title: row.title,
      target: row.target_price === null ? null : Number(row.target_price),
      price: await currentPrice(row.asin),
      recommendation: row.current_recommendation ?? null,
    });
  }
  return items;
}

export async function loadAlerts(): Promise<SentAlert[]> {
  const res = await chrome.storage.local.get(ALERTS_KEY);
  return Array.isArray(res[ALERTS_KEY]) ? res[ALERTS_KEY] : [];
}

export async function checkPrices(): Promise<number> {
  const items = await loadWatchlist();
  const stored = await chrome.storage.local.get(ALERTED_KEY);
  const alerted: Record<string, number> = stored[ALERTED_KEY] ?? {};
  const alerts = await loadAlerts();
  let sent = 0;

  for (const item of items) {
    if (item.target === null || item.price === null) continue;
    if (item.price > item.target) continue;
    // One alert per target. Setting a new target can alert again.
    if (alerted[item.asin] === item.target) continue;

    chrome.notifications.create(`buywise-${item.asin}`, {
      type: "basic",
      iconUrl: "logo.png",
      title: `Price drop: ${item.title.slice(0, 60)}`,
      message: `Now $${item.price.toFixed(2)}, at or below your $${item.target.toFixed(2)} target.`,
    });
    alerted[item.asin] = item.target;
    alerts.unshift({ asin: item.asin, title: item.title, price: item.price, target: item.target, at: new Date().toISOString() });
    sent += 1;
  }

  await chrome.storage.local.set({ [ALERTED_KEY]: alerted, [ALERTS_KEY]: alerts.slice(0, 20) });
  return sent;
}