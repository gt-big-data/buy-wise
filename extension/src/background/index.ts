// Service worker — handles background tasks and proxies backend requests.
// Content scripts can't fetch localhost directly from an HTTPS page (Chrome MV3
// Private Network Access policy), so all backend calls go through here instead.

import { BACKEND_URL } from "../config";
import { getUserId } from "../userId";
import { checkPrices, loadAlerts, loadWatchlist } from "./alerts";

chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.log("BuyWise extension installed");
  getUserId();
  // Ask the three profile questions once, on a fresh install only (not on updates).
  if (reason === chrome.runtime.OnInstalledReason.INSTALL) {
    chrome.tabs.create({ url: chrome.runtime.getURL("profile.html?welcome") });
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "BUYWISE_FETCH") {
    const { asin } = message;
    Promise.all([
      fetch(`${BACKEND_URL}/predict/${asin}`).then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail ?? `predict: ${r.status}`);
        }
        return r.json();
      }),
      fetch(`${BACKEND_URL}/price-history/${asin}`).then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail ?? `price-history: ${r.status}`);
        }
        return r.json();
      }),
    ])
      .then(([predict, history]) => sendResponse({ ok: true, predict, history }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep message channel open for async response
  }

  if (message?.type === "BUYWISE_POST_ACTIVITY") {
    getUserId()
      .then((userId) =>
        fetch(`${BACKEND_URL}/activity`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            asin: message.asin,
            action: message.action,
            user_id: userId,
            timestamp: new Date().toISOString(),
          }),
        })
      )
      .catch(() => {});
    return false;
  }

  if (message?.type === "BUYWISE_OPEN_DASHBOARD") {
    chrome.tabs.create({ url: chrome.runtime.getURL("dashboard.html") });
    return false;
  }

  if (message?.type === "BUYWISE_GET_WATCHLIST") {
    Promise.all([loadWatchlist(), loadAlerts()])
      .then(([items, alerts]) => sendResponse({ ok: true, items, alerts }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message?.type === "BUYWISE_CHECK_PRICES") {
    checkPrices()
      .then((sent) => sendResponse({ ok: true, sent }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message?.type === "BUYWISE_WATCHLIST_ADD") {
    const { asin, target_price } = message;
    getUserId()
      .then((userId) =>
        fetch(`${BACKEND_URL}/watchlist`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, asin, target_price: target_price ?? null }),
        })
      )
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail ?? `watchlist: ${r.status}`);
        }
        return r.json();
      })
      .then((data) => {
        sendResponse({ ok: true, ...data });
        // Check right away, so a target at or above today's price alerts now.
        checkPrices().catch(() => {});
      })
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  return false;
});

// Clicking any BuyWise notification opens the dashboard.
chrome.notifications.onClicked.addListener(() => {
  chrome.tabs.create({ url: chrome.runtime.getURL("dashboard.html") });
});

// Re-check watched prices every hour and when Chrome starts. Prices only change
// when the daily job (#90) runs, so hourly is plenty.
chrome.alarms.get("buywise-check-prices").then((alarm) => {
  if (!alarm) chrome.alarms.create("buywise-check-prices", { periodInMinutes: 60 });
});
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "buywise-check-prices") checkPrices().catch(() => {});
});
chrome.runtime.onStartup.addListener(() => {
  checkPrices().catch(() => {});
});