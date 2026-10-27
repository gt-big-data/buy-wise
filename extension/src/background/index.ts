// Service worker — handles background tasks and proxies backend requests.
// Content scripts can't fetch localhost directly from an HTTPS page (Chrome MV3
// Private Network Access policy), so all backend calls go through here instead.

const BACKEND_URL = "http://localhost:8000";

chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.log("BuyWise extension installed");
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
    fetch(`${BACKEND_URL}/activity`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        asin: message.asin,
        action: message.action,
        user_id: "1",
        timestamp: new Date().toISOString(),
      }),
    }).catch(() => {});
    return false;
  }

  if (message?.type === "BUYWISE_WATCHLIST_ADD") {
    const { asin } = message;
    fetch(`${BACKEND_URL}/watchlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: 1, asin }),
    })
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail ?? `watchlist: ${r.status}`);
        }
        return r.json();
      })
      .then((data) => sendResponse({ ok: true, ...data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  return false;
});
