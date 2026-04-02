// Service worker — handles background tasks and proxies backend requests.
// Content scripts can't fetch localhost directly from an HTTPS page (Chrome MV3
// Private Network Access policy), so all backend calls go through here instead.

const BACKEND_URL = "http://localhost:8000";

chrome.runtime.onInstalled.addListener(() => {
  console.log("BuyWise extension installed");
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
        timestamp: new Date().toISOString(),
      }),
    }).catch(() => {});
    return false;
  }

  return false;
});
