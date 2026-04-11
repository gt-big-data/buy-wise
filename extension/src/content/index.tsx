import React from "react";
import { createRoot, Root } from "react-dom/client";
import BuyWisePanel from "../components/BuyWisePanel";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import { isAmazonProductPage, extractASIN, extractProductImage } from "./amazon";
import { BuyWiseData } from "../types";
import "./styles.css";

let currentRoot: Root | null = null;
let currentContainer: HTMLDivElement | null = null;

function removeExistingUI(): void {
  if (currentRoot) {
    currentRoot.unmount();
    currentRoot = null;
  }
  if (currentContainer && currentContainer.parentNode) {
    currentContainer.parentNode.removeChild(currentContainer);
  }
  currentContainer = null;
}

function createContainer(): HTMLDivElement {
  const existing = document.getElementById("buywise-root");
  if (existing) existing.remove();
  const container = document.createElement("div");
  container.id = "buywise-root";
  document.body.appendChild(container);
  return container;
}

async function fetchBuyWiseData(asin: string): Promise<BuyWiseData> {
  const response = await chrome.runtime.sendMessage({ type: "BUYWISE_FETCH", asin });
  if (!response?.ok) throw new Error(response?.error ?? "unknown error from background");

  const { predict, history } = response;
  const watchedList = await chrome.storage.local.get(["buywise_watchlist"]);
  const isWatched = Array.isArray(watchedList.buywise_watchlist) && watchedList.buywise_watchlist.some((w: any) => w.asin === predict.asin);

  return {
    asin: predict.asin,
    productTitle: document.title.replace(/^Amazon\.com\s*:\s*/i, "").trim(),
    imageUrl: extractProductImage() || undefined,
    currentPrice: history.current_price,
    predictedBestPrice: history.predicted_best_price,
    expectedSavings: predict.potential_savings,
    confidence: Math.round(predict.confidence),
    recommendation: predict.recommendation as "BUY" | "WAIT",
    why: predict.why,
    chartTitle: history.chart_title,
    points: history.points,
    isWatched,
  };
}

function postActivity(asin: string, action: string): void {
  chrome.runtime.sendMessage({ type: "BUYWISE_POST_ACTIVITY", asin, action }).catch(() => {});
}

async function mountFloatingPanel(): Promise<boolean> {
  if (!isAmazonProductPage()) return false;

  const asin = extractASIN();
  if (!asin) return false;

  removeExistingUI();
  currentContainer = createContainer();
  currentRoot = createRoot(currentContainer);

  currentRoot.render(
    <div className="buywise-floating-shell">
      <LoadingState />
    </div>
  );

  let data: BuyWiseData;
  try {
    data = await fetchBuyWiseData(asin);
  } catch (err) {
    console.error("BuyWise: failed to fetch data", err);
    const isNotTracked = err instanceof Error && err.message === "not_tracked";
    currentRoot.render(
      <div className="buywise-floating-shell">
        <ErrorState
          title={isNotTracked ? "Product not tracked" : "Couldn't load recommendation"}
          message={
            isNotTracked
              ? "BuyWise doesn't have price history for this product yet. Try a more popular item."
              : "BuyWise couldn't reach the server. Make sure the backend is running."
          }
          onRetry={isNotTracked ? undefined : () => { mountFloatingPanel(); }}
        />
      </div>
    );
    return true;
  }

  const handleClose = () => removeExistingUI();

  const handleActionClick = () => {
    const action = data.recommendation === "BUY" ? "purchased" : "dismissed";
    postActivity(data.asin, action);
    chrome.storage.local.set({
      lastBuyWiseAction: {
        asin: data.asin,
        recommendation: data.recommendation,
        timestamp: new Date().toISOString(),
        source: "page-floating-popup",
      },
    });
  };

  const handleWatchlistClick = () => {
    postActivity(data.asin, "watchlisted");
    chrome.storage.local.get(["buywise_watchlist"], (res) => {
      const currentList = Array.isArray(res.buywise_watchlist) ? res.buywise_watchlist : [];
      if (!currentList.some((w: any) => w.asin === data.asin)) {
         currentList.push({
            asin: data.asin,
            productTitle: data.productTitle,
            targetPrice: data.predictedBestPrice,
            addedAt: new Date().toISOString()
         });
         chrome.storage.local.set({ buywise_watchlist: currentList });
      }
    });
    
    chrome.storage.local.set({
      lastBuyWiseAction: {
        asin: data.asin,
        action: "watchlisted",
        timestamp: new Date().toISOString(),
        source: "page-floating-popup-watchlist",
      },
    });
  };

  currentRoot.render(
    <BuyWisePanel
      data={data}
      onClose={handleClose}
      onActionClick={handleActionClick}
      onWatchlistClick={handleWatchlistClick}
      showCloseButton={true}
      title="BuyWise"
      floating={true}
    />
  );

  return true;
}

function init(): void {
  console.log("BuyWise content script loaded");
  if (!isAmazonProductPage()) return;
  window.setTimeout(() => { mountFloatingPanel(); }, 500);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "BUYWISE_OPEN_PANEL") {
    mountFloatingPanel().then((opened) => sendResponse({ opened }));
    return true;
  }
  if (message?.type === "BUYWISE_CLOSE_PANEL") {
    removeExistingUI();
    sendResponse({ closed: true });
    return true;
  }
  if (message?.type === "BUYWISE_IS_PANEL_OPEN") {
    sendResponse({ isOpen: Boolean(currentContainer) });
    return true;
  }
  return false;
});

init();
