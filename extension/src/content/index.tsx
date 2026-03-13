import React from "react";
import { createRoot, Root } from "react-dom/client";
import BuyWisePanel from "../components/BuyWisePanel";
import { isAmazonProductPage, extractASIN } from "./amazon";
import { getMockBuyWiseData } from "./mockData";
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
  if (existing) {
    existing.remove();
  }

  const container = document.createElement("div");
  container.id = "buywise-root";
  document.body.appendChild(container);
  return container;
}

function mountFloatingPanel(): boolean {
  if (!isAmazonProductPage()) {
    return false;
  }

  const asin = extractASIN();
  if (!asin) {
    return false;
  }

  const data = getMockBuyWiseData(asin);

  removeExistingUI();
  currentContainer = createContainer();
  currentRoot = createRoot(currentContainer);

  const handleClose = () => {
    removeExistingUI();
  };

  const handleActionClick = () => {
    console.log("BuyWise floating action clicked", {
      asin: data.asin,
      recommendation: data.recommendation
    });

    chrome.storage.local.set({
      lastBuyWiseAction: {
        asin: data.asin,
        recommendation: data.recommendation,
        timestamp: new Date().toISOString(),
        source: "page-floating-popup"
      }
    });
  };

  currentRoot.render(
    <BuyWisePanel
      data={data}
      onClose={handleClose}
      onActionClick={handleActionClick}
      showCloseButton={true}
      title="BuyWise"
      floating={true}
    />
  );

  return true;
}

function init(): void {
  console.log("BuyWise content script loaded");

  if (!isAmazonProductPage()) {
    return;
  }

  window.setTimeout(() => {
    mountFloatingPanel();
  }, 500);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "BUYWISE_OPEN_PANEL") {
    const opened = mountFloatingPanel();
    sendResponse({ opened });
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
