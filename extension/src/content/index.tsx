import React from "react";
import { createRoot, Root } from "react-dom/client";
import { isAmazonProductPage, extractASIN } from "./amazon";
import { extractOffers } from "./offers";
import { OfferListResult } from "./offerList";
import { readVersions } from "./versions";
import { buildDecisionRequest } from "./decisionRequest";
import { Decision, toPanelData } from "./decision";
import { loadProfile, resolvedAnswers } from "../profile";
import "./styles.css";
import Panel, { PanelStatus } from "./Panel";
import { openOffersPanel } from "./offerPanel";
import "./panel.css";
import { pickFakeData, PanelData } from "./fakeData";

let currentRoot: Root | null = null;
let currentContainer: HTMLDivElement | null = null;
// The product the panel answers for. Stays set while the endpoint says there's
// nothing to show, so picking another size or color still gets its own answer.
// Cleared when the shopper closes the panel.
let currentAsin: string | null = null;

function unmountPanel(): void {
  currentRoot?.unmount();
  currentRoot = null;
  currentContainer?.remove();
  currentContainer = null;
}

function removeExistingUI(): void {
  unmountPanel();
  currentAsin = null;
}

function mountContainer(): Root {
  unmountPanel();
  document.getElementById("buywise-root")?.remove();
  currentContainer = document.createElement("div");
  currentContainer.id = "buywise-root";
  currentContainer.classList.add("bw-floating");
  document.body.appendChild(currentContainer);
  currentRoot = createRoot(currentContainer);
  return currentRoot;
}

const markUrl = () => chrome.runtime.getURL("mark.png");

// Developer mode: ?bw_state=better_offer, wait or nothing_better shows fake data
function mountFakePanel(data: PanelData): boolean {
  if (!isAmazonProductPage()) return false;
  removeExistingUI();
  const root = mountContainer();
  const render = () =>
    root.render(
      <Panel data={data} markUrl={markUrl()} onClose={removeExistingUI} onSeeOffers={() => openOffersPanel()} offers={extractOffers()} />
    );
  render();
  whenOtherSellersLoad(() => {
    if (root === currentRoot) render();
  });
  return true;
}

const OTHER_SELLERS_WAIT_MS = 10000;

// Amazon fills in the "Other sellers" box after the page loads; call back once it's there
function whenOtherSellersLoad(onLoad: () => void): void {
  if (extractOffers().otherOffersCount !== null) return;
  const started = Date.now();
  const timer = window.setInterval(() => {
    if (extractOffers().otherOffersCount !== null) {
      window.clearInterval(timer);
      onLoad();
    } else if (Date.now() - started > OTHER_SELLERS_WAIT_MS) {
      window.clearInterval(timer);
    }
  }, 500);
}

const SETTLE_MAX_MS = 2000;

// Amazon fills in the buy-box price and the version swatches' prices a moment
// after load. Wait for them, briefly, so the first request carries them.
function pageSettled(): Promise<void> {
  const ready = () => {
    if (extractOffers().oneTimePrice === null) return false;
    const v = readVersions();
    return v.status !== "ok" || v.versions.some((x) => !x.selected && x.price !== null);
  };
  return new Promise((resolve) => {
    if (ready()) return resolve();
    const started = Date.now();
    const timer = window.setInterval(() => {
      if (ready() || Date.now() - started > SETTLE_MAX_MS) {
        window.clearInterval(timer);
        resolve();
      }
    }, 200);
  });
}

async function showPanel(): Promise<boolean> {
  const fake = pickFakeData(window.location.href);
  if (fake) return mountFakePanel(fake);
  return mountLivePanel();
}

// The panel on a real product page: collect what's on the page, ask the decision
// endpoint, and show its answer. Asks again once the shopper opens Amazon's full
// offer list, since that has every seller rather than only the buy box.
async function mountLivePanel(): Promise<boolean> {
  if (!isAmazonProductPage()) return false;
  const asin = extractASIN();
  if (!asin) return false;

  removeExistingUI();
  currentAsin = asin;
  const root = mountContainer();
  const isCurrent = () => root === currentRoot;

  let data: PanelData | null = null;
  let status: PanelStatus = "loading";
  let partialOffers = true;
  let fullList: OfferListResult | null = null;
  let lastRequest = 0;

  const render = () => {
    if (!isCurrent()) return;
    root.render(
      <Panel
        data={data}
        status={status}
        markUrl={markUrl()}
        onClose={removeExistingUI}
        onSeeOffers={() => openOffersPanel()}
        onRetry={ask}
        offers={extractOffers()}
        partialOffers={partialOffers}
        onOfferList={onOfferList}
      />
    );
  };

  async function ask(): Promise<void> {
    const id = ++lastRequest;
    if (!data) {
      status = "loading";
      render();
    }
    const profile = resolvedAnswers((await loadProfile()).answers);
    const request = buildDecisionRequest({ asin: asin!, profile, list: fullList });
    const res = await chrome.runtime
      .sendMessage({ type: "BUYWISE_DECISION", request })
      .catch((err: Error) => ({ ok: false, error: err.message }));
    // A newer request, a variant change or a close happened while this one ran
    if (!isCurrent() || id !== lastRequest) return;

    if (!res?.ok) {
      console.warn("BuyWise decision:", res?.error);
      // A failed re-ask keeps the answer already on screen
      if (data) return;
      status = res?.error === "not_tracked" ? "not_tracked" : res?.error === "timeout" ? "timeout" : "error";
      render();
      return;
    }

    const next = toPanelData(res.decision as Decision, new Date(), request.page_price);
    if (!next) {
      // Nothing worth showing: step out of the way, but keep following variant changes
      unmountPanel();
      return;
    }
    data = next;
    status = "ready";
    partialOffers = request.offers_source === "page";
    render();
  }

  function onOfferList(list: OfferListResult): void {
    if (list.status !== "ok") return; // the section itself says it couldn't read the list
    fullList = list;
    ask();
  }

  render();
  await pageSettled();
  if (!isCurrent()) return true;
  await ask();
  // Refresh the other-sellers summary once Amazon fills it in
  whenOtherSellersLoad(render);
  return true;
}

// Picking a color or size swaps the product in place without reloading the
// page, so the panel would keep answering for the old version (#73). Follow it
// while the panel is open, or hidden because there was nothing to show. A panel
// the shopper closed stays closed.
function followVariantChanges(): void {
  window.setInterval(() => {
    if (!currentAsin) return;
    const asin = extractASIN();
    if (asin && asin !== currentAsin) mountLivePanel();
  }, 1000);
}

function init(): void {
  console.log("BuyWise content script loaded");
  if (!isAmazonProductPage()) return;
  window.setTimeout(() => {
    showPanel();
  }, 500);
  followVariantChanges();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "BUYWISE_OPEN_PANEL") {
    showPanel().then((opened) => sendResponse({ opened }));
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
