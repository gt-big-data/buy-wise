import React, { useEffect, useState } from "react";
import "../content/styles.css";
import {
  extractASINFromUrl,
  isAmazonProductPageUrl
} from "../content/amazon";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

type PopupState =
  | { status: "loading" }
  | { status: "product-opened"; asin: string }
  | { status: "non-product" }
  | { status: "error"; message: string };

const App: React.FC = () => {
  const [popupState, setPopupState] = useState<PopupState>({ status: "loading" });

  useEffect(() => {
    const loadPopup = async (): Promise<void> => {
      try {
        const tabs = await chrome.tabs.query({
          active: true,
          currentWindow: true
        });

        const activeTab = tabs[0];
        const url = activeTab?.url;
        const tabId = activeTab?.id;

        if (!url || !isAmazonProductPageUrl(url)) {
          setPopupState({ status: "non-product" });
          return;
        }

        const asin = extractASINFromUrl(url);

        if (!asin) {
          setPopupState({
            status: "error",
            message: "We found an Amazon page, but couldn’t identify the product yet."
          });
          return;
        }

        if (typeof tabId !== "number") {
          setPopupState({
            status: "error",
            message: "BuyWise couldn’t access the current tab."
          });
          return;
        }

        try {
          await chrome.tabs.sendMessage(tabId, {
            type: "BUYWISE_OPEN_PANEL"
          });

          setPopupState({
            status: "product-opened",
            asin
          });
        } catch (error) {
          console.error("Failed to reopen BuyWise panel", error);
          setPopupState({
            status: "error",
            message: "BuyWise couldn’t reopen the recommendation on this page. Try refreshing the tab, then open the extension again."
          });
        }
      } catch (error) {
        console.error("BuyWise popup failed to load", error);
        setPopupState({
          status: "error",
          message: "BuyWise couldn’t load this page."
        });
      }
    };

    loadPopup();
  }, []);

  if (popupState.status === "loading") {
    return (
      <div className="buywise-popup-root">
        <div className="buywise-popup-inner">
          <div className="buywise-popup-title">BuyWise</div>
          <LoadingState />
        </div>
      </div>
    );
  }

if (popupState.status === "non-product") {
  const mockDashboard = {
    watchlist: [
      { title: "Sony XM5", rec: "WAIT" },
      { title: "AirPods Pro", rec: "BUY" }
    ],
    alerts: [{ title: "Samsung SSD", change: "WAIT → BUY" }],
    recent: [
      { title: "Kindle", rec: "WAIT" },
      { title: "Anker Charger", rec: "BUY" }
    ],
    summary: {
      savings: "$214",
      accuracy: "88%"
    }
  };

  return (
    <div className="buywise-popup-root">
      <div
        className="buywise-popup-inner"
        style={{ width: 380, maxWidth: 380 }}
      >
        <div className="buywise-popup-title">BuyWise</div>

        <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
          <div className="buywise-popup-card" style={{ flex: 1, padding: 10 }}>
            <div className="buywise-popup-helper-title">Saved</div>
            <div className="buywise-popup-badge">
              {mockDashboard.summary.savings}
            </div>
          </div>

          <div className="buywise-popup-card" style={{ flex: 1, padding: 10 }}>
            <div className="buywise-popup-helper-title">Accuracy</div>
            <div className="buywise-popup-badge">
              {mockDashboard.summary.accuracy}
            </div>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 8
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div className="buywise-popup-card" style={{ padding: 10 }}>
              <div className="buywise-popup-helper-title">Alerts</div>
              {mockDashboard.alerts.map((item) => (
                <div key={item.title} className="buywise-popup-helper">
                  <div className="buywise-popup-badge">{item.change}</div>
                  <p className="buywise-popup-muted buywise-popup-muted--tight">
                    {item.title}
                  </p>
                </div>
              ))}
            </div>

            <div className="buywise-popup-card" style={{ padding: 10 }}>
              <div className="buywise-popup-helper-title">Recent</div>
              {mockDashboard.recent.map((item) => (
                <div key={item.title} className="buywise-popup-helper">
                  <div className="buywise-popup-badge">{item.rec}</div>
                  <p className="buywise-popup-muted buywise-popup-muted--tight">
                    {item.title}
                  </p>
                </div>
              ))}
            </div>
          </div>
          <div className="buywise-popup-card" style={{ padding: 10 }}>
            <div className="buywise-popup-helper-title">Watchlist</div>
            {mockDashboard.watchlist.map((item) => (
              <div key={item.title} className="buywise-popup-helper">
                <div className="buywise-popup-badge">{item.rec}</div>
                <p className="buywise-popup-muted buywise-popup-muted--tight">
                  {item.title}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

  if (popupState.status === "product-opened") {
    return (
      <div className="buywise-popup-root">
        <div className="buywise-popup-inner">
          <div className="buywise-popup-title">BuyWise</div>
          <div className="buywise-popup-card">
            <div className="buywise-popup-badge">Recommendation opened</div>
            <p className="buywise-popup-muted">
              The full BuyWise panel is in the top-right corner of this product page. Close it there when you are done.
            </p>
            <div className="buywise-popup-helper">
              <div className="buywise-popup-helper-title">Detected product</div>
              <p className="buywise-popup-muted buywise-popup-muted--tight">
                ASIN: {popupState.asin}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (popupState.status === "error") {
    return (
      <div className="buywise-popup-root">
        <div className="buywise-popup-inner">
          <div className="buywise-popup-title">BuyWise</div>
          <ErrorState />
          {popupState.message ? (
            <p className="buywise-popup-error-detail">{popupState.message}</p>
          ) : null}
        </div>
      </div>
    );
  }

  return null;
};

export default App;
