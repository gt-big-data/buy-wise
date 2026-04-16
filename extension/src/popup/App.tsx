import React, { useEffect, useState } from "react";
import "../content/styles.css";
import {
  extractASINFromUrl,
  isAmazonProductPageUrl
} from "../content/amazon";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

const BACKEND_URL = "http://localhost:8000";
const USER_ID = 1;

type PopupState =
  | { status: "loading" }
  | { status: "product-opened"; asin: string }
  | { status: "non-product" }
  | { status: "error"; message: string };

type WatchlistItem = {
  asin: string;
  title: string;
  current_recommendation: string | null;
  recommendation_changed: boolean;
};

type ActivityItem = {
  activity_id: number;
  asin: string;
  product_title: string | null;
  recommendation_shown: string;
  action: string;
};

type DashboardData = {
  watchlist: WatchlistItem[];
  recent: ActivityItem[];
};

const App: React.FC = () => {
  const [popupState, setPopupState] = useState<PopupState>({ status: "loading" });
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);

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

          setDashboardLoading(true);
          try {
            const [watchlistRes, activityRes] = await Promise.all([
              fetch(`${BACKEND_URL}/watchlist/${USER_ID}`),
              fetch(`${BACKEND_URL}/activity/recent?user_id=${USER_ID}&limit=10`),
            ]);
            const watchlistData = watchlistRes.ok ? await watchlistRes.json() : { watchlist: [] };
            const activityData = activityRes.ok ? await activityRes.json() : { items: [] };
            setDashboard({
              watchlist: watchlistData.watchlist ?? [],
              recent: activityData.items ?? [],
            });
          } catch {
            setDashboard({ watchlist: [], recent: [] });
          } finally {
            setDashboardLoading(false);
          }
          return;
        }

        const asin = extractASINFromUrl(url);

        if (!asin) {
          setPopupState({
            status: "error",
            message: "We found an Amazon page, but couldn't identify the product yet."
          });
          return;
        }

        if (typeof tabId !== "number") {
          setPopupState({
            status: "error",
            message: "BuyWise couldn't access the current tab."
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
            message: "BuyWise couldn't reopen the recommendation on this page. Try refreshing the tab, then open the extension again."
          });
        }
      } catch (error) {
        console.error("BuyWise popup failed to load", error);
        setPopupState({
          status: "error",
          message: "BuyWise couldn't load this page."
        });
      }
    };

    loadPopup();
  }, []);

  if (popupState.status === "loading") {
    return (
      <div className="buywise-popup-root">
        <div className="buywise-popup-inner">
          <img src={chrome.runtime.getURL("logo.png")} alt="BuyWise" className="buywise-popup-logo-img" />
          <LoadingState />
        </div>
      </div>
    );
  }

  if (popupState.status === "non-product") {
    const watchlist = dashboard?.watchlist ?? [];
    const recent = dashboard?.recent ?? [];
    const alerts = watchlist.filter((w) => w.recommendation_changed);

    return (
      <div className="buywise-popup-root">
        <div className="buywise-popup-inner buywise-dashboard-inner">
          <img src={chrome.runtime.getURL("logo.png")} alt="BuyWise" className="buywise-popup-logo-img" />

          <div className="buywise-dashboard-summary-row">
            <div className="buywise-popup-card buywise-dashboard-summary-card">
              <div className="buywise-popup-helper-title">Watching</div>
              <div className="buywise-popup-badge">
                {dashboardLoading ? "…" : `${watchlist.length} items`}
              </div>
            </div>

            <div className="buywise-popup-card buywise-dashboard-summary-card">
              <div className="buywise-popup-helper-title">Activity</div>
              <div className="buywise-popup-badge">
                {dashboardLoading ? "…" : `${recent.length} recent`}
              </div>
            </div>
          </div>

          <div className="buywise-dashboard-grid">
            <div className="buywise-popup-card buywise-dashboard-section-card">
              <div className="buywise-popup-helper-title">Watchlist</div>
              {dashboardLoading ? (
                <p className="buywise-popup-muted buywise-popup-muted--tight">Loading…</p>
              ) : watchlist.length === 0 ? (
                <p className="buywise-popup-muted buywise-popup-muted--tight">Nothing watched yet</p>
              ) : (
                watchlist.slice(0, 5).map((item) => (
                  <div key={item.asin} className="buywise-popup-helper">
                    <div className={`buywise-popup-badge buywise-popup-badge--${(item.current_recommendation ?? "buy").toLowerCase()}`}>
                      {item.current_recommendation ?? "—"}
                    </div>
                    <p className="buywise-popup-muted buywise-popup-muted--tight">
                      {item.title}
                    </p>
                  </div>
                ))
              )}
            </div>

            {alerts.length > 0 && (
              <div className="buywise-popup-card buywise-dashboard-section-card">
                <div className="buywise-popup-helper-title">Alerts</div>
                {alerts.map((item) => (
                  <div key={item.asin} className="buywise-popup-helper">
                    <div className="buywise-popup-badge buywise-popup-badge--alert">
                      Rec changed → {item.current_recommendation}
                    </div>
                    <p className="buywise-popup-muted buywise-popup-muted--tight">
                      {item.title}
                    </p>
                  </div>
                ))}
              </div>
            )}

            <div className="buywise-popup-card buywise-dashboard-section-card">
              <div className="buywise-popup-helper-title">Recent</div>
              {dashboardLoading ? (
                <p className="buywise-popup-muted buywise-popup-muted--tight">Loading…</p>
              ) : recent.length === 0 ? (
                <p className="buywise-popup-muted buywise-popup-muted--tight">No recent activity</p>
              ) : (
                recent.slice(0, 4).map((item) => (
                  <div key={item.activity_id} className="buywise-popup-helper">
                    <div className={`buywise-popup-badge buywise-popup-badge--${item.recommendation_shown.toLowerCase()}`}>
                      {item.recommendation_shown}
                    </div>
                    <p className="buywise-popup-muted buywise-popup-muted--tight">
                      {item.product_title ?? item.asin}
                    </p>
                  </div>
                ))
              )}
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
          <img src={chrome.runtime.getURL("logo.png")} alt="BuyWise" className="buywise-popup-logo-img" />
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
          <img src={chrome.runtime.getURL("logo.png")} alt="BuyWise" className="buywise-popup-logo-img" />
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
