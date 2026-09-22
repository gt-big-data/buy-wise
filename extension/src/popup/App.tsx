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
  target_price?: number | null;
  pred_14d?: number | null;
};

type ActivityItem = {
  activity_id: number;
  asin: string;
  product_title: string | null;
  recommendation_shown: string;
  action: string;
};

type SummaryData = {
  accuracy_when_followed_pct: number;
  estimated_savings_usd: number;
  watchlist_count: number;
};

type DashboardData = {
  watchlist: WatchlistItem[];
  recent: ActivityItem[];
  summary: SummaryData | null;
};

const ACTION_LABEL: Record<string, string> = {
  purchased: "Bought",
  dismissed: "Passed",
  added_to_watchlist: "Watching",
};

const App: React.FC = () => {
  const [popupState, setPopupState] = useState<PopupState>({ status: "loading" });
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);

  useEffect(() => {
    const loadPopup = async (): Promise<void> => {
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        const activeTab = tabs[0];
        const url = activeTab?.url;
        const tabId = activeTab?.id;

        if (!url || !isAmazonProductPageUrl(url)) {
          setPopupState({ status: "non-product" });
          setDashboardLoading(true);
          try {
            const [watchlistRes, activityRes, summaryRes] = await Promise.all([
              fetch(`${BACKEND_URL}/watchlist/${USER_ID}`),
              fetch(`${BACKEND_URL}/activity/recent?user_id=${USER_ID}&limit=10`),
              fetch(`${BACKEND_URL}/dashboard/summary`),
            ]);
            const watchlistData = watchlistRes.ok ? await watchlistRes.json() : { watchlist: [] };
            const activityData = activityRes.ok ? await activityRes.json() : { items: [] };
            const summaryData = summaryRes.ok ? await summaryRes.json() : null;
            setDashboard({
              watchlist: watchlistData.watchlist ?? [],
              recent: activityData.items ?? [],
              summary: summaryData,
            });
          } catch {
            setDashboard({ watchlist: [], recent: [], summary: null });
          } finally {
            setDashboardLoading(false);
          }
          return;
        }

        const asin = extractASINFromUrl(url);
        if (!asin) {
          setPopupState({ status: "error", message: "We found an Amazon page, but couldn't identify the product yet." });
          return;
        }
        if (typeof tabId !== "number") {
          setPopupState({ status: "error", message: "BuyWise couldn't access the current tab." });
          return;
        }

        try {
          await chrome.tabs.sendMessage(tabId, { type: "BUYWISE_OPEN_PANEL" });
          setPopupState({ status: "product-opened", asin });
        } catch {
          setPopupState({ status: "error", message: "BuyWise couldn't reopen the recommendation on this page. Try refreshing the tab, then open the extension again." });
        }
      } catch {
        setPopupState({ status: "error", message: "BuyWise couldn't load this page." });
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
    const summary = dashboard?.summary ?? null;
    const alerts = watchlist.filter((w) => w.recommendation_changed);

    const accuracy = summary?.accuracy_when_followed_pct ?? null;
    const savings = summary?.estimated_savings_usd ?? null;

    return (
      <div className="buywise-popup-root">
        {/* ── Header ── */}
        <div className="buywise-db-header">
          <img src={chrome.runtime.getURL("logo.png")} alt="BuyWise" className="buywise-db-logo" />
          <div className="buywise-db-stats">
            <div className="buywise-db-stat">
              <span className="buywise-db-stat__num">
                {dashboardLoading ? "—" : watchlist.length}
              </span>
              <span className="buywise-db-stat__label">Tracked</span>
            </div>
            <div className="buywise-db-stat-divider" />
            <div className="buywise-db-stat buywise-db-stat--green">
              <span className="buywise-db-stat__num">
                {dashboardLoading ? "—" : accuracy !== null ? `${Math.round(accuracy)}%` : "—"}
              </span>
              <span className="buywise-db-stat__label">Accuracy</span>
            </div>
            <div className="buywise-db-stat-divider" />
            <div className="buywise-db-stat">
              <span className="buywise-db-stat__num">
                {dashboardLoading ? "—" : savings !== null ? `$${Math.round(savings)}` : "—"}
              </span>
              <span className="buywise-db-stat__label">Saved</span>
            </div>
          </div>
        </div>

        <div className="buywise-db-body">
          {/* ── Alerts ── */}
          {alerts.length > 0 && (
            <div className="buywise-db-alerts-strip">
              <span className="buywise-db-alerts-icon">⚡</span>
              <span className="buywise-db-alerts-text">
                {alerts.length} watchlist item{alerts.length > 1 ? "s" : ""} changed recommendation
              </span>
            </div>
          )}

          {/* ── Watchlist ── */}
          <div className="buywise-db-section">
            <div className="buywise-db-section-header">
              <span className="buywise-db-section-title">Watchlist</span>
              {watchlist.length > 0 && (
                <span className="buywise-db-section-count">{watchlist.length}</span>
              )}
            </div>

            {dashboardLoading ? (
              <div className="buywise-db-empty">Loading…</div>
            ) : watchlist.length === 0 ? (
              <div className="buywise-db-empty">
                Visit an Amazon product page to start tracking.
              </div>
            ) : (
              <div className="buywise-db-list">
                {watchlist.slice(0, 5).map((item) => {
                  const rec = (item.current_recommendation ?? "buy").toLowerCase();
                  const changed = item.recommendation_changed;
                  return (
                    <div key={item.asin} className={`buywise-db-item buywise-db-item--${rec}`}>
                      <div className="buywise-db-item__left">
                        <span className={`buywise-db-chip buywise-db-chip--${rec}`}>
                          {item.current_recommendation ?? "—"}
                        </span>
                        <span className="buywise-db-item__title">
                          {item.title ?? item.asin}
                        </span>
                      </div>
                      {changed && (
                        <span className="buywise-db-item__alert" title="Recommendation changed">⚡</span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Recent Activity ── */}
          <div className="buywise-db-section">
            <div className="buywise-db-section-header">
              <span className="buywise-db-section-title">Recent Activity</span>
            </div>

            {dashboardLoading ? (
              <div className="buywise-db-empty">Loading…</div>
            ) : recent.length === 0 ? (
              <div className="buywise-db-empty">No activity yet.</div>
            ) : (
              <div className="buywise-db-list">
                {recent.slice(0, 5).map((item) => (
                  <div key={item.activity_id} className="buywise-db-activity-item">
                    <span className={`buywise-db-chip buywise-db-chip--${item.recommendation_shown.toLowerCase()}`}>
                      {item.recommendation_shown}
                    </span>
                    <span className="buywise-db-activity__title">
                      {item.product_title ?? item.asin}
                    </span>
                    <span className="buywise-db-activity__action">
                      {ACTION_LABEL[item.action] ?? item.action}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Footer nudge ── */}
          <div className="buywise-db-footer">
            Navigate to an Amazon product to get a recommendation.
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
              The full BuyWise panel is in the top-right corner of this product page.
            </p>
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
