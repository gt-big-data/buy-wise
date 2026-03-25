import React, { useEffect, useState } from "react";
import "../content/styles.css";
import {
  extractASINFromUrl,
  isAmazonProductPageUrl
} from "../content/amazon";
import ActionButton from "../components/ActionButton";
import ErrorState from "../components/ErrorState";
import InfoBlurb from "../components/InfoBlurb";
import LoadingState from "../components/LoadingState";
import PriceChart from "../components/PriceChart";
import RecommendationBanner from "../components/RecommendationBanner";
import { PricePoint } from "../types";

type PopupState =
  | { status: "loading" }
  | { status: "product-opened"; asin: string }
  | { status: "non-product" }
  | { status: "error"; message: string };

const popupShellStyle: React.CSSProperties = {
  width: 360,
  minHeight: 260,
  background: "#f7f6f4",
  fontFamily: "Arial, Helvetica, sans-serif",
  color: "#1d1d1d"
};

const wrapStyle: React.CSSProperties = {
  padding: 20
};

const titleStyle: React.CSSProperties = {
  fontSize: 30,
  fontWeight: 900,
  marginBottom: 12
};

const cardStyle: React.CSSProperties = {
  background: "#ffffff",
  borderRadius: 16,
  padding: 18,
  boxShadow: "0 8px 22px rgba(0,0,0,0.12)",
  border: "1px solid #ece8df"
};

const mutedStyle: React.CSSProperties = {
  fontSize: 15,
  lineHeight: 1.5,
  color: "#4e4e4e",
  margin: 0
};

const badgeStyle: React.CSSProperties = {
  display: "inline-block",
  background: "#efdf93",
  color: "#1d6b25",
  fontWeight: 800,
  borderRadius: 999,
  padding: "6px 12px",
  fontSize: 13,
  marginBottom: 12
};

const helperBoxStyle: React.CSSProperties = {
  marginTop: 14,
  background: "#f4f2eb",
  borderRadius: 12,
  padding: 14,
  border: "1px solid #e9e2d3"
};

const helperTitleStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 800,
  marginBottom: 6,
  color: "#2f2f2f"
};

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
            message: "BuyWise couldn’t reopen the recommendation on this page."
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
      <div style={popupShellStyle}>
        <div style={wrapStyle}>
          <div style={titleStyle}>BuyWise</div>
          <div style={cardStyle}>
            <div style={badgeStyle}>Loading...</div>
            <p style={mutedStyle}>Checking the current page.</p>
          </div>
        </div>
      </div>
    );
  }

  if (popupState.status === "product-opened") {
    return (
      <div style={popupShellStyle}>
        <div style={wrapStyle}>
          <div style={titleStyle}>BuyWise</div>
          <div style={cardStyle}>
            <div style={badgeStyle}>Recommendation opened</div>
            <p style={mutedStyle}>
              The full BuyWise recommendation, graph, and explanation have been opened
              in the top-right corner of this product page.
            </p>

            <div style={helperBoxStyle}>
              <div style={helperTitleStyle}>Detected product</div>
              <p style={{ ...mutedStyle, marginBottom: 0 }}>
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
      <div style={popupShellStyle}>
        <div style={wrapStyle}>
          <div style={titleStyle}>BuyWise</div>
          <div style={cardStyle}>
            <div style={badgeStyle}>Couldn’t load recommendation</div>
            <p style={mutedStyle}>{popupState.message}</p>
          </div>
        </div>
      </div>
    );
  }

  const mockPoints: PricePoint[] = [
    { label: "W-4", actual: 120 },
    { label: "W-3", actual: 115 },
    { label: "W-2", actual: 118 },
    { label: "W-1", actual: 110 },
    { label: "Now", actual: 108, predicted: 108 },
    { label: "W+1", predicted: 102 },
    { label: "W+2", predicted: 95 },
    { label: "W+3", predicted: 91 },
    { label: "W+4", predicted: 89 }
  ];

  return (
    <div style={{ ...popupShellStyle, width: 420, minHeight: "unset" }}>
      <div style={wrapStyle}>
        <div style={titleStyle}>BuyWise — Component Showcase</div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <div style={helperTitleStyle}>RecommendationBanner (BUY)</div>
            <RecommendationBanner
              recommendation="BUY"
              confidence={82}
              expectedSavings={19}
              onActionClick={() => {}}
            />
          </div>

          <div>
            <div style={helperTitleStyle}>RecommendationBanner (WAIT)</div>
            <RecommendationBanner
              recommendation="WAIT"
              confidence={67}
              expectedSavings={12}
              onActionClick={() => {}}
            />
          </div>

          <div>
            <div style={helperTitleStyle}>ActionButton (BUY)</div>
            <ActionButton recommendation="BUY" onClick={() => {}} />
          </div>

          <div>
            <div style={helperTitleStyle}>ActionButton (WAIT)</div>
            <ActionButton recommendation="WAIT" onClick={() => {}} />
          </div>

          <div>
            <div style={helperTitleStyle}>PriceChart</div>
            <PriceChart
              title="Price history & forecast"
              points={mockPoints}
              predictedBestPrice={89}
            />
          </div>

          <div>
            <div style={helperTitleStyle}>InfoBlurb</div>
            <InfoBlurb text="Our model uses 30 days of price history to predict where prices will go in the next 4 weeks." />
          </div>

          <div>
            <div style={helperTitleStyle}>LoadingState</div>
            <LoadingState />
          </div>

          <div>
            <div style={helperTitleStyle}>ErrorState</div>
            <ErrorState />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
