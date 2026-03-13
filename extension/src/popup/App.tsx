import React, { useEffect, useState } from "react";
import {
  extractASINFromUrl,
  isAmazonProductPageUrl
} from "../content/amazon";

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

  return (
    <div style={popupShellStyle}>
      <div style={wrapStyle}>
        <div style={titleStyle}>BuyWise</div>

        <div style={cardStyle}>
          <div style={badgeStyle}>No product detected</div>

          <p style={mutedStyle}>
            Open an Amazon product page to see a BuyWise recommendation, savings estimate,
            confidence score, and price graph.
          </p>

          <div style={helperBoxStyle}>
            <div style={helperTitleStyle}>Works on pages like:</div>
            <p style={{ ...mutedStyle, marginBottom: 0 }}>
              amazon.com/.../dp/XXXXXXXXXX
              <br />
              amazon.com/gp/product/XXXXXXXXXX
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
