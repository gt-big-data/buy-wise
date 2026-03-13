import React, { useEffect, useState } from "react";
import BuyWisePanel from "../components/BuyWisePanel";
import LoadingState from "../components/LoadingState";
import { BuyWiseData } from "../types";
import { getMockBuyWiseData } from "../content/mockData";
import {
  extractASINFromUrl,
  isAmazonProductPageUrl
} from "../content/amazon";
import "../content/styles.css";

type PopupState =
  | { status: "loading" }
  | { status: "product"; data: BuyWiseData }
  | { status: "non-product" }
  | { status: "error"; message: string };

const popupShellStyle: React.CSSProperties = {
  width: 460,
  minHeight: 560,
  maxHeight: 700,
  overflowY: "auto",
  background: "#f7f6f4",
  fontFamily: "Arial, Helvetica, sans-serif",
  color: "#1d1d1d"
};

const nonProductWrapStyle: React.CSSProperties = {
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

        const data = getMockBuyWiseData(asin);
        setPopupState({ status: "product", data });
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
        <div style={{ padding: 16 }}>
          <LoadingState />
        </div>
      </div>
    );
  }

  if (popupState.status === "product") {
    return (
      <div style={popupShellStyle}>
        <div
          style={{
            padding: 14
          }}
        >
          <div
            style={{
              background: "#f7f6f4",
              borderRadius: 16
            }}
          >
            <BuyWisePanel
              data={popupState.data}
              onActionClick={() => {
                console.log("BuyWise popup action clicked", {
                  asin: popupState.data.asin,
                  recommendation: popupState.data.recommendation
                });

                chrome.storage.local.set({
                  lastBuyWiseAction: {
                    asin: popupState.data.asin,
                    recommendation: popupState.data.recommendation,
                    timestamp: new Date().toISOString(),
                    source: "popup"
                  }
                });
              }}
              showCloseButton={false}
              title="BuyWise"
              embedded={true}
            />
          </div>
        </div>
      </div>
    );
  }

  if (popupState.status === "error") {
    return (
      <div style={popupShellStyle}>
        <div style={nonProductWrapStyle}>
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
      <div style={nonProductWrapStyle}>
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
