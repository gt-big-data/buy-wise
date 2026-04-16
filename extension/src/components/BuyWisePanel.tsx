import React, { useCallback, useRef, useState } from "react";
import { BuyWiseData } from "../types";
import InfoBlurb from "./InfoBlurb";
import PriceChart from "./PriceChart";
import RecommendationBanner from "./RecommendationBanner";

type BuyWisePanelProps = {
  data: BuyWiseData;
  onClose?: () => void;
  onActionClick: () => void;
  onWatchlistClick?: () => void;
  showCloseButton?: boolean;
  title?: string;
  floating?: boolean;
};

const DISMISS_MS = 240;

const BuyWisePanel: React.FC<BuyWisePanelProps> = ({
  data,
  onClose,
  onActionClick,
  onWatchlistClick,
  showCloseButton = false,
  title = "BuyWise",
  floating = false,
}) => {
  const [isDismissing, setIsDismissing] = useState(false);
  const [isWatched, setIsWatched] = useState(Boolean(data.isWatched));
  const [showToast, setShowToast] = useState(false);
  const dismissStartedRef = useRef(false);

  const handleWatchlist = useCallback(() => {
     if (isWatched) return;
     setIsWatched(true);
     setShowToast(true);
     setTimeout(() => setShowToast(false), 3000);
     if (onWatchlistClick) onWatchlistClick();
  }, [isWatched, onWatchlistClick]);

  const handleRequestClose = useCallback(() => {
    if (!onClose || dismissStartedRef.current) {
      return;
    }
    dismissStartedRef.current = true;
    setIsDismissing(true);
    window.setTimeout(() => {
      onClose();
    }, DISMISS_MS);
  }, [onClose]);

  const shellClass =
    (floating ? "buywise-floating-shell" : "buywise-embedded-shell") +
    (isDismissing ? " buywise-panel-is-dismissing" : "");

  return (
    <div className={shellClass}>
      <div
        className={
          floating ? "buywise-floating-modal" : "buywise-embedded-modal"
        }
      >
        {showCloseButton && onClose ? (
          <button
            className="buywise-close-button"
            onClick={handleRequestClose}
            type="button"
            aria-label="Close BuyWise"
          >
            ×
          </button>
        ) : null}

        <div className="buywise-panel-body">
          <header className="buywise-panel-top">
            <div className="buywise-header">
              <div className="buywise-logo buywise-logo-no-underline">
                <img src={chrome.runtime.getURL("logo.png")} alt="BuyWise" className="buywise-logo-img" />
              </div>
            </div>

            <section className="buywise-product-summary" aria-label="Product">
              <div className="buywise-header-product-info">
                {data.imageUrl && (
                  <img
                    src={data.imageUrl}
                    alt="Product"
                    className="buywise-product-image"
                  />
                )}
                <h2 className="buywise-product-summary-title">
                  {data.productTitle || "This product"}
                </h2>
              </div>
            </section>
          </header>

          <section
            className="buywise-panel-section buywise-panel-section--recommendation"
            aria-label="Recommendation"
          >
            <RecommendationBanner
              recommendation={data.recommendation}
              confidence={data.confidence}
              expectedSavings={data.expectedSavings}
              onActionClick={onActionClick}
              onWatchlistClick={handleWatchlist}
              isWatched={isWatched}
            />
          </section>

          <section className="buywise-panel-section buywise-panel-section--detail">
            <InfoBlurb text={data.why} />
          </section>

          <section className="buywise-panel-section buywise-panel-section--chart">
            <PriceChart
              title={data.chartTitle}
              points={data.points}
              predictedBestPrice={data.predictedBestPrice}
            />
          </section>
        </div>
        
        {showToast && (
          <div className="buywise-toast-wrap">
            <div className="buywise-toast">
              <span className="buywise-toast-icon">✓</span>
              Item added to your BuyWise Watchlist!
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BuyWisePanel;
