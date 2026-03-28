import React, { useCallback, useRef, useState } from "react";
import { BuyWiseData } from "../types";
import InfoBlurb from "./InfoBlurb";
import PriceChart from "./PriceChart";
import RecommendationBanner from "./RecommendationBanner";

type BuyWisePanelProps = {
  data: BuyWiseData;
  onClose?: () => void;
  onActionClick: () => void;
  showCloseButton?: boolean;
  title?: string;
  floating?: boolean;
};

const DISMISS_MS = 240;

const BuyWisePanel: React.FC<BuyWisePanelProps> = ({
  data,
  onClose,
  onActionClick,
  showCloseButton = false,
  title = "BuyWise",
  floating = false
}) => {
  const [isDismissing, setIsDismissing] = useState(false);
  const dismissStartedRef = useRef(false);

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
        className={floating ? "buywise-floating-modal" : "buywise-embedded-modal"}
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
              <div className="buywise-logo buywise-logo-no-underline">{title}</div>
            </div>

            <section className="buywise-product-summary" aria-label="Product">
              <h2 className="buywise-product-summary-title">
                {data.productTitle || "This product"}
              </h2>
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
      </div>
    </div>
  );
};

export default BuyWisePanel;
