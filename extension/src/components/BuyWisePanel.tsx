import React from "react";
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
  embedded?: boolean;
};

const BuyWisePanel: React.FC<BuyWisePanelProps> = ({
  data,
  onClose,
  onActionClick,
  showCloseButton = false,
  title = "BuyWise",
  embedded = true
}) => {
  return (
    <div className={embedded ? "buywise-embedded-shell" : "buywise-overlay"}>
      <div className={embedded ? "buywise-embedded-modal" : "buywise-modal"}>
        {showCloseButton && onClose ? (
          <button
            className="buywise-close-button"
            onClick={onClose}
            type="button"
            aria-label="Close BuyWise"
          >
            ×
          </button>
        ) : null}

        <div className="buywise-header">
          <div className="buywise-logo buywise-logo-no-underline">{title}</div>
        </div>

        <RecommendationBanner
          recommendation={data.recommendation}
          confidence={data.confidence}
          expectedSavings={data.expectedSavings}
          onActionClick={onActionClick}
        />

        <InfoBlurb text={data.why} />

        <PriceChart
          title={data.chartTitle}
          points={data.points}
          predictedBestPrice={data.predictedBestPrice}
        />
      </div>
    </div>
  );
};

export default BuyWisePanel;
