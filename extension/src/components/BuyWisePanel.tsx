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
  floating?: boolean;
};

const BuyWisePanel: React.FC<BuyWisePanelProps> = ({
  data,
  onClose,
  onActionClick,
  showCloseButton = false,
  title = "BuyWise",
  floating = false
}) => {
  return (
    <div className={floating ? "buywise-floating-shell" : "buywise-embedded-shell"}>
      <div className={floating ? "buywise-floating-modal" : "buywise-embedded-modal"}>
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
