import React from "react";
import { Recommendation } from "../types";
import ActionButton from "./ActionButton";

type RecommendationBannerProps = {
  recommendation: Recommendation;
  confidence: number;
  expectedSavings: number;
  onActionClick: () => void;
};

const RecommendationBanner: React.FC<RecommendationBannerProps> = ({
  recommendation,
  confidence,
  expectedSavings,
  onActionClick
}) => {
  return (
    <div className="buywise-banner">
      <div className="buywise-banner-left">
        <div className="buywise-recommendation-word">{recommendation}</div>
        <ActionButton recommendation={recommendation} onClick={onActionClick} />
      </div>

      <div className="buywise-banner-right">
        <div className="buywise-banner-copy">
          We&apos;re <span className="buywise-emphasis">{confidence}%</span> confident
          you&apos;ll save <span className="buywise-emphasis">${expectedSavings.toFixed(0)}</span> in 7 days
        </div>
      </div>
    </div>
  );
};

export default RecommendationBanner;
