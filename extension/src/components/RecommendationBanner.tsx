import React from "react";
import { Recommendation } from "../types";
import ActionButton from "./ActionButton";

type RecommendationBannerProps = {
  recommendation: Recommendation;
  confidence: number;
  expectedSavings: number;
  onActionClick: () => void;
  actionLoading?: boolean;
};

const RecommendationBanner: React.FC<RecommendationBannerProps> = ({
  recommendation,
  confidence,
  expectedSavings,
  onActionClick,
  actionLoading = false
}) => {
  const themeClass = `buywise-banner--${recommendation.toLowerCase()}`;
  const visualConfidence = Math.max(0, Math.min(100, confidence));

  return (
    <div className={`buywise-banner ${themeClass}`}>
      <div className="buywise-banner-left">
        <div className="buywise-recommendation-word" aria-label={`Recommendation: ${recommendation}`}>
          {recommendation}
        </div>
        <ActionButton
          recommendation={recommendation}
          onClick={onActionClick}
          isLoading={actionLoading}
        />
      </div>

      <div className="buywise-banner-right">
        <div className="buywise-banner-copy">
          <div className="buywise-banner-kicker">7-day guidance</div>
          <div className="buywise-banner-headline">
            {recommendation === "WAIT"
              ? "Odds point to a near-term price drop."
              : "The current price already looks strong."}
          </div>
          <div className="buywise-banner-metrics">
            <div className="buywise-confidence-meter" aria-label={`${visualConfidence}% confidence`}>
              <svg viewBox="0 0 42 42" className="buywise-confidence-ring" aria-hidden="true">
                <circle className="buywise-confidence-ring__track" cx="21" cy="21" r="16" />
                <circle
                  className="buywise-confidence-ring__progress"
                  cx="21"
                  cy="21"
                  r="16"
                  strokeDasharray={`${visualConfidence} 100`}
                />
              </svg>
              <div className="buywise-confidence-ring__label">
                <span className="buywise-confidence-value">{visualConfidence}%</span>
                <span className="buywise-confidence-caption">confidence</span>
              </div>
            </div>

            <div className="buywise-savings-pill">
              <span className="buywise-savings-pill__label">Projected savings</span>
              <span className="buywise-savings-pill__value">${expectedSavings.toFixed(0)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationBanner;
