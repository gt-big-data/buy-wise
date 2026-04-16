import React, { useRef } from "react";
import { Recommendation } from "../types";
import ActionButton from "./ActionButton";

type RecommendationBannerProps = {
  recommendation: Recommendation;
  confidence: number;
  expectedSavings: number;
  onActionClick: () => void;
  onWatchlistClick?: () => void;
  actionLoading?: boolean;
  isWatched?: boolean;
};

const RecommendationBanner: React.FC<RecommendationBannerProps> = ({
  recommendation,
  confidence,
  expectedSavings,
  onActionClick,
  onWatchlistClick,
  actionLoading = false,
  isWatched = false
}) => {
  const themeClass = `buywise-banner--${recommendation.toLowerCase()}`;
  const visualConfidence = Math.max(0, Math.min(100, confidence));

  const handleDynamicClick = () => {
    if (recommendation === "WAIT" && !isWatched && onWatchlistClick) {
       onWatchlistClick();
    } else {
       onActionClick();
    }
  };

  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left - rect.width / 2) / 25;
    const y = (e.clientY - rect.top - rect.height / 2) / 25;
    
    cardRef.current.style.setProperty('--mouse-x', `${x}px`);
    cardRef.current.style.setProperty('--mouse-y', `${y}px`);
  };

  const handleMouseLeave = () => {
    if (!cardRef.current) return;
    cardRef.current.style.setProperty('--mouse-x', '0px');
    cardRef.current.style.setProperty('--mouse-y', '0px');
  };

  return (
    <div className={`buywise-decision-card ${themeClass}`}>
      <div 
        className="buywise-decision-main"
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <div className="buywise-banner-kicker">
           7-day prediction 
        </div>
        <div className="buywise-recommendation-word" aria-label={`Recommendation: ${recommendation}`}>
          {recommendation}
        </div>
        
        <div className="buywise-timeline-indicator">
           {recommendation === "WAIT" ? (
             <><span className="active">Now</span> → <span>Drop expected</span> → <span>Buy soon</span></>
           ) : (
             <><span className="active">Best Price Now</span> → <span>Trending up</span></>
           )}
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

          {expectedSavings > 0 && (
            <span className="animate-savings" style={{ display: 'inline-flex' }}>
              <div className="buywise-savings-pill">
                <span className="buywise-savings-pill__label">Projected savings</span>
                <span className="buywise-savings-pill__value">${expectedSavings.toFixed(0)}</span>
              </div>
            </span>
          )}
        </div>

        <div className="buywise-action-wrapper">
          <ActionButton
            recommendation={recommendation}
            onClick={handleDynamicClick}
            isLoading={actionLoading}
            isWatched={isWatched}
          />
        </div>
      </div>
      
      <div className="buywise-insight-strip animate-insight">
          ✨ {recommendation === "WAIT" ? "Prices for similar tracked items actively dropped 10-15% this week." : "Price matches historical all-time lows for this category."}
      </div>
    </div>
  );
};

export default RecommendationBanner;
