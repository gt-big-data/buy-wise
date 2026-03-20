import React from "react";

const LoadingState: React.FC = () => {
  return (
    <div className="buywise-status-card buywise-status-card--loading">
      <div className="buywise-status-graphic" aria-hidden="true">
        <svg viewBox="0 0 120 120" className="buywise-loading-orbit">
          <circle className="buywise-loading-orbit__track" cx="60" cy="60" r="42" />
          <circle className="buywise-loading-orbit__pulse" cx="60" cy="60" r="42" />
          <path
            className="buywise-loading-orbit__arc"
            d="M60 18 A42 42 0 0 1 98 40"
            fill="none"
          />
          <circle className="buywise-loading-orbit__dot" cx="98" cy="40" r="5" />
        </svg>
      </div>

      <div className="buywise-status-body">
        <div className="buywise-status-eyebrow">Preparing recommendation</div>
        <div className="buywise-status-title">Analyzing this item...</div>
        <div className="buywise-status-text">
          BuyWise is checking recent price movement and building a short-term forecast.
        </div>

        <div className="buywise-skeleton-stack" aria-hidden="true">
          <span className="buywise-skeleton buywise-skeleton--short" />
          <span className="buywise-skeleton buywise-skeleton--full" />
          <span className="buywise-skeleton buywise-skeleton--medium" />
        </div>
      </div>
    </div>
  );
};

export default LoadingState;
