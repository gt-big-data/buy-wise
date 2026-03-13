import React from "react";

const LoadingState: React.FC = () => {
  return (
    <div className="buywise-status-card">
      <div className="buywise-spinner" />
      <div className="buywise-status-title">Analyzing this item...</div>
      <div className="buywise-status-text">
        BuyWise is checking recent price movement and building a short-term forecast.
      </div>
    </div>
  );
};

export default LoadingState;
