import React from "react";

const ErrorState: React.FC = () => {
  return (
    <div className="buywise-status-card">
      <div className="buywise-status-title">No forecast available</div>
      <div className="buywise-status-text">
        BuyWise couldn&apos;t generate a recommendation for this product yet.
      </div>
    </div>
  );
};

export default ErrorState;
