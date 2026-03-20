import React from "react";

type ErrorStateProps = {
  title?: string;
  message?: string;
  onRetry?: () => void;
};

const ErrorState: React.FC<ErrorStateProps> = ({
  title = "No forecast available",
  message = "BuyWise couldn’t generate a recommendation for this product yet.",
  onRetry
}) => {
  return (
    <div className="buywise-status-card buywise-status-card--error" role="alert">
      <div className="buywise-error-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" className="buywise-error-icon__svg">
          <path d="M12 3.75 2.75 19.5h18.5L12 3.75Zm0 4.5c.41 0 .75.34.75.75v5.25a.75.75 0 1 1-1.5 0V9c0-.41.34-.75.75-.75Zm0 9a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z" />
        </svg>
      </div>

      <div className="buywise-status-body">
        <div className="buywise-status-title">{title}</div>
        <div className="buywise-status-text">{message}</div>
        {onRetry ? (
          <button type="button" className="buywise-retry-button" onClick={onRetry}>
            Try again
          </button>
        ) : null}
      </div>
    </div>
  );
};

export default ErrorState;
