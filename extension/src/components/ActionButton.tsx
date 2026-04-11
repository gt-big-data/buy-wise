import React from "react";
import { Recommendation } from "../types";

type ActionButtonProps = {
  recommendation: Recommendation;
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  labelOverride?: string;
  isWatched?: boolean;
};

const ActionButton: React.FC<ActionButtonProps> = ({
  recommendation,
  onClick,
  isLoading = false,
  disabled = false,
  labelOverride,
  isWatched = false
}) => {
  const defaultLabel = isWatched 
    ? (
       <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
          Watching <span className="buywise-live-dot" />
       </span>
    )
    : (recommendation === "WAIT" ? "Add to Watchlist" : "Buy");
  const buttonLabel = isLoading ? "Working..." : labelOverride ?? defaultLabel;
  const watchedClass = isWatched ? " is-watched" : "";

  return (
    <button
      className={`buywise-action-button buywise-ripple buywise-action-button--${recommendation.toLowerCase()}${watchedClass}`}
      onClick={onClick}
      type="button"
      disabled={disabled || isLoading}
      aria-busy={isLoading}
    >
      <span className="buywise-action-button__content">
        {isLoading ? <span className="buywise-button-loader" aria-hidden="true" /> : null}
        <span>{buttonLabel}</span>
      </span>
    </button>
  );
};

export default ActionButton;
