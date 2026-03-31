import React from "react";
import { Recommendation } from "../types";

type ActionButtonProps = {
  recommendation: Recommendation;
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  labelOverride?: string;
};

const ActionButton: React.FC<ActionButtonProps> = ({
  recommendation,
  onClick,
  isLoading = false,
  disabled = false,
  labelOverride
}) => {
  const defaultLabel = recommendation === "WAIT" ? "Add to Watchlist" : "Buy With Confidence";
  const buttonLabel = isLoading ? "Working..." : labelOverride ?? defaultLabel;

  return (
    <button
      className={`buywise-action-button buywise-action-button--${recommendation.toLowerCase()}`}
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
