import React from "react";
import { Recommendation } from "../types";

type ActionButtonProps = {
  recommendation: Recommendation;
  onClick: () => void;
};

const ActionButton: React.FC<ActionButtonProps> = ({
  recommendation,
  onClick
}) => {
  const buttonLabel = recommendation === "WAIT" ? "Add to Watchlist 🔔" : "Buy With Confidence";

  return (
    <button className="buywise-action-button" onClick={onClick} type="button">
      {buttonLabel}
    </button>
  );
};

export default ActionButton;
