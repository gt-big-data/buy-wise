import React, { useId, useState } from "react";

type InfoBlurbProps = {
  text: string;
};

const InfoBlurb: React.FC<InfoBlurbProps> = ({ text }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const contentId = useId();

  return (
    <div className={`buywise-info-row ${isExpanded ? "buywise-info-row--expanded" : ""}`}>
      <button
        className="buywise-info-toggle"
        type="button"
        onClick={() => setIsExpanded((current) => !current)}
        aria-expanded={isExpanded}
        aria-controls={contentId}
      >
        <span className="buywise-info-icon" aria-hidden="true">
          i
        </span>
        <span className="buywise-info-content">
          <span className="buywise-info-title">Why this recommendation</span>
          <span className="buywise-info-preview">
            {isExpanded ? "Hide the longer explanation" : "Tap to see the reasoning behind this forecast"}
          </span>
        </span>
        <span className="buywise-info-chevron" aria-hidden="true">
          ▾
        </span>
      </button>

      <div id={contentId} className="buywise-info-panel" hidden={!isExpanded}>
        <div className="buywise-info-text">{text}</div>
      </div>
    </div>
  );
};

export default InfoBlurb;
