import React from "react";

type InfoBlurbProps = {
  text: string;
};

const InfoBlurb: React.FC<InfoBlurbProps> = ({ text }) => {
  return (
    <div className="buywise-info-row">
      <span className="buywise-info-icon">i</span>
      <div className="buywise-info-content">
        <div className="buywise-info-title">Find out more</div>
        <div className="buywise-info-text">{text}</div>
      </div>
    </div>
  );
};

export default InfoBlurb;
