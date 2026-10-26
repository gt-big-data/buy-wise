import React from "react";
import { PanelData } from "./fakeData";

const money = (n: number) => `$${n.toFixed(2)}`;

type Props = {
  data: PanelData;
  markUrl: string;
  onClose: () => void;
  onSeeOffers: () => void;
};

export default function Panel({ data, markUrl, onClose, onSeeOffers }: Props) {
  return (
    <div className="bw-panel">
      <div className="bw-head">
        <img src={markUrl} alt="" width={16} height={18} />
        <span className="bw-name">BuyWise</span>
        <span>checked {data.offersChecked} offers</span>
        <button type="button" className="bw-close" aria-label="Hide BuyWise on this page" onClick={onClose}>
          ×
        </button>
      </div>
      <div className="bw-body">
        {data.state === "better_offer" && <BetterOffer data={data} onSeeOffers={onSeeOffers} />}
        {data.state === "wait" && <Wait data={data} />}
        {data.state === "nothing_better" && <NothingBetter data={data} />}
      </div>
    </div>
  );
}

function BetterOffer({ data, onSeeOffers }: { data: Extract<PanelData, { state: "better_offer" }>; onSeeOffers: () => void }) {
  const { amazon, best } = data;
  return (
    <>
      <p className="bw-headline">
        Same item, new, <span className="bw-good">{money(amazon.price - best.price)} less</span> from another seller.
      </p>
      <dl className="bw-rows">
        <div>
          <dt>Amazon.com, new</dt>
          <dd className="bw-mono">{money(amazon.price)}</dd>
        </div>
        <div>
          <dt className="bw-strong">{best.seller}, new</dt>
          <dd className="bw-mono bw-strong bw-good-ink">{money(best.price)}</dd>
        </div>
      </dl>
      <p className="bw-note">
        {best.stars} stars, {best.years} years selling{best.freeReturns ? ", free returns" : ""}.{" "}
        {!best.fullWarranty && <span className="bw-caution">May not include the manufacturer warranty.</span>}
      </p>
      <button type="button" className="bw-button" onClick={onSeeOffers}>
        See offer
      </button>
    </>
  );
}

function Wait({ data }: { data: Extract<PanelData, { state: "wait" }> }) {
  return (
    <>
      <p className="bw-headline">
        Wait. {Math.round(data.dropChance * 100)}% chance this drops 8% or more in the next 14 days.
      </p>
      <dl className="bw-rows">
        <div>
          <dt>Price now</dt>
          <dd className="bw-mono">{money(data.price)}</dd>
        </div>
        <div>
          <dt>Expected low</dt>
          <dd className="bw-mono">{money(data.expectedLow)}</dd>
        </div>
      </dl>
      <p className="bw-note">
        It dropped {data.drops} times in the last {data.months} months. No other seller has it cheaper right now.
      </p>
    </>
  );
}

function NothingBetter({ data }: { data: Extract<PanelData, { state: "nothing_better" }> }) {
  return (
    <>
      <p className="bw-headline">This is the best offer for you.</p>
      {data.nextBest && (
        <p className="bw-note">
          Next best is {money(data.nextBest.price)} and arrives {data.nextBest.daysLater} days later.
        </p>
      )}
    </>
  );
}