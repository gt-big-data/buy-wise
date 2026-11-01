import React, { useState } from "react";
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
  const daysLater = best.daysOut - amazon.daysOut;

  let reason = `${best.condition}, ${money(amazon.price - best.price)} less. ${best.seller} has ${best.stars} stars over ${best.years} years`;
  reason += best.freeReturns ? " and free returns." : ".";
  if (daysLater > 0) reason += ` Arrives ${daysLater} day${daysLater > 1 ? "s" : ""} later.`;

  return (
    <>
      <p className="bw-headline">
        Same item, <span className="bw-good">{money(amazon.price - best.price)} less</span> from another seller.
      </p>
      <div className="bw-compare">
        <div className="bw-offer">
          <p className="bw-offer-label">Amazon's pick</p>
          <p className="bw-offer-price bw-mono">{money(amazon.price)}</p>
          <p className="bw-offer-seller">{amazon.seller}</p>
          <p className="bw-offer-meta">
            {amazon.condition}, arrives {amazon.arrives}
          </p>
        </div>
        <div className="bw-offer bw-offer--best">
          <p className="bw-offer-label">Best for you</p>
          <p className="bw-offer-price bw-mono bw-good-ink">{money(best.price)}</p>
          <p className="bw-offer-seller">{best.seller}</p>
          <p className="bw-offer-meta">
            {best.condition}, arrives {best.arrives}
          </p>
        </div>
      </div>
      <p className="bw-note">
        <span className="bw-strong">Why: </span>
        {reason}{" "}
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
        <WatchFlow price={data.price} expectedLow={data.expectedLow} />
      </p>
    </>
  );
}

// Tap Watch, set a target, get notified. Nothing is saved yet; that's #74.
function WatchFlow({ price, expectedLow }: { price: number; expectedLow: number }) {
  const [step, setStep] = useState<"start" | "target" | "watching">("start");
  const [target, setTarget] = useState(String(Math.round(expectedLow)));
  const targetPrice = Number(target);
  const validTarget = targetPrice > 0 && targetPrice < price;

  if (step === "start") {
    return (
      <button type="button" className="bw-button" onClick={() => setStep("target")}>
        Watch this price
      </button>
    );
  }

  if (step === "target") {
    return (
      <div className="bw-watch">
        <label className="bw-watch-label" htmlFor="bw-target">
          Tell me when it drops to
        </label>
        <div className="bw-watch-row">
          <span className="bw-money-input">
            $
            <input
              id="bw-target"
              className="bw-mono"
              value={target}
              inputMode="decimal"
              onChange={(e) => setTarget(e.target.value.replace(/[^0-9.]/g, ""))}
            />
          </span>
          <button type="button" className="bw-button bw-button--inline" disabled={!validTarget} onClick={() => setStep("watching")}>
            Watch
          </button>
        </div>
        <p className="bw-note">
          {validTarget ? `Expected low is ${money(expectedLow)}.` : `Pick a price below today's ${money(price)}.`}
        </p>
      </div>
    );
  }

  return (
    <div className="bw-watch">
      <p className="bw-strong">Watching at {money(targetPrice)}.</p>
      <p className="bw-note" style={{ marginTop: 4 }}>
        We'll send a Chrome notification when it hits {money(targetPrice)} or lower.
      </p>
      <div className="bw-watch-row" style={{ marginTop: 10 }}>
        <button type="button" className="bw-link" onClick={() => sendTestAlert(targetPrice)}>
          Preview the alert
        </button>
        <button type="button" className="bw-link" onClick={() => setStep("start")}>
          Stop watching
        </button>
      </div>
    </div>
  );
}

function sendTestAlert(target: number): void {
  const title = document.getElementById("productTitle")?.textContent?.trim() ?? "This product";
  chrome.runtime.sendMessage({ type: "BUYWISE_TEST_ALERT", title, target }).catch(() => {});
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