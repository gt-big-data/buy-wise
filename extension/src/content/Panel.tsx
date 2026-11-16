import React, { useState } from "react";
import { PanelData } from "./fakeData";
import { extractASIN } from "./amazon";
import { PageOffers } from "./offers";
import OffersSection from "./OffersSection";
import { OfferListResult } from "./offerList";
import { PANEL_STATUS, checkedOffers } from "../../../website/app/preview/copy";

const money = (n: number) => `$${n.toFixed(2)}`;

// ready: data holds the endpoint's answer. The rest are the panel with no answer.
export type PanelStatus = "ready" | "loading" | "error" | "timeout" | "not_tracked";

type Props = {
  data: PanelData | null;
  status?: PanelStatus;
  markUrl: string;
  onClose: () => void;
  onSeeOffers: () => void;
  onRetry?: () => void;
  offers?: PageOffers;
  // The answer only saw the buy box; Amazon's full list hasn't been read yet
  partialOffers?: boolean;
  onOfferList?: (list: OfferListResult) => void;
};

export default function Panel({
  data,
  status = "ready",
  markUrl,
  onClose,
  onSeeOffers,
  onRetry,
  offers,
  partialOffers,
  onOfferList,
}: Props) {
  const [minimized, setMinimized] = useState(false);
  const answer = status === "ready" ? data : null;

  // Stays mounted while minimized so the offer list keeps updating behind the chip
  return (
    <>
      {minimized && (
        <button type="button" className="bw-chip" onClick={() => setMinimized(false)}>
          <img src={markUrl} alt="" width={14} height={16} />
          BuyWise
        </button>
      )}
      <div className="bw-panel" hidden={minimized}>
        <div className="bw-head">
          <img src={markUrl} alt="" width={16} height={18} />
          <span className="bw-name">BuyWise</span>
          {answer && <span>{checkedOffers(answer.offersChecked)}</span>}
          <button type="button" className="bw-close" aria-label="Hide BuyWise on this page" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="bw-body" aria-live="polite">
          {!answer && <NoAnswer status={status} onRetry={onRetry} />}
          {answer?.state === "better_offer" && <BetterOffer data={answer} onSeeOffers={onSeeOffers} />}
          {answer?.state === "wait" && <Wait data={answer} />}
          {answer?.state === "nothing_better" && <NothingBetter data={answer} />}
          {answer && partialOffers && <p className="bw-note">{PANEL_STATUS.partialOffers}</p>}
        </div>
        {offers && <OffersSection offers={offers} onCompareOpen={() => setMinimized(true)} onList={onOfferList} />}
      </div>
    </>
  );
}

function NoAnswer({ status, onRetry }: { status: PanelStatus; onRetry?: () => void }) {
  if (status === "loading") return <p className="bw-note">{PANEL_STATUS.loading}</p>;
  if (status === "not_tracked") return <p className="bw-note">{PANEL_STATUS.notTracked}</p>;
  return (
    <>
      <p className="bw-note bw-caution">{status === "timeout" ? PANEL_STATUS.timeout : PANEL_STATUS.error}</p>
      {onRetry && (
        <button type="button" className="bw-button bw-button--quiet" onClick={onRetry}>
          {PANEL_STATUS.retry}
        </button>
      )}
    </>
  );
}

function BetterOffer({ data, onSeeOffers }: { data: Extract<PanelData, { state: "better_offer" }>; onSeeOffers: () => void }) {
  const { amazon, best } = data;

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
        {data.reason}{" "}
        <span className="bw-caution">May not include the manufacturer warranty.</span>
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
      <WatchFlow expectedLow={data.expectedLow} />
    </>
  );
}
// Tap Watch, set a target, get notified. The target is saved to the backend's
// watchlist, and the background script checks prices against it (#74).
function WatchFlow({ expectedLow }: { expectedLow: number }) {
  const [step, setStep] = useState<"start" | "target" | "saving" | "watching">("start");
  const [target, setTarget] = useState(String(Math.round(expectedLow)));
  const [error, setError] = useState<string | null>(null);
  const targetPrice = Number(target);

  async function watch() {
    setStep("saving");
    setError(null);
    const asin = extractASIN();
    const res = await chrome.runtime
      .sendMessage({ type: "BUYWISE_WATCHLIST_ADD", asin, target_price: targetPrice })
      .catch(() => null);
    if (res?.ok) {
      setStep("watching");
    } else {
      setError(res?.error?.includes("not found") ? "BuyWise doesn't track this product yet." : "Couldn't save. Is the backend running?");
      setStep("target");
    }
  }

  if (step === "start") {
    return (
      <button type="button" className="bw-button" onClick={() => setStep("target")}>
        Watch this price
      </button>
    );
  }

  if (step === "target" || step === "saving") {
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
          <button type="button" className="bw-button bw-button--inline" disabled={!(targetPrice > 0) || step === "saving"} onClick={watch}>
            {step === "saving" ? "Saving..." : "Watch"}
          </button>
        </div>
        <p className={error ? "bw-note bw-caution" : "bw-note"}>{error ?? `Expected low is ${money(expectedLow)}.`}</p>
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
        <button type="button" className="bw-link" onClick={() => chrome.runtime.sendMessage({ type: "BUYWISE_OPEN_DASHBOARD" })}>
          See everything you're watching
        </button>
      </div>
    </div>
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