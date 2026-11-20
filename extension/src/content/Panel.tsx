import React, { useState } from "react";
import { PanelData } from "./fakeData";
import { extractASIN } from "./amazon";
import { PageOffers } from "./offers";
import OffersSection from "./OffersSection";
import { OfferListResult } from "./offerList";
import { LABEL, PANEL_STATUS, SOLD_BY_AMAZON, checkedOffers, dropChanceVsUsual, dropPrice, headline, ifDropsLine, trustLine } from "../../../website/app/preview/copy";
import { defaultTarget, targetText } from "./watchTarget";
import type { RankedOffer } from "./decision";

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
  // Set when the dashboard replays a past call: the panel as it would have looked that
  // day, with nothing to click and no claims about that day's other sellers.
  replayDate?: string;
  ranking?: RankedOffer[];
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
  replayDate,
  ranking,
}: Props) {
  const [minimized, setMinimized] = useState(false);
  const answer = status === "ready" ? data : null;
  const tone = !answer
    ? "neutral"
    : answer.state === "better_offer"
      ? "good"
      : answer.state === "wait"
        ? "wait"
        : answer.timing?.status === "buy"
          ? "buy"
          : "neutral";

  return (
    <>
      {minimized && (
        <button type="button" className="bw-chip" onClick={() => setMinimized(false)}>
          <img src={markUrl} alt="" width={14} height={16} />
          BuyWise
        </button>
      )}
      <div className={`bw-panel bw-tone--${tone}`} hidden={minimized}>
        <div className="bw-head">
          <img className="bw-logo" src={markUrl} alt="" width={18} height={20} />
          <span className="bw-name">BuyWise</span>
          {replayDate ? (
            <span className="bw-head-chip">{replayDate}</span>
          ) : (
            answer && answer.offersChecked > 0 && <span className="bw-head-chip">{checkedOffers(answer.offersChecked)}</span>
          )}
          {!replayDate && (
            <button
              type="button"
              className="bw-dash-link"
              onClick={() => chrome.runtime.sendMessage({ type: "BUYWISE_OPEN_DASHBOARD" }).catch(() => undefined)}
            >
              Dashboard
            </button>
          )}
          {!replayDate && (
            <button type="button" className="bw-close" aria-label="Hide BuyWise on this page" onClick={onClose}>
              ×
            </button>
          )}
        </div>
        <div className="bw-body" aria-live="polite">
          {!answer && <NoAnswer status={status} onRetry={onRetry} />}
          {answer?.state === "better_offer" && <BetterOffer data={answer} onSeeOffers={onSeeOffers} />}
          {answer?.state === "wait" && <Wait data={answer} replay={!!replayDate} />}
          {answer?.state === "nothing_better" && <NothingBetter data={answer} />}
          {answer && partialOffers && <p className="bw-fine">{PANEL_STATUS.partialOffers}</p>}
        </div>
        {offers && (
          <OffersSection
            offers={offers}
            recommendedPrice={answer?.state === "better_offer" ? answer.best.price : null}
            onCompareOpen={() => document.getElementById("buywise-root")?.classList.add("bw-left")}
            onList={onOfferList}
            ranking={ranking}
          />
        )}
      </div>
    </>
  );
}

function NoAnswer({ status, onRetry }: { status: PanelStatus; onRetry?: () => void }) {
  if (status === "loading")
    return (
      <div className="bw-hero">
        <div className="bw-loader" aria-hidden="true" />
        <p className="bw-sub">{PANEL_STATUS.loading}</p>
      </div>
    );
  if (status === "not_tracked") return <p className="bw-sub">{PANEL_STATUS.notTracked}</p>;
  return (
    <div className="bw-hero">
      <p className="bw-sub bw-caution">{status === "timeout" ? PANEL_STATUS.timeout : PANEL_STATUS.error}</p>
      {onRetry && (
        <button type="button" className="bw-button bw-button--quiet" onClick={onRetry}>
          {PANEL_STATUS.retry}
        </button>
      )}
    </div>
  );
}

function BetterOffer({ data, onSeeOffers }: { data: Extract<PanelData, { state: "better_offer" }>; onSeeOffers: () => void }) {
  const { amazon, best } = data;
  const rating =
    best.stars !== null && best.ratingCount !== null && best.positivePercent !== null
      ? trustLine({ stars: best.stars, count: best.ratingCount, positive: best.positivePercent })
      : SOLD_BY_AMAZON;
  const warrantyMayNotApply = best.condition.trim().toLowerCase() !== "new" || best.fullWarranty === false;
  const saving = amazon.price - best.price;
  const pctOff = Math.round((saving / amazon.price) * 100);

  return (
    <>
      <div className="bw-hero">
        <span className="bw-badge">Better deal found</span>
        <p className="bw-big">
          {money(saving)} <span className="bw-big-sub">less · {pctOff}% off</span>
        </p>
        <p className="bw-sub">Same item from another seller.</p>
      </div>

      <div className="bw-best">
        <div className="bw-best-top">
          <span className="bw-best-label">Best for you</span>
          <span className="bw-best-price bw-mono">{money(best.price)}</span>
        </div>
        <p className="bw-best-seller">{best.seller}</p>
        <p className="bw-best-meta">
          {best.condition} · arrives {best.arrives}
        </p>
        <p className="bw-pill-row">
          <span className="bw-pill bw-pill--good">{rating}</span>
          {best.freeReturns && <span className="bw-pill">Free returns</span>}
          {warrantyMayNotApply && <span className="bw-pill bw-pill--warn">Warranty may not apply</span>}
        </p>
        <button type="button" className="bw-cta" onClick={onSeeOffers}>
          See this offer →
        </button>
      </div>

      <p className="bw-vs">
        Add to Cart is <s className="bw-mono">{money(amazon.price)}</s> from {amazon.seller}, arrives {amazon.arrives}
      </p>
      <Passed offers={data.passed} />
    </>
  );
}

function Passed({ offers }: { offers?: { seller: string; condition: string; price: number; why: string }[] }) {
  if (!offers || offers.length === 0) return null;
  return (
    <details className="bw-details">
      <summary>
        Why not the {offers.length} cheaper offer{offers.length === 1 ? "" : "s"}?
      </summary>
      <ul className="bw-passed">
        {offers.map((o) => (
          <li key={o.seller + o.price}>
            <p className="bw-passed-top">
              <span className="bw-strong">{o.seller}</span>
              <span className="bw-mono">{money(o.price)}</span>
            </p>
            <p className="bw-passed-why">
              {o.condition} · {o.why}
            </p>
          </li>
        ))}
      </ul>
    </details>
  );
}

function Wait({ data, replay }: { data: Extract<PanelData, { state: "wait" }>; replay: boolean }) {
  const note = [
    data.drops !== null && data.drops > 0 && `It dropped ${data.drops} times in the last ${data.months} months.`,
    !replay && "No other seller has it cheaper right now.",
  ].filter(Boolean);
  return (
    <>
      <div className="bw-hero">
        <span className="bw-badge">Wait</span>
        <p className="bw-title">{headline.wait(data.price)}</p>
      </div>
      <div className="bw-stats">
        <div>
          <p className="bw-stat-label">Price now</p>
          <p className="bw-stat bw-mono">{money(data.price)}</p>
        </div>
        <div>
          <p className="bw-stat-label">{LABEL.ifDrops}</p>
          <p className="bw-stat bw-mono bw-stat--tone">{ifDropsLine(data.price).replace(" or less", "")}</p>
        </div>
        <div>
          <p className="bw-stat-label">Drop chance</p>
          <p className="bw-stat bw-mono">{Math.round(data.dropChance * 100)}%</p>
          <p className="bw-stat-foot">{dropChanceVsUsual(data.dropChance).includes("usually") ? "usually 18%" : ""}</p>
        </div>
      </div>
      {data.history && <PriceChart history={data.history} target={dropPrice(data.price)} />}
      {note.length > 0 && <p className="bw-fine">{note.join(" ")}</p>}
      {!replay && data.expectedLow !== null && <WatchFlow price={data.price} expectedLow={data.expectedLow} />}
    </>
  );
}

function WatchFlow({ price, expectedLow }: { price: number; expectedLow: number }) {
  const [step, setStep] = useState<"start" | "target" | "saving" | "watching">("start");
  const [target, setTarget] = useState(() => targetText(defaultTarget(price, expectedLow)));
  const [error, setError] = useState<string | null>(null);
  const targetPrice = Number(target);

  async function watch() {
    setStep("saving");
    setError(null);
    const asin = extractASIN();
    const res = await chrome.runtime
      .sendMessage({ type: "BUYWISE_WATCHLIST_ADD", asin, target_price: targetPrice, recommendation: "WAIT" })
      .catch(() => null);
    if (res?.ok) {
      setStep("watching");
    } else {
      setError(res?.error?.includes("not found") ? "BuyWise doesn't track this product yet." : "Couldn't save right now. Try again in a moment.");
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
        <p className={error ? "bw-note bw-caution" : "bw-note"}>{error ?? `If it drops, it will be ${ifDropsLine(price)}.`}</p>
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
  const buy = data.timing?.status === "buy";
  return (
    <>
      <div className="bw-hero">
        <span className="bw-badge">{buy ? "Buy now" : "Best offer"}</span>
        <p className="bw-title">{buy ? "Good time to buy, and this is the best offer for you." : "This is the best offer for you."}</p>
        {data.pickWhy && (
          <p className="bw-sub">
            {data.pickWhy}
            {data.passedCount
              ? ` We checked ${data.passedCount} cheaper offer${data.passedCount === 1 ? "" : "s"}; none was worth switching to.`
              : " No other seller has it for less."}
          </p>
        )}
      </div>
      {buy && data.history && <PriceChart history={data.history} />}
      {data.timing && (
        <p className={`bw-timing ${buy ? "bw-timing--buy" : ""}`}>
          {buy ? data.timing.text.replace("Good time to buy. ", "") : data.timing.text}
        </p>
      )}
      <Passed offers={data.passed} />
      {data.nextBest && (
        <p className="bw-fine">
          Next best is {money(data.nextBest.price)} and arrives {data.nextBest.daysLater} days later.
        </p>
      )}
    </>
  );
}

const monthDay = (iso: string) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { month: "short", year: "numeric" });

// Daily prices as a step line. The last point is today's page price; a long gap before
// it (our data ends in April 2026) is drawn dashed so it doesn't read as a smooth trend.
function PriceChart({ history, target }: { history: [string, number][]; target?: number }) {
  if (history.length < 3) return null;
  const W = 320;
  const H = 96;
  const pad = 6;
  const t = history.map(([d]) => new Date(`${d}T12:00:00`).getTime());
  const prices = history.map(([, p]) => p);
  const lo = Math.min(...prices, target ?? Infinity);
  const hi = Math.max(...prices);
  const span = hi - lo || 1;
  const t0 = t[0];
  const tSpan = t[t.length - 1] - t0 || 1;
  const gapIdx = t.length - 1;
  const gap = t[gapIdx] - t[gapIdx - 1] > 30 * 86_400_000;
  // compress a long gap to a fixed slice of the width
  const gapW = gap ? 0.12 : 0;
  const body = gap ? t[gapIdx - 1] - t0 || 1 : tSpan;
  const x = (i: number) =>
    pad + (W - 2 * pad) * (gap && i === gapIdx ? 1 : ((t[i] - t0) / body) * (1 - gapW));
  const y = (p: number) => pad + (H - 2 * pad) * (1 - (p - lo) / span);
  const end = gap ? gapIdx - 1 : gapIdx;
  let d = `M${x(0)},${y(prices[0])}`;
  for (let i = 1; i <= end; i++) d += ` H${x(i)} V${y(prices[i])}`;
  const last = history.length - 1;
  return (
    <figure className="bw-chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Price history">
        <path d={`${d} V${H - pad} H${x(0)} Z`} className="bw-chart-fill" />
        <path d={d} className="bw-chart-line" />
        {gap && <path d={`M${x(end)},${y(prices[end])} L${x(last)},${y(prices[last])}`} className="bw-chart-gap" />}
        {target !== undefined && (
          <>
            <line x1={pad} x2={W - pad} y1={y(target)} y2={y(target)} className="bw-chart-target" />
            <text x={W - pad} y={y(target) - 4} textAnchor="end" className="bw-chart-label">
              {money(target)}
            </text>
          </>
        )}
        <circle cx={x(last)} cy={y(prices[last])} r={4} className="bw-chart-now" />
      </svg>
      <figcaption>
        <span>{monthDay(history[0][0])}</span>
        <span>
          low <b>{money(Math.min(...prices))}</b> · high <b>{money(hi)}</b>
        </span>
        <span>Today</span>
      </figcaption>
    </figure>
  );
}
