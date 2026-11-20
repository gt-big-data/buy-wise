import React, { useEffect, useState } from "react";
import Panel from "../content/Panel";
import "../content/panel.css";
import { outcome, replayDay, replayPanel, type ReplayCall, type ReplayDetail } from "./pastCall";

const money = (n: number) => `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const short = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
const UNREACHABLE = "Couldn't reach BuyWise right now. Try again in a moment.";

const KIND_LABEL: Record<ReplayCall["kind"], string> = {
  wait_right: "Right Wait",
  wait_wrong: "Wrong Wait",
  buy_right: "Right Buy",
  buy_wrong: "Missed drop",
};

function PriceChart({ c }: { c: ReplayDetail }) {
  const W = 420;
  const H = 170;
  const L = 52;
  const R = 8;
  const T = 10;
  const B = 24;
  const prices = c.prices.map(([, p]) => p);
  const lo = Math.min(...prices);
  const hi = Math.max(...prices);
  const pad = (hi - lo || hi * 0.05) * 0.12;
  const yMin = lo - pad;
  const yMax = hi + pad;
  const n = c.prices.length;
  const x = (i: number) => L + (i / Math.max(n - 1, 1)) * (W - L - R);
  const y = (p: number) => T + (1 - (p - yMin) / (yMax - yMin)) * (H - T - B);
  const callAt = c.prices.findIndex(([d]) => d === c.predicted_on);
  const lowAt = c.low_on ? c.prices.findIndex(([d]) => d === c.low_on) : -1;
  let path = `M${x(0)},${y(prices[0])}`;
  for (let i = 1; i < n; i++) path += ` H${x(i)} V${y(prices[i])}`;

  return (
    <figure className="db-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Daily price from ${replayDay(c.prices[0][0])} to ${replayDay(c.end_on)}`}>
        {callAt >= 0 && (
          <rect className="db-chart-window" x={x(callAt)} y={T} width={x(n - 1) - x(callAt)} height={H - T - B} />
        )}
        <line className="db-chart-grid" x1={L} x2={W - R} y1={y(hi)} y2={y(hi)} />
        <line className="db-chart-grid" x1={L} x2={W - R} y1={y(lo)} y2={y(lo)} />
        <text className="db-chart-label" x={L - 6} y={y(hi) + 4} textAnchor="end">
          {money(hi)}
        </text>
        {hi !== lo && (
          <text className="db-chart-label" x={L - 6} y={y(lo) + 4} textAnchor="end">
            {money(lo)}
          </text>
        )}
        <path className="db-chart-line" d={path} />
        {callAt >= 0 && (
          <>
            <line className="db-chart-call" x1={x(callAt)} x2={x(callAt)} y1={T} y2={H - B} />
            <circle className="db-chart-dot" cx={x(callAt)} cy={y(prices[callAt])} r={3.5} />
          </>
        )}
        {lowAt >= 0 && (c.kind === "wait_right" || c.kind === "buy_wrong") && <circle className="db-chart-low" cx={x(lowAt)} cy={y(c.low)} r={3.5} />}
        <text className="db-chart-label" x={L} y={H - 6}>
          {short(c.prices[0][0])}
        </text>
        {callAt >= 0 && (
          <text className="db-chart-label db-chart-strong" x={x(callAt)} y={H - 6} textAnchor="middle">
            {short(c.predicted_on)}
          </text>
        )}
        <text className="db-chart-label" x={W - R} y={H - 6} textAnchor="end">
          {short(c.end_on)}
        </text>
      </svg>
      <figcaption className="db-muted">
        Real daily prices, {replayDay(c.prices[0][0])} to {replayDay(c.end_on)}. The line marks the call; the shaded part is
        the {c.horizon_days} days it was judged on.
      </figcaption>
    </figure>
  );
}

function Detail({ c }: { c: ReplayDetail }) {
  return (
    <div className="db-replay">
      <div className="db-replay-panel">
        <Panel
          data={replayPanel(c)}
          markUrl="mark.png"
          onClose={() => {}}
          onSeeOffers={() => {}}
          replayDate={`as of ${replayDay(c.predicted_on)}`}
        />
        {c.said === "BUY" && (
          <p className="db-muted">
            Drop chance that day: {Math.round(c.drop_chance * 100)}%, too low to say Wait.
          </p>
        )}
      </div>
      <div className="db-replay-main">
        <PriceChart c={c} />
        <p className={c.right ? "db-replay-outcome db-good" : "db-replay-outcome db-caution"}>{outcome(c)}</p>
      </div>
    </div>
  );
}

export default function Replay() {
  const [rule, setRule] = useState<string | null>(null);
  const [calls, setCalls] = useState<ReplayCall[]>([]);
  const [chosen, setChosen] = useState<ReplayCall | null>(null);
  const [detail, setDetail] = useState<ReplayDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    chrome.runtime
      .sendMessage({ type: "BUYWISE_GET_REPLAY" })
      .then((res) => {
        if (!res?.ok) return setError(res?.error === "no_track_record" ? "No past calls to replay yet." : UNREACHABLE);
        setRule(res.replay.rule);
        setCalls(res.replay.calls);
        if (res.replay.calls.length) setChosen(res.replay.calls[0]);
      })
      .catch(() => setError(UNREACHABLE));
  }, []);

  useEffect(() => {
    if (!chosen) return;
    setDetail(null);
    chrome.runtime
      .sendMessage({ type: "BUYWISE_GET_REPLAY", asin: chosen.asin, day: chosen.predicted_on })
      .then((res) => (res?.ok ? setDetail(res.replay) : setError(UNREACHABLE)))
      .catch(() => setError(UNREACHABLE));
  }, [chosen]);

  return (
    <section>
      <h2>Replay a past call</h2>
      <p className="db-muted">
        Pick a call to see the panel the model would have shown that day and what the price did next. Prices and dates are
        real. {rule}
      </p>
      {error && <p className="db-caution db-note">{error}</p>}
      {calls.length > 0 && (
        <ul className="db-list">
          {calls.map((c) => {
            const on = chosen?.asin === c.asin && chosen.predicted_on === c.predicted_on;
            return (
              <li key={c.asin + c.predicted_on} className={on ? "db-item db-item--pick db-item--chosen" : "db-item db-item--pick"}>
                <button type="button" className="db-replay-pick" aria-pressed={on} onClick={() => setChosen(c)}>
                  <span className="db-item-main">
                    <span className="db-item-title">
                      {c.brand ? `${c.brand} ` : ""}
                      <span className="db-mono">{c.asin}</span>
                    </span>
                    <span className="db-muted">
                      {KIND_LABEL[c.kind]}: said {c.said === "WAIT" ? "Wait" : "Buy"} on {replayDay(c.predicted_on)} at{" "}
                      {money(c.price)}
                    </span>
                  </span>
                  <span className={c.right ? "db-good db-mono" : "db-caution db-mono"}>
                    {c.right ? "+" : "-"}
                    {money(c.amount)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {detail && <Detail c={detail} />}
    </section>
  );
}
