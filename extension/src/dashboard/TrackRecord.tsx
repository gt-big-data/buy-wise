import React, { useEffect, useState } from "react";

// "How BuyWise has done" (#93), from the backend's /track-record (#85).

type Call = { asin: string; predicted_on: string; said: "WAIT" | "BUY"; price: number; low: number; end: number; amount: number };

type Record = {
  products: number;
  first_day: string;
  last_day: string;
  calls: number;
  hit_rate: number;
  base_rate: number;
  wait_calls: number;
  wait_right: number;
  saved: number;
  lost_waiting: number;
  missed_drops: number;
  wins: Call[];
  misses: Call[];
};

const money = (n: number) => `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (n: number) => `${Math.round(n * 100)}%`;
const month = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { month: "short", year: "numeric" });
const day = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

function whatHappened(c: Call): string {
  if (c.said === "WAIT" && c.low < c.price) return `Dropped to ${money(c.low)}.`;
  if (c.said === "WAIT") return `Didn't drop. Was ${money(c.end)} two weeks later.`;
  return `Dropped to ${money(c.low)} anyway.`;
}

function CallList({ calls, right }: { calls: Call[]; right: boolean }) {
  return (
    <ul className="db-list">
      {calls.map((c) => (
        <li key={c.asin + c.predicted_on} className="db-item">
          <div className="db-item-main">
            <a className="db-item-title db-mono" href={`https://www.amazon.com/dp/${c.asin}`} target="_blank" rel="noopener">
              {c.asin}
            </a>
            <p className="db-muted">
              Said {c.said === "WAIT" ? "Wait" : "Buy"} on {day(c.predicted_on)} at {money(c.price)}. {whatHappened(c)}
            </p>
          </div>
          <div className="db-item-side">
            <p className={right ? "db-good db-mono" : "db-caution db-mono"}>
              {right ? "+" : "-"}
              {money(c.amount)}
            </p>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function TrackRecord() {
  const [record, setRecord] = useState<Record | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    chrome.runtime
      .sendMessage({ type: "BUYWISE_GET_TRACK_RECORD" })
      .then((res) => {
        if (res?.ok) setRecord(res.record);
        else if (res?.error === "no_track_record") setError("No track record yet. Run backend/scripts/track_record.py replay.");
        else setError("Couldn't reach BuyWise. Is the backend running?");
      })
      .catch(() => setError("Couldn't reach BuyWise. Is the backend running?"));
  }, []);

  return (
    <section>
      <h2>How BuyWise has done</h2>
      {error && <p className="db-caution">{error}</p>}
      {record && (
        <>
          <p className="db-muted">
            Replayed over {record.products} products from {month(record.first_day)} to {month(record.last_day)}, each call
            checked against the next 14 days of real prices. These are the model's calls, not ones shoppers saw.
          </p>
          <div className="db-stats">
            <div>
              <p className="db-stat db-mono">{pct(record.wait_right)}</p>
              <p className="db-muted">
                of Wait calls dropped, vs {pct(record.base_rate)} if you waited at random
              </p>
            </div>
            <div>
              <p className="db-stat db-mono db-good">{money(record.saved)}</p>
              <p className="db-muted">saved by right Wait calls</p>
            </div>
            <div>
              <p className="db-stat db-mono db-caution">{money(record.lost_waiting + record.missed_drops)}</p>
              <p className="db-muted">lost or missed when it was wrong</p>
            </div>
          </div>
          <p className="db-muted db-note">
            {money(record.lost_waiting)} paid extra by waiting when the price didn't drop, and {money(record.missed_drops)} in
            drops that came after it said Buy. Overall {pct(record.hit_rate)} of {record.calls.toLocaleString("en-US")} calls
            were right, but most were Buy calls on prices that didn't move.
          </p>

          <div className="db-columns">
            <div>
              <h3>Biggest wins</h3>
              <CallList calls={record.wins} right={true} />
            </div>
            <div>
              <h3>Biggest misses</h3>
              <CallList calls={record.misses} right={false} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}