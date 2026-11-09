import React, { useEffect, useState } from "react";
import { choiceText, loadProfile, QUESTIONS, resolvedAnswers, type StoredProfile } from "../profile";
import { PROFILE_LABEL } from "../../../website/app/preview/copy";
import { TRACK_RECORD } from "./fakeDashboard";
import type { SentAlert, WatchedItem } from "../background/alerts";

const money = (n: number) => `$${n.toFixed(2)}`;

export default function Dashboard() {
  const [profile, setProfile] = useState<StoredProfile | null>(null);
  const [watching, setWatching] = useState<WatchedItem[] | null>(null);
  const [alerts, setAlerts] = useState<SentAlert[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  async function loadWatching() {
    const res = await chrome.runtime.sendMessage({ type: "BUYWISE_GET_WATCHLIST" }).catch(() => null);
    if (res?.ok) {
      setWatching(res.items);
      setAlerts(res.alerts);
      setError(null);
    } else {
      setError("Couldn't reach BuyWise. Is the backend running?");
    }
  }

  async function checkNow() {
    setChecking(true);
    await chrome.runtime.sendMessage({ type: "BUYWISE_CHECK_PRICES" }).catch(() => null);
    await loadWatching();
    setChecking(false);
  }

  useEffect(() => {
    loadProfile().then(setProfile);
    loadWatching();
  }, []);

  return (
    <div className="db">
      <header className="db-head">
        <img src="mark.png" alt="" width={22} height={25} />
        <span className="db-name">BuyWise</span>
      </header>

      <section>
        <div className="db-section-head">
          <h2>Watching</h2>
          <button type="button" className="db-button db-button--small" disabled={checking} onClick={checkNow}>
            {checking ? "Checking..." : "Check prices now"}
          </button>
        </div>
        {error && <p className="db-caution">{error}</p>}
        {watching && watching.length === 0 && (
          <p className="db-muted">Nothing yet. Tap Watch on a product page and it shows up here.</p>
        )}
        {watching && watching.length > 0 && (
          <ul className="db-list">
            {watching.map((w) => {
              const atTarget = w.price !== null && w.target !== null && w.price <= w.target;
              return (
                <li key={w.asin} className="db-item">
                  <div className="db-item-main">
                    <a className="db-item-title" href={`https://www.amazon.com/dp/${w.asin}`} target="_blank" rel="noopener">
                      {w.title}
                    </a>
                    <p className="db-muted">{w.target !== null ? `Target ${money(w.target)}` : "No target set"}</p>
                  </div>
                  <div className="db-item-side">
                    <p className="db-mono">{w.price !== null ? money(w.price) : "No price yet"}</p>
                    <p className={atTarget ? "db-good" : "db-muted"}>{atTarget ? "At your target" : "Waiting"}</p>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section>
        <h2>Recent alerts</h2>
        {alerts.length === 0 ? (
          <p className="db-muted">No alerts yet. You'll get one when a watched price hits your target.</p>
        ) : (
          <ul className="db-list">
            {alerts.map((a) => (
              <li key={a.asin + a.at} className="db-item db-item--alert">
                <span className="db-muted db-when">
                  {new Date(a.at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}
                </span>
                <span>
                  {a.title} dropped to {money(a.price)}, at or below your {money(a.target)} target.
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <div className="db-section-head">
          <h2>How BuyWise has done</h2>
          <span className="db-sample">Sample data until #85</span>
        </div>
        <p className="db-muted">
          Since {TRACK_RECORD.since}. Every call is checked 14 days later against what the price actually did.
        </p>
        <div className="db-stats">
          <div>
            <p className="db-stat db-mono">
              {TRACK_RECORD.right} of {TRACK_RECORD.calls}
            </p>
            <p className="db-muted">calls were right ({Math.round((TRACK_RECORD.right / TRACK_RECORD.calls) * 100)}%)</p>
          </div>
          <div>
            <p className="db-stat db-mono db-good">{money(TRACK_RECORD.saved)}</p>
            <p className="db-muted">saved when it was right</p>
          </div>
          <div>
            <p className="db-stat db-mono db-caution">{money(TRACK_RECORD.lost)}</p>
            <p className="db-muted">cost you when it was wrong</p>
          </div>
        </div>

        <h3>Recent calls</h3>
        <ul className="db-list">
          {TRACK_RECORD.recent.map((c) => (
            <li key={c.title} className="db-item">
              <div className="db-item-main">
                <p className="db-item-title">{c.title}</p>
                <p className="db-muted">
                  Said {c.said}. {c.happened}.
                </p>
              </div>
              <div className="db-item-side">
                <p className={c.right ? "db-good" : "db-caution"}>{c.right ? "Right" : "Wrong"}</p>
                <p className="db-muted db-mono">
                  {c.right ? "+" : "-"}
                  {money(c.amount)}
                </p>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>Your profile</h2>
        {profile && (
          <dl className="db-profile">
            {QUESTIONS.map((q) => (
              <div key={q}>
                <dt className="db-muted">{PROFILE_LABEL[q]}</dt>
                <dd>{choiceText(q, resolvedAnswers(profile.answers)[q])}</dd>
              </div>
            ))}
          </dl>
        )}
        <button type="button" className="db-button" onClick={() => chrome.runtime.openOptionsPage()}>
          Edit profile
        </button>
      </section>
    </div>
  );
}