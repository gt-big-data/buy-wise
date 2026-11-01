import React, { useEffect, useState } from "react";
import { choiceText, loadProfile, QUESTIONS, resolvedAnswers, type StoredProfile } from "../profile";
import { PROFILE_LABEL } from "../../../website/app/preview/copy";
import { ALERTS, TRACK_RECORD, WATCHING } from "./fakeDashboard";

const money = (n: number) => `$${n.toFixed(2)}`;

export default function Dashboard() {
  const [profile, setProfile] = useState<StoredProfile | null>(null);

  useEffect(() => {
    loadProfile().then(setProfile);
  }, []);

  return (
    <div className="db">
      <header className="db-head">
        <img src="mark.png" alt="" width={22} height={25} />
        <span className="db-name">BuyWise</span>
        <span className="db-sample">Sample data</span>
      </header>

      <section>
        <h2>Watching</h2>
        <ul className="db-list">
          {WATCHING.map((w) => {
            const atTarget = w.price <= w.target;
            return (
              <li key={w.title} className="db-item">
                <div className="db-item-main">
                  <p className="db-item-title">{w.title}</p>
                  <p className="db-muted">
                    Target {money(w.target)}
                    {w.price < w.startPrice ? ` · down ${money(w.startPrice - w.price)} since you started` : ""}
                  </p>
                </div>
                <div className="db-item-side">
                  <p className="db-mono">{money(w.price)}</p>
                  <p className={atTarget ? "db-good" : "db-muted"}>
                    {atTarget ? "At your target" : `${Math.round(w.dropChance * 100)}% chance of a drop`}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      <section>
        <h2>Recent alerts</h2>
        <ul className="db-list">
          {ALERTS.map((a) => (
            <li key={a.text} className="db-item db-item--alert">
              <span className="db-muted db-when">{a.when}</span>
              <span>{a.text}</span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>How BuyWise has done</h2>
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