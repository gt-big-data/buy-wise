import React, { useEffect, useState } from "react";
import { PROFILE_QUESTION, PROFILE_TEXT, assumedLine } from "../../../website/app/preview/copy";
import {
  DEFAULT_ANSWERS,
  PROFILE_CHOICES,
  QUESTIONS,
  choiceText,
  loadProfile,
  saveProfile,
  type ProfileAnswers,
  type ProfileQuestion,
} from "../profile";

// Opened with ?welcome right after install; otherwise it's the edit screen the
// popup and chrome://extensions "Options" link to.
const isWelcome = new URLSearchParams(window.location.search).has("welcome");

type Notice = "allSet" | "skipped" | "saved" | null;

const ProfilePage: React.FC = () => {
  const [answers, setAnswers] = useState<ProfileAnswers | null>(null);
  const [welcome, setWelcome] = useState(isWelcome);
  const [notice, setNotice] = useState<Notice>(null);

  useEffect(() => {
    loadProfile().then((p) => setAnswers(p.answers));
  }, []);

  if (!answers) return null;

  const pick = (q: ProfileQuestion, choice: string) => {
    setAnswers({ ...answers, [q]: choice });
    setNotice(null);
  };

  const finish = async (skip: boolean) => {
    // Skipping keeps whatever was picked so far; the rest stay unanswered and use the defaults.
    const saved = await saveProfile(answers);
    setAnswers(saved.answers);
    setWelcome(false);
    setNotice(skip ? "skipped" : welcome ? "allSet" : "saved");
  };

  return (
    <div className="bw-profile">
      <div className="bw-profile__card">
        <div className="bw-profile__head">
          <img src={chrome.runtime.getURL("logo.png")} alt="" className="bw-profile__logo" />
          <span className="bw-profile__title">{welcome ? PROFILE_TEXT.welcome : PROFILE_TEXT.editTitle}</span>
        </div>
        {welcome && <p className="bw-profile__intro">{PROFILE_TEXT.intro}</p>}
        {notice && (
          <p className="bw-profile__notice" role="status">
            {notice === "saved" ? PROFILE_TEXT.saved : PROFILE_TEXT[notice]}
          </p>
        )}

        {QUESTIONS.map((q) => {
          const selected = answers[q] as string | null;
          return (
            <fieldset key={q} className="bw-profile__q">
              <legend className="bw-profile__q-text">{PROFILE_QUESTION[q]}</legend>
              {Object.entries(PROFILE_CHOICES[q]).map(([choice, text]) => (
                <label key={choice} className={`bw-profile__opt${selected === choice ? " is-selected" : ""}`}>
                  <input
                    type="radio"
                    name={q}
                    value={choice}
                    checked={selected === choice}
                    onChange={() => pick(q, choice)}
                  />
                  {text}
                </label>
              ))}
              {!welcome && selected === null && (
                <p className="bw-profile__assumed">{assumedLine(choiceText(q, DEFAULT_ANSWERS[q]))}</p>
              )}
            </fieldset>
          );
        })}

        <p className="bw-profile__device">{PROFILE_TEXT.onDevice}</p>
        <div className="bw-profile__actions">
          <button type="button" className="bw-profile__btn" onClick={() => finish(false)}>
            {welcome ? PROFILE_TEXT.done : PROFILE_TEXT.save}
          </button>
          {welcome && (
            <button type="button" className="bw-profile__link" onClick={() => finish(true)}>
              {PROFILE_TEXT.skip}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
