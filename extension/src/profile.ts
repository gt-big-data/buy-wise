// The shopper's first-run profile. It lives in chrome.storage.local only and is
// never sent to the backend.

import { PROFILE_CHOICES, type ProfileQuestion } from "../../website/app/preview/copy";

export { PROFILE_CHOICES, type ProfileQuestion };

export const PROFILE_KEY = "buywise_profile";
export const QUESTIONS = Object.keys(PROFILE_CHOICES) as ProfileQuestion[];

export type Choice<Q extends ProfileQuestion> = keyof (typeof PROFILE_CHOICES)[Q];

// null means the shopper skipped or hasn't answered that question yet.
export type ProfileAnswers = { [Q in ProfileQuestion]: Choice<Q> | null };

export type StoredProfile = {
  answers: ProfileAnswers;
  // Set the first time the shopper finishes or skips the first-run screen, so it never shows again.
  seenAt: string | null;
};

// What we assume for an unanswered question. Never recommend used items to someone
// who hasn't said they'd take one; otherwise pick the middle option.
export const DEFAULT_ANSWERS: { [Q in ProfileQuestion]: Choice<Q> } = {
  condition: "new_only",
  patience: "week_or_two",
  alerts: "on",
};

export const EMPTY_ANSWERS: ProfileAnswers = { condition: null, patience: null, alerts: null };

function isChoice(q: ProfileQuestion, value: unknown): boolean {
  return typeof value === "string" && value in PROFILE_CHOICES[q];
}

// Anything unreadable in storage (an old shape, a renamed choice) counts as unanswered.
export function parseProfile(raw: unknown): StoredProfile {
  const obj = (raw ?? {}) as Partial<StoredProfile>;
  const answers = { ...EMPTY_ANSWERS } as Record<ProfileQuestion, unknown>;
  for (const q of QUESTIONS) {
    const value = obj.answers?.[q];
    if (isChoice(q, value)) answers[q] = value;
  }
  return {
    answers: answers as ProfileAnswers,
    seenAt: typeof obj.seenAt === "string" ? obj.seenAt : null,
  };
}

// The answers recommendations should use: what the shopper said, else the default.
export function resolvedAnswers(answers: ProfileAnswers): { [Q in ProfileQuestion]: Choice<Q> } {
  return {
    condition: answers.condition ?? DEFAULT_ANSWERS.condition,
    patience: answers.patience ?? DEFAULT_ANSWERS.patience,
    alerts: answers.alerts ?? DEFAULT_ANSWERS.alerts,
  };
}

export function choiceText(q: ProfileQuestion, choice: string): string {
  return (PROFILE_CHOICES[q] as Record<string, string>)[choice];
}

export async function loadProfile(): Promise<StoredProfile> {
  const res = await chrome.storage.local.get([PROFILE_KEY]);
  return parseProfile(res[PROFILE_KEY]);
}

export async function saveProfile(answers: ProfileAnswers): Promise<StoredProfile> {
  const current = await loadProfile();
  const next: StoredProfile = { answers, seenAt: current.seenAt ?? new Date().toISOString() };
  await chrome.storage.local.set({ [PROFILE_KEY]: next });
  return next;
}
