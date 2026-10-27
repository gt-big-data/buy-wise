import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_ANSWERS,
  EMPTY_ANSWERS,
  PROFILE_KEY,
  loadProfile,
  parseProfile,
  resolvedAnswers,
  saveProfile,
} from "../src/profile";

// A fake chrome.storage.local backed by a plain object
let store: Record<string, unknown>;
beforeEach(() => {
  store = {};
  vi.stubGlobal("chrome", {
    storage: {
      local: {
        get: async (keys: string[]) => Object.fromEntries(keys.filter((k) => k in store).map((k) => [k, store[k]])),
        set: async (items: Record<string, unknown>) => void Object.assign(store, structuredClone(items)),
      },
    },
  });
});

describe("parseProfile", () => {
  it("treats a missing profile as unanswered and never seen", () => {
    expect(parseProfile(undefined)).toEqual({ answers: EMPTY_ANSWERS, seenAt: null });
  });

  it("drops choices it doesn't recognise", () => {
    const parsed = parseProfile({ answers: { condition: "refurbished", patience: "soon", alerts: 3 }, seenAt: "x" });
    expect(parsed).toEqual({ answers: { condition: null, patience: "soon", alerts: null }, seenAt: "x" });
  });
});

describe("resolvedAnswers", () => {
  it("fills skipped questions with the defaults", () => {
    expect(resolvedAnswers(EMPTY_ANSWERS)).toEqual(DEFAULT_ANSWERS);
    expect(resolvedAnswers({ ...EMPTY_ANSWERS, condition: "any_good" }).condition).toBe("any_good");
  });

  it("assumes new only, so a skipper is never pushed toward used items", () => {
    expect(DEFAULT_ANSWERS.condition).toBe("new_only");
  });
});

describe("saveProfile", () => {
  it("stores answers under one key and marks the first run as seen", async () => {
    const saved = await saveProfile({ condition: "like_new", patience: "deal", alerts: "off" });
    expect(saved.seenAt).not.toBeNull();
    expect(store[PROFILE_KEY]).toEqual(saved);
    expect(await loadProfile()).toEqual(saved);
  });

  it("keeps the original seenAt when answers are edited later", async () => {
    const first = await saveProfile(EMPTY_ANSWERS);
    const second = await saveProfile({ ...EMPTY_ANSWERS, patience: "soon" });
    expect(second.seenAt).toBe(first.seenAt);
    expect(second.answers.patience).toBe("soon");
  });
});
