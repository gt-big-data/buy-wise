import { describe, expect, it } from "vitest";
import { toPanelData } from "../src/content/decision";
import { FAKE, pickFakeData } from "../src/content/fakeData";

const TODAY = new Date(2026, 10, 6);

describe("toPanelData", () => {
  it("maps a better offer and counts delivery days from today", () => {
    const d = { ...FAKE.better_offer, amazon: { ...FAKE.better_offer.amazon!, arrives: "2026-11-08" } };
    const p = toPanelData(d, TODAY);
    expect(p?.state).toBe("better_offer");
    if (p?.state !== "better_offer") return;
    expect(p.amazon.daysOut).toBe(2);
    expect(p.amazon.arrives).toBe("Sun, Nov 8");
    expect(p.best.fullWarranty).toBe(false);
    expect(p.reason).toContain("$80.99 less");
  });

  it("hides the panel when show is false", () => {
    expect(toPanelData({ ...FAKE.wait, show: false }, TODAY)).toBeNull();
  });

  it("hides the panel when the answer is missing what its status needs", () => {
    expect(toPanelData({ ...FAKE.better_offer, best: null }, TODAY)).toBeNull();
    expect(toPanelData({ ...FAKE.wait, wait: null }, TODAY)).toBeNull();
  });

  it("maps wait and nothing better", () => {
    const w = toPanelData(FAKE.wait, TODAY);
    expect(w).toMatchObject({ state: "wait", dropChance: 0.74, expectedLow: 285 });
    const n = toPanelData(FAKE.nothing_better, TODAY);
    expect(n).toMatchObject({ state: "nothing_better", nextBest: { price: 254, daysLater: 9 } });
  });
});

describe("pickFakeData", () => {
  it("only returns data in dev mode", () => {
    expect(pickFakeData("https://www.amazon.com/dp/B0H964GJMD")).toBeNull();
    expect(pickFakeData("https://www.amazon.com/dp/B0H964GJMD?bw_state=nope")).toBeNull();
    expect(pickFakeData("https://www.amazon.com/dp/B0H964GJMD?bw_state=wait")?.state).toBe("wait");
  });
});