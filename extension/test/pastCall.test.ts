import { describe, expect, it } from "vitest";
import { outcome, replayPanel, type ReplayDetail } from "../src/dashboard/pastCall";

const CALL: ReplayDetail = {
  asin: "B0D2PNH494",
  brand: "Hisense",
  predicted_on: "2025-09-25",
  said: "WAIT",
  kind: "wait_right",
  right: true,
  price: 999.99,
  drop_chance: 0.4702,
  low: 696.99,
  end: 696.99,
  amount: 303,
  horizon_days: 14,
  expected_low: 954.58,
  drops: 4,
  months: 4,
  low_on: "2025-10-06",
  end_on: "2025-10-09",
  prices: [["2025-09-25", 999.99], ["2025-10-06", 696.99]],
  model_version: "clean@2026-11-06T17:36:46+00:00",
};

describe("replay", () => {
  it("says what happened with real dates", () => {
    expect(outcome(CALL)).toBe("Dropped to $696.99 on Oct 6, 2025. $303.00 saved by waiting.");
    expect(outcome({ ...CALL, said: "BUY", kind: "buy_right", low: 1345.98, end: 1645.98, amount: 298.04, end_on: "2025-12-11" }))
      .toBe("Didn't drop. Buying that day was right: it was $1,645.98 on Dec 11, 2025, $298.04 higher.");
    expect(outcome({ ...CALL, kind: "wait_wrong", right: false, low: 999.99, end: 1055.99, amount: 56, low_on: null }))
      .toBe("Didn't drop 8%. It was $1,055.99 on Oct 9, 2025, $56.00 more than buying that day.");
    expect(outcome({ ...CALL, said: "BUY", kind: "buy_wrong", right: false })).toBe(
      "A miss: it dropped to $696.99 on Oct 6, 2025. Waiting would have saved $303.00.",
    );
  });

  it("shows the Wait panel with that day's numbers, or no reason to wait", () => {
    expect(replayPanel(CALL)).toEqual({
      state: "wait", offersChecked: 0, price: 999.99, expectedLow: 954.58, dropChance: 0.4702, drops: 4, months: 4,
    });
    expect(replayPanel({ ...CALL, said: "BUY", kind: "buy_right" }).state).toBe("nothing_better");
  });
});
