import { describe, expect, it } from "vitest";
import { defaultTarget, targetText } from "../src/content/watchTarget";

describe("defaultTarget", () => {
  it("goes about 5% below the price when the expected low is within 1% of it", () => {
    const t = defaultTarget(99, 98.64);
    expect(t).toBe(94);
    expect(t).toBeLessThan(99);
  });

  it("uses the expected low, floored, when it's well below the price", () => {
    expect(defaultTarget(349, 285.37)).toBe(285);
    expect(defaultTarget(9.99, 7.456)).toBe(7.45);
  });

  it("is always strictly below the price", () => {
    for (const [price, low] of [[20, 19.99], [100, 100], [5, 6], [1, 0.995], [450, 0], [20.5, 20.1]]) {
      const t = defaultTarget(price, low);
      expect(t).toBeGreaterThan(0);
      expect(t).toBeLessThan(price);
    }
  });

  it("prints whole dollars without cents", () => {
    expect(targetText(94)).toBe("94");
    expect(targetText(7.45)).toBe("7.45");
  });
});
