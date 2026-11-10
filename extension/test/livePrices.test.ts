import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { followLivePrices } from "../src/content/livePrices";
const page = vi.hoisted(() => ({ asin: "B08N5WRWNW", price: null as number | null }));
vi.mock("../src/content/amazon", () => ({ extractASIN: () => page.asin }));
vi.mock("../src/content/offers", () => ({ extractOffers: () => ({ oneTimePrice: page.price }) }));
const send = vi.fn();
beforeEach(() => {
  vi.useFakeTimers();
  page.asin = "B08N5WRWNW";
  page.price = null;
  send.mockReset().mockResolvedValue({ ok: true });
  vi.stubGlobal("chrome", { runtime: { sendMessage: send } });
});
afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); });

test("waits for the buy box, saves once, and follows variants", async () => {
  followLivePrices();
  await vi.advanceTimersByTimeAsync(1000);
  expect(send).not.toHaveBeenCalled();
  page.price = 50;
  await vi.advanceTimersByTimeAsync(3000);
  expect(send).toHaveBeenCalledTimes(1);
  expect(send).toHaveBeenLastCalledWith({ type: "BUYWISE_RECORD_PRICE", asin: page.asin, price: 50 });
  page.asin = "B000000001";
  await vi.advanceTimersByTimeAsync(1000);
  expect(send).toHaveBeenCalledTimes(2);
  expect(send).toHaveBeenLastCalledWith({ type: "BUYWISE_RECORD_PRICE", asin: page.asin, price: 50 });
});

test("retries unsuccessful writes but stops after a bounded window", async () => {
  page.price = 50;
  send.mockResolvedValue({ ok: false });
  followLivePrices();
  await vi.advanceTimersByTimeAsync(30000);
  expect(send).toHaveBeenCalledTimes(20);
});
