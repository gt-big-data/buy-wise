import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../src/config", () => ({ BACKEND_URL: "http://localhost:8000" }));
vi.mock("../src/userId", () => ({ getUserId: async () => "demo" }));
import { checkPrices } from "../src/background/alerts";

let storage: Record<string, any>;
let price: number;
const notify = vi.fn(async () => "notification");
beforeEach(() => {
  storage = {};
  price = 110;
  notify.mockClear();
  vi.stubGlobal("chrome", {
    storage: { local: {
      get: async (key: string) => ({ [key]: storage[key] }),
      set: async (values: object) => Object.assign(storage, values),
    } },
    notifications: { create: notify },
  });
  vi.stubGlobal("fetch", vi.fn(async (url: string) => ({
    ok: true,
    json: async () => url.includes("/watchlist/")
      ? { watchlist: [{ asin: "B08N5WRWNW", title: "Demo", target_price: 100 }] }
      : { current_price: price },
  })));
});

test("a stored live price drop alerts once, including simultaneous checks", async () => {
  expect(await checkPrices()).toBe(0);
  price = 90;
  await Promise.all([checkPrices(), checkPrices()]);
  expect(notify).toHaveBeenCalledTimes(1);
  expect(storage.buywise_alerts[0]).toMatchObject({ price: 90, target: 100 });
  expect(await checkPrices()).toBe(0);
});

test("notification failure can be retried", async () => {
  price = 90;
  notify.mockRejectedValueOnce(new Error("notification unavailable"));
  await expect(checkPrices()).rejects.toThrow("notification unavailable");
  expect(storage.buywise_alerted).toBeUndefined();
  expect(await checkPrices()).toBe(1);
});
