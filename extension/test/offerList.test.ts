import { readdirSync, readFileSync, existsSync } from "fs";
import { join } from "path";
import { describe, expect, it, vi } from "vitest";
import { parseOfferList } from "../src/content/offerList";

const FIXTURES = join(__dirname, "fixtures/aod");

function load(name: string): Document {
  return new DOMParser().parseFromString(readFileSync(join(FIXTURES, name), "utf8"), "text/html");
}

vi.spyOn(console, "warn").mockImplementation(() => {});

// Every <name>.html with a hand-checked <name>.expected.json next to it
const fixtures = readdirSync(FIXTURES)
  .filter((f) => f.endsWith(".html") && existsSync(join(FIXTURES, f.replace(/\.html$/, ".expected.json"))));

describe("parseOfferList fixtures", () => {
  it.each(fixtures)("%s matches its expected offers", (name) => {
    const expected = JSON.parse(readFileSync(join(FIXTURES, name.replace(/\.html$/, ".expected.json")), "utf8"));
    expect(parseOfferList(load(name))).toEqual(expected);
  });
});

describe("parseOfferList layout changes", () => {
  it("reports unreadable when an offer row loses its price", () => {
    const doc = load("synthetic-mixed.html");
    doc.querySelectorAll("#aod-offer [data-csa-c-price-to-pay]")[1].removeAttribute("data-csa-c-price-to-pay");
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "row 2: no price", offers: [] });
  });

  it("reports unreadable when the seller markup is renamed", () => {
    const doc = load("synthetic-mixed.html");
    doc.querySelectorAll("#aod-offer-soldBy").forEach((el) => (el.id = "aod-offer-seller"));
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "pinned offer: no seller" });
  });

  it("reports unreadable when rows are renamed but the count says there are offers", () => {
    const doc = load("synthetic-mixed.html");
    doc.querySelectorAll("#aod-offer").forEach((el) => (el.id = "aod-offer-card"));
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "no offer rows, count says 12" });
  });

  it("reports unreadable for markup it doesn't recognise at all", () => {
    const doc = new DOMParser().parseFromString("<div id='aod-container'><p>new layout</p></div>", "text/html");
    expect(parseOfferList(doc)).toMatchObject({ status: "unreadable", reason: "no offer rows and no offer count" });
  });
});
