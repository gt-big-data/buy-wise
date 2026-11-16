import { readdirSync, readFileSync, existsSync } from "fs";
import { join } from "path";
import { describe, expect, it, vi } from "vitest";
import { parseCount, parseUnitPrice, readVersions } from "../src/content/versions";

const FIXTURES = join(__dirname, "fixtures/versions");

function load(name: string): Document {
  return new DOMParser().parseFromString(readFileSync(join(FIXTURES, name), "utf8"), "text/html");
}

vi.spyOn(console, "warn").mockImplementation(() => {});

// Every <name>.html with a hand-checked <name>.expected.json next to it
const fixtures = readdirSync(FIXTURES).filter(
  (f) => f.endsWith(".html") && existsSync(join(FIXTURES, f.replace(/\.html$/, ".expected.json")))
);

describe("readVersions fixtures", () => {
  it.each(fixtures)("%s matches its expected versions", (name) => {
    const expected = JSON.parse(readFileSync(join(FIXTURES, name.replace(/\.html$/, ".expected.json")), "utf8"));
    expect(readVersions(load(name))).toEqual(expected);
  });
});

describe("readVersions without a readable picker", () => {
  it("reports none for a product with no versions", () => {
    const doc = new DOMParser().parseFromString('<span id="productTitle">Thing</span>', "text/html");
    expect(readVersions(doc)).toEqual({ status: "none", dimensions: [], versions: [] });
  });

  it("reports unreadable for an older dropdown picker instead of none", () => {
    const doc = new DOMParser().parseFromString(
      '<div id="twister"><select id="native_dropdown_selected_size_name"><option>Large</option></select></div>',
      "text/html"
    );
    expect(readVersions(doc)).toMatchObject({ status: "unreadable", versions: [] });
  });

  it("reports unreadable when a row's swatches lose their ASINs", () => {
    const doc = load("synthetic-pack-counts.html");
    doc.querySelectorAll("li[data-asin]").forEach((li) => li.removeAttribute("data-asin"));
    expect(readVersions(doc)).toMatchObject({
      status: "unreadable",
      reason: "total_product_quantity: no swatches read",
    });
  });
});

describe("parseCount", () => {
  it.each([
    ["24 Count", 24],
    ["White - 6 Pack", 6],
    ["12 Ounce (Pack of 2)", 2],
    ["5 Pound (Pack of 1)", 1],
    ["1,440 Count", 1440],
    ["100 Ct", 100],
    ["2-Pack", 2],
    ["Large", null],
    ["2.0 TB", null],
    ["0 Count", null],
  ])("%s -> %s", (value, count) => {
    expect(parseCount(value)).toBe(count);
  });
});

describe("parseUnitPrice", () => {
  it.each([
    ["$1.31 per count", { unitPrice: 1.31, unitLabel: "count" }],
    ["($0.82 / count)", { unitPrice: 0.82, unitLabel: "count" }],
    ["$1.25 per Ounce", { unitPrice: 1.25, unitLabel: "ounce" }],
    ["$1,049.99", null],
    ["In Stock", null],
  ])("%s", (text, expected) => {
    expect(parseUnitPrice(text)).toEqual(expected);
  });
});
