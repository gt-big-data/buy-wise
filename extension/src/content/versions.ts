import { parsePrice } from "./offers";

// Reads the product's other versions (sizes, colors, pack counts) from the
// swatches under the price. Each swatch carries the ASIN it switches to and,
// on most pages, that version's price and Amazon's per-unit price. Everything
// read here is already on the page; nothing is fetched or clicked.

export interface Version {
  asin: string;
  dimension: string; // Amazon's key: "size_name", "color_name", "total_product_quantity", ...
  value: string; // "Large", "White - 6 Pack", "24 Count"
  selected: boolean; // the version on screen
  available: boolean;
  price: number | null; // null when the swatch shows no price (often out of stock)
  // Amazon's own per-unit price, e.g. 0.82 per "count". Units vary by product
  // and aren't always consistent (a 6-pack of shirts can say "$3.33 per pack").
  unitPrice: number | null;
  unitLabel: string | null;
  // How many items the version holds, read from its name ("6 Pack", "24 Count",
  // "Pack of 2"); null when the name doesn't say. pricePerItem = price / count.
  count: number | null;
  pricePerItem: number | null;
}

export interface Dimension {
  key: string;
  label: string | null; // "Size", "Color", "Total product quantity"
  selectedValue: string | null;
}

export interface VersionsResult {
  // none: the product has no versions. unreadable: it has a version picker we couldn't read.
  status: "ok" | "none" | "unreadable";
  reason?: string;
  dimensions: Dimension[];
  versions: Version[];
}

const ROW = '[id^="inline-twister-row-"]';
// Older or unusual pickers we don't read yet. Their presence without inline
// rows means there are versions we missed, not that there are none.
const OTHER_PICKERS = '#twister .a-dropdown-container, select[id^="native_dropdown_selected_"], [id^="variation_"] li[data-defaultasin]';

function text(el: Element | null | undefined): string | null {
  if (!el) return null;
  const copy = el.cloneNode(true) as Element;
  copy.querySelectorAll("style, script").forEach((s) => s.remove());
  return copy.textContent?.replace(/\s+/g, " ").trim() || null;
}

// The first "$12.34" in a string ("$57.74 with 7 percent savings" -> 57.74)
function firstMoney(s: string | null): number | null {
  const m = s?.match(/\$\s?([\d,]+(?:\.\d+)?)/);
  return m ? parsePrice(m[1]) : null;
}

// "$1.31 per count", "($0.82 / count)" -> [1.31, "count"]
export function parseUnitPrice(s: string | null): { unitPrice: number; unitLabel: string } | null {
  const m = s?.match(/\$\s?([\d,]+(?:\.\d+)?)\s*(?:per|\/)\s*([a-z][a-z .]*?)\s*\)?$/i);
  if (!m) return null;
  const unitPrice = parsePrice(m[1]);
  return unitPrice === null ? null : { unitPrice, unitLabel: m[2].trim().toLowerCase() };
}

// "White - 6 Pack" -> 6, "24 Count" -> 24, "12 Ounce (Pack of 2)" -> 2, "Large" -> null
export function parseCount(value: string): number | null {
  const packOf = value.match(/pack\s+of\s+(\d[\d,]*)/i);
  const m = packOf ?? value.match(/(\d[\d,]*)\s*-?\s*(?:count|ct|pack|pk|pcs|pieces?|rolls?|pairs?|bars?|pods?|bottles?|cans?)\b/i);
  const n = m ? Number(m[1].replace(/,/g, "")) : NaN;
  return n > 0 ? n : null;
}

function readSwatch(li: HTMLElement, dimension: string): Version | null {
  const asin = (li.dataset.asin || li.dataset.defaultasin || "").trim().toUpperCase();
  if (!/^[A-Z0-9]{10}$/.test(asin)) return null; // carousel arrows and "see more" tiles
  const value =
    text(li.querySelector(".swatch-title-text-display, .swatch-title-text")) ??
    li.querySelector("img")?.getAttribute("alt")?.trim() ??
    null;
  if (!value) return null;

  const price =
    firstMoney(text(li.querySelector(".apex-pricetopay-accessibility-label"))) ??
    firstMoney(text(li.querySelector(".a-price .a-offscreen"))) ??
    firstMoney(text(li.querySelector(".a-price")));
  const unit =
    Array.from(li.querySelectorAll(".a-size-mini"))
      .map((el) => parseUnitPrice(text(el)))
      .find((u) => u !== null) ?? null;
  const count = parseCount(value);

  return {
    asin,
    dimension,
    value,
    selected: li.dataset.initiallyselected === "true",
    available: li.dataset.initiallyunavailable !== "true",
    price,
    unitPrice: unit?.unitPrice ?? null,
    unitLabel: unit?.unitLabel ?? null,
    count,
    pricePerItem: price !== null && count !== null ? Math.round((price / count) * 100) / 100 : null,
  };
}

// "Size: Large" -> ["Size", "Large"]
function readTitle(root: ParentNode, key: string): [string | null, string | null] {
  const title = text(root.querySelector(`[id="inline-twister-dim-title-${key}"]`));
  const m = title?.match(/^([^:]+):\s*(.*)$/);
  return m ? [m[1].trim(), m[2].trim() || null] : [title, null];
}

export function readVersions(root: ParentNode = document): VersionsResult {
  const rows = Array.from(root.querySelectorAll<HTMLElement>(ROW));
  if (rows.length === 0) {
    if (root.querySelector(OTHER_PICKERS)) return unreadable("version picker in a layout we don't read");
    return { status: "none", dimensions: [], versions: [] };
  }

  const dimensions: Dimension[] = [];
  const versions: Version[] = [];
  for (const row of rows) {
    const key = row.id.slice("inline-twister-row-".length);
    const [label, titleValue] = readTitle(root, key);
    const seen = new Set<string>();
    let read = 0;
    for (const li of Array.from(row.querySelectorAll<HTMLElement>("li[data-asin], li[data-defaultasin]"))) {
      const v = readSwatch(li, key);
      if (!v || seen.has(v.asin)) continue;
      seen.add(v.asin);
      versions.push(v);
      read++;
    }
    if (read === 0) return unreadable(`${key}: no swatches read`);
    const selected = versions.find((v) => v.dimension === key && v.selected);
    dimensions.push({ key, label, selectedValue: selected?.value ?? titleValue });
  }
  return { status: "ok", dimensions, versions };
}

function unreadable(reason: string): VersionsResult {
  console.warn("BuyWise versions:", reason);
  return { status: "unreadable", reason, dimensions: [], versions: [] };
}
