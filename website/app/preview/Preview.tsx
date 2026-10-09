"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import {
  LABEL,
  OFFER_TAG,
  PROFILE_LABEL,
  SKIP,
  STATUS,
  WARRANTY,
  bestOtherLine,
  checked,
  date,
  dateRange,
  deliveryDelta,
  fulfillmentLine,
  headline,
  missLine,
  money,
  patternLine,
  pct,
  profileSummary,
  ratingLine,
  savingsVs,
  sellerLine,
  watchFacts,
  watchingConfirm,
  type OfferTag,
  type Status,
} from "./copy";
import { PROFILE_OPTIONS, bose, boseHistory, garmin, profile, sony, trackRecord, watching, type Offer } from "./data";

/* ───────────────────────── decisions derived from data ───────────────────────── */

const featured = garmin.offers.find((o) => o.featured)!;
const eligible = garmin.offers.filter((o) => !o.skip);
const best = eligible.filter((o) => !o.featured).sort((a, b) => a.price - b.price)[0];
const fullWarrantyAlt =
  best.warranty === "full"
    ? null
    : eligible.filter((o) => o !== best && !o.featured && o.warranty === "full" && o.price < featured.price).sort((a, b) => a.price - b.price)[0] ?? null;

const tagFor = (o: Offer): OfferTag => (o.skip ? "skipped" : o === best ? "best" : o.featured ? "featured" : "eligible");
const offerNote = (o: Offer) => {
  if (o.skip) return SKIP[o.skip];
  if (o.featured) return WARRANTY[o.warranty];
  return [savingsVs(featured.price, o.price), deliveryDelta(featured.arrives, o.arrives), WARRANTY[o.warranty]].join(" · ");
};
const rankedOffers = [...garmin.offers].sort((a, b) => {
  const order: OfferTag[] = ["best", "featured", "eligible", "skipped"];
  return order.indexOf(tagFor(a)) - order.indexOf(tagFor(b)) || a.price - b.price;
});

/* ───────────────────────── shared pieces ───────────────────────── */

function StatusTag({ status }: { status: Status }) {
  const cls: Record<Status, string> = {
    better_offer: "bg-ink text-white",
    buy_now: "bg-ink text-white",
    wait: "border border-ink text-ink",
    buy_here: "border border-line-strong text-ink-2",
  };
  return <span className={`inline-block whitespace-nowrap rounded-[4px] px-2 py-0.5 text-[11.5px] font-medium ${cls[status]}`}>{STATUS[status]}</span>;
}

function OfferTagChip({ tag }: { tag: OfferTag }) {
  const cls: Record<OfferTag, string> = {
    best: "bg-ink text-white",
    featured: "border border-line-strong text-ink-2",
    eligible: "border border-line text-muted",
    skipped: "text-faint",
  };
  return <span className={`inline-block whitespace-nowrap rounded-[4px] px-2 py-0.5 text-[11.5px] ${cls[tag]}`}>{OFFER_TAG[tag]}</span>;
}

function Sparkline({ data, w = 120, h = 32, stroke = "var(--color-ink-2)" }: { data: number[]; w?: number; h?: number; stroke?: string }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const pts = data
    .map((v, i) => `${((i / (data.length - 1)) * w).toFixed(1)},${(h - 2 - ((v - min) / (max - min || 1)) * (h - 4)).toFixed(1)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} aria-hidden="true">
      <polyline points={pts} fill="none" stroke={stroke} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function PriceChart() {
  const data = boseHistory;
  const w = 320;
  const h = 96;
  const min = 260;
  const max = 360;
  const y = (v: number) => h - ((v - min) / (max - min)) * h;
  const pts = data.map((v, i) => `${((i / (data.length - 1)) * w).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h + 18}`} className="w-full" role="img" aria-label="Price over the last 120 days">
      {[Math.min(...data), Math.max(...data)].map((v) => (
        <g key={v}>
          <line x1="0" x2={w} y1={y(v)} y2={y(v)} stroke="var(--color-line)" strokeDasharray="2 3" />
          <text x={w} y={y(v) - 4} textAnchor="end" className="fill-[var(--color-faint)] font-mono text-[9px]">
            {money(v)}
          </text>
        </g>
      ))}
      <polyline points={pts} fill="none" stroke="var(--color-ink)" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx={w} cy={y(data[data.length - 1])} r="3" fill="var(--color-ink)" />
      <text x="0" y={h + 14} className="fill-[var(--color-faint)] font-mono text-[9px]">
        120 days ago
      </text>
      <text x={w} y={h + 14} textAnchor="end" className="fill-[var(--color-faint)] font-mono text-[9px]">
        today
      </text>
    </svg>
  );
}

function Btn({ children, kind = "primary", onClick }: { children: React.ReactNode; kind?: "primary" | "secondary" | "link"; onClick?: () => void }) {
  const cls = {
    primary: "h-9 rounded-ui bg-ink px-3.5 text-[13.5px] font-medium text-white hover:bg-ink-2",
    secondary: "h-9 rounded-ui border border-line-strong bg-card px-3.5 text-[13.5px] font-medium hover:border-ink",
    link: "text-[13px] text-muted underline-offset-2 hover:text-ink hover:underline",
  }[kind];
  return (
    <button type="button" onClick={onClick} className={cls}>
      {children}
    </button>
  );
}

function PanelShell({ children, meta, profileLine = true }: { children: React.ReactNode; meta: string; profileLine?: boolean }) {
  return (
    <div className="rounded-ui border border-line-strong bg-card text-ink">
      <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5 text-[12.5px] text-muted">
        <Image src="/mark.png" alt="" width={14} height={16} />
        <span className="font-semibold text-ink">BuyWise</span>
        <span>{meta}</span>
      </div>
      {children}
      {profileLine && (
        <p className="border-t border-line px-3.5 py-2.5 text-[12px] text-faint">
          For you: {profileSummary(profile)} · <span className="text-muted underline underline-offset-2">Edit</span>
        </p>
      )}
    </div>
  );
}

function Verdict({ status, text }: { status: Status; text: string }) {
  return (
    <div>
      <StatusTag status={status} />
      <p className="mt-2 text-[16px] font-semibold leading-snug tracking-tight">{text}</p>
    </div>
  );
}

function Row({ label, value, tone }: { label: string; value: React.ReactNode; tone?: "caution" | "good" }) {
  const color = tone === "caution" ? "text-caution" : tone === "good" ? "text-brand-ink" : "text-ink-2";
  return (
    <div className="flex justify-between gap-4 border-t border-line py-2 text-[13px]">
      <span className="flex-none text-muted">{label}</span>
      <span className={`text-right ${color}`}>{value}</span>
    </div>
  );
}

/* ───────────────────────── the panel states ───────────────────────── */

function BetterOfferPanel({ onCompare }: { onCompare?: () => void }) {
  return (
    <PanelShell meta={checked(garmin.offersChecked)}>
      <div className="px-3.5 pb-3.5 pt-3">
        <Verdict status="better_offer" text={headline.better_offer(featured.price - best.price, best.condition)} />
        <p className="mt-1 text-[13px] text-muted">
          {best.seller} · {sellerLine(best.rating)}
        </p>
        <div className="mt-3">
          <Row label={LABEL.price} value={money(best.price)} />
          <Row label={LABEL.arrives} value={`${date(best.arrives)} · ${deliveryDelta(featured.arrives, best.arrives)}`} />
          <Row label={LABEL.returns} value={fulfillmentLine(best)} />
          <Row label={LABEL.warranty} value={WARRANTY[best.warranty]} tone={best.warranty === "full" ? undefined : "caution"} />
        </div>
        {fullWarrantyAlt && (
          <div className="mt-2 rounded-ui bg-paper px-3 py-2 text-[12.5px]">
            <p className="text-muted">With full warranty</p>
            <p className="text-ink-2">
              {fullWarrantyAlt.seller} · {money(fullWarrantyAlt.price)} · {savingsVs(featured.price, fullWarrantyAlt.price)}
            </p>
          </div>
        )}
        <div className="mt-3 flex items-center gap-2">
          <Btn>View offer</Btn>
          <Btn kind="secondary" onClick={onCompare}>
            Compare all {garmin.offersChecked}
          </Btn>
        </div>
      </div>
    </PanelShell>
  );
}

function WaitPanel() {
  const [mode, setMode] = useState<"target" | "signal">("target");
  const [target, setTarget] = useState("290");
  const [watched, setWatched] = useState(false);
  const p = bose.pattern;
  return (
    <PanelShell meta={checked(bose.offersChecked)}>
      <div className="px-3.5 pb-3.5 pt-3">
        <Verdict status="wait" text={headline.wait(bose.dropChance)} />
        <div className="mt-3">
          <PriceChart />
        </div>
        <Row label={LABEL.expectedLow} value={money(bose.expectedLow)} />
        <Row label={LABEL.pattern} value={patternLine(p.drops, p.months, p.minDays, p.maxDays)} />
        <Row label={LABEL.bestOther} value={bestOtherLine(null)} />

        {!watched ? (
          <div className="mt-3 rounded-ui border border-line p-3">
            <p className="text-[13px] font-medium">Watch this product</p>
            <div className="mt-2 flex gap-1 rounded-ui bg-paper p-0.5 text-[12.5px]">
              {(
                [
                  ["target", "At my price"],
                  ["signal", "When it's a good time"],
                ] as const
              ).map(([k, l]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setMode(k)}
                  className={`flex-1 rounded-[4px] px-2 py-1.5 ${mode === k ? "bg-card font-medium shadow-[0_0_0_1px_var(--color-line)]" : "text-muted"}`}
                >
                  {l}
                </button>
              ))}
            </div>
            {mode === "target" && (
              <label className="mt-2.5 flex items-center gap-2 text-[13px] text-muted">
                {LABEL.target}
                <span className="flex items-center rounded-ui border border-line-strong bg-card px-2">
                  $
                  <input
                    value={target}
                    onChange={(e) => setTarget(e.target.value.replace(/[^0-9.]/g, ""))}
                    className="w-14 bg-transparent py-1 font-mono text-ink outline-none"
                    inputMode="decimal"
                  />
                </span>
              </label>
            )}
            <div className="mt-3 flex items-center gap-3">
              <Btn onClick={() => setWatched(true)}>Watch</Btn>
              <Btn kind="link">Buy now anyway</Btn>
            </div>
          </div>
        ) : (
          <div className="mt-3 rounded-ui border border-line bg-paper p-3 text-[13px]">
            <p>{watchingConfirm(mode === "target" ? Number(target) || null : null)}</p>
            <button type="button" onClick={() => setWatched(false)} className="mt-1.5 text-[12.5px] text-muted underline underline-offset-2">
              Change
            </button>
          </div>
        )}
      </div>
    </PanelShell>
  );
}

function QuietPanel() {
  const [open, setOpen] = useState(false);
  const s = sony.skippedCheaper;
  return (
    <PanelShell meta={checked(sony.offersChecked)} profileLine={open}>
      <div className="px-3.5 py-3">
        <Verdict status="buy_here" text={headline.buy_here()} />
        <button type="button" onClick={() => setOpen(!open)} className="mt-1 text-[12.5px] text-muted underline underline-offset-2">
          {open ? "Hide details" : "Details"}
        </button>
        {open && (
          <div className="mt-2">
            <Row label={LABEL.bestOther} value={bestOtherLine(sony.bestOther)} />
            <Row label={LABEL.dropChance} value={pct(sony.dropChance)} />
            <Row label={OFFER_TAG.skipped} value={`${money(s.price)}, ${s.condition.toLowerCase()} · ${SKIP[s.reason]}`} />
          </div>
        )}
      </div>
    </PanelShell>
  );
}

/* ───────────────────────── product page frame ───────────────────────── */

function ProductPage({
  title,
  price,
  arrives,
  rating,
  imageLabel,
  children,
}: {
  title: string;
  price: number;
  arrives: string;
  rating: { stars: number; count: number };
  imageLabel: string;
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-ui border border-line bg-white">
      <div className="flex items-center gap-3 border-b border-line bg-[#f2f3f1] px-4 py-2 text-[12px] text-faint">
        <span className="font-medium text-muted">Store</span>
        <span className="h-6 flex-1 rounded-[4px] bg-white" />
        <span>Simplified product page</span>
      </div>
      <div className="grid gap-6 p-5 md:grid-cols-[1fr_300px]">
        <div className="min-w-0">
          <div className="grid h-44 place-items-center rounded-ui bg-[#eceeea] text-[12px] text-faint">{imageLabel}</div>
          <p className="mt-4 text-[17px] font-medium leading-snug">{title}</p>
          <p className="mt-1 text-[12.5px] text-muted">{ratingLine(rating)}</p>
          <p className="mt-3 font-mono text-2xl">{money(price)}</p>
          <div className="mt-4 space-y-1.5">
            {[90, 75, 82, 60].map((w, i) => (
              <div key={i} className="h-2 rounded-full bg-[#eceeea]" style={{ width: `${w}%` }} />
            ))}
          </div>
        </div>
        <div className="min-w-0 space-y-3">
          <div className="rounded-ui border border-line p-3.5 text-[13px]">
            <p className="font-mono text-lg">{money(price)}</p>
            <p className="mt-1 text-muted">Delivery {date(arrives)}</p>
            <p className="mt-1 text-brand-ink">In stock</p>
            <div className="mt-3 space-y-1.5">
              <div className="rounded-full bg-[#f7ca00] py-1.5 text-center text-[13px]">Add to Cart</div>
              <div className="rounded-full bg-[#f5a623] py-1.5 text-center text-[13px]">Buy Now</div>
            </div>
            <p className="mt-2 text-[12px] text-muted">Ships from and sold by Amazon.com</p>
          </div>
          {children}
        </div>
      </div>
    </div>
  );
}

/* ───────────────────────── other screens ───────────────────────── */

function AllOffers() {
  return (
    <div className="rounded-ui border border-line-strong bg-card">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3 text-[13px] text-muted">
        <Image src="/mark.png" alt="" width={14} height={16} />
        <span className="font-semibold text-ink">All offers</span>
        <span>· Garmin Forerunner 265, 46mm · {checked(garmin.offersChecked)}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-[13px]">
          <thead className="text-[12px] text-faint">
            <tr className="border-b border-line">
              {["Seller", "Condition", "Price", "Delivery and returns", "BuyWise"].map((h) => (
                <th key={h} className="px-4 py-2 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rankedOffers.map((o) => {
              const tag = tagFor(o);
              const dim = tag === "skipped";
              return (
                <tr key={o.seller + o.condition} className={`border-b border-line align-top ${dim ? "text-faint" : ""}`}>
                  <td className="px-4 py-3">
                    <p className={dim ? "" : "font-medium text-ink"}>{o.seller}</p>
                    <p className="text-[12px] text-faint">{sellerLine(o.rating)}</p>
                  </td>
                  <td className="px-4 py-3">{o.condition}</td>
                  <td className="px-4 py-3 font-mono">{money(o.price)}</td>
                  <td className="px-4 py-3">
                    {dateRange(o.arrives, o.arrivesBy)}
                    <p className="text-[12px] text-faint">{fulfillmentLine(o)}</p>
                  </td>
                  <td className="w-[230px] px-4 py-3">
                    <OfferTagChip tag={tag} />
                    <p className={`mt-1.5 text-[12px] ${!dim && o.warranty === "may_not_apply" ? "text-caution" : "text-muted"}`}>{offerNote(o)}</p>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="border-t border-line px-4 py-4">
        <p className="text-[13px] font-medium">Other versions</p>
        <ul className="mt-2 grid gap-2 md:grid-cols-3">
          {garmin.versions.map((v) => (
            <li key={v.name} className={`rounded-ui border p-3 text-[13px] ${v.current ? "border-ink" : "border-line"}`}>
              <div className="flex justify-between gap-2">
                <span className="font-medium">{v.name}</span>
                <span className="font-mono">{money(v.price)}</span>
              </div>
              <p className="mt-1 text-[12.5px] text-muted">
                {v.current ? "This version" : `${savingsVs(featured.price, v.price)} · ${v.differs.join(" · ")}`}
              </p>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-[12px] text-faint">Differences come from the listing and recurring points in reviews.</p>
      </div>
    </div>
  );
}

function BeforeYouBuy() {
  const [answer, setAnswer] = useState<null | "soon" | "norush">(null);
  return (
    <div className="w-full max-w-[340px] rounded-ui border border-line-strong bg-card">
      <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5 text-[12.5px] text-muted">
        <Image src="/mark.png" alt="" width={14} height={16} />
        <span className="font-semibold text-ink">BuyWise</span>
        <span>One question</span>
      </div>
      <div className="p-3.5">
        <p className="text-[15px] font-medium">When do you need it?</p>
        <p className="mt-0.5 text-[12.5px] text-muted">
          {OFFER_TAG.best}: {money(best.price)}, arrives {date(best.arrives)} · {OFFER_TAG.featured}: {money(featured.price)}, arrives{" "}
          {date(featured.arrives)}
        </p>
        <div className="mt-3 grid grid-cols-2 gap-2">
          {(
            [
              ["soon", `By ${date(featured.arrives)}`],
              ["norush", "No rush"],
            ] as const
          ).map(([k, l]) => (
            <button
              key={k}
              type="button"
              onClick={() => setAnswer(k)}
              className={`h-9 rounded-ui border text-[13px] ${answer === k ? "border-ink bg-ink text-white" : "border-line-strong hover:border-ink"}`}
            >
              {l}
            </button>
          ))}
        </div>
        {answer && (
          <div className="mt-3 border-t border-line pt-3">
            {answer === "soon" ? (
              <Verdict status="buy_here" text={headline.buy_here()} />
            ) : (
              <>
                <Verdict status="better_offer" text={headline.better_offer(featured.price - best.price, best.condition)} />
                <div className="mt-2.5">
                  <Btn>View offer</Btn>
                </div>
              </>
            )}
          </div>
        )}
        <label className="mt-3 flex items-center gap-2 text-[12px] text-faint">
          <input type="checkbox" className="accent-[var(--color-ink)]" /> Remember for similar purchases
        </label>
      </div>
    </div>
  );
}

function Choice({ q, options, selected }: { q: string; options: readonly string[]; selected: string }) {
  const [sel, setSel] = useState(selected);
  return (
    <div className="border-t border-line py-4">
      <p className="text-[14px] font-medium">{q}</p>
      <div className="mt-2.5 flex flex-col gap-1.5">
        {options.map((o) => (
          <button
            key={o}
            type="button"
            onClick={() => setSel(o)}
            className={`flex items-center gap-2.5 rounded-ui border px-3 py-2 text-left text-[13.5px] ${sel === o ? "border-ink" : "border-line hover:border-line-strong"}`}
          >
            <span className={`grid size-3.5 place-items-center rounded-full border ${sel === o ? "border-ink" : "border-line-strong"}`}>
              {sel === o && <span className="size-1.5 rounded-full bg-ink" />}
            </span>
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}

function FirstRun() {
  return (
    <div className="w-full max-w-[380px] rounded-ui border border-line-strong bg-card p-5">
      <div className="flex items-center gap-2">
        <Image src="/mark.png" alt="" width={20} height={22} />
        <span className="font-semibold">Welcome to BuyWise</span>
      </div>
      <p className="mt-2 text-[13.5px] text-muted">Three questions so BuyWise only speaks up when it matters to you. You can change them anytime.</p>
      <div className="mt-4">
        <Choice q="What condition would you buy?" options={PROFILE_OPTIONS.condition} selected={profile.condition} />
        <Choice q="How patient are you?" options={PROFILE_OPTIONS.patience} selected={profile.patience} />
        <Choice q="Price alerts" options={PROFILE_OPTIONS.alerts} selected={profile.alerts} />
      </div>
      <div className="border-t border-line pt-4 text-[12.5px] text-muted">
        <p>
          {PROFILE_LABEL.prime}: {profile.prime ? "Yes" : "No"} (detected)
        </p>
        <p className="mt-1">Stored on this device only.</p>
      </div>
      <div className="mt-4 flex items-center gap-3">
        <Btn>Done</Btn>
        <Btn kind="link">Skip for now</Btn>
      </div>
    </div>
  );
}

function Dashboard() {
  const [tab, setTab] = useState<"watching" | "record" | "profile">("watching");
  const ready = watching.filter((w) => w.status === "buy_now").length;
  return (
    <div className="w-full max-w-[400px] overflow-hidden rounded-ui border border-line-strong bg-card">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <Image src="/mark.png" alt="" width={18} height={20} />
        <span className="font-semibold">BuyWise</span>
        <span className="ml-auto text-[12px] text-muted">
          {ready} of {watching.length} ready to buy
        </span>
      </div>
      <div className="flex border-b border-line text-[13px]">
        {(
          [
            ["watching", "Watching"],
            ["record", "Track record"],
            ["profile", "Profile"],
          ] as const
        ).map(([k, l]) => (
          <button
            key={k}
            type="button"
            onClick={() => setTab(k)}
            className={`flex-1 border-b-2 py-2.5 ${tab === k ? "border-ink font-medium text-ink" : "border-transparent text-muted hover:text-ink"}`}
          >
            {l}
          </button>
        ))}
      </div>

      {tab === "watching" && (
        <ul>
          {watching.map((w) => (
            <li key={w.title} className="border-b border-line px-4 py-3 last:border-b-0">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-[13.5px] font-medium">{w.title}</p>
                  <p className="mt-0.5 font-mono text-[13px]">
                    {money(w.price)}
                    {w.startPrice > w.price && <span className="ml-2 text-faint line-through">{money(w.startPrice)}</span>}
                  </p>
                </div>
                <StatusTag status={w.status} />
              </div>
              <div className="mt-2 flex items-end justify-between gap-3">
                <p className="text-[12px] text-muted">{watchFacts(w)}</p>
                <Sparkline data={w.history} w={72} h={22} stroke={w.status === "buy_now" ? "var(--color-brand)" : "var(--color-ink-2)"} />
              </div>
            </li>
          ))}
        </ul>
      )}

      {tab === "record" && (
        <div className="px-4 py-4">
          <p className="text-[12.5px] text-muted">Since {date(trackRecord.since)}, scored against what happened next.</p>
          <div className="mt-3 grid grid-cols-3 border-y border-line">
            {[
              [`${trackRecord.correct}/${trackRecord.shown}`, "Recommendations right"],
              [money(trackRecord.saved), "Saved by following"],
              [pct(trackRecord.quietShare), "Pages marked Buy here"],
            ].map(([v, l], i) => (
              <div key={l} className={`py-3 ${i ? "border-l border-line pl-3" : ""}`}>
                <p className="font-mono text-[15px]">{v}</p>
                <p className="text-[11.5px] text-muted">{l}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[13px] font-medium">Misses</p>
          <ul className="mt-1">
            {trackRecord.misses.map((m) => (
              <li key={m.title} className="border-b border-line py-2.5 text-[12.5px] last:border-b-0">
                <p className="font-medium text-ink">{m.title}</p>
                <p className="text-muted">{missLine(m.said, m.outcome)}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {tab === "profile" && (
        <div className="px-4 py-2">
          {(
            [
              [PROFILE_LABEL.condition, profile.condition],
              [PROFILE_LABEL.patience, profile.patience],
              [PROFILE_LABEL.alerts, profile.alerts],
              [PROFILE_LABEL.prime, profile.prime ? "Yes (detected)" : "No (detected)"],
              [PROFILE_LABEL.minSaving, `${money(profile.minSaving.dollars)} or ${pct(profile.minSaving.share)}`],
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-line py-2.5 text-[13px] last:border-b-0">
              <span className="text-muted">{k}</span>
              <span>{v}</span>
            </div>
          ))}
          <p className="py-3 text-[12px] text-faint">Stored on this device only.</p>
        </div>
      )}
    </div>
  );
}

function AlertScreen() {
  const w = watching[0];
  const p = bose.pattern;
  return (
    <div className="flex w-full flex-col gap-6">
      <div className="w-full max-w-[360px] rounded-[10px] border border-line bg-white/95 p-3 shadow-[0_8px_24px_rgb(17_19_17/0.12)]">
        <div className="flex items-center gap-2 text-[11.5px] text-faint">
          <Image src="/mark.png" alt="" width={14} height={16} />
          BuyWise · now
        </div>
        <p className="mt-1 text-[13.5px] font-medium">
          {STATUS.buy_now}: {w.title}
        </p>
        <p className="text-[13px] text-ink-2">{headline.buy_now_target(w.price, w.target!)}</p>
      </div>
      <div className="w-full max-w-[340px]">
        <p className="mb-2 text-[12px] text-faint">Opening the product page from the alert:</p>
        <PanelShell meta={checked(bose.offersChecked)}>
          <div className="px-3.5 pb-3.5 pt-3">
            <Verdict status="buy_now" text={headline.buy_now_target(w.price, w.target!)} />
            <div className="mt-3">
              <Row label={LABEL.sinceWatching} value={savingsVs(w.startPrice, w.price)} tone="good" />
              <Row label={LABEL.pattern} value={patternLine(p.drops, p.months, p.minDays, p.maxDays)} />
              <Row label={LABEL.bestOther} value={bestOtherLine(null)} />
            </div>
            <div className="mt-3 flex items-center gap-3">
              <Btn>Add to Cart</Btn>
              <Btn kind="link">Keep watching</Btn>
            </div>
          </div>
        </PanelShell>
      </div>
    </div>
  );
}

/* ───────────────────────── screens + notes ───────────────────────── */

type Screen = {
  id: string;
  name: string;
  summary: string;
  decided: string[];
  open: string[];
  render: (go: (id: string) => void) => React.ReactNode;
};

const centered = (bg: string, node: React.ReactNode) => <div className={`grid w-full place-items-center rounded-ui border border-line p-8 ${bg}`}>{node}</div>;

const screens: Screen[] = [
  {
    id: "better",
    name: "Better offer",
    summary: "A cheaper offer fits this shopper's profile. The panel sits under the buy box.",
    decided: [
      "Every panel starts with a status and one sentence built from numbers.",
      "Trade-offs use the same row labels on every screen.",
      "When the best price carries a warranty risk, the cheapest full-warranty option is shown too.",
    ],
    open: ["How to show an offer whose price is hidden until checkout.", "Whether to lead with the seller's name or a trust grade."],
    render: (go) => (
      <ProductPage title={garmin.title} price={featured.price} arrives={featured.arrives} rating={garmin.rating} imageLabel="Product photo">
        <BetterOfferPanel onCompare={() => go("offers")} />
      </ProductPage>
    ),
  },
  {
    id: "wait",
    name: "Wait and watch",
    summary: "No better offer today, and a drop is likely. The shopper can hand off the waiting.",
    decided: [
      "Chance of a drop, never 'confidence'.",
      "The pattern row says whether drops on this product are regular or rare.",
      "Watch at a target price, or when it's a good time.",
    ],
    open: ["Where fresh prices come from after the Keepa month.", "Whether the expected low should be a range."],
    render: () => (
      <ProductPage title={bose.title} price={bose.price} arrives={bose.arrives} rating={bose.rating} imageLabel="Product photo">
        <WaitPanel />
      </ProductPage>
    ),
  },
  {
    id: "quiet",
    name: "Buy here",
    summary: "The most common result. Amazon's pick is already the best offer for this shopper.",
    decided: ["One line, details on tap.", "Details list what was checked, so a quiet panel still reads as a result."],
    open: ["Whether the panel should shrink to just the logo after a few seconds."],
    render: () => (
      <ProductPage title={sony.title} price={sony.price} arrives={sony.arrives} rating={sony.rating} imageLabel="Product photo">
        <QuietPanel />
      </ProductPage>
    ),
  },
  {
    id: "offers",
    name: "All offers",
    summary: "Every offer, ranked for this shopper. Each note is the same three facts: price difference, delivery difference, warranty.",
    decided: ["Skipped offers stay visible with a reason from a fixed list.", "Versions show price difference and listed differences."],
    open: ["Which review points count as a real difference between versions.", "Whether shoppers want to re-sort this list."],
    render: () => <AllOffers />,
  },
  {
    id: "check",
    name: "Before you buy",
    summary: "Asked only when the answer changes the result. Here the cheaper offer arrives later.",
    decided: ["At most one question per purchase.", "The answer leads straight to a status."],
    open: ["Which other questions ever change a result, such as gifts or returns."],
    render: () => centered("bg-white", <BeforeYouBuy />),
  },
  {
    id: "first",
    name: "First run",
    summary: "Three questions at install. The answers are the exact values the panel and dashboard show.",
    decided: ["Skippable, editable, stored on the device.", "Prime is detected, not asked."],
    open: ["Whether a fourth question earns its place. Analysis tells us which answers change results."],
    render: () => centered("bg-white", <FirstRun />),
  },
  {
    id: "dashboard",
    name: "Dashboard",
    summary: "The extension popup. Every watched product shows its status and the same three facts.",
    decided: ["Statuses match the panel exactly.", "The track record lists misses."],
    open: ["A full-page view for people watching many products.", "Sharing a watched product."],
    render: () => centered("bg-white", <Dashboard />),
  },
  {
    id: "alert",
    name: "Alert",
    summary: "A watched product hits its target. The notification and the page say the same thing.",
    decided: ["The notification uses the same status and headline as the panel."],
    open: ["Email or phone alerts.", "How many alerts per week is too many."],
    render: () => centered("bg-[#e9ebe7]", <AlertScreen />),
  },
];

export default function Preview() {
  const [active, setActiveState] = useState(screens[0].id);
  const screen = screens.find((s) => s.id === active) ?? screens[0];

  // ?s=<id> links straight to a screen, so people can share one in a PR or chat.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("s");
    if (id && screens.some((s) => s.id === id)) setActiveState(id);
  }, []);
  const setActive = (id: string) => {
    setActiveState(id);
    window.history.replaceState(null, "", `?s=${id}`);
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-paper">
        <div className="mx-auto flex max-w-[1240px] flex-wrap items-center gap-x-4 gap-y-1 px-4 py-3 sm:px-6">
          <a href="/" className="flex items-center gap-2">
            <Image src="/mark.png" alt="" width={20} height={22} />
            <span className="font-semibold">BuyWise</span>
          </a>
          <span className="text-[13px] text-muted">Product preview: where we&apos;re headed by the end of fall 2026</span>
          <span className="ml-auto text-[12px] text-faint">Synthetic data. A direction, not a spec.</span>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1240px] gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[200px_1fr]">
        <nav className="flex gap-1 overflow-x-auto lg:flex-col lg:overflow-visible" aria-label="Screens">
          {screens.map((s, i) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setActive(s.id)}
              className={`flex flex-none items-center gap-2.5 rounded-ui px-3 py-2 text-left text-[13.5px] ${
                active === s.id ? "bg-card font-medium text-ink shadow-[0_0_0_1px_var(--color-line)]" : "text-muted hover:text-ink"
              }`}
            >
              <span className="font-mono text-[11px] text-faint">{String(i + 1).padStart(2, "0")}</span>
              {s.name}
            </button>
          ))}
        </nav>

        <main className="min-w-0">
          <div className="mb-4">
            <h1 className="text-[22px] font-semibold tracking-tight">{screen.name}</h1>
            <p className="mt-1 max-w-[720px] text-[14.5px] text-muted">{screen.summary}</p>
          </div>
          <div className="grid gap-6 xl:grid-cols-[1fr_260px]">
            <div className="min-w-0">{screen.render(setActive)}</div>
            <aside className="grid content-start gap-5 text-[13px]">
              <div>
                <p className="border-b border-ink pb-2 font-medium">Decided</p>
                <ul className="mt-1">
                  {screen.decided.map((d) => (
                    <li key={d} className="border-b border-line py-2 text-ink-2">
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="border-b border-ink pb-2 font-medium">Open for ideas</p>
                <ul className="mt-1">
                  {screen.open.map((d) => (
                    <li key={d} className="border-b border-line py-2 text-muted">
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}
