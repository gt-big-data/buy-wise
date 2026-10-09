"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import { boseHistory, garmin, trackRecord, watching, type Offer } from "./data";

const money = (n: number) => `$${n.toFixed(2)}`;

/* ───────────────────────── shared pieces ───────────────────────── */

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
      {[279, 349].map((v) => (
        <g key={v}>
          <line x1="0" x2={w} y1={y(v)} y2={y(v)} stroke="var(--color-line)" strokeDasharray="2 3" />
          <text x={w} y={y(v) - 4} textAnchor="end" className="fill-[var(--color-faint)] font-mono text-[9px]">
            ${v}
          </text>
        </g>
      ))}
      <polyline points={pts} fill="none" stroke="var(--color-ink)" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx={w} cy={y(349)} r="3" fill="var(--color-ink)" />
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

function PanelShell({ children, meta }: { children: React.ReactNode; meta: string }) {
  return (
    <div className="rounded-ui border border-line-strong bg-card text-ink">
      <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5 text-[12.5px] text-muted">
        <Image src="/mark.png" alt="" width={14} height={16} />
        <span className="font-semibold text-ink">BuyWise</span>
        <span>{meta}</span>
        <span className="ml-auto text-faint" aria-hidden="true">
          ⌄
        </span>
      </div>
      {children}
    </div>
  );
}

function Row({ label, value, tone }: { label: string; value: React.ReactNode; tone?: "caution" | "good" }) {
  const color = tone === "caution" ? "text-caution" : tone === "good" ? "text-brand-ink" : "text-ink-2";
  return (
    <div className="flex justify-between gap-4 border-t border-line py-2 text-[13px]">
      <span className="text-muted">{label}</span>
      <span className={`text-right ${color}`}>{value}</span>
    </div>
  );
}

function ProfileLine() {
  return (
    <p className="border-t border-line px-3.5 py-2.5 text-[12px] text-faint">
      For you: new only · can wait a few days · Prime.{" "}
      <span className="text-muted underline underline-offset-2">Edit</span>
    </p>
  );
}

/* ───────────────────────── the three panel states ───────────────────────── */

function BetterOfferPanel({ onCompare }: { onCompare?: () => void }) {
  const best = garmin.offers[0];
  return (
    <PanelShell meta="checked 28 offers">
      <div className="px-3.5 pb-3.5 pt-3">
        <p className="text-[17px] font-semibold leading-snug tracking-tight">
          Same watch, new, <span className="text-brand">$80.99 less</span>.
        </p>
        <p className="mt-1 text-[13px] text-muted">
          {money(best.price)} from {best.seller} · {best.rating} · {best.years} years selling
        </p>
        <div className="mt-3">
          <Row label="Arrives" value="Sun, Oct 18 (2 days later)" />
          <Row label="Returns" value="Free 30-day returns through Amazon" tone="good" />
          <Row label="Warranty" value="Seller isn't an authorized Garmin dealer" tone="caution" />
        </div>
        <p className="mt-2 rounded-ui bg-paper px-3 py-2 text-[12.5px] text-ink-2">
          Want the warranty? Runner&apos;s Depot is authorized and $20 less than Amazon.
        </p>
        <div className="mt-3 flex items-center gap-2">
          <Btn>View offer</Btn>
          <Btn kind="secondary" onClick={onCompare}>
            Compare all 28
          </Btn>
        </div>
      </div>
      <ProfileLine />
    </PanelShell>
  );
}

function WaitPanel() {
  const [mode, setMode] = useState<"target" | "signal">("target");
  const [target, setTarget] = useState("290");
  const [watched, setWatched] = useState(false);
  return (
    <PanelShell meta="14-day outlook">
      <div className="px-3.5 pb-3.5 pt-3">
        <p className="text-[17px] font-semibold leading-snug tracking-tight">Likely to drop. Worth waiting.</p>
        <p className="mt-1 text-[13px] text-muted">
          <span className="font-mono text-ink">74%</span> chance it falls 8% or more in the next 14 days. Expected low
          around <span className="font-mono text-ink">$285</span>.
        </p>
        <div className="mt-3">
          <PriceChart />
        </div>
        <Row label="Pattern" value="Steady: 4 drops in 4 months, each lasted 5–8 days" tone="good" />
        <Row label="Other sellers" value="None cheaper right now" />

        {!watched ? (
          <div className="mt-3 rounded-ui border border-line p-3">
            <p className="text-[13px] font-medium">Watch it for me</p>
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
            {mode === "target" ? (
              <label className="mt-2.5 flex items-center gap-2 text-[13px] text-muted">
                Notify me at or below
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
            ) : (
              <p className="mt-2.5 text-[13px] text-muted">We&apos;ll notify you when a drop starts, or if waiting stops being worth it.</p>
            )}
            <div className="mt-3 flex items-center gap-3">
              <Btn onClick={() => setWatched(true)}>Watch</Btn>
              <Btn kind="link">Buy now anyway</Btn>
            </div>
          </div>
        ) : (
          <div className="mt-3 rounded-ui border border-line bg-paper p-3 text-[13px]">
            <p className="font-medium">Watching.</p>
            <p className="mt-0.5 text-muted">
              {mode === "target" ? `We'll notify you at $${target || "—"} or lower.` : "We'll notify you when it's a good time to buy."} You can close
              this tab.
            </p>
            <button type="button" onClick={() => setWatched(false)} className="mt-1.5 text-[12.5px] text-muted underline underline-offset-2">
              Change
            </button>
          </div>
        )}
      </div>
      <ProfileLine />
    </PanelShell>
  );
}

function QuietPanel() {
  const [open, setOpen] = useState(false);
  return (
    <PanelShell meta="checked 14 offers">
      <div className="px-3.5 py-3">
        <p className="text-[14.5px] font-medium">This is the best deal for you.</p>
        <button type="button" onClick={() => setOpen(!open)} className="mt-0.5 text-[12.5px] text-muted underline underline-offset-2">
          {open ? "Hide details" : "What we checked"}
        </button>
        {open && (
          <div className="mt-2">
            <Row label="Cheapest new elsewhere" value="$254.00, arrives in 9 days" />
            <Row label="Used, Very Good" value="$211.00, skipped (new only)" />
            <Row label="Chance of a drop" value="4% in the next 14 days" />
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
  children,
  imageLabel,
}: {
  title: string;
  price: number;
  arrives: string;
  rating: string;
  children: React.ReactNode;
  imageLabel: string;
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
          <p className="mt-1 text-[12.5px] text-muted">{rating}</p>
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
            <p className="mt-1 text-muted">Delivery {arrives}</p>
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

const verdictLabel: Record<Offer["verdict"], { text: string; cls: string }> = {
  best: { text: "Best for you", cls: "bg-ink text-white" },
  featured: { text: "Amazon's pick", cls: "border border-line-strong text-ink-2" },
  ok: { text: "Good option", cls: "border border-line text-muted" },
  skipped: { text: "Skipped", cls: "text-faint" },
};

function AllOffers() {
  return (
    <div className="rounded-ui border border-line-strong bg-card">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3 text-[13px] text-muted">
        <Image src="/mark.png" alt="" width={14} height={16} />
        <span className="font-semibold text-ink">All offers</span>
        <span>· Garmin Forerunner 265, 46mm · 28 checked, 5 shown</span>
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
            {garmin.offers.map((o) => (
              <tr key={o.seller + o.condition} className={`border-b border-line align-top ${o.verdict === "skipped" ? "text-faint" : ""}`}>
                <td className="px-4 py-3">
                  <p className={o.verdict === "skipped" ? "" : "font-medium text-ink"}>{o.seller}</p>
                  <p className="text-[12px] text-faint">
                    {o.rating ? `${o.rating} · ${o.years} yrs` : "Sold by Amazon"}
                  </p>
                </td>
                <td className="px-4 py-3">{o.condition}</td>
                <td className="px-4 py-3 font-mono">{money(o.price)}</td>
                <td className="px-4 py-3">
                  {o.arrives}
                  <p className="text-[12px] text-faint">
                    {o.fulfilled === "Amazon" ? "Fulfilled by Amazon" : "Ships from seller"} · {o.returns}
                  </p>
                </td>
                <td className="w-[220px] px-4 py-3">
                  <span className={`inline-block whitespace-nowrap rounded-[4px] px-2 py-0.5 text-[12px] ${verdictLabel[o.verdict].cls}`}>
                    {verdictLabel[o.verdict].text}
                  </span>
                  {o.note && <p className={`mt-1.5 text-[12px] ${o.verdict === "best" ? "text-caution" : "text-muted"}`}>{o.note}</p>}
                </td>
              </tr>
            ))}
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
              <p className="mt-1 text-[12.5px] text-muted">{v.current ? "You're viewing this one." : v.reviews}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function BeforeYouBuy() {
  const [answer, setAnswer] = useState<null | "soon" | "norush">(null);
  return (
    <div className="max-w-[340px] rounded-ui border border-line-strong bg-card">
      <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5 text-[12.5px] text-muted">
        <Image src="/mark.png" alt="" width={14} height={16} />
        <span className="font-semibold text-ink">BuyWise</span>
        <span>quick check</span>
      </div>
      <div className="p-3.5">
        <p className="text-[15px] font-medium">When do you need it?</p>
        <p className="mt-0.5 text-[12.5px] text-muted">The cheaper offer arrives two days later. This is the only thing we&apos;ll ask.</p>
        <div className="mt-3 grid grid-cols-2 gap-2">
          {(
            [
              ["soon", "By Friday"],
              ["norush", "No rush"],
            ] as const
          ).map(([k, l]) => (
            <button
              key={k}
              type="button"
              onClick={() => setAnswer(k)}
              className={`h-9 rounded-ui border text-[13.5px] ${answer === k ? "border-ink bg-ink text-white" : "border-line-strong hover:border-ink"}`}
            >
              {l}
            </button>
          ))}
        </div>
        {answer === "soon" && (
          <p className="mt-3 border-t border-line pt-3 text-[13px]">
            Then Amazon&apos;s pick is right for you. <span className="text-muted">Arrives Friday, full warranty.</span>
          </p>
        )}
        {answer === "norush" && (
          <div className="mt-3 border-t border-line pt-3 text-[13px]">
            <p>
              Save <span className="font-medium text-brand">$80.99</span> with TrailTech Outfitters. <span className="text-muted">Arrives Sunday.</span>
            </p>
            <div className="mt-2.5">
              <Btn>Switch to this offer</Btn>
            </div>
          </div>
        )}
        <label className="mt-3 flex items-center gap-2 text-[12px] text-faint">
          <input type="checkbox" className="accent-[var(--color-ink)]" /> Remember for this kind of purchase
        </label>
      </div>
    </div>
  );
}

function Choice({ q, options, hint, initial = 0 }: { q: string; options: string[]; hint?: string; initial?: number }) {
  const [sel, setSel] = useState(initial);
  return (
    <div className="border-t border-line py-4">
      <p className="text-[14px] font-medium">{q}</p>
      {hint && <p className="mt-0.5 text-[12.5px] text-muted">{hint}</p>}
      <div className="mt-2.5 flex flex-col gap-1.5">
        {options.map((o, i) => (
          <button
            key={o}
            type="button"
            onClick={() => setSel(i)}
            className={`flex items-center gap-2.5 rounded-ui border px-3 py-2 text-left text-[13.5px] ${sel === i ? "border-ink" : "border-line hover:border-line-strong"}`}
          >
            <span className={`grid size-3.5 place-items-center rounded-full border ${sel === i ? "border-ink" : "border-line-strong"}`}>
              {sel === i && <span className="size-1.5 rounded-full bg-ink" />}
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
      <p className="mt-2 text-[13.5px] text-muted">Three questions so we only speak up when it matters to you. Takes 20 seconds, and you can change them anytime.</p>
      <div className="mt-4">
        <Choice q="Would you buy used or open-box?" options={["Yes, if it's like new", "Yes, any good condition", "New only"]} initial={2} />
        <Choice
          q="How patient are you with purchases?"
          options={["I usually need things soon", "I can wait a week or two", "I'll wait for a real deal"]}
          initial={1}
        />
        <Choice q="Should we watch prices for you?" options={["Yes, notify me", "Only when I ask", "No notifications"]} />
      </div>
      <div className="mt-1 border-t border-line pt-4 text-[12.5px] text-muted">
        <p>
          <span className="text-ink">Detected:</span> Prime member. We use it for delivery estimates.
        </p>
        <p className="mt-1">Your answers stay on this device.</p>
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
  const statusCls = {
    target: "bg-ink text-white",
    buy: "border border-ink text-ink",
    waiting: "border border-line text-muted",
  };
  const statusText = { target: "Target hit", buy: "Good time to buy", waiting: "Waiting" };
  return (
    <div className="w-full max-w-[400px] overflow-hidden rounded-ui border border-line-strong bg-card">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <Image src="/mark.png" alt="" width={18} height={20} />
        <span className="font-semibold">BuyWise</span>
        <span className="ml-auto text-[12px] text-faint">1 alert</span>
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
                    {w.was > w.price && <span className="ml-2 text-faint line-through">{money(w.was)}</span>}
                  </p>
                </div>
                <span className={`whitespace-nowrap rounded-[4px] px-2 py-0.5 text-[11.5px] ${statusCls[w.status]}`}>{statusText[w.status]}</span>
              </div>
              <div className="mt-2 flex items-end justify-between gap-3">
                <p className="text-[12px] text-muted">
                  {w.status === "target" ? `Below your $${w.target} target. Lowest in 90 days.` : w.note}
                </p>
                <Sparkline data={w.history} w={88} h={24} stroke={w.status === "target" ? "var(--color-brand)" : "var(--color-ink-2)"} />
              </div>
            </li>
          ))}
        </ul>
      )}

      {tab === "record" && (
        <div className="px-4 py-4">
          <p className="text-[12.5px] text-muted">Since {trackRecord.since}, scored against what actually happened.</p>
          <div className="mt-3 grid grid-cols-3 border-y border-line">
            {[
              [`${trackRecord.played}/${trackRecord.shown}`, "played out"],
              [`$${trackRecord.saved}`, "saved"],
              [`${trackRecord.quietPct}%`, "pages we stayed quiet"],
            ].map(([v, l], i) => (
              <div key={l} className={`py-3 ${i ? "border-l border-line pl-3" : ""}`}>
                <p className="font-mono text-lg">{v}</p>
                <p className="text-[11.5px] text-muted">{l}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[13px] font-medium">Where we were wrong</p>
          <ul className="mt-1">
            {trackRecord.misses.map((m) => (
              <li key={m.title} className="border-b border-line py-2.5 text-[12.5px] last:border-b-0">
                <p className="font-medium text-ink">{m.title}</p>
                <p className="text-muted">{m.what}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {tab === "profile" && (
        <div className="px-4 py-2">
          {[
            ["Used or open-box", "New only"],
            ["Patience", "Can wait a week or two"],
            ["Notifications", "On"],
            ["Prime", "Yes (detected)"],
            ["Minimum saving to show", "$10 or 5%"],
          ].map(([k, v]) => (
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
  return (
    <div className="flex w-full flex-col gap-6">
      <div className="w-full max-w-[360px] rounded-[10px] border border-line bg-white/95 p-3 shadow-[0_8px_24px_rgb(17_19_17/0.12)]">
        <div className="flex items-center gap-2 text-[11.5px] text-faint">
          <Image src="/mark.png" alt="" width={14} height={16} />
          BuyWise · now
        </div>
        <p className="mt-1 text-[13.5px] font-medium">Bose QuietComfort Headphones: $279.00</p>
        <p className="text-[13px] text-ink-2">Below your $290 target and the lowest in 90 days. Drops like this usually last 5–8 days.</p>
      </div>
      <div className="w-full max-w-[340px]">
        <p className="mb-2 text-[12px] text-faint">When they open the page from the alert:</p>
        <PanelShell meta="you're watching this">
          <div className="px-3.5 pb-3.5 pt-3">
            <p className="text-[17px] font-semibold leading-snug tracking-tight">
              Good time to buy. <span className="text-brand">$70 less</span> than when you started watching.
            </p>
            <div className="mt-3">
              <Row label="Price now" value="$279.00" tone="good" />
              <Row label="Your target" value="$290.00" />
              <Row label="Other sellers" value="None cheaper" />
            </div>
            <div className="mt-3 flex items-center gap-3">
              <Btn>Add to Cart at $279</Btn>
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

const screens: Screen[] = [
  {
    id: "better",
    name: "Better offer",
    summary: "A cheaper sensible offer exists on the page. The panel sits right under the buy box.",
    decided: [
      "One headline with the dollar difference, then the trade-offs in plain rows.",
      "Warranty and seller risks are shown, never hidden.",
      "A safer alternative is offered when the best price carries a risk.",
    ],
    open: ["How to show an offer whose price is hidden until checkout.", "Whether to show the seller's name or a trust grade first."],
    render: (go) => (
      <ProductPage title={garmin.title} price={449.99} arrives="Friday, Oct 16" rating={garmin.rating} imageLabel="Watch photo">
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
      "A stability line tells you if this is a steady pattern or a volatile dip.",
      "Watch at a target price, or when it's a good time. One tap either way.",
    ],
    open: ["Where fresh prices come from after the Keepa month.", "Whether to show the expected low as a single number or a range."],
    render: () => (
      <ProductPage title="Bose QuietComfort Wireless Noise Cancelling Headphones" price={349} arrives="Thursday, Oct 15" rating="4.5 out of 5 · 14,388 ratings" imageLabel="Headphones photo">
        <WaitPanel />
      </ProductPage>
    ),
  },
  {
    id: "quiet",
    name: "You're good",
    summary: "The most common state. Amazon's pick is already the best sensible deal for this shopper.",
    decided: ["One line. Details only on tap.", "It still says what was checked, so silence reads as a result."],
    open: ["Whether the quiet state should collapse to just the logo after a few seconds."],
    render: () => (
      <ProductPage title="Sony WH-1000XM5 Wireless Noise Canceling Headphones" price={248} arrives="Wednesday, Oct 14" rating="4.4 out of 5 · 22,105 ratings" imageLabel="Headphones photo">
        <QuietPanel />
      </ProductPage>
    ),
  },
  {
    id: "offers",
    name: "All offers",
    summary: "Every offer on the page, ranked for this shopper, with the reason each one was picked or skipped. Other versions underneath.",
    decided: ["Skipped offers stay visible with a reason.", "Versions show price and what reviews say about the difference."],
    open: ["How to summarize reviews across versions in one line.", "Whether shoppers want to sort this themselves."],
    render: () => <AllOffers />,
  },
  {
    id: "check",
    name: "Before you buy",
    summary: "Shown only when the answer changes the recommendation, here because the cheaper offer is slower.",
    decided: ["At most one question per purchase.", "Answers can be remembered for similar purchases."],
    open: ["Which other questions ever flip an answer (gift, returns)."],
    render: () => (
      <div className="grid w-full place-items-center rounded-ui border border-line bg-white p-8">
        <BeforeYouBuy />
      </div>
    ),
  },
  {
    id: "first",
    name: "First run",
    summary: "Three questions at install. Everything else is inferred or asked in the moment.",
    decided: ["Skippable, editable later, stored on the device.", "Prime is detected, not asked."],
    open: ["Whether a fourth question earns its place. Analysis tells us which answers change outcomes."],
    render: () => (
      <div className="grid w-full place-items-center rounded-ui border border-line bg-white p-8">
        <FirstRun />
      </div>
    ),
  },
  {
    id: "dashboard",
    name: "Dashboard",
    summary: "The extension popup: what you're watching, how BuyWise has done, and your profile.",
    decided: ["Track record includes the misses.", "Watched items lead with status, not just price."],
    open: ["A full-page version for people watching many items.", "Sharing a watched item with a friend."],
    render: () => (
      <div className="grid w-full place-items-center rounded-ui border border-line bg-white p-8">
        <Dashboard />
      </div>
    ),
  },
  {
    id: "alert",
    name: "Alert",
    summary: "A watched product hits its target. The notification says why now; the page confirms it.",
    decided: ["Price, target and how long drops usually last, in one notification.", "Opening the page shows the change since you started watching."],
    open: ["Email or phone alerts in addition to the browser.", "How often is too often."],
    render: () => (
      <div className="grid w-full place-items-center rounded-ui border border-line bg-[#e9ebe7] p-8">
        <AlertScreen />
      </div>
    ),
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
