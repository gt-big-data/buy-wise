import Image from "next/image";
import { GITHUB_URL, building, built, done, stats, team } from "@/content/site";

const wrap = "mx-auto w-full max-w-[1120px] px-4 sm:px-6";

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-brand-line bg-brand-tint px-2.5 py-1 font-mono text-[12.5px] uppercase tracking-wider text-brand-ink">
      <i className="block size-1.5 rounded-full bg-brand" />
      {children}
    </span>
  );
}

function Chip({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "caution" | "dark" }) {
  const tones = {
    neutral: "border-line bg-surface text-ink-2",
    caution: "border-caution-line bg-caution-tint text-caution",
    dark: "border-[#2b342e] bg-[#1b221d] text-[#dfe6e0]",
  };
  return <span className={`rounded-full border px-2.5 py-1 text-xs ${tones[tone]}`}>{children}</span>;
}

function SectionHead({ eyebrow, title, lede }: { eyebrow: string; title: React.ReactNode; lede?: string }) {
  return (
    <div className="mb-12 flex flex-col items-start gap-3.5">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2 className="text-[clamp(28px,4vw,40px)] font-semibold leading-[1.1] tracking-tight">{title}</h2>
      {lede && <p className="max-w-[620px] text-lg text-muted">{lede}</p>}
    </div>
  );
}

function Nav() {
  return (
    <nav className="sticky top-0 z-20 border-b border-line bg-white/80 backdrop-blur-md">
      <div className={`${wrap} flex h-16 items-center justify-between`}>
        <a href="#top" className="flex items-center gap-2.5">
          <Image src="/mark.png" alt="" width={30} height={33} priority />
          <span className="text-[19px] font-bold tracking-tight">
            Buy<span className="text-brand">Wise</span>
          </span>
        </a>
        <div className="flex items-center gap-7 text-[14.5px] text-muted">
          {[["How it works", "#how"], ["Prototype", "#prototype"], ["Progress", "#progress"], ["Team", "#team"]].map(
            ([label, href]) => (
              <a key={href} href={href} className="hidden hover:text-ink md:inline">
                {label}
              </a>
            ),
          )}
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener"
            className="inline-flex h-9 items-center rounded-[10px] border border-line bg-white px-3.5 text-sm font-medium text-ink transition hover:-translate-y-px hover:bg-surface"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}

function ConceptPanel() {
  return (
    <div aria-label="Concept preview of the BuyWise panel">
      <div className="flex items-center gap-4 rounded-[20px] border border-line bg-white px-5 py-4 shadow-card">
        <div className="grid size-16 flex-none place-items-center rounded-xl bg-gradient-to-br from-[#e9ece8] to-[#d9ded8]">
          <div className="size-7 rounded-full border-[5px] border-ink-2 bg-ink" />
        </div>
        <div>
          <p className="text-[15px] font-semibold">Running smartwatch, 46mm</p>
          <p className="text-[13px] text-muted">Sold by Amazon.com</p>
        </div>
        <div className="ml-auto hidden text-right sm:block">
          <p className="font-mono text-[17px] font-semibold">$449.99</p>
          <span className="mt-1.5 inline-block rounded-full bg-[#ffd84d] px-3 py-1 text-xs font-semibold text-[#3a2f00]">
            Add to Cart
          </span>
        </div>
      </div>

      <div className="relative -mt-2.5 ml-3 mr-0 overflow-hidden rounded-[20px] border border-line bg-white shadow-float sm:ml-12 sm:mr-4">
        <div className="flex items-center gap-2 border-b border-line px-4 py-3 text-[13px] text-muted">
          <Image src="/mark.png" alt="" width={18} height={20} />
          <strong className="font-semibold text-ink">BuyWise</strong> checked 28 offers
          <span className="ml-auto rounded-full bg-brand-tint px-2 py-0.5 font-mono text-[11px] text-brand-ink">CONCEPT</span>
        </div>
        <div className="p-[18px]">
          <p className="text-xl font-semibold leading-tight tracking-tight">
            Same watch, new — <span className="text-brand">$80.99 less</span> from another seller.
          </p>
          <div className="my-3.5 overflow-hidden rounded-xl border border-line text-sm">
            <div className="flex justify-between px-3.5 py-2.5">
              <span className="text-muted">Amazon.com · new</span>
              <span className="font-mono font-medium">$449.99</span>
            </div>
            <div className="flex justify-between border-t border-line bg-brand-tint px-3.5 py-2.5">
              <span className="font-medium">Third-party seller · new</span>
              <span className="font-mono font-semibold text-brand-ink">$369.00</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Chip>Established seller</Chip>
            <Chip>Free returns</Chip>
            <Chip tone="caution">Check warranty coverage</Chip>
          </div>
        </div>
      </div>
      <p className="ml-3 mt-3.5 text-[12.5px] text-faint sm:ml-12">
        Concept of the next panel. Prices are from our April 2026 data on a real product.
      </p>
    </div>
  );
}

function Hero() {
  return (
    <header
      id="top"
      className="bg-[radial-gradient(900px_420px_at_85%_-10%,#e7f6ea,transparent_60%),radial-gradient(700px_360px_at_-10%_10%,#f3f8ea,transparent_60%)] py-14 md:pb-[72px] md:pt-[88px]"
    >
      <div className={`${wrap} grid items-center gap-14 md:grid-cols-[1.05fr_.95fr]`}>
        <div className="min-w-0">
          <Eyebrow>A GT Big Data project · Georgia Tech</Eyebrow>
          <h1 className="my-5 text-[clamp(40px,6vw,64px)] font-bold leading-[1.02] tracking-tight">
            Same product.
            <br />
            <span className="text-brand">Better price.</span>
            <br />
            Same page.
          </h1>
          <p className="max-w-[620px] text-lg text-muted">
            BuyWise is a Chrome extension that reads every offer on an Amazon product page, not just the one behind
            the button, and tells you in one plain line when there&apos;s a better way to buy, and why.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="#how"
              className="inline-flex h-[42px] items-center rounded-[10px] bg-ink px-[18px] font-medium text-white transition hover:-translate-y-px hover:bg-[#1f2621]"
            >
              See how it works →
            </a>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener"
              className="inline-flex h-[42px] items-center rounded-[10px] border border-line bg-white px-[18px] font-medium transition hover:-translate-y-px hover:bg-surface"
            >
              View the code
            </a>
          </div>
          <p className="mt-[18px] text-[13.5px] text-faint">In development · Demo day December 2026</p>
        </div>
        <div className="min-w-0">
          <ConceptPanel />
        </div>
      </div>
    </header>
  );
}

function Stats() {
  return (
    <div className="border-y border-line bg-surface">
      <div className={`${wrap} grid grid-cols-2 md:grid-cols-4`}>
        {stats.map((s, i) => (
          <div
            key={s.label}
            className={`px-4 py-6 sm:px-6 sm:py-[30px] ${i % 2 ? "border-l border-line" : ""} ${
              i > 1 ? "border-t border-line md:border-t-0" : ""
            } ${i === 2 ? "md:border-l" : ""}`}
          >
            <p className="font-mono text-2xl font-semibold tracking-tight sm:text-[30px]">{s.value}</p>
            <p className="mt-1 text-sm text-muted">{s.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function HowItWorks() {
  const card = "flex flex-col rounded-[20px] border p-7 shadow-card";
  return (
    <section id="how" className="py-[72px] md:py-[104px]">
      <div className={wrap}>
        <SectionHead
          eyebrow="How it works"
          title={
            <>
              One product page. Dozens of sellers.
              <br />
              One button.
            </>
          }
          lede="You know the big “Add to Cart” button? Only one seller is behind it. The rest are a click away, and almost nobody looks."
        />
        <div className="grid gap-5 md:grid-cols-3">
          <div className={`${card} border-line bg-white`}>
            <span className="font-mono text-[13px] text-brand-ink">01</span>
            <h3 className="mb-2.5 mt-3.5 text-[21px] font-semibold leading-tight tracking-tight">Amazon is a marketplace</h3>
            <p className="text-[15px] text-muted">
              Amazon sells on a product page, and so do outside businesses: authorized dealers, liquidators, resellers.
              Each sets its own price, often adjusted by bots many times a day.
            </p>
          </div>
          <div className={`${card} border-line bg-white`}>
            <span className="font-mono text-[13px] text-brand-ink">02</span>
            <h3 className="mb-2.5 mt-3.5 text-[21px] font-semibold leading-tight tracking-tight">
              Amazon picks one for the button
            </h3>
            <p className="text-[15px] text-muted">
              The “featured offer” is chosen for a typical shopper and for Amazon. It isn&apos;t always the best deal for
              you, and some of the lowest prices are hidden until checkout.
            </p>
            <div className="mt-[18px] flex flex-wrap gap-1.5">
              {["Price + shipping", "Delivery speed", "Seller track record", "In stock"].map((c) => (
                <Chip key={c}>{c}</Chip>
              ))}
            </div>
          </div>
          <div className={`${card} border-ink bg-ink`}>
            <span className="font-mono text-[13px] text-lime">03</span>
            <h3 className="mb-2.5 mt-3.5 text-[21px] font-semibold leading-tight tracking-tight text-white">
              BuyWise works for the buyer
            </h3>
            <p className="text-[15px] text-[#b9c2bb]">
              We read every offer, weigh what makes a cheaper one worth it or not, and tell you plainly, with the reason.
            </p>
            <div className="mt-[18px] flex flex-wrap gap-1.5">
              {["Seller trust", "Warranty", "Shipping", "Returns", "Condition"].map((c) => (
                <Chip key={c} tone="dark">
                  {c}
                </Chip>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-5 flex items-center gap-3.5 rounded-[14px] border border-brand-line bg-brand-tint px-5 py-4 text-[15px] text-ink-2">
          <span className="grid size-7 flex-none place-items-center rounded-full bg-brand font-bold text-white">✓</span>
          <p>
            <b>Most of the time, we stay quiet.</b> If Amazon&apos;s pick is already the best sensible deal, BuyWise says so
            and gets out of your way.
          </p>
        </div>
      </div>
    </section>
  );
}

function Prototype() {
  return (
    <section id="prototype" className="border-y border-line bg-surface py-[72px] md:py-[104px]">
      <div className={`${wrap} grid items-center gap-16 md:grid-cols-[.9fr_1.1fr]`}>
        <div className="min-w-0">
          <figure className="flex justify-center rounded-3xl border border-line bg-gradient-to-b from-white to-surface-2 p-7 shadow-float">
            <Image
              src="/panel.jpg"
              alt="The BuyWise extension panel showing a 14-day WAIT outlook with a 64% chance of a drop, projected savings and a price chart"
              width={600}
              height={1377}
              className="w-full max-w-[380px] rounded-[14px]"
            />
          </figure>
          <p className="mt-3.5 text-center text-[13px] text-faint">The real extension panel, shown here with demo data.</p>
        </div>
        <div className="min-w-0">
          <Eyebrow>Built, end to end</Eyebrow>
          <h2 className="mt-4 text-[clamp(28px,4vw,40px)] font-semibold leading-[1.1] tracking-tight">
            We shipped a working prototype first.
          </h2>
          <p className="mt-3.5 max-w-[620px] text-lg text-muted">
            Last spring the team built BuyWise from scratch: a full pipeline from raw price history to a recommendation
            on the product page.
          </p>
          <div className="mt-7 grid gap-3 sm:grid-cols-2">
            {built.map((b) => (
              <div key={b.title} className="rounded-xl border border-line bg-white px-4 py-3.5">
                <p className="font-semibold">{b.title}</p>
                <p className="text-[13.5px] text-muted">{b.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function Track({
  title,
  pill,
  items,
  state,
}: {
  title: string;
  pill: string;
  items: { title: string; detail: string }[];
  state: "done" | "now";
}) {
  return (
    <div className="rounded-[20px] border border-line bg-white p-7">
      <h3 className="flex items-center gap-2.5 text-lg font-semibold">
        {title}
        <span
          className={`rounded-full px-2.5 py-0.5 font-mono text-[11.5px] font-medium uppercase tracking-wider ${
            state === "done" ? "bg-brand-tint text-brand-ink" : "bg-[#fff3d6] text-caution"
          }`}
        >
          {pill}
        </span>
      </h3>
      <ul className="mt-5">
        {items.map((it, i) => (
          <li key={it.title} className={`grid grid-cols-[22px_1fr] gap-2.5 text-[15px] ${i ? "border-t border-line py-3.5" : "pb-3.5"}`}>
            {state === "done" ? (
              <span className="mt-[3px] grid size-[18px] place-items-center rounded-full bg-brand text-[11px] font-bold text-white">✓</span>
            ) : (
              <span className="mt-[3px] size-[18px] rounded-full border-2 border-[#e2b93b]" />
            )}
            <div>
              {it.title}
              <p className="mt-0.5 text-[13.5px] text-muted">{it.detail}</p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Progress() {
  return (
    <section id="progress" className="py-[72px] md:py-[104px]">
      <div className={wrap}>
        <SectionHead
          eyebrow="Progress"
          title="Research first, then the product."
          lede="This fall we stress-tested our own premise, found where the money really is, and are building toward it."
        />
        <div className="grid gap-5 md:grid-cols-2">
          <Track title="What we've done" pill="Done" items={done} state="done" />
          <Track title="What we're building" pill="Fall 2026" items={building} state="now" />
        </div>
      </div>
    </section>
  );
}

function initials(name: string) {
  return name === "Name Surname" ? "—" : name.split(" ").map((p) => p[0]).slice(0, 2).join("");
}

function Team() {
  return (
    <section id="team" className="pb-[72px] md:pb-[104px]">
      <div className={wrap}>
        <SectionHead eyebrow="Team" title="The people building BuyWise." lede="Built by students at GT Big Data." />
        <div className="flex flex-col gap-10">
          {team.map((g) => (
            <div key={g.group}>
              <h3 className="mb-4 font-mono text-sm font-medium uppercase tracking-widest text-muted">{g.group}</h3>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(230px,1fr))] gap-3.5">
                {g.members.map((m, i) => (
                  <div key={`${m.name}-${i}`} className="flex items-center gap-3.5 rounded-[14px] border border-line bg-white p-[18px]">
                    {m.photo ? (
                      <Image src={m.photo} alt="" width={46} height={46} className="size-[46px] rounded-full object-cover" />
                    ) : (
                      <div className="grid size-[46px] flex-none place-items-center rounded-full border border-dashed border-[#cdd3cc] bg-surface-2 font-semibold text-faint">
                        {initials(m.name)}
                      </div>
                    )}
                    <div className="min-w-0">
                      <p className="font-semibold">{m.name}</p>
                      <p className="text-[13px] text-muted">{m.role}</p>
                      {(m.linkedin || m.github) && (
                        <p className="mt-0.5 flex gap-2 font-mono text-xs text-faint">
                          {m.linkedin && <a href={m.linkedin} className="hover:text-brand-ink">LinkedIn</a>}
                          {m.github && <a href={m.github} className="hover:text-brand-ink">GitHub</a>}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Cta() {
  return (
    <section className="pb-24">
      <div className={wrap}>
        <div className="relative flex flex-col items-start gap-8 overflow-hidden rounded-[28px] bg-ink p-9 text-white md:flex-row md:items-center md:justify-between md:p-14">
          <div className="pointer-events-none absolute -right-[120px] -top-[120px] size-[360px] rounded-full bg-[radial-gradient(circle,rgb(163_214_92/.35),transparent_70%)]" />
          <div>
            <h2 className="text-[clamp(28px,4vw,40px)] font-semibold leading-[1.1] tracking-tight">
              Shop like you checked every seller.
            </h2>
            <p className="mt-3 max-w-[520px] text-[#b9c2bb]">
              BuyWise is in development. Follow along on GitHub, or come see it at GT Big Data demo day this December.
            </p>
          </div>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener"
            className="relative inline-flex h-[42px] flex-none items-center rounded-[10px] bg-white px-[18px] font-medium text-ink transition hover:-translate-y-px"
          >
            Follow on GitHub →
          </a>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <>
      <Nav />
      <Hero />
      <Stats />
      <HowItWorks />
      <Prototype />
      <Progress />
      <Team />
      <Cta />
      <footer className="border-t border-line py-8 text-[13.5px] text-faint">
        <div className={`${wrap} flex flex-wrap justify-between gap-5`}>
          <span>© 2026 BuyWise · A GT Big Data project at Georgia Tech</span>
          <span>Student project. Not affiliated with or endorsed by Amazon.</span>
        </div>
      </footer>
    </>
  );
}
