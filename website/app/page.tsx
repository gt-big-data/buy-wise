import Image from "next/image";
import { GITHUB_URL, PLACEHOLDER, building, built, done, stats, steps, team, type Member } from "@/content/site";

const wrap = "mx-auto w-full max-w-[1080px] px-4 sm:px-6";
const label = "text-[13px] font-medium text-muted";
const h2 = "text-[clamp(28px,4vw,38px)] font-semibold leading-[1.12] tracking-tight";
const button =
  "inline-flex h-10 items-center rounded-ui px-4 text-[15px] font-medium transition-colors";

function Nav() {
  return (
    <nav className="sticky top-0 z-20 border-b border-line bg-paper">
      <div className={`${wrap} flex h-14 items-center justify-between`}>
        <a href="#top" className="flex items-center gap-2">
          <Image src="/mark.png" alt="" width={24} height={27} priority />
          <span className="text-[17px] font-semibold tracking-tight">BuyWise</span>
        </a>
        <div className="flex items-center gap-6 text-sm text-muted">
          {[["How it works", "#how"], ["Prototype", "#prototype"], ["Progress", "#progress"], ["Team", "#team"]].map(
            ([text, href]) => (
              <a key={href} href={href} className="hidden hover:text-ink md:inline">
                {text}
              </a>
            ),
          )}
          <a href={GITHUB_URL} target="_blank" rel="noopener" className="text-ink underline-offset-4 hover:underline">
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}

function ConceptPanel() {
  return (
    <figure className="m-0">
      <div className="flex items-center gap-4 rounded-ui border border-line bg-card px-4 py-3.5">
        <div className="grid size-14 flex-none place-items-center rounded-ui bg-[#e8eae6]">
          <div className="size-6 rounded-full border-4 border-ink-2 bg-ink" />
        </div>
        <div className="min-w-0">
          <p className="text-[15px] font-medium">Running smartwatch, 46mm</p>
          <p className="text-[13px] text-muted">Sold by Amazon.com</p>
        </div>
        <div className="ml-auto hidden text-right sm:block">
          <p className="font-mono text-base font-medium">$449.99</p>
          <span className="mt-1 inline-block rounded-ui bg-[#f7ca00] px-2.5 py-0.5 text-xs font-medium text-[#1a1400]">
            Add to Cart
          </span>
        </div>
      </div>

      <div className="mt-2 rounded-ui border border-line-strong bg-card sm:ml-10">
        <div className="flex items-center gap-2 border-b border-line px-4 py-2.5 text-[13px] text-muted">
          <Image src="/mark.png" alt="" width={16} height={18} />
          <span className="font-medium text-ink">BuyWise</span>
          <span>checked 28 offers</span>
        </div>
        <div className="p-4">
          <p className="text-lg font-semibold leading-snug tracking-tight">
            Same watch, new, <span className="text-brand">$80.99 less</span> from another seller.
          </p>
          <dl className="mt-3 border-y border-line text-sm">
            <div className="flex justify-between py-2">
              <dt className="text-muted">Amazon.com, new</dt>
              <dd className="font-mono">$449.99</dd>
            </div>
            <div className="flex justify-between border-t border-line py-2">
              <dt className="font-medium">Third-party seller, new</dt>
              <dd className="font-mono font-medium text-brand-ink">$369.00</dd>
            </div>
          </dl>
          <p className="mt-3 text-[13px] leading-relaxed text-muted">
            Seller has years of history and free returns.{" "}
            <span className="text-caution">May not include the manufacturer warranty.</span>
          </p>
        </div>
      </div>
      <figcaption className="mt-3 text-[12.5px] text-faint sm:ml-10">
        Concept for the next version of the panel. Prices are from our April 2026 data on a real product.
      </figcaption>
    </figure>
  );
}

function Hero() {
  return (
    <header id="top" className="border-b border-line py-14 md:py-20">
      <div className={`${wrap} grid items-center gap-12 md:grid-cols-[1.05fr_.95fr]`}>
        <div className="min-w-0">
          <p className={label}>A GT Big Data project at Georgia Tech</p>
          <h1 className="mb-5 mt-4 text-[clamp(40px,6vw,60px)] font-bold leading-[1.03] tracking-tight">
            Same product.
            <br />
            <span className="text-brand">Better price.</span>
            <br />
            Same page.
          </h1>
          <p className="max-w-[560px] text-lg leading-relaxed text-muted">
            BuyWise is a Chrome extension for Amazon. It reads every offer on a product page and tells you in one line
            when there&apos;s a better way to buy, and why.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#how" className={`${button} bg-ink text-white hover:bg-ink-2`}>
              How it works
            </a>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener"
              className={`${button} border border-line-strong bg-card hover:border-ink`}
            >
              View the code
            </a>
          </div>
          <p className="mt-5 text-[13.5px] text-faint">In development. Demo day is December 2026.</p>
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
    <div className="border-b border-line">
      <div className={`${wrap} grid grid-cols-2 md:grid-cols-4`}>
        {stats.map((s, i) => (
          <div
            key={s.label}
            className={`py-6 pr-4 ${i % 2 ? "border-l border-line pl-4 md:pl-6" : ""} ${
              i > 1 ? "border-t border-line md:border-t-0" : ""
            } ${i === 2 ? "md:border-l md:pl-6" : ""}`}
          >
            <p className="font-mono text-2xl font-medium tracking-tight sm:text-[28px]">{s.value}</p>
            <p className="mt-1 text-sm text-muted">{s.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function HowItWorks() {
  return (
    <section id="how" className="border-b border-line py-16 md:py-24">
      <div className={`${wrap} grid gap-12 md:grid-cols-[.8fr_1.2fr]`}>
        <div>
          <p className={label}>How it works</p>
          <h2 className={`${h2} mt-3`}>One product page, dozens of sellers, one button.</h2>
          <p className="mt-4 max-w-[440px] text-[17px] leading-relaxed text-muted">
            The Add to Cart button belongs to one seller. Everyone else selling the same item is a click away, and almost
            nobody looks.
          </p>
        </div>
        <ol className="border-t border-ink">
          {steps.map((s, i) => (
            <li key={s.title} className="grid grid-cols-[40px_1fr] gap-x-4 border-b border-line py-6">
              <span className="font-mono text-sm text-faint">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <h3 className="text-lg font-semibold tracking-tight">{s.title}</h3>
                <p className="mt-1.5 text-[15px] leading-relaxed text-muted">{s.body}</p>
                {s.factors && <p className="mt-2 text-[13.5px] text-ink-2">{s.factors}</p>}
              </div>
            </li>
          ))}
          <li className="grid grid-cols-[40px_1fr] gap-x-4 py-6">
            <span className="font-mono text-sm text-faint">04</span>
            <div>
              <h3 className="text-lg font-semibold tracking-tight">Most of the time, it stays quiet</h3>
              <p className="mt-1.5 text-[15px] leading-relaxed text-muted">
                When Amazon&apos;s pick is already the best sensible deal, BuyWise says so and gets out of the way.
              </p>
            </div>
          </li>
        </ol>
      </div>
    </section>
  );
}

function Prototype() {
  return (
    <section id="prototype" className="border-b border-line bg-card py-16 md:py-24">
      <div className={`${wrap} grid items-center gap-14 md:grid-cols-[.85fr_1.15fr]`}>
        <figure className="m-0 min-w-0">
          <div className="flex justify-center rounded-ui border border-line bg-paper p-6">
            <Image
              src="/panel.jpg"
              alt="The BuyWise extension panel showing a 14-day WAIT outlook with a 64% chance of a drop, projected savings and a price chart"
              width={600}
              height={1377}
              className="w-full max-w-[360px] rounded-ui"
            />
          </div>
          <figcaption className="mt-3 text-[13px] text-faint">The current extension panel, shown with demo data.</figcaption>
        </figure>
        <div className="min-w-0">
          <p className={label}>The prototype</p>
          <h2 className={`${h2} mt-3`}>We shipped a working prototype first.</h2>
          <p className="mt-4 max-w-[560px] text-[17px] leading-relaxed text-muted">
            Last spring the team built BuyWise from scratch, from raw price history to a recommendation on the product
            page.
          </p>
          <dl className="mt-8 grid border-t border-line sm:grid-cols-2">
            {built.map((b, i) => (
              <div key={b.title} className={`border-b border-line py-4 ${i % 2 ? "sm:border-l sm:pl-5" : "sm:pr-5"}`}>
                <dt className="font-medium">{b.title}</dt>
                <dd className="mt-0.5 text-[14px] text-muted">{b.detail}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}

function Track({ title, status, items }: { title: string; status: string; items: { title: string; detail: string }[] }) {
  return (
    <div>
      <div className="flex items-baseline justify-between border-b border-ink pb-3">
        <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
        <span className="text-[13px] text-muted">{status}</span>
      </div>
      <ul>
        {items.map((it) => (
          <li key={it.title} className="border-b border-line py-4">
            <p className="font-medium">{it.title}</p>
            <p className="mt-0.5 text-[14px] leading-relaxed text-muted">{it.detail}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Progress() {
  return (
    <section id="progress" className="border-b border-line py-16 md:py-24">
      <div className={wrap}>
        <p className={label}>Progress</p>
        <h2 className={`${h2} mt-3`}>Research first, then the product.</h2>
        <p className="mt-4 max-w-[560px] text-[17px] leading-relaxed text-muted">
          This fall we tested our own premise, measured where shoppers actually lose money, and started building toward it.
        </p>
        <div className="mt-12 grid gap-12 md:grid-cols-2">
          <Track title="Done" status="Spring and early fall 2026" items={done} />
          <Track title="In progress" status="Fall 2026" items={building} />
        </div>
      </div>
    </section>
  );
}

function initials(name: string) {
  return name === PLACEHOLDER ? "" : name.split(" ").map((p) => p[0]).slice(0, 2).join("");
}

function ProfileLink({ href, children }: { href: string; children: React.ReactNode }) {
  return href ? (
    <a href={href} target="_blank" rel="noopener" className="text-ink underline-offset-2 hover:underline">
      {children}
    </a>
  ) : (
    <span className="text-faint">{children}</span>
  );
}

function MemberCard({ m }: { m: Member }) {
  const isPlaceholder = m.name === PLACEHOLDER;
  return (
    <li className="flex flex-col rounded-ui border border-line bg-card p-5">
      <div className="flex items-center gap-3">
        {m.photo ? (
          <Image src={m.photo} alt="" width={48} height={48} className="size-12 rounded-full object-cover" />
        ) : (
          <div className="grid size-12 flex-none place-items-center rounded-full bg-[#e8eae6] font-medium text-muted">
            {initials(m.name)}
          </div>
        )}
        <div className="min-w-0">
          <p className={`font-semibold ${isPlaceholder ? "text-faint" : ""}`}>{m.name}</p>
          <p className="text-[13px] text-muted">{m.role}</p>
        </div>
      </div>
      <p className={`mt-4 font-mono text-[13px] ${isPlaceholder ? "text-faint" : "text-ink-2"}`}>
        {m.major}, &apos;{m.gradYear}
      </p>
      <ul className="mb-4 mt-2 list-disc space-y-1 pl-4 text-[14px] marker:text-line-strong">
        {m.interests.map((it, i) => (
          <li key={i} className={isPlaceholder ? "text-faint" : "text-ink-2"}>
            {it}
          </li>
        ))}
      </ul>
      <p className="mt-auto flex gap-4 border-t border-line pt-3 text-[13px]">
        <ProfileLink href={m.linkedin}>LinkedIn</ProfileLink>
        <ProfileLink href={m.github}>GitHub</ProfileLink>
      </p>
    </li>
  );
}

function Team() {
  return (
    <section id="team" className="border-b border-line py-16 md:py-24">
      <div className={wrap}>
        <p className={label}>Team</p>
        <h2 className={`${h2} mt-3`}>The people building BuyWise.</h2>
        <div className="mt-12 flex flex-col gap-12">
          {team.map((g) => (
            <div key={g.group}>
              <h3 className="mb-5 border-b border-ink pb-3 text-[15px] font-semibold">{g.group}</h3>
              <ul className="grid grid-cols-[repeat(auto-fill,minmax(230px,1fr))] gap-4">
                {g.members.map((m, i) => (
                  <MemberCard key={`${m.name}-${i}`} m={m} />
                ))}
              </ul>
            </div>
          ))}
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
      <section className="py-16 md:py-20">
        <div className={`${wrap} flex flex-col items-start justify-between gap-6 md:flex-row md:items-end`}>
          <div>
            <h2 className={h2}>See it at demo day in December.</h2>
            <p className="mt-3 text-[17px] text-muted">Until then, the code and our findings are on GitHub.</p>
          </div>
          <a href={GITHUB_URL} target="_blank" rel="noopener" className={`${button} bg-ink text-white hover:bg-ink-2`}>
            Follow on GitHub
          </a>
        </div>
      </section>
      <footer className="border-t border-line py-7 text-[13px] text-faint">
        <div className={`${wrap} flex flex-wrap justify-between gap-4`}>
          <span>© 2026 BuyWise, a GT Big Data project at Georgia Tech</span>
          <span>Student project. Not affiliated with or endorsed by Amazon.</span>
        </div>
      </footer>
    </>
  );
}
