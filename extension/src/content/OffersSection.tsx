import React, { useCallback, useEffect, useRef, useState } from "react";
import { PageOffers } from "./offers";
import { Offer, OfferListResult } from "./offerList";
import { openOffersPanel, watchOfferList } from "./offerPanel";
import type { RankedOffer } from "./decision";

// Other sellers, in the panel's style. Shows Amazon's summary box right away and
// the full list once the shopper asks for it (Amazon only loads it on click).

const money = (n: number) => `$${n.toFixed(2)}`;
const total = (o: Offer) => o.price + (o.shippingCost ?? 0);
const sellerRating = (o: Offer) =>
  /^amazon/i.test(o.seller)
    ? "Run by Amazon, Amazon returns"
    : o.stars !== null && o.ratingCount !== null && o.positivePercent !== null
      ? `${o.stars.toFixed(1)}★ · ${o.ratingCount.toLocaleString("en-US")} ratings · ${Math.round(o.positivePercent)}% positive`
      : "No seller ratings shown";

type Props = {
  offers: PageOffers;
  // Amazon's offer list slides in where the panel floats, so step aside while it's open
  onCompareOpen?: () => void;
  // Each time the full list is read (first load, more rows on scroll)
  onList?: (list: OfferListResult) => void;
  // The panel's recommended offer, if it named one. A cheaper offer here is one we passed on.
  recommendedPrice?: number | null;
  // BuyWise's order and verdict for each offer, from the decision endpoint
  ranking?: RankedOffer[];
};

const VERDICT: Record<string, { label: string; tone: "good" | "plain" | "skip" }> = {
  best: { label: "Best for you", tone: "good" },
  add_to_cart: { label: "Add to Cart", tone: "plain" },
  costs_more: { label: "Good seller, costs more", tone: "plain" },
  small_saving: { label: "Saves too little", tone: "plain" },
  condition_profile: { label: "Not your condition", tone: "skip" },
  slow_shipping: { label: "Arrives too late", tone: "skip" },
  low_ratings: { label: "Low seller ratings", tone: "skip" },
};

function verdictFor(o: Offer, ranking: RankedOffer[]): { rank: number; verdict: string | null } {
  const i = ranking.findIndex((r) => r.seller === o.seller && Math.abs(r.price - total(o)) < 0.01);
  return i >= 0 ? { rank: i, verdict: ranking[i].verdict } : { rank: Number.MAX_SAFE_INTEGER, verdict: null };
}

export default function OffersSection({ offers, onCompareOpen, onList, recommendedPrice = null, ranking = [] }: Props) {
  const [list, setList] = useState<OfferListResult | null>(null);
  const [showSkipped, setShowSkipped] = useState(false);
  const [comparing, setComparing] = useState(false);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => () => stopRef.current?.(), []);

  const handleCompare = useCallback(() => {
    setComparing(true);
    stopRef.current?.();
    stopRef.current = watchOfferList((result) => {
      setList(result);
      onList?.(result);
    });
    if (!openOffersPanel()) {
      stopRef.current();
      setList({ status: "unreadable", reason: "no offers link on the page", totalCount: null, offers: [] });
      return;
    }
    onCompareOpen?.();
  }, [onCompareOpen, onList]);

  if (offers.otherOffersCount === null) return null;

  const lowest = offers.otherOffersLowestPrice;
  const savings = lowest !== null && offers.oneTimePrice !== null ? offers.oneTimePrice - lowest : null;
  const ranked =
    list?.status === "ok"
      ? [...list.offers]
          .map((o) => ({ o, ...verdictFor(o, ranking) }))
          .sort((a, b) => a.rank - b.rank || total(a.o) - total(b.o))
      : [];
  const isSkip = (v: string | null) => v !== null && VERDICT[v]?.tone === "skip";
  const skippedCount = ranked.filter((r) => isSkip(r.verdict)).length;
  const visible = showSkipped ? ranked : ranked.filter((r) => !isSkip(r.verdict));
  const shown = list?.offers.filter((o) => !o.pinned).length ?? 0;
  const cheapestNotRecommended = recommendedPrice !== null && lowest !== null && lowest < recommendedPrice - 0.005;

  return (
    <section className="bw-offers" aria-label="Other sellers">
      <p className="bw-strong">
        {offers.otherOffersCount} other offer{offers.otherOffersCount === 1 ? "" : "s"}
        {offers.otherOffersLabel ? ` (${offers.otherOffersLabel})` : ""}
        {lowest !== null && (
          <>
            , from <span className="bw-mono">{money(lowest)}</span>
          </>
        )}
      </p>
      {savings !== null && savings >= 0.01 && (
        <p className="bw-note bw-note--tight">
          {cheapestNotRecommended && "The cheapest is "}
          <span className="bw-good">{money(savings)} less</span> than the Add to Cart price
          {cheapestNotRecommended && ", from an offer we don't recommend"}
        </p>
      )}

      {!comparing && (
        <button type="button" className="bw-button bw-button--quiet" onClick={handleCompare}>
          Compare all offers
        </button>
      )}

      {comparing && list === null && <p className="bw-note">Opening Amazon's offer list…</p>}
      {list?.status === "unreadable" && (
        <p className="bw-note bw-caution">Couldn't read Amazon's offer list. Showing the summary only.</p>
      )}
      {list?.status === "no_offers" && <p className="bw-note">No other offers right now.</p>}

      {list?.status === "ok" && (
        <>
          <ul className="bw-offer-list">
            {visible.map(({ o, verdict }, i) => (
              <li key={`${o.sellerId ?? o.seller}-${i}`} className={verdict ? `bw-verdict--${VERDICT[verdict]?.tone ?? "plain"}` : undefined}>
                <div className="bw-offer-row">
                  <span className="bw-mono bw-strong">{money(total(o))}</span>
                  <span>{o.condition ?? "Unknown condition"}</span>
                  {verdict && VERDICT[verdict] ? (
                    <span className={`bw-tag bw-tag--${VERDICT[verdict].tone}`}>{VERDICT[verdict].label}</span>
                  ) : (
                    o.pinned && <span className="bw-tag">Add to Cart</span>
                  )}
                </div>
                <p className="bw-offer-detail">
                  {o.seller}
                  {o.shippingCost
                    ? ` · ${money(o.price)} + ${money(o.shippingCost)} shipping`
                    : o.shippingCost === 0
                      ? " · free shipping"
                      : ""}
                  {o.deliveryDate ? ` · ${o.deliveryDate}` : ""}
                </p>
                <p className="bw-offer-detail">{sellerRating(o)}</p>
              </li>
            ))}
          </ul>
          {skippedCount > 0 && (
            <button type="button" className="bw-button bw-button--quiet" onClick={() => setShowSkipped((s) => !s)}>
              {showSkipped ? "Hide offers we'd skip" : `Show ${skippedCount} offer${skippedCount === 1 ? "" : "s"} we'd skip`}
            </button>
          )}
          {list.totalCount !== null && list.totalCount > shown && (
            <p className="bw-note">
              Showing {shown} of {list.totalCount}. Scroll Amazon's list to load more.
            </p>
          )}
        </>
      )}
    </section>
  );
}
