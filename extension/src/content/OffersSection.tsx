import React, { useCallback, useEffect, useRef, useState } from "react";
import { PageOffers } from "./offers";
import { Offer, OfferListResult } from "./offerList";
import { openOffersPanel, watchOfferList } from "./offerPanel";

// Other sellers, in the panel's style. Shows Amazon's summary box right away and
// the full list once the shopper asks for it (Amazon only loads it on click).

const money = (n: number) => `$${n.toFixed(2)}`;
const total = (o: Offer) => o.price + (o.shippingCost ?? 0);

type Props = {
  offers: PageOffers;
  // Amazon's offer list slides in where the panel floats, so step aside while it's open
  onCompareOpen?: () => void;
};

export default function OffersSection({ offers, onCompareOpen }: Props) {
  const [list, setList] = useState<OfferListResult | null>(null);
  const [comparing, setComparing] = useState(false);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => () => stopRef.current?.(), []);

  const handleCompare = useCallback(() => {
    setComparing(true);
    stopRef.current?.();
    stopRef.current = watchOfferList(setList);
    if (!openOffersPanel()) {
      stopRef.current();
      setList({ status: "unreadable", reason: "no offers link on the page", totalCount: null, offers: [] });
      return;
    }
    onCompareOpen?.();
  }, [onCompareOpen]);

  if (offers.otherOffersCount === null) return null;

  const lowest = offers.otherOffersLowestPrice;
  const savings = lowest !== null && offers.oneTimePrice !== null ? offers.oneTimePrice - lowest : null;
  const sorted = list?.status === "ok" ? [...list.offers].sort((a, b) => total(a) - total(b)) : [];
  const shown = list?.offers.filter((o) => !o.pinned).length ?? 0;

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
          <span className="bw-good">{money(savings)} less</span> than the Add to Cart price
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
            {sorted.map((o, i) => (
              <li key={`${o.sellerId ?? o.seller}-${i}`}>
                <div className="bw-offer-row">
                  <span className="bw-mono bw-strong">{money(total(o))}</span>
                  <span>{o.condition ?? "Unknown condition"}</span>
                  {o.pinned && <span className="bw-tag">Add to Cart</span>}
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
              </li>
            ))}
          </ul>
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
