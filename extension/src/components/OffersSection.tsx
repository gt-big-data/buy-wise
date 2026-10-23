import React, { useCallback, useEffect, useRef, useState } from "react";
import { PageOffers } from "../content/offers";
import { Offer, OfferListResult } from "../content/offerList";
import { openOffersPanel, watchOfferList } from "../content/offerPanel";

type OffersSectionProps = {
  offers: PageOffers;
  onCompareOpen?: () => void;
};

const money = (n: number) => `$${n.toFixed(2)}`;
const total = (o: Offer) => o.price + (o.shippingCost ?? 0);

const OffersSection: React.FC<OffersSectionProps> = ({ offers, onCompareOpen }) => {
  const [list, setList] = useState<OfferListResult | null>(null);
  const [comparing, setComparing] = useState(false);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => () => stopRef.current?.(), []);

  const handleCompare = useCallback(() => {
    setComparing(true);
    stopRef.current?.();
    stopRef.current = watchOfferList(setList);
    if (!openOffersPanel()) {
      setList({ status: "unreadable", reason: "no offers link on the page", totalCount: null, offers: [] });
    }
    onCompareOpen?.();
  }, [onCompareOpen]);

  if (offers.otherOffersCount === null) return null;

  const lowest = offers.otherOffersLowestPrice;
  const savings = lowest !== null && offers.oneTimePrice !== null ? offers.oneTimePrice - lowest : null;
  const sorted = list?.status === "ok" ? [...list.offers].sort((a, b) => total(a) - total(b)) : [];

  return (
    <section className="buywise-panel-section buywise-offers" aria-label="Other sellers">
      <div className="buywise-offers-summary">
        <div className="buywise-info-title">
          {offers.otherOffersCount} other offer{offers.otherOffersCount === 1 ? "" : "s"}
          {offers.otherOffersLabel ? ` (${offers.otherOffersLabel})` : ""}
          {lowest !== null ? `, from ${money(lowest)}` : ""}
        </div>
        {savings !== null && savings >= 0.01 && (
          <div className="buywise-info-preview">{money(savings)} less than the Add to Cart price</div>
        )}
      </div>

      {!comparing && (
        <button type="button" className="buywise-offers-button" onClick={handleCompare}>
          Compare offers
        </button>
      )}

      {comparing && list === null && <div className="buywise-info-preview">Opening Amazon's offer list…</div>}

      {list?.status === "unreadable" && (
        <div className="buywise-info-preview">Couldn't read Amazon's offer list. Showing the summary only.</div>
      )}

      {list?.status === "no_offers" && <div className="buywise-info-preview">No other offers right now.</div>}

      {list?.status === "ok" && (
        <>
          <ul className="buywise-offers-list">
            {sorted.map((o, i) => (
              <li key={`${o.sellerId ?? o.seller}-${i}`} className="buywise-offers-row">
                <div className="buywise-offers-row-main">
                  <span className="buywise-offers-price">{money(total(o))}</span>
                  <span className="buywise-offers-condition">{o.condition ?? "Unknown condition"}</span>
                  {o.pinned && <span className="buywise-offers-tag">Add to Cart</span>}
                </div>
                <div className="buywise-offers-row-detail">
                  {o.seller}
                  {o.shippingCost ? ` · ${money(o.price)} + ${money(o.shippingCost)} shipping` : o.shippingCost === 0 ? " · free shipping" : ""}
                  {o.deliveryDate ? ` · ${o.deliveryDate}` : ""}
                </div>
              </li>
            ))}
          </ul>
          {list.totalCount !== null && list.totalCount > list.offers.filter((o) => !o.pinned).length && (
            <div className="buywise-info-preview">
              Showing {list.offers.filter((o) => !o.pinned).length} of {list.totalCount}. Scroll Amazon's list to load more.
            </div>
          )}
        </>
      )}
    </section>
  );
};

export default OffersSection;
