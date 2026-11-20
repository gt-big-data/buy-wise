"""Shopper-facing text for the decision endpoint, mirroring website/app/preview/copy.ts.

copy.ts is the source of truth (see CONTRIBUTING, "Product text"). Every sentence the
backend sends to the panel is built here from the same templates, word for word, so
the same fact reads the same way in the panel, the preview and the dashboard.
If you change a template, change it in copy.ts first, then here;
tests/test_analysis_signals.py checks the two still match.
"""
from __future__ import annotations


def money(n: float) -> str:
    return f"${n:.2f}"


def pct(p: float) -> str:
    # copy.ts uses Math.round, which rounds .5 up; Python's round() rounds to even.
    return f"{int(p * 100 + 0.5)}%"


# copy.ts SKIP: why an offer wasn't considered.
SKIP = {
    "condition_profile": "Condition doesn't match your profile",
    "low_ratings": "Too few seller ratings",
    "slow_shipping": "Arrives more than a week later",
}

# copy.ts headline: one sentence per status, always built from numbers.
def headline_better_offer(savings: float, condition: str) -> str:
    return f"Same item, {condition.lower()}, {money(savings)} less."


def headline_wait(p: float) -> str:
    return f"{pct(p)} chance it drops 8% or more in the next 14 days."


def headline_buy_here() -> str:
    return "This is the best offer for you."


# Not in copy.ts yet: the version verdict (T2). Proposed in this PR; Data Viz to add
# them to copy.ts when the panel shows version verdicts.
VERSION_LINE = {
    "this_worse": "Reviews say this version is worse than the other one",
    "other_worse": "Reviews say the other version is worse than this one",
    "same": "Reviews show no real difference between versions",
    "mixed": "Reviews are mixed between versions",
}
