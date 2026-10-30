# Labeling guide: version pairs

We are labeling 100 pairs of product versions to check whether our methods can tell when reviews show a real quality difference. Two people label every pair, working separately. Don't discuss a pair until both of you have finished it.

## What you see

Each pair is two versions from the same family (same `parent_asin`), called A and B.

- The product title and what differs between the versions, if known.
- Up to 30 reviews for each version, with star ratings. Only pairs with at least 30 reviews on both sides are included.

You don't see prices. Knowing which version is cheaper could make you expect it to be worse. We work out which one is cheaper afterward.

## What counts as a difference

Judge the product itself. Ignore:

- shipping, packaging, and seller or customer service problems
- differences the shopper chose on purpose: a larger size is not better just for being larger, and a color is not better unless reviewers say it holds up differently
- one-off defects mentioned by a single reviewer

A quality difference is real if both of these are true:

1. At least 3 reviewers on one version raise the same issue.
2. That issue comes up at least twice as often on that version as on the other.

Star ratings alone are not enough. If the stars differ but the text doesn't say why, use **Can't tell**.

## Labels

Pick exactly one.

| Label | Use when |
|---|---|
| A is worse | A has a real difference against it, and B has none against it |
| B is worse | B has a real difference against it, and A has none against it |
| Mixed | Each version is worse on a different topic |
| No real difference | The reviews are useful, and neither version has a real difference against it |
| Can't tell | Too few reviews discuss quality, or you can't tell which version they mean |

**Mixed** gets its own label because "No real difference" would be false, and shoppers may care about one topic more than the other.

If you hesitate between **No real difference** and **Can't tell**, ask whether the reviews discuss quality at all. If they do and the versions sound alike, it's **No real difference**. If they don't, it's **Can't tell**.

## What to record

Add one row per pair and per labeler to the labels CSV:

| Column | Example |
|---|---|
| `parent_asin` | B07CJYMRWM |
| `asin_a` | B013J7WUGC |
| `asin_b` | B07CJYMRWM |
| `labeler` | your initials |
| `label` | A is worse |
| `topics` | battery; durability |
| `note` | optional, your own words |

Don't copy review text into the CSV or any other file in the repo. We commit the labels, not the reviews.

## Before labeling for real

1. Both labelers label the same 5 pairs separately.
2. Compare answers and find where you disagreed.
3. Change this guide so the next labeler wouldn't disagree in the same way.
4. Don't include these 5 pairs in the 100.

After that, don't resolve disagreements by changing labels. We report how often the two labelers agree, so the labels have to stay independent.
