# Contributing to BuyWise

## Getting set up

Follow [ONBOARDING.md](ONBOARDING.md) to get the backend and extension running locally. [ROADMAP.md](ROADMAP.md) covers what we're building this semester and why.

## How work is organized

Work happens in pods of 2–3 people, usually from different subteams. Each pod owns one feature from design to a working demo. A pod's feature is tracked as a GitHub issue, and every PR for it links back to that issue.

## Branches

- Never commit directly to `main`.
- Branch from an up-to-date `main` and give the branch a short, descriptive name, prefixed with the area: `offers/variant-reader`, `watching/alert-schedule`, `viz/panel-tokens`.
- Keep a branch focused on one change. Big features land as a series of PRs, not one large one.

## Pull requests

Every change goes in through a pull request. The description says:

1. **What changed.**
2. **How you checked it.** Be specific: "the offer reader returns the right sellers on these 20 pages" rather than "added the offer reader." Screenshots are welcome for UI changes.
3. **The issue it belongs to**, if any.

Before opening a PR, run whatever applies to what you touched:

```bash
# backend
cd backend && .venv/bin/python -m pytest tests -q

# extension
cd extension && npm run build

# website
cd website && npm run build
```

## Reviews

- Someone other than the author reviews every PR.
- For analysis and model work, the reviewer re-runs it and gets the same result, not just reads the code.
- Any model or scoring rule that would change what users see has to beat the baselines in `backend/ml/evaluate.py` first.

## Merging

- Merge after an approval and passing checks.
- Use **squash and merge**, then delete the branch.
- If `main` moved since you branched, update your branch and re-run the checks before merging.

## What not to commit

- `.env` files, API keys, or any credentials. Use `.env.example` for new settings.
- Large data files. Raw Keepa pulls and other datasets live outside the repo.
- Personal information about shoppers or members.

## AI tools

Use them. If we find a bug or issue in something you built, we'll come to you about it, so make sure you understand what you submitted.

## Product text

Text shown to shoppers in the preview and extension comes from shared templates (`website/app/preview/copy.ts` for now). Add or change a template there rather than writing one-off sentences in a component, so the same fact reads the same way everywhere.

## Questions

Ask in the project's Discord channels rather than in DMs, so the answer helps everyone.
