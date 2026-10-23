# Daily price refresh: production setup

## What this job does

The `Daily price refresh` GitHub Action runs once each UTC day. It reads every
tracked product from the shared MySQL database, gets its latest Keepa price,
upserts exactly one dated price observation, and stores a fresh model prediction.
It fails the workflow if *any* product fails. A manual re-run is safe: the job
updates that product's UTC-day snapshot instead of duplicating it.

Both track-record reporting and price alerts should read `prices` with
`source = 'keepa_daily'`. This gives them the same daily source of truth without
spending Keepa tokens independently.

## Recommended production choice

Use a hosted MySQL database plus this scheduled GitHub Action. It is the lowest
maintenance option for a semester project: there is no always-on application
server just to run one daily command. GitHub-hosted Actions are free for public
repositories; private repositories receive included monthly minutes, and this
one short daily Python run is well below the usual allowance. See GitHub's
[Actions billing documentation](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Railway can host a small shared MySQL database. Its Free plan includes $1/month
of usage credit; Hobby is $5/month and applies that subscription to usage, so
the team should expect a paid plan for a database that must remain available.
Its current pricing is published at [Railway pricing](https://docs.railway.com/pricing/plans).
The owner should keep the Railway project until final grading, export a final
database backup, then delete the project to stop charges.

The backend API does **not** need to be hosted for the scheduled job or the two
features to share data. It does need hosting before the extension can work for
people who are not running it locally. Deploy it later against the same database
when that is needed; do not give the extension database credentials.

## One-time setup, before Oct 23

1. A team-owned Railway project provisions MySQL. Store its connection details
   in the team password manager, not in the repository. Limit database access
   to the team lead and the owners of #85 and #74; use a separate least-privilege
   application user if the provider supports it.
2. Apply `backend/db/schema.sql` to a new database. For an existing database,
   apply `backend/db/migrations/002_daily_price_snapshots.sql` after migration
   001.
3. In the GitHub repository's **Settings → Secrets and variables → Actions**,
   add `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and
   `KEEPA_API_KEY`. Never put any value in source, Actions output, or an issue.
4. The designated run owner enables GitHub notifications for **failed Actions**
   by email. Scheduled-workflow notifications go to the user who creates or
   last changes the cron schedule; GitHub documents this behavior at
   [workflow-run notifications](https://docs.github.com/en/actions/concepts/workflows-and-actions/notifications-for-workflow-runs).
5. Run **Actions → Daily price refresh → Run workflow** once. Verify a
   `keepa_daily` row and a new prediction for one ASIN in the database. Leave
   the workflow enabled for its first scheduled run.

## Access and handoff

Give #85 and #74 owners read-only database credentials (or a small read-only API
once the backend is deployed). The database owner retains schema and credential
management. Before the semester ends, export a MySQL dump and decide whether to
delete the hosted database or transfer its ownership; otherwise its provider
will continue billing.
