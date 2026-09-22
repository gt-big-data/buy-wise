# Project skills

One folder per skill, each containing a `SKILL.md` with YAML frontmatter
(`name`, `description`). Claude Code picks them up automatically and they're
shared with the team through git.

| Skill | What it does |
|---|---|
| `humanizer/` | Strips AI writing tells from prose. Used on FINDINGS.md and PLAN.md. |

To add one: `mkdir .claude/skills/<name>` and write `SKILL.md`. The
`description` field is what Claude matches against, so make it say when to
use the skill, not just what it is.
