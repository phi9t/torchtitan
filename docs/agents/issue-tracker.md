# Issue tracker: Git tickets

Dispatchable multi-agent issues live as committed markdown files in `.tickets/`.
This follows the Corenius Collab Lite shape described in
`docs/agents/multi-agent-collab.md`.

`.scratch/` may still hold private notes, research, or planning scratch. It is
not a dispatch queue and does not grant a worker authority.

## Conventions

- One dispatchable ticket per file: `.tickets/<exact-id>.md`.
- IDs are exact and stable, for example `T-qwen-formal-cleanup`.
- Ticket state is recorded in frontmatter as `status: open`, `in_progress`, or
  `closed`.
- Dependencies are recorded in frontmatter as `deps: [T-other-ticket]`.
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create or update a committed `.tickets/<exact-id>.md` file unless the user asks
for scratch planning only.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. Keep exploratory maps in `.scratch/`; promote only
dispatchable, accepted implementation work into `.tickets/`.

- **Map**: `.scratch/<effort>/map.md` — the Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `open`/`claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Create**: set `Status: open` before the ticket is claimed.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
