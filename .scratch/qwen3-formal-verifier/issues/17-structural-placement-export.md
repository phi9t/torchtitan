# 17 — Structural DTensor placement export

**What to build:** Export real per-parameter DTensor placements and check the
sharding contract, replacing the two booleans per rank in use today.

**Blocked by:** none. Independent of the protocol model work.

**Status:** ready-for-agent

## What is lost today

The run observes 74 parameters per rank with real placements -- 115
`Shard(dim)` and 33 `Replicate` across ranks, with strided flags, zero
`Partial`. The formal facts collapse all of it to `PlacementHasDpShard` and
`PlacementHasTpShard`: two booleans per rank. `PlacementValid` therefore
checks that a 2x2 DPxTP run shards along both axes somewhere, which is true of
almost any non-degenerate configuration.

## What becomes checkable with structure

- No `Partial` placement survives into the optimizer step. A `Partial`
  reaching the optimizer means an unreduced gradient and silently wrong math
  -- exactly the class of bug `.claude/rules/distributed.md` warns about, and
  currently invisible to the formal layer.
- Every parameter's placement is consistent with the mesh axis it claims, and
  sharded dimensions divide evenly by the axis degree.
- TP-sharded and DP-sharded dimensions are distinguished per parameter rather
  than aggregated, so a parameter sharded on the wrong axis is detectable.

## Acceptance criteria

- [ ] Placements exported per parameter with axis and dim, not aggregated to
      booleans. Sizes grow: measure the effect on `ScoutBFacts.tla` parse time
      and report it as its own token, since that file is already about 1.4 MB
      and parse cost is charged before a single state is generated.
- [ ] Keep the two booleans, or derive them from the structural facts, so the
  existing `PlacementValid` result does not silently change meaning.
- [ ] An invariant that no `Partial` placement is present at the optimizer
  boundary, and a negative that injects one, derived from the facts.
- [ ] Divisibility of every sharded dim by its axis degree, checked against
      the mesh facts already exported.
- [ ] Raw schema changes, so digests move: own commit, intentional
  `--update-artifacts`, stated in the ticket.
