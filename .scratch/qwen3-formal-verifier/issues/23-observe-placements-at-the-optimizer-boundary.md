# 23 — Observe placements at the optimizer boundary

**What to build:** Capture tensor placements at the optimizer pre-hook, and
capture gradient placements at all, so the no-unreduced-gradient check can be
about gradients.

**Blocked by:** none. Needs one fresh 4-GPU run, so it pairs with ticket 15.

**Status:** ready-for-agent

## Why ticket 17 could not finish this

Ticket 17 added a structural placement export and a
`ScoutBNoPartialAtOptimizer` invariant. Measured while building it:

- `tensor_placements` is a **single snapshot**, taken in `run_scout_b` after
  `config.build()` and before `trainer.train()`. It is not observed at the
  optimizer boundary.
- **No gradient placement is observed anywhere.** The one `gradient.ready` event
  carries a parameter's gradient digest, not a placement.

The optimizer boundary itself *is* observed -- `optimizer.started` and
`optimizer.mutated` per rank, from AdamW pre/post hooks, with
`parameter_sha256` before and after. So the boundary exists; the placements are
not sampled at it.

Consequence: a `Partial` **gradient** -- an unreduced gradient reaching the
optimizer, which is the silently-wrong-math case
`.claude/rules/distributed.md` warns about -- is invisible to this evidence.
Ticket 17's invariant is therefore about parameters at a boundary the trace
contains, not about gradients, and it says so in the module, the Lean mirror,
the exporter docstring, the README and its own ticket. That labelling is correct
and should not be quietly upgraded later; close the gap instead.

## Acceptance criteria

- [ ] Capture a placement snapshot in the optimizer **pre**-hook, so the facts
  describe the state the optimizer actually consumes rather than the state after
  construction.
- [ ] Capture gradient placements, not only gradient digests. Without them the
  central check cannot be stated.
- [ ] Restate `ScoutBNoPartialAtOptimizer` over gradients, keeping the parameter
  version as a separate, separately-named invariant. Do not widen the existing
  name to cover the new evidence -- a reader who saw the old claim must not be
  told the new one silently.
- [ ] A negative that injects a `Partial` gradient, derived from the facts.
- [ ] This changes the raw schema, so every digest moves and it needs the
  two-run gate sequence recorded in ticket 18. It also affects ticket 08's
  overhead measurement, since the snapshot is per optimizer step rather than
  once per run -- report the cost.

## Note on scope

Zero `Partial` placements were observed in the current trace, so this check is
expected to pass on a healthy run either way. Its value is fault detection, and
it is untested against a real fault until the v1 fault suite injects one; until
then it proves the property holds on this run, not that it would catch a
violation in the wild. The derived negative is what stands in for that.
