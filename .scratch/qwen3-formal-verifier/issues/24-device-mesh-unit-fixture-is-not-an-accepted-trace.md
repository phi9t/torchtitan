# 24 — The four-rank unit fixture is not a trace DeviceMeshValid accepts

**What to build:** Make the unit-test four-rank bundle satisfy the invariants
the real trace satisfies, so end-to-end checker runs against a fixture become
repeatable.

**Blocked by:** none. Small.

**Status:** ready-for-agent

## What was found

While implementing ticket 15, driving the real checkers against the unit-test
four-rank bundle showed it is **not** a trace `DeviceMeshValid` accepts. It
shards both axes of `tok_embeddings.weight` on dim 0, unstrided, with a local
shape half of global in each dim -- which contradicts both
`LocalShapeReflectsSharding` and `StridedShardIsAnOuterComposedShard` from
ticket 17.

Only the bundle's *projection* had ever been unit-tested, so nothing noticed.
Moving the tp shard to dim 1 reportedly fixes it.

## Why it matters more than it looks

Every phase since ticket 12 has wanted to run a real checker against a fixture
rather than a sealed trace, because the sealed trace needs a 4-GPU gate run.
Each one has had to improvise a derived fixture and delete it. A fixture that
the shipped invariants actually accept would make that repeatable, and would
let the TLC- and Lean-executing tests widen from hand-written mini-inputs to a
full four-rank shape.

It is also a small honesty point: the fixture is named as though it stands in
for the observed run, and it does not.

## Acceptance criteria

- [ ] The fixture satisfies every invariant `DeviceMeshValid.cfg` names, checked
  by running TLC against facts exported from it -- not by reasoning about it.
- [ ] The existing projection tests still pass unchanged, or the changes to them
  are explained. The fixture exists for them first.
- [ ] A test asserts the fixture is checker-accepted, so it cannot drift back.
  That test is the point of the ticket; without it this recurs.
- [ ] Note in the fixture's docstring that it is checker-accepted and must stay
  so, since the next person to add a parameter will not guess.
