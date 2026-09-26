# 18 — Evidence-contract residue

**What to build:** Close the four open findings from review 7. Each is a place
where the evidence contract is weaker than it appears.

**Blocked by:** none. Independent of all modelling work.

**Status:** ready-for-agent

## 1. Inferred attribution presented as observation

Producer and stream attribution comes from a **positional zip** of Flight
Recorder entries to Kineto kernels. It is an inference, and a plausible one,
but the exported facts and `CollectiveProducerCorrelation` read as though the
producer were observed. Anyone reasoning from these facts will over-trust
them.

- [ ] Label the inferred fields as inferred, in the exported facts and in the
  predicate name or its comment -- not only in a design note.
- [ ] State the positional-zip assumption where a reader of the facts will see
  it.
- [ ] `resource_id` is opaque; say so rather than letting it read as a stream
  identity.

Ticket 09's Kineto bridge is the real fix, via a joined key instead of
position. This ticket makes the current state honest in the meantime; it does
not substitute for 09.

## 2. The skip guard covers only the owning stages

The undeclared-skip guard is what stops a pytest stage passing while silently
skipping its substance. It is wired into the owning stages only, so a skip in
another pytest stage still passes unnoticed.

- [ ] Apply it to every pytest stage.
- [ ] Note in the runner why it exists: the first version used `grep -E
      "^SKIPPED"`, which matched **zero** lines because pytest emits
      `\x1b[33mSKIPPED`. The gate passed partly *because the guard did
      nothing*, and it had been "verified in both directions" against
      hand-written plain text. The fix is `--color=no` plus ANSI stripping,
      verified against real pytest output.

## 3. The empty-log guard is not at the sealing boundary

`run_logged` refuses an empty log, which is how the `lint_paths_file`
collision became visible at all -- a stage aborted early and wrote a zero-byte
log. But sealing does not itself require every declared stage log to be
non-empty.

- [ ] Require, at the sealing boundary, that every stage in `stages.tsv` has a
      non-empty log. A missing or empty log at seal time must fail the seal,
      not be sealed as evidence of nothing.

## 4. Source manifest coverage narrows after commit

The manifest covered 87 entries while work was uncommitted and 13 afterwards,
because coverage is derived from working-tree status. So the sealed manifest
is weakest exactly when the tree is clean -- the state a reviewer sees.

- [ ] Decide and record what the manifest is *for*: pinning the process
      documents that authorize the run (stable, and the current roster
      achieves it), or pinning the code that produced it (needs `HEAD`
      content, not status).
- [ ] If the latter, derive coverage from a declared roster plus `HEAD`, not
      from working-tree status. The roster completeness check added for review
      finding R2 is the right mechanism -- shape validation cannot catch an
      absent entry, only a roster comparison can.
- [ ] Either way, the sealed bundle should state which guarantee it makes.
