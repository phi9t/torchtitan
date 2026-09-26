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

## 5. Unchecked `git diff --check` in the source recheck

`run_lint_and_verify_source` in both runners ends:

```
"${GIT_BIN}" -C "${REPO_ROOT}" diff --check
printf 'SCOUT_B_SOURCE_RECHECK result=success head=%s\n' "${HEAD_ID}"
```

Every other step in that function uses an explicit `|| return 1`. This one does
not, and errexit cannot cover it either: `scout_run_logged` does `set +e` before
invoking the function, and errexit is not function-local in bash, so the body
runs with errexit off. `result=success` therefore prints unconditionally and is
not evidence that the recheck's last step passed.

- [ ] Check the status explicitly, like every other step in that function.
- [ ] Audit the same function for any other unguarded command.

## 6. The public verifier mutates the bundle it verifies

`_secure_managed_path` calls `current.mkdir()` for missing parents, so
`verify_evidence_bundle` cannot run against a read-only copy of a sealed bundle
-- which is precisely the situation a third party verifying evidence is in.

- [ ] Verification is read-only. Creating a directory is a sealing concern, not
  a verification one.

## 7. Lint-time manifest validation is weaker than seal-time

`source_manifest_lint_paths`, which the `lint` stage consumes, runs
`_validate_process_section` without `repo_root` and without the roster
comparison, and has no `..` rejection of its own. The roster check runs later,
at finalize. So at lint time the path list is shape-validated only.

- [ ] Apply the same validation at both points, or state why lint may be
  weaker.

## 8. Ticket-text accuracy

Findings from the ticket-03 review that are documentation-only but mislead a
reader:

- [ ] Ticket 03's producer-correlation scope note cites `scout_b.py:2439-2455`,
  which is `_operation_family`/`_flight_snapshot`. The positional zip is at
  `scout_b.py:2548` (kernel sort), `:2556-2559` (entry sort), `:2566`
  (`zip(entries, kernels, strict=True)`).
- [ ] Ticket 03 reports "180 owning tests (1 declared skip)"; the sealed log
  says `182 passed, 1 skipped`.
- [ ] Ticket 03 reports Pyrefly with zero errors without noting the sealed log
  also contains `WARN ... Invalid search-path: /workspace/pytorch does not
  exist` and `0 errors (1 suppressed)`.
- [ ] Ticket 03 cites an `outputs/...` bundle path as its evidence location
  without stating that `outputs` is gitignored, so a reviewer cloning the branch
  does not have it. Distinguish this sense of "checked-in" from the tracked
  `formal/ScoutB*Facts.*` fixtures.

## 9. The skip guard catches less than its name suggests

Confirmed working against real pytest output, in both directions, including
colour bytes -- that part of the contract holds. But it matches `SKIPPED` only.

- [ ] `xfail`, collection-time `importorskip`, a deleted test, and a renamed
  file dropping out of an explicit file list are all invisible. Assert a
  collected-test count, or state the limit.
- [ ] `cuda_pytest` is gated by a module-level `pytest.mark.skipif` and exits 0
  when skipped; it is currently caught only by accident, because a skipped run
  leaves `normalized/scout_b.json` absent and `artifact_sync` then fails.
  `focused_pytest` is likewise covered only because `owning_pytest` happens to
  list the same file. Make both couplings explicit.
