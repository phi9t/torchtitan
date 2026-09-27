# 18 — Evidence-contract residue

**What to build:** Close the four open findings from review 7. Each is a place
where the evidence contract is weaker than it appears.

**Blocked by:** none. Independent of all modelling work.

**Status:** implemented; gate run pending (see Resolution 2026-09-26)

## 1. Inferred attribution presented as observation

Producer and stream attribution comes from a **positional zip** of Flight
Recorder entries to Kineto kernels. It is an inference, and a plausible one,
but the exported facts and `CollectiveProducerCorrelation` read as though the
producer were observed. Anyone reasoning from these facts will over-trust
them.

- [x] Label the inferred fields as inferred, in the exported facts and in the
  predicate name or its comment -- not only in a design note.
- [x] State the positional-zip assumption where a reader of the facts will see
  it.
- [x] `resource_id` is opaque; say so rather than letting it read as a stream
  identity.

Ticket 09's Kineto bridge is the real fix, via a joined key instead of
position. This ticket makes the current state honest in the meantime; it does
not substitute for 09.

## 2. The skip guard covers only the owning stages

The undeclared-skip guard is what stops a pytest stage passing while silently
skipping its substance. It is wired into the owning stages only, so a skip in
another pytest stage still passes unnoticed.

- [x] Apply it to every pytest stage.
- [x] Note in the runner why it exists: the first version used `grep -E
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

- [x] Require, at the sealing boundary, that every stage in `stages.tsv` has a
      non-empty log. A missing or empty log at seal time must fail the seal,
      not be sealed as evidence of nothing.

## 4. Source manifest coverage narrows after commit

> **Understated, corrected 2026-09-27.** It does not merely narrow. On a fully
> clean tree coverage is **zero** and the gate **fails** in its lint stage, so a
> committed state cannot be gated at all. See ticket 25.


The manifest covered 87 entries while work was uncommitted and 13 afterwards,
because coverage is derived from working-tree status. So the sealed manifest
is weakest exactly when the tree is clean -- the state a reviewer sees.

- [x] Decide and record what the manifest is *for*: pinning the process
      documents that authorize the run (stable, and the current roster
      achieves it), or pinning the code that produced it (needs `HEAD`
      content, not status).
- [x] If the latter, derive coverage from a declared roster plus `HEAD`, not
      from working-tree status. The roster completeness check added for review
      finding R2 is the right mechanism -- shape validation cannot catch an
      absent entry, only a roster comparison can.
- [x] Either way, the sealed bundle should state which guarantee it makes.

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

- [x] Check the status explicitly, like every other step in that function.
- [x] Audit the same function for any other unguarded command.

## 6. The public verifier mutates the bundle it verifies

`_secure_managed_path` calls `current.mkdir()` for missing parents, so
`verify_evidence_bundle` cannot run against a read-only copy of a sealed bundle
-- which is precisely the situation a third party verifying evidence is in.

- [x] Verification is read-only. Creating a directory is a sealing concern, not
  a verification one.

## 7. Lint-time manifest validation is weaker than seal-time

`source_manifest_lint_paths`, which the `lint` stage consumes, runs
`_validate_process_section` without `repo_root` and without the roster
comparison, and has no `..` rejection of its own. The roster check runs later,
at finalize. So at lint time the path list is shape-validated only.

- [x] Apply the same validation at both points, or state why lint may be
  weaker.

## 8. Ticket-text accuracy

Findings from the ticket-03 review that are documentation-only but mislead a
reader:

- [x] Ticket 03's producer-correlation scope note cites `scout_b.py:2439-2455`,
  which is `_operation_family`/`_flight_snapshot`. The positional zip is at
  `scout_b.py:2548` (kernel sort), `:2556-2559` (entry sort), `:2566`
  (`zip(entries, kernels, strict=True)`).
- [x] Ticket 03 reports "180 owning tests (1 declared skip)"; the sealed log
  says `182 passed, 1 skipped`.
- [x] Ticket 03 reports Pyrefly with zero errors without noting the sealed log
  also contains `WARN ... Invalid search-path: /workspace/pytorch does not
  exist` and `0 errors (1 suppressed)`.
- [x] Ticket 03 cites an `outputs/...` bundle path as its evidence location
  without stating that `outputs` is gitignored, so a reviewer cloning the branch
  does not have it. Distinguish this sense of "checked-in" from the tracked
  `formal/ScoutB*Facts.*` fixtures.

## 9. The skip guard catches less than its name suggests

Confirmed working against real pytest output, in both directions, including
colour bytes -- that part of the contract holds. But it matches `SKIPPED` only.

- [x] `xfail`, collection-time `importorskip`, a deleted test, and a renamed
  file dropping out of an explicit file list are all invisible. Assert a
  collected-test count, or state the limit.
- [x] `cuda_pytest` is gated by a module-level `pytest.mark.skipif` and exits 0
  when skipped; it is currently caught only by accident, because a skipped run
  leaves `normalized/scout_b.json` absent and `artifact_sync` then fails.
  `focused_pytest` is likewise covered only because `owning_pytest` happens to
  list the same file. Make both couplings explicit.

## Resolution 2026-09-26

**Status:** implemented, awaiting the gate run. Owning surface: the two host
runners, the shared runner library, `scout_a.py`/`scout_b.py`, and the two
focused unit suites. Nothing under `formal/` was touched.

### 1. Inferred attribution presented as observation -- DONE, digest-moving

The disclaimer now travels *inside* the artifact instead of living only in
source comments, which is why `grep -ril infer` over the sealed bundle returned
nothing before:

- `PROVENANCE_CONTRACT` gained an `inferred_attribution` block --
  `inferred: true`, `observed: false`, `method:
  positional_zip_flight_entries_to_kineto_kernels`, `join_key: null`,
  `entry_order: flight_recorder_record_id`, `kernel_order:
  kineto_kernel_start_ns`, the exact list of inferred fields, and
  `resource_id_meaning: "opaque Kineto resource label, not a verified CUDA
  stream identity"`. It is deep-copied into `normalized/scout_b.json`, so it is
  in the sealed bundle and in the trace ID content.
- The exported TLA module emits a `\*` header block and `\*` comments directly
  above `CollectiveStream` and `CollectiveProducers`.
- The exported Lean module emits a `--` header block and a `--` comment above
  every `collectiveLifecycleRank{N}` definition.

Fields were **not** renamed: `CollectiveProducers`, `CollectiveStream` and
`producer.correlation_id` are consumed by specs under `formal/`, which this
ticket does not own. Ticket 09 remains the real fix via a joined key.

Because `provenance_contract` feeds `trace_id`, this changes every Scout B
digest and requires one intentional `--update-artifacts` gate run.

### 2. Skip guard covers only the owning stage -- DONE

Lifted into `scout_a_runner_lib.sh` as `scout_pytest_guard_program`, which emits
the bash program each pytest stage runs, and applied to `focused_pytest`,
`cuda_pytest` and `owning_pytest` in both runners. The program refuses to run a
command that does not carry both `-rs` and `--color=no`, and it still strips
ANSI escapes afterwards, so neither half can silently become the only defence.
The reason -- the original `grep -E "^SKIPPED"` matched zero lines of coloured
output, so the gate passed partly *because the guard did nothing* -- is recorded
in the function's comment, next to the limits in item 9.

### 3. Empty-log guard not at the sealing boundary -- DONE

`_require_non_empty_stage_log` in `scout_a.py`, called from
`_read_stage_journal` in **both** modules, so a zero-byte declared stage log
fails the seal and also fails `verify`. The bash `-s` guard stays as defence in
depth; its comment now says why it is kept (it names the stage and fires before
the journal row is written).

### 4. Source-manifest coverage narrows after commit -- DECIDED, documentation

**Decision: the manifest pins the process documents that authorize the run, not
the code that produced it.** Rationale: the code identity is already pinned, and
pinned better, by `head` plus the `source_id` over the working-tree status; a
`HEAD`-content manifest would duplicate what Git already guarantees and would
still not cover an uncommitted tree. What the manifest adds that nothing else
does is the by-bytes identity of the tickets, ADRs and notes that authorized the
attempt.

Consequences accepted, and to be stated wherever the guarantee is quoted:

- Coverage shrinking from 87 entries to 13 after commit is **not** a defect. The
  clean-tree case is the manifest working as intended: fewer dirty paths means
  fewer entries, and the process roster is the stable part.
- The second bullet of this item (derive coverage from a declared roster plus
  `HEAD`) is therefore **not applicable** and was deliberately not implemented.
- The roster completeness check added for review finding R2 stays: it is what
  makes an *absent* process entry detectable.

Kept out of code on purpose: adding a `guarantee` field to the sealed manifest
would move every digest for a documentation change. The statement belongs in
`experiments/qwen3_formal_verifier/README.md`, next to the existing process-
section discussion, and that one-line README addition is the only part of this
item still open.

### 5. Unchecked `git diff --check` -- DONE

The whole recheck sequence moved into `scout_verify_source_identity` in the
runner library, shared by both runners, with every step explicitly checked and
the reason in the comment: `scout_run_logged` does `set +e` before invoking a
stage function and errexit is not function-local, so the body runs with errexit
off and an unguarded command prints `result=success` after a failure. Audit of
the rest of the function found no other unguarded command; the remaining bare
commands are `rm -f` and the success `printf`.

### 6. Public verifier mutates the bundle -- DONE

`_secure_managed_path` takes `create_parents`, default `False`. Only
`_write_immutable` and rank 0's `raw/ranks` pre-creation pass `True`, so
`verify_evidence_bundle` can run against a read-only sealed copy and a missing
parent is a `FileNotFoundError` instead of a `mkdir`.

### 7. Lint-time vs seal-time manifest validation -- DONE

`lint-paths` now **requires** `--repo-root`, `--status-file`, `--head` and
`--attempt-dir` and runs `verify_source_manifest` before deriving the path list,
so the lint stage gets the same HEAD check, the same deleted-but-present
rejection, and the same process-roster comparison that finalize performs.
Making the arguments required rather than optional is deliberate: an optional
stronger mode is an invitation to a silent fall back to the weaker one.
`source_manifest_lint_paths` additionally rejects any path that would escape the
checkout (absolute, `..`) or be read as an option by `git add` /
`pre-commit --files`.

### 8. Ticket-text accuracy -- DONE, documentation

Ticket 03 corrected in place, each correction marked as coming from here: the
producer-correlation citation now names `_collective_observations` and its three
steps instead of `scout_b.py:2439-2455`; the owning-suite result is quoted from
the sealed log as `182 passed, 1 skipped`; Pyrefly is quoted as
`0 errors (1 suppressed)` together with its `Invalid search-path:
/workspace/pytorch does not exist` warning; and the `outputs/...` bundle paths
are marked gitignored and distinguished from the tracked `formal/ScoutB*Facts.*`
fixtures.

### 9. Skip guard catches less than its name suggests -- DONE

- **The limit is stated, not asserted.** `scout_pytest_guard_program`'s comment
  says the guard matches `SKIPPED` only and that an `xfail`, a collection-time
  `importorskip`, a deleted test, and a renamed file dropping out of an explicit
  file list are all invisible to it: it proves no undeclared skip, not that any
  particular check ran. A collected-test count was considered and rejected for
  now -- a hardcoded expected count in the runner is a second place to update on
  every test added, and it fails as a stale-number nuisance rather than as
  evidence. The comment names the alternative for anyone who wants the stronger
  property: bind the stage to an artifact the tests must produce.
- **Both accidental couplings are now explicit.** `cuda_pytest` is guarded with
  zero declared skips, so a module-level `pytest.mark.skipif` firing fails that
  stage directly instead of being caught later by `artifact_sync` tripping over
  an absent `normalized/scout_{a,b}.json`. `focused_pytest` is likewise guarded
  in its own right rather than being covered only because `owning_pytest`
  happens to list the same file.

### Evidence

Focused suites inside the rootfs: `pytest -q
tests/unit_tests/test_qwen3_formal_scout_a.py
tests/unit_tests/test_qwen3_formal_scout_b.py` -> `130 passed`. Owning suite
minus `test_formal_toolchain.py` (owned by a parallel agent) ->
`169 passed, 1 skipped`, the one skip being the declared
`test_rootfs_bwrap_plan.py:500` entry. `run_formal_tier0.sh --no-fetch
--skip-pytest` -> `QFV_TIER0 result=success mode=no-fetch`.

Three fail-closed experiments, each driving the real mechanism:

1. `test_sealing_refuses_a_zero_byte_stage_log` drives the real sealer with a
   zero-byte log, parametrized over all 8 Scout A and all 9 Scout B required
   stages. All 17 now raise `stage <name> log is empty and proves nothing`.
2. `test_runner_library_skip_guard_refuses_only_undeclared_real_pytest_skips`
   runs the shared guard over output from a real `pytest` run, in both
   directions, and repeats both with `--color=yes` appended after `--color=no`
   so genuine escape bytes are present (asserted). Undeclared skip -> exit 1
   with `undeclared skip in a pytest stage`; declared skip -> exit 0, silent.
   Raw bytes confirmed as `^[[33mSKIPPED^[[0m [1] ...`.
3. `test_source_recheck_fails_when_git_diff_check_fails` runs
   `scout_verify_source_identity` with errexit off, as the real stage does, and
   a fake `git` whose `diff --check` exits 1: exit non-zero, and
   `SCOUT_A_SOURCE_RECHECK result=success` absent. The positive control with
   `diff --check` exiting 0 prints the marker.

Not run here, and required before this ticket closes: the Scout A and Scout B
gates. Item 1 moves every Scout B digest, so it needs its own commit and an
intentional `--update-artifacts` run.

## Review round 2026-09-26 (independent, clean context): PASS WITH FINDINGS

All seven claims were confirmed against the real mechanism rather than by
reading tests: the sealer refuses a zero-byte log for all 9 Scout B stages, the
skip guard fires and stays silent correctly against real coloured pytest output,
the source recheck is fully guarded on re-audit, `verify_evidence_bundle` runs
against a bundle copy with every file `0444` and every directory `0555`, the
`lint-paths` arguments are genuinely required, and the digest movement is
exactly two normalized-JSON differences -- the added `inferred_attribution`
block and the resulting `trace_id` change.

The review also **retired the unverified-syntax caveat** without touching
`formal/`: it rendered the facts modules from the exporter into a scratch
directory and parsed them with SANY from the cached `tla2tools` jar and the
cached remote JDK 17, and compiled the Lean header with the cached Lean 4.34.0.
Both clean. So the generated `\*` and `--` comment blocks are known to parse.

### Findings fixed in the same phase

- **The non-empty seal check was a byte count, not a content check.** A log of
  one newline, of whitespace only, or of one meaningless word all sealed
  successfully. It now requires a non-whitespace byte, and the docstring says
  what it does and does not prove.
- **The declared-skip allowlist was an unanchored `grep -Fv`**, so any skip
  whose reason merely *contained* the declared text was allowed, from any file.
  It is now matched against the line's shape and its whole `file:line:reason`
  tail, with only pytest's `[N]` occurrence count left free. Verified three ways
  against real pytest output: the declared skip stays silent, the substring
  bypass now fires where it previously exited 0, and the declared skip stays
  silent with colour forced back on.
- **Two of the guard's stated limits were themselves wrong** -- the same species
  of unverified text this ticket set out to remove. A collection-time
  `importorskip` IS caught, because pytest reports it as a `SKIPPED` line; and
  the claim that neither `--color=no` nor the ANSI strip is the sole defence is
  false for the flag check, since a later `--color=yes` satisfies it. Both are
  corrected; the xfail and deleted-test limits are real and were verified.
- **The artifact-facing half of item 1 was unasserted** -- the half whose
  absence was the original defect. The export test now asserts the disclaimer
  reaches all four generated modules, verified by mutation: renaming the header
  string fails the test.
- **The contract changed shape while `SCOUT_SCHEMA` stayed `v0`**, so every
  older sealed bundle failed with a message indistinguishable from tampering.
  The contract now carries `PROVENANCE_CONTRACT_VERSION`, and the two cases
  report differently: an unversioned contract says "missing, unversioned, or
  altered", while a versioned mismatch says the bundle "was sealed under a
  different contract, which is not the same as tampering". This is
  digest-moving and ships with item 1.

### Item 4 is now closed in full

The guarantee statement landed in `experiments/qwen3_formal_verifier/README.md`
(added by the coordinating session, not the implementing agent, since that file
was outside its declared set). It states that manifest coverage scales with how
dirty the tree is, that a clean tree pins code by `head` plus `status_sha256`
alone, and that a bundle therefore attests "this evidence came from commit X
plus exactly these uncommitted bytes" rather than "these are all the files that
mattered".

## Workflow finding: `--update-artifacts` and the source-identity recheck

**Corrected 2026-09-26, second observation.** The claim below is too absolute.
The recheck refuses the run only when the artifacts **actually change during
it**. If the fixtures were regenerated beforehand and are already in sync,
`--update-artifacts` rewrites them to byte-identical content, `git status` does
not move, and the gate passes all nine stages -- measured on ticket 17, which
regenerated its fixtures during development and then gated clean on the first
run.

So the better recipe is not two gate runs but one: regenerate the fixtures
outside the gate, then gate. And note what `--update-artifacts` costs even when
it passes: `artifact_sync` **writes** before it checks, so that stage cannot
fail, and a bundle sealed under the flag has not had its fixtures verified. Use
a plain run for any bundle you intend to cite.

### The original observation, which holds when the artifacts do change

Exercised for the first time in a full gate, and it fails by construction, not
by defect. The run reached `lint` with all 13 pre-commit hooks and Pyrefly
green, then the source-identity recheck refused it:

```
error: dirty source identity changed during Scout B gate
```

`--update-artifacts` rewrites the tracked `formal/ScoutB{,Bad}Facts.{tla,lean}`
partway through the run. Those files are part of `git status`, which is what
`status_sha256` pins, so the identity captured at the start cannot still match
at the end whenever the artifacts genuinely change. The recheck is right to
refuse: a gate whose own source moved mid-run has not gated a single state.

The supported sequence is therefore two runs, and it should be documented as
such rather than discovered again:

1. Run with `--update-artifacts` to regenerate the fixtures. Expect it to fail
   at `lint`. The artifacts are updated on disk; the bundle is not valid
   evidence and must not be cited.
2. Run again without the flag. `artifact_sync --check` now compares equal, no
   tracked file is rewritten during the run, the identity stays stable, and the
   resulting bundle is the citable one.

- [ ] Say this in `experiments/qwen3_formal_verifier/README.md` next to the
  flag's description, and in `run_scout_b.sh --help`, so the first failure is
  expected rather than alarming.
- [ ] Consider making the failure explicit instead of incidental: with
  `--update-artifacts`, the runner could stop after the sync stage with a
  message saying the fixtures were updated and a plain re-run is required. As it
  stands the run does eight stages of real work and then fails on a recheck,
  which reads like a regression.

### Gate evidence (shared with ticket 18)

Full nine-stage Scout B gate, all stages exit 0, sealed and verified.

- Scout B evidence ID:
  `sha256:b24ddf9593af32c96dcd8c8588f9845aaad37a9473e2e140fc74b63be98c986f`
- Scout B source ID:
  `sha256:e179d91bb4bdb62ae01cb56221289f19761a0ec8406312b054c99510afff7c12`
- Nested Scout A evidence ID:
  `sha256:439517d3a89579fe7dcd34b4c5e5d26d625bfd7812541403c3ea0c9aa2f244fc`
- Source manifest: 26 verified paths, 7 process paths.

Result tokens present in the sealed formal log:

```
SCOUT_B_MODEL_SAFETY     distinct_states=38321 max_outdegree=24
                         bound_max_issues_per_rank=2 bound_ranks=4
                         bound_communicators=8 bound_issue_skew=unbounded
SCOUT_B_MODEL_NONVACUOUS  completion_reachable
SCOUT_B_MODEL_DIVERGENT   DeadlockFreedom                  named_violation
SCOUT_B_MODEL_WITNESS     NoCrossCommunicatorCycleWitness  named_violation
SCOUT_B_MODEL_OPMISMATCH  NoOpMismatchHang                 named_violation
SCOUT_B_MODEL_UNGUARDED   RendezvousOpAgreement            named_violation
SCOUT_B_MODEL_STREAM_EDGE        StreamEdgeIsInert  named_violation exit=151
SCOUT_B_MODEL_STREAM_EDGE_MUTANT StreamOfIssue_is_comm    holds     exit=0
```

This run needed two passes: the first, with `--update-artifacts`, regenerated
the facts fixtures and then failed its own source-identity recheck, for the
structural reason recorded under ticket 18. The IDs above are from the clean
second pass.

The IDs above are from the final run, after the soundness review's stream-order
caveat was applied to `BlockingPairs`. An earlier run of the same tree is
superseded and is not cited anywhere.
