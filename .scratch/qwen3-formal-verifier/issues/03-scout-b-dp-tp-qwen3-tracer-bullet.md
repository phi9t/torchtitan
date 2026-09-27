# 03 — Scout B: 2x2 DPxTP Qwen3 tracer bullet

**What to build:** Widen Scout A to one real four-rank Qwen3 optimizer step and
carry observed 2x2 DPxTP topology, placement, collective, executor, and causal
evidence through the same TLA+ and Lean path.

**Blocked by:** 02 — Scout A: single-rank Qwen3 tracer bullet.

**Status:** review-pending

- [x] The run uses four CUDA ranks through the normal
  `qwen3_debugmodel`/`ConfigManager` path with data-parallel replicate degree 1,
  data-parallel shard degree 2, tensor-parallel degree 2, and every other mesh
  degree 1. It preserves Scout A's model, precision, optimizer, seed,
  deterministic input, and checkpoint policy.
- [x] Local batch remains 1 and sequence length remains 128, giving global
  batch 2, 256 tokens per optimizer step, and one gradient-accumulation step.
  If runtime validation disproves a required divisibility assumption, the
  ticket returns to design review rather than silently changing this profile.
- [x] Raw evidence includes every rank's mesh coordinates, process-group
  membership, tensor placement, collective work enqueue/start/complete
  lifecycle, stream/executor assignment, and step end. Producer correlation is
  **inferred, not observed** -- see the scope note below -- so it is recorded
  as a derived field rather than claimed as observed evidence.
- [x] Normalization preserves per-rank source order and adds cross-rank order
  only from explicit causal or synchronization evidence.
- [x] A missing rank, duplicated rank, inconsistent run/attempt/topology/model/
  config identity, or unsupported cross-rank total order marks the bundle
  incomplete before either formal exporter can run. TP peers with the same DP
  coordinate must agree on their local first-batch identity; the ordered global
  input-bundle identity is keyed by DP coordinate, and different DP coordinates
  are not required to consume the same local batch.
- [x] Independently maintained TLA+ semantics check the bounded 2x2 mesh and
  collective lifecycle; a controlled invalid fact produces the expected named
  invariant violation rather than a checker error.
- [x] Independently maintained Lean definitions check the same finite DPxTP
  facts without `sorryAx` or project-defined axioms and reject one controlled
  false DPxTP proposition.
- [x] Synthetic DPxTP traces remain unit fixtures only and are not reported as
  Scout B runtime evidence.
- [ ] Focused and distributed integration tests, artifact synchronization,
  no-fetch cache reuse, lint, and separate Standards and Spec reviews by a fresh
  clean-context Codex reviewer pass from the same final worktree state. Missing
  review output or unresolved blocking findings block the ticket.
- [x] The report labels the result as one bounded 2x2 execution, not proof of
  arbitrary-topology or complete distributed-training correctness.

## Gate evidence

> **Stale as of 2026-09-26.** The bundle below was sealed at HEAD
> `366737c37`. Commits `25315472c`, `fee966fea` and `f93102cc2` have landed
> since, so 13 of the 87 verified manifest entries no longer match the tree and
> the public verifier now refuses the bundle with "source manifest no longer
> matches the current source tree". Every artifact under review is among the 13.
> The new invariants added by `fee966fea` have no sealed evidence at all.
>
> This ticket's final gate is deliberately deferred to the end of the
> formalization sequence (tickets 12-20), because re-sealing now would go stale
> again at the next phase. The ticket cannot be marked resolved until that final
> gate runs and this section is rewritten from the new sealed logs.

The complete supported gate passed from this worktree state at HEAD
`366737c37478d2253491742a74179febcf4ea54f`, all nine stages exit 0:
`source_manifest`, `focused_pytest`, `cuda_pytest`, `owning_pytest`,
`artifact_sync`, `scout_a_regression`, `formal_networked`, `formal_no_fetch`,
`lint`.

- Scout B sealed bundle:
  `outputs/qwen3_formal_verifier/evidence-task03-fix13/qfv-scout-b-seed42/four-rank-dp2-tp2-v1`
- Scout B evidence ID:
  `sha256:b95e3f3e74e6874eeeef47aaa70005b26328eacc5f5762b68ea096c7d05fa266`
- Nested Scout A sealed bundle:
  `outputs/qwen3_formal_verifier/evidence-task03-fix13/scout-a-regression/qfv-scout-a-seed42-fix1/single-rank-cuda-fix1`
- Nested Scout A evidence ID:
  `sha256:afa04eb38c64d7218abadd5dc32ff43171aae8981a0934f06d4de783823b50c4`
- Shared source ID:
  `sha256:63b39450b548fb39b13bafb062e2214350d3abb21f56926476da31bebdf8dab1`
  (`qwen3.formal.scout.source-manifest.v1`; 87 verified entries, 12 process
  documents recorded informationally)

Results, read from the sealed logs of this attempt: 31 focused Scout B tests,
the real four-rank DP-shard-2/TP-2 CUDA step, the owning suite at
`182 passed, 1 skipped`, checked-in artifact synchronization in `--check` mode,
the complete nested Scout A gate (69 focused, 149 owning, its own formal suites
and seal), both networked and no-fetch formal suites, thirteen lint hooks,
Pyrefly at `0 errors (1 suppressed)`, source recheck, sealing, and public
verification.

> Corrected by ticket 18 item 8, against the sealed logs rather than memory:
> the owning stage log says `182 passed, 1 skipped`, not "180 owning tests
> (1 declared skip)". The Pyrefly log is not a bare zero either -- it contains
> `WARN /workspace/torchtitan/pyproject.toml: Invalid search-path:
> /workspace/pytorch does not exist` and reports `0 errors (1 suppressed)`, so
> one suppression is in force and one configured search path is absent.
> Evidence-location caveat: the `outputs/...` bundle paths above are
> **gitignored**. A reviewer cloning this branch does not get them, and they are
> not "checked in" in the sense that the tracked `formal/ScoutB*Facts.*`
> fixtures are. Treat the bundle paths as machine-local evidence on the machine
> that ran the gate, and the tracked fact modules as the only artifacts a clone
> carries.

The one skip is `test_rootfs_bwrap_plan.py:500: host perf is not dynamically
linked`. It is declared in both runners; any other skip fails the stage.

TLC reports the named `ScoutBCollectiveProducerCorrelation` violation for the
negative model. Lean `validDPxTP` is axiom-free, all three event mutations are
rejected, and the controlled producer proposition is rejected.

### Review history

An earlier attempt at this change (preserved as tag `qfv-task03-fix2-snapshot`)
passed its gate but failed independent review with six findings, none Critical
(`.superpowers/sdd/spec/task-03-fix2-codex-review.md`). All six were addressed:

- process-section validation now enforces an exact key set, `verified=false`, a
  hex `sha256_at_seal`, a valid octal mode, and rejects a `deleted` record whose
  file is actually present -- the forgery that would have dropped a live
  document from lint while still passing verification;
- `source_manifest_lint_paths` now calls that validation instead of trusting the
  section, which it previously did not do at all;
- the executable-source guard became an allowlist (`.md`/`.txt` only, matched
  case-insensitively), which closes extensionless files, `.cfg`, and any
  unlisted interpreted or formal source; symlinks and executable mode bits are
  rejected as well;
- read-path `assert`s became `ValueError`, per the repository convention that
  data loaded from disk is user input;
- the content-versus-membership boundary is stated precisely and pinned by test:
  tracker text may drift without changing `source_id`, while adding or removing
  a tracker path changes `status_sha256` and therefore does change it;
- this gate ran from the same final state under review *at that time*. That is
  no longer so: three commits have landed since, and the same-final-state
  finding is therefore reopened. See the stale-evidence note above.

### A defect this process caught in its own guard

The skip guard added in an earlier round was **dead code**. Pytest colourises
output, so the sealed log carried `\x1b[33mSKIPPED`, and the guard's `^SKIPPED`
anchor matched nothing. A gate then passed partly because the guard silently
matched zero lines. It had been "verified both directions" against plain-text
output written by hand rather than produced by pytest.

It is now fixed twice over -- `--color=no` on the pytest invocation and ANSI
stripping before matching, so neither alone is load-bearing -- and verified
against real pytest output. The sealed log of this attempt shows one
guard-visible `SKIPPED` line where previous attempts showed zero.

The general lesson is recorded because it recurred three times in this effort:
a check verified against hand-authored fixtures confirms the author's
assumptions instead of testing the system. Fixtures must come from the tool
whose output they imitate.

### Evidence-contract hardening from review

Five independent review rounds ran against successive states of this change.
Each found something real; the fixes are in the source and summarised here so a
reader does not have to reconstruct them from review artifacts:

- process-section forgery: exact key set, `verified=false`, hex digest, valid
  mode, no entry claiming deleted while present, and roster completeness
  against the captured status -- omission was the hole shape validation missed;
- executable source in process paths: an allowlist (`.md`/`.txt`), not a
  denylist, which closes extensionless files and unlisted languages;
- generated checker products sealed as verified source: fails closed, scoped to
  the formal directory so it neither misses `<Spec>_TTrace_<n>.bin` nor rejects
  legitimate source elsewhere;
- checker crashes after a valid diagnostic: JVM frames, including
  module-qualified ones, and abort markers count as infrastructure failure, and
  each classifier requires the completion signal TLC actually emits for its
  case -- a search summary for a state-space run, `Finished in` for an
  initial-state violation;
- a lint path producer that crashed mid-emit could lint a partial set and still
  seal: its exit status is now checked rather than discarded by process
  substitution;
- a skipped test could seal as a passing stage: skips must be declared, and an
  undeclared one fails the stage;
- the documented Scout B command could not pass artifact synchronisation, since
  the default attempt ID did not match the attempt the checked-in fixtures were
  generated under.

### Toolchain hazard, and its resolution

An earlier run of this gate failed at `formal_networked` because upstream
regenerated the TLA+ `v1.8.0` asset at 2026-09-25T02:05:27Z under a new asset
id, changing its digest and tripping Bazel's integrity check. `v1.8.0` is a
rolling prerelease: the same URL serves different bytes over time, so a digest
pinned against it is not a pin but a tripwire. It had already recurred once, on
2026-09-23.

This is now resolved rather than worked around. The pin moved to **TLA+ 1.7.4**,
the newest stable release (published 2024-08-05, asset unmodified since
2024-08-08). Equivalence was verified before switching: all three valid models
pass and all three negative models produce the same named invariant violations
with the same exit code 12. Lean was already pinned to a stable release
(`v4.34.0`, published 2026-09-14, unmodified since), and its digest was
confirmed equal to the digest GitHub publishes for the asset.

### Superseded evidence

Bundles under `evidence-task03-fix1-final-v2` were sealed under the
`source-manifest.v0` schema and are refused by both public verifiers with
`source manifest has an unsupported schema`. They remain on disk as historical
evidence and are no longer machine re-verifiable -- an intentional, named cost.

### Scope note on producer correlation

Producer correlation is **not** observed correspondence. `_collective_observations`
in `scout_b.py` groups NCCL Flight Recorder entries and Kineto kernels by
operation family, sorts the kernels by GPU `start_ns()` (kernel sort), sorts the
entries by `record_id` (entry sort), requires the two counts to be equal, and
then pairs them positionally with `zip(entries, kernels, strict=True)`.

> Corrected by ticket 18 item 8: this note previously cited
> `scout_b.py:2439-2455`, which is `_operation_family`/`_flight_snapshot`, not
> the zip. The positional zip lives in `_collective_observations`: the kernel
> sort, the entry sort, and the `zip(entries, kernels, strict=True)` a few lines
> later. Line numbers are deliberately not restated here, because they moved
> again when ticket 18 added the exported inference disclaimer.

Equal counts plus a sorted zip do not establish that the i-th Flight record
describes the same work as the i-th kernel. The pairing assumes issue order
matches kernel start order within an operation family, which holds for a single
in-order stream but is not guaranteed across asynchronous streams or multiple
process groups. A work can therefore receive another work's stream and
correlation identity without any check failing.

The count-equality guard is real and would catch a dropped or extra record. The
TLA+ `ScoutBCollectiveProducerCorrelation` invariant and its Lean counterpart
check the structural consistency of the producer identities that this pairing
produced; they do not and cannot establish that the pairing is correct, because
the facts they consume are its output.

Establishing real correspondence needs a shared key the two sources both carry
-- a correlation ID joined across Flight Recorder and Kineto rather than
position. Ticket 09's Kineto bridge is where that belongs. Until then, any claim
that Scout B establishes producer-to-work correlation is too strong, and this
ticket does not make it.

### Scope note on the attempt identity

The attempt ID is hashed into every event ID and therefore into every exported
digest, so the checked-in `ScoutB*Facts` fixtures are keyed to
`four-rank-dp2-tp2-v1`. A re-seal under a different attempt ID reports them stale
even when the trace is semantically identical. This gate reused the canonical
attempt ID rather than passing `--update-artifacts`, which would have accepted a
new canonical trace silently. Ticket 04 owns whether the fixture should become
attempt-ID independent.

### Review round 2026-09-26 (independent, clean context): FAIL

Two blocking findings, both confirmed:

1. **The sealed evidence is for a superseded state**, and the ticket asserted
   the opposite. Recorded above; the final gate is deferred to the end of the
   sequence rather than re-run per phase.
2. **Sealing does not verify that any stage log contains anything.**
   `_read_stage_journal` calls `_require_regular_readonly` and discards the
   bytes, so mode `0444` and regular-file status are checked but not size. The
   reviewer drove the real sealer with nine zero-byte logs and it accepted all
   nine. The `-s` guard lives only in bash, so it is not part of the sealed
   contract. Tracked as ticket 18 item 3.

Non-blocking findings were folded into ticket 18 (items 5-9) and a new ticket 20
(host paths in the identity chain). Two are worth restating here because they
correct this ticket's own text: the producer-correlation scope note cites the
wrong code, and the reported test counts and Pyrefly result do not match the
sealed logs.

What the review confirmed as sound, so it is not re-litigated: the digest chain
has no circularity and nothing is self-certifying; immutability, symlink and
path-traversal handling are real; the source-manifest allowlist has no bypass
and the roster comparison does close omission; and the skip guard genuinely
works in both directions against real coloured pytest output.

### 2026-09-27: the final-state criterion is unsatisfiable, not merely unmet

The final re-seal was attempted from the clean committed tree at `d8b1016d5`,
after every other phase closed. It **fails**, in the nested Scout A gate's
`lint` stage, whose log is ten lines with no hook run.

The manifest from a clean tree has zero verified and zero process entries, so
`source_manifest_lint_paths` yields nothing and the lint stage's "at least one
file" assertion fails. Tracked as ticket 25.

So this ticket's criterion -- re-execute the supported commands from the final
worktree state -- cannot be satisfied as the gate is written, because the final
state is a committed state, a committed state is a clean tree, and a clean tree
has no lint paths. Every bundle this project has sealed came from a dirty tree.

This does not mean the work was ungated. Each phase ran a real nine-stage gate
whose manifest pinned the uncommitted bytes under review, and those bundles
verify. What is absent is the final seal from the committed state, which is a
narrower claim than "never gated" and should be read as such. This ticket stays
open until ticket 25 lands.
