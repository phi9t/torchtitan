# Falcon FWA mechanism campaign -- methodology spec

Status: approved 2026-09-07

Technical review: amended 2026-09-07 before ticket publication

Parent evidence: `../falcon-fast-weight-attention/`

Primary-source alignment review:
`../../docs/research/2026-09-07-falcon-fast-weight-attention-paper-alignment.md`

## Problem Statement

The process-faithful local Falcon campaign found a repeatable advantage for
Falcon-1A with delayed writes and QK-L2 normalization over the local
QK-RMSNorm variants. The result does not yet identify why the advantage
exists. QK-L2 and QK-RMSNorm change feature scale as well as the effective
ridge coefficient, write magnitude, read magnitude, and memory horizon. The
existing matrix also omits the same-step/L2 interaction cell and the behavioral
metrics needed to distinguish those explanations.

The campaign's addition result cannot support a transfer claim because its ID
evaluation examples are members of the finite training bank. Its OOD examples
are width-disjoint and valid, but every observed OOD score is zero. Historical
run records also predate the full run-attempt evidence contract.

Finally, the checkout implements context-conditioned beta with an uncoupled
lambda coefficient. The paper's strongest reported variants use
context-conditioned eta and lambda, whose exact unpublished parameterization
cannot be reconstructed faithfully. The next work must explain the local result
before spending more scale or presenting it as a paper-aligned result.

## Solution

Preserve Campaign B as the record of the completed local comparison. Campaign
B owns historical-evidence normalization, the repaired addition protocol, and
the final closeout. Conduct new normalization, alignment, and effective-scale
experiments in this separate mechanism campaign. Canonical evidence repair is a
hard gate before any new GPU training, and the mechanism decision is a hard
gate before Campaign B's final closeout.

The mechanism campaign first proves the full RMS-to-L2 coordinate transform at
the deterministic kernel seam. It then runs a five-arm, three-seed, 2,000-step
screen using the same model, data, optimizer, batch, and evaluation protocol.
Every arm records state dynamics as well as loss. Only contrasts that survive
the declared replication and practical-effect rule advance to an 8,000-step
confirmation. The screen has a one-B200-hour ceiling; confirmation has a
three-B200-hour ceiling.

Campaign B repairs addition evaluation with deterministic online training
that excludes fixed evaluation identities. Three fixed evaluation draws cover
each ID and OOD width. Results report exact suffix accuracy, token accuracy,
width-macro accuracy, output-position accuracy, and carry-count accuracy.

The output is an evidence-backed decision: the local A5 gain is explained by
effective scaling, alignment contributes a separate effect, the empirical
contrast does not replicate, or an implementation invariant fails and the work
walks back. State metrics explain the resulting behavior; they are not a route
around the scale theorem. A paper-aligned implementation campaign is considered
only after that decision and never by guessing the paper's unpublished
context-conditioned-eta rule.

## User Stories

1. As a researcher, I want every historical result traceable to an immutable
   raw artifact, so that later normalization does not rewrite experimental
   history.
2. As a researcher, I want legacy evidence to identify unavailable provenance
   explicitly, so that missing fields are never fabricated.
3. As a researcher, I want training runs and run attempts represented
   separately, so that restarts and duplicate summaries are not counted as new
   experiments.
4. As a researcher, I want training-bank accuracy labeled separately from
   held-out accuracy, so that memorization is not reported as transfer.
5. As a researcher, I want addition training identities excluded from every
   evaluation draw, so that ID accuracy measures generalization to new operands.
6. As a researcher, I want OOD widths excluded from training by construction,
   so that length extrapolation remains a real transfer test.
7. As a researcher, I want addition results broken down by width, output
   position, and carry count, so that aggregate zero or nonzero accuracy can be
   interpreted mechanistically.
8. As a researcher, I want RMS/L2 scale equivalence tested without training,
   so that algebraic parameterization effects are separated from learned
   effects before GPU time is spent.
9. As a researcher, I want the missing same-step/L2 cell included, so that an
   alignment-by-normalization interaction can be identified.
10. As a researcher, I want a scale-compensated RMS arm, so that a trivial
    effective-lambda explanation can be tested directly.
11. As a researcher, I want matched seeds and data order across arms, so that
    paired differences are more informative than unrelated trajectories.
12. As a researcher, I want several disjoint held-out regions evaluated for
    each checkpoint, so that checkpoint ranking is not tied to one fixed prefix.
13. As a researcher, I want eta, gamma, effective memory horizon, state norm,
    and clamp rates recorded by layer and head, so that scalar loss differences
    can be connected to memory behavior.
14. As a research lead, I want explicit compute ceilings and preflights, so
    that a diagnostic screen cannot silently turn into another hero campaign.
15. As a research lead, I want only discriminating contrasts promoted from
    2,000 to 8,000 steps, so that null arms stop consuming science compute.
16. As a research lead, I want a written stop decision when the L2 advantage
    is explained by scale compensation, so that the branch is not rescued by
    unplanned tuning.
17. As a research lead, I want paper-aligned variants kept behind a new design
    gate, so that an unpublished parameterization is not reverse-engineered and
    mislabeled as reproduction.
18. As a future reviewer, I want facts, inferences, hypotheses, and omissions
    separated in the closeout, so that the strength of every claim is obvious.

## Implementation Decisions

- The work remains a repo-local research program. It depends on TorchTitan
  core, and TorchTitan core does not depend on it.
- Campaign B remains immutable as the historical science/hero decision line.
  Factual corrections and claim downgrades are allowed; its frozen hero arm is
  not tuned again retroactively.
- New mechanism arms live in this campaign and never masquerade as Campaign B
  hero arms.
- Campaign B delivers the canonical evidence and repaired-evaluation seams;
  this campaign consumes them. Cross-campaign gates remain explicit even when
  the local Markdown tracker cannot enforce them automatically.
- Raw historical outcomes remain unchanged. A canonical import ledger records
  their hashes, source locations, inferred relationships, and explicit
  `legacy_import` provenance. Missing identifiers or environment fields remain
  missing rather than being invented.
- A training run is the logical experiment lineage. A run attempt is one
  launch or recovery attempt. Derived tables and live-attach artifacts refer
  to attempts and are not counted as additional training runs.
- Addition problem identity is operand width plus the unordered operand pair.
  Evaluation identities are fixed before training, and the online training
  stream rejects both operand orientations for every evaluation identity.
- Addition training widths remain 1 through 16 and OOD widths remain 17
  through 24 for comparability with Campaign B. Three fixed evaluation draws
  are mutually disjoint, contain eight problem identities per width, and are
  evaluated for every model. Each draw has an immutable split manifest and
  digest; operand orientation within a draw is deterministic.
- Addition reports whole-suffix exact accuracy and token accuracy, each as a
  micro-average and a width-macro average. It also reports output-position and
  carry-count strata. Output position is zero-based from the least-significant
  generated digit. Carry count is the number of digit columns whose outgoing
  carry is one, including the most-significant column. Training-bank accuracy,
  if retained for continuity, is labeled diagnostic-only.
- Campaign B's repaired addition matrix trains A0 and A5 for 2,000 optimizer
  steps at seeds 0, 1, and 2 with matched online streams and evaluates every
  final checkpoint on all three fixed draws. GDN uses the same matrix only when
  its 100-step steady-state projection, including evaluation, fits its separate
  two-B200-hour ceiling; otherwise it receives an omission record. The final
  checkpoint is predeclared, so evaluation accuracy is not used for checkpoint
  selection.
- The mechanism screen has five arms:
  - M0: delayed + QK-RMSNorm;
  - M1: same-step + QK-RMSNorm;
  - M2: delayed + QK-L2;
  - M3: same-step + QK-L2;
  - M4: delayed + scale-compensated QK-RMSNorm, using the complete coordinate
    transform relative to M2.
- Let `d` be head dimension. With RMS epsilon set to L2 epsilon divided by `d`,
  the native RMS map equals `sqrt(d)` times the L2 map; M4 does not apply an
  extra post-normalization multiply. Relative to M2, M4 also scales lambda and
  the NLMS-denominator epsilon by `d`. The kernel exposes the two epsilon roles
  separately while preserving current defaults for every existing arm. This is
  one conceptual scale-compensation knob.
- The deterministic proof requires matching outputs and backward gradients and
  compares the coordinate-adjusted state `sqrt(d) * S_rms` with `S_l2`; raw
  state tensors are not expected to match. M4 is ineligible for training until
  kernel and full-mixer output, state, and gradient comparisons pass with fp32
  relative and absolute tolerances of `1e-5`, and a short deterministic
  trajectory passes its full-precision loss and gradient guard.
- The screen uses the existing 4-layer, hidden-256, sequence-512 science model,
  plain FineWeb GPT-2 token stream, 16,384 tokens per optimizer step, existing
  optimizer recipe, and matched seeds and data order. Only the declared arm
  knob changes.
- The screen runs 2,000 steps for three seeds per arm. One arm and seed is one
  logical training run with its own attempt bundle. Its total allocation is
  capped at one B200 GPU-hour, summing device wall time across actual-shape
  preflights, successful runs, failed attempts, retries, and screen evaluation;
  parallel execution does not discount GPU-hours. CPU-only tests are excluded.
  Exceeding the ceiling requires a new human decision.
- Every screen checkpoint is evaluated on at least three fixed, nonoverlapping
  regions of the held-out token shard. The primary scalar is the mean
  token-weighted CE; PPL is reported as a derived presentation metric.
- Each arm contrast is paired by seed and held-out region. The decision unit is
  the per-seed mean of those fixed-region CE differences. A directional effect
  survives the screen only when all three seed means have the same sign and the
  absolute grand mean is at least 0.01 CE, approximately a one-percent PPL
  ratio. Region ranges are reported as heterogeneity diagnostics and are not
  treated as sampling uncertainty. A contrast that misses this rule is
  unresolved at this budget, not statistically equivalent.
- M4-to-M2 equivalence begins as a deterministic implementation invariant, not
  a failure-to-reject decision from three seeds. The 2,000-step screen also
  requires every paired seed's region-mean CE difference to fall within
  `[-0.01, 0.01]` and the grand mean within `[-0.005, 0.005]`; this is the
  predeclared practical-equivalence margin. If those gates hold and M2/M4
  improve over M0 under the directional rule, the A5 result closes as a
  parameterization/preconditioning effect and receives no larger normalization
  run. A training-time M4/M2 mismatch outside the margin is an implementation
  walk-back, not a new mechanism.
- An alignment interaction exists only when M1 versus M0 or M3 versus M2 passes
  the directional effect rule. Otherwise alignment is reported as
  unresolved/tied at this budget.
- State metrics explain how the uncompensated RMS arm differs from the
  scale-equivalent M2/M4 pair. They do not support a distinct L2-geometry claim
  when the scale theorem holds. Numerical tolerance applies only to
  same-input/same-weight seam tests. State metrics from independently trained
  trajectories are diagnostic and use the predeclared practical CE margins;
  only a failed seam invariant forces a kernel/mixer walk-back.
- Only the minimum arms needed to confirm a surviving contrast advance to
  8,000 steps with three seeds. The confirmation allocation is capped at three
  B200 GPU-hours, summing all confirmation attempts and evaluation. No
  hero-scale or paper-scale run is authorized.
- Confirmation uses the same per-seed directional rule and 0.01-CE practical
  threshold as the screen and must retain the screen's direction. Passing at
  2,000 steps but not 8,000 steps closes as non-replication; thresholds are not
  tuned after seeing confirmation data.
- GDN is not part of the mechanism screen. Campaign B may run a 100-step GDN
  addition preflight; full replicated GDN transfer runs proceed only when their
  projected total is at most two B200 GPU-hours. Otherwise the control is an
  explicit performance-based omission.
- Paper `ctxeta-ctxlambda`, Falcon-1, and Falcon-3A work requires a later design
  decision after the mechanism closeout. No implementation may claim exact
  paper alignment while the reference parameterization remains unpublished.
- Every GPU preflight records free disk, projected peak bytes, retained
  checkpoints, and a safety reserve. Screen runs retain at most one final
  model-only checkpoint per arm and seed; evidence normalization points to raw
  checkpoints rather than copying them.
- A proper replicated addition evaluation with zero OOD exact-suffix accuracy
  closes Campaign B with a negative `no_transfer` conclusion. It is completed
  evaluation evidence, not a successful transfer gate.

## Testing Decisions

- Tests target the highest stable behavior seams approved on 2026-09-07:
  evidence normalization, addition dataset/evaluation, Falcon kernel and mixer,
  and campaign promotion validation.
- Evidence normalization has two explicit records. A `LegacyImportRecord` has a
  deterministic import ID, immutable source path and digest, nullable
  source-declared run/attempt IDs, explicit inferred linkage, schema version,
  configuration and source digests, device/GPU/elapsed facts when available,
  and missing-field reasons. A new `FalconAttemptBundle` has nonempty run and
  attempt IDs plus lane, mode, arm, claim label, evidence tier, environment
  class, process/rank/mesh/device identity, clocks, steps, phases, data and
  checkpoint lineage, artifact index, and outcome. Evidence tests prove
  hashing, deduplication, idempotence, and rejection of digest conflicts,
  corrupted artifacts, and incomplete new-run bundles.
- Addition tests prove deterministic online generation, commutation-aware
  identity exclusion across train/ID/OOD, mutual draw disjointness, eight
  examples per width per draw, stable manifest digests, suffix masking, and the
  declared exact, token, width, position, and carry aggregations.
- The scale-equivalence test compares outputs, coordinate-adjusted final state,
  and backward gradients at the Falcon kernel boundary in fp32. It covers the
  actual science head dimension, separate epsilon roles, and the tolerance
  justified by the complete transform.
- Mixer tests prove every mechanism arm changes only its declared knob and that
  state instrumentation does not alter outputs or gradients when enabled.
- Runner smoke tests execute exactly one arm and seed and produce a complete
  attempt bundle without a science budget. Matrix orchestration may launch
  these runners but may not collapse several logical runs into one identity.
- Promotion tests consume result bundles rather than source constants. They
  enforce three seeds, disjoint held-out regions, the declared directional
  effect rule, deterministic M2/M4 equivalence, compute ceilings, and explicit
  omission records.
- Every feature or bug fix is developed red-green inside the rootfs. Focused
  tests run before the complete Falcon unit suite and changed-file lint.
- GPU runs begin only after the complete affected-arm fixed-bank overfit gate is
  rerun and the full arm matrix passes a few-step dry-run at the actual science
  shape with finite forward/backward values, correct M2/M4 equivalence, and
  a full-precision loss or gradient-norm difference above `1e-6` for every
  non-equivalent arm pair on at least one post-update step. Shrunk-model smoke
  evidence is not screen eligibility evidence.
- Mechanism diagnostics run on fixed probe batches at steps 0, 100, 500, 1,000,
  and 2,000; confirmation adds steps 4,000 and 8,000. Per valid token, layer,
  and head they record count, mean, standard deviation, and p05/p50/p95 for
  pre/post-normalization key energy, beta, actual lambda, eta, gamma,
  instantaneous e-folding horizon `-1/log(gamma)`, raw and
  scale-canonicalized state norm, update norm, and read norm, plus clamp
  fraction. Delayed sentinel step zero is excluded. Probe time and bytes are
  recorded and counted against the GPU and disk budgets.

## Out of Scope

- Retuning or replacing Campaign B's frozen hero arm.
- Repeating the completed 20,000-step hero runs.
- Paper-scale 49.2B-token training.
- Claiming a Table 1, downstream, or addition reproduction.
- Guessing the paper's unpublished context-conditioned-eta projection.
- Falcon-1, Falcon-3, Falcon-3A, sliding-window state, or a production
  chunk-parallel kernel in the mechanism screen.
- GDN performance engineering.
- Context parallelism, online RL, or promotion into core model packages.
- Remote experiment tracking as a canonical evidence store.
- Commit, push, pull-request creation, or merge without separate authority.

## Further Notes

- Campaign B's A5 result remains a valid local empirical observation even if
  M4 later explains it through scale compensation.
- Existing addition OOD scores are valid width-disjoint measurements and are
  all zero. Existing ID scores are training-bank diagnostics only.
- The official paper repository currently contains no executable model code or
  configurations. The primary-source review documents every exact match,
  approximation, divergence, and unknown that constrains this campaign.
- No ADR is created by this spec. These choices are campaign-local and remain
  reversible until a mechanism is promoted into a shared model surface.
