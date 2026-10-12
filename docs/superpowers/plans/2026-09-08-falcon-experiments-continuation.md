# Falcon Experiments Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Each ticket gets a fresh implementer, an independent reviewer, and a
> finalizer. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Campaign B with uncontaminated replicated evaluation and run a
bounded mechanism campaign that determines whether A5's local L2 advantage is
explained by effective scale, alignment, non-replication, or an implementation
failure.

**Architecture:** Keep the immutable Campaign B history separate from new
mechanism experiments. Reuse the native Falcon attempt-bundle contract for all
new work, build pure dataset/evaluation/decision modules under
`torchtitan/experiments/falcon/`, and keep rootfs entrypoints and generated
artifacts under `experiments/falcon/`. GPU work is gated by deterministic
proofs, actual-shape dry-runs, disk projections, complete matrix projections,
and predeclared stop rules.

**Tech Stack:** Python 3.12, PyTorch/TorchTitan Trainer, pytest, JSON evidence
bundles, nanoGPT FineWeb token bins, bwrap rootfs, NVIDIA B200.

**Specs:**

- `.scratch/falcon-fast-weight-attention/spec.md`
- `.scratch/falcon-fwa-mechanism/spec.md`
- `.claude/skills/running-experiment-campaigns/SKILL.md`
- `docs/research/2026-09-07-falcon-fast-weight-attention-paper-alignment.md`

## Global Constraints

- Run every Python, PyTorch, model, parser, evaluator, test, preflight, and GPU
  command through `scripts/rootfs/enter_rootfs.sh`; the host shell only
  orchestrates and inspects Git state or logs.
- Preserve Campaign B raw artifacts and all three 20,000-step hero checkpoints.
  Evidence code points to them; it never rewrites or copies them.
- Preserve historical A0-A5 meanings and A5 as the frozen Campaign B winner.
  New M0-M4 arms belong only to the mechanism campaign.
- One arm and seed is one logical run. Each launch or recovery is a separate
  attempt. Matrix orchestration never shares identity across cells.
- No paper-scale run, new hero, `ctxeta` guess, Falcon-3A, GDN optimization,
  chunk-parallel production work, remote evidence store, or core-model change.
- No commit, push, pull request, merge, stash, or destructive Git action without
  separate user authority.
- The mechanism preflight plus 15-run screen and evaluation must remain at or
  below one summed B200-hour. Confirmation must remain at or below three summed
  B200-hours.
- Campaign B A0/A5 addition preflight, six training runs, retries,
  checkpoints, diagnostics, and evaluation must remain at or below one summed
  B200-hour, with at most one retry per logical run.
- Full GDN addition runs are authorized only when the 100-step projection is at
  or below two summed B200-hours; otherwise record an omission.
- Every GPU preflight records free disk, projected peak bytes, retained files,
  and safety reserve before launch.
- Smoke proves plumbing, deterministic comparison proves identity,
  `representative_small` describes the mechanism screen, and replicated
  transfer claims require three seeds/draws. Never call this Table 1 or paper
  reproduction.

---

## Baseline at Plan Publication

- Campaign B tickets 01-09 are resolved. B09's canonical ledger contains 55
  pre-B09 records, including 52 Campaign B rows: 20 valid, 23 smoke, 2 invalid,
  2 incomplete, and 5 omitted. It indexes 83 artifacts and 33 source paths.
- B09 final verification is 31 focused tests, 100 Falcon tests, canonical CLI
  count 55, and ledger SHA-256
  `4c7e150629f5595e7e4600a82ad2129f870eec277d1f78f624bb0ef7d4b57cd8`.
- Step 4 used the 4-layer, hidden-256, head-dimension-32, sequence-512 model at
  16,384 tokens/step. Completed 8,000-step LM means were A0 CE 4.673918, A5 CE
  4.729747, A4 CE 4.795631, A1 CE 4.797328 (one seed), and A2 CE 4.807353.
- Step 5 completed A0, A5, and A1 at 20,000 steps / 327,680,000 tokens each.
  Their validation PPLs were 87.63, 95.04, and 99.48 respectively.
- Prior addition ID values are training-bank diagnostics. All valid width-OOD
  exact-suffix results are zero. No held-out addition transfer claim exists.

## File Structure

| File | Responsibility |
| --- | --- |
| `torchtitan/experiments/falcon/falcon.py` | Recurrent/masked kernel math and separate QK/NLMS epsilons. |
| `torchtitan/experiments/falcon/model.py` | `FalconConfig`, M4 transform materialization, mixer diagnostics hooks. |
| `torchtitan/experiments/falcon/config_registry.py` | Immutable historical A arms and separate M0-M4 materialization. |
| `torchtitan/experiments/falcon/addition.py` | Pure addition identities, manifests, online sampling, masks, and metrics. |
| `torchtitan/experiments/falcon/lm_eval.py` | Fixed LM region registry, digest validation, and token-weighted aggregation. |
| `torchtitan/experiments/falcon/mechanism.py` | Diagnostic aggregation and scale-canonicalized state quantities. |
| `torchtitan/experiments/falcon/promotion.py` | Pure budget, coverage, equivalence, directional-effect, and terminal-decision validation. |
| `torchtitan/experiments/falcon/evidence.py` | Existing native attempt publication and canonical legacy verification; do not merge campaign policy into it. |
| `experiments/falcon/addition_runner.py` | Exactly one Campaign B addition arm/seed per invocation. |
| `experiments/falcon/checkpoint_eval.py` | Read-only fixed-region evaluation of one checkpoint. |
| `experiments/falcon/mechanism_runner.py` | Exactly one mechanism arm/seed per invocation with bounded probes. |
| `experiments/falcon/campaign_driver.py` | Preflight/matrix orchestration that launches single-attempt subprocesses and invokes pure validators. |
| `experiments/falcon/run.sh` | Rootfs-aware shell dispatch only; no Python business logic. |
| `tests/unit_tests/test_falcon_scale_transform.py` | Kernel, mixer, model, trajectory, validation, and compatibility proof. |
| `tests/unit_tests/test_falcon_addition_protocol.py` | Disjoint draws, online exclusion, masks, and addition metric behavior. |
| `tests/unit_tests/test_falcon_lm_eval.py` | Region registry, digest, weighting, and immutable checkpoint behavior. |
| `tests/unit_tests/test_falcon_mechanism.py` | Single-attempt ownership and instrumentation invariance/formulas. |
| `tests/unit_tests/test_falcon_promotion.py` | Screen, confirmation, compute, coverage, retry, and omission boundaries. |
| `tests/unit_tests/test_falcon_closeout.py` | Full Campaign B matrix coverage and claim/no-regression decisions. |

### Task 1: Record the B09 Gate and Freeze the Baseline

**Files:**

- Modify: `.scratch/falcon-fast-weight-attention/issues/09-step6-eval-close.md`
- Modify: `.scratch/falcon-fast-weight-attention/map.md`
- Modify: `.scratch/falcon-fast-weight-attention/sdd/progress.md`
- Verify: `experiments/falcon/results/evidence/legacy-ledger-v1.json`

**Interfaces:**

- Consumes: `verify_known_pre_b09_ledger(repo_root: Path, ledger_path: Path) -> int`.
- Produces: resolved B09 gate and unblocked B10/B11 tracker state.

- [x] **Step 1: Record finalizer acceptance and all seven acceptance checks.**

- [x] **Step 2: Pin the inventory counts and ledger digest in the ticket Answer.**

- [x] **Step 3: Move the Campaign B frontier from B09 to B10/B11.**

- [x] **Step 4: Re-run canonical verification before the first downstream implementation starts.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m torchtitan.experiments.falcon.evidence verify-ledger'
  ```

  Expected: stdout `55`; the ledger digest remains the baseline digest above.

### Task 2: M01 -- Prove and Materialize the RMS/L2 Scale Transform

**Files:**

- Modify: `torchtitan/experiments/falcon/falcon.py`
- Modify: `torchtitan/experiments/falcon/model.py`
- Modify: `torchtitan/experiments/falcon/config_registry.py`
- Create: `tests/unit_tests/test_falcon_scale_transform.py`
- Modify: `tests/unit_tests/test_falcon_kernel.py`
- Modify: `tests/unit_tests/test_falcon_config_registry.py`
- Modify: `torchtitan/experiments/falcon/HARNESS.md`
- Modify: `torchtitan/experiments/falcon/README.md`

**Interfaces:**

- Consumes: existing `falcon_recurrent_forward`,
  `falcon_masked_parallel_forward`, `FalconConfig`, and `apply_arm` seams.
- Produces: `qk_norm_eps: float`, `nlms_denom_eps: float`,
  `scale_compensation: Literal["none", "rms_to_l2"]`, and a separate
  `apply_mechanism_arm(config, arm_id)` mapping for M0-M4.

- [ ] **Step 1: Write a red kernel theorem test at `head_dim=32`.**

  Construct matched M2 and M4 graphs with distinct nonzero base epsilons. Assert
  output equality, `S_l2 == sqrt(32) * S_rms`, and gradients of raw Q, K, V,
  beta, and the base lambda at `rtol=atol=1e-5`. Use a fixed random cotangent;
  form `lambda_rms = 32 * lambda_base` inside autograd.

- [ ] **Step 2: Run the theorem test and require a failure caused by the shared `eps` API.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_scale_transform.py -x'
  ```

- [ ] **Step 3: Split the epsilon roles in both recurrent and masked paths.**

  Default both new arguments to `1e-6`; keep `eps_gamma` unchanged. Reject
  negative or non-finite values with the exact field name. Keep zero valid for
  direct worked examples.

- [ ] **Step 4: Add the single high-level compensation knob.**

  For `d=head_dim`, M4 resolves QK epsilon to `base/d`, multiplies actual
  post-softplus lambda by `d`, and resolves NLMS epsilon to `base*d`. Do not
  scale beta, values, `eps_gamma`, projection parameters, or add a normalization
  multiplier. Reject compensation with a non-Falcon mixer or `phi != "rms"`.

- [ ] **Step 5: Add M0-M4 materialization without changing `_ARM_KNOBS`.**

  ```python
  M0 = dict(alignment="delayed", phi="rms", scale_compensation="none")
  M1 = dict(alignment="same_step", phi="rms", scale_compensation="none")
  M2 = dict(alignment="delayed", phi="l2", scale_compensation="none")
  M3 = dict(alignment="same_step", phi="l2", scale_compensation="none")
  M4 = dict(alignment="delayed", phi="rms", scale_compensation="rms_to_l2")
  ```

  Every M arm also fixes `mixer="falcon"` and `variant="falcon1a"`.

- [ ] **Step 6: Add full-mixer and full-model proof tests.**

  Load identical state dicts into M2/M4 mixers and small LMs with
  `head_dim=32`; compare outputs/logits, input gradients, CE, and every
  parameter gradient at `1e-5`.

- [ ] **Step 7: Add the short deterministic optimizer trajectory.**

  Run matched CPU fp32 models over the same fixed token batches for several
  steps. Compare pre-update losses/gradients and post-update parameter states
  every step. This is deterministic identity evidence, not a quality claim.

- [ ] **Step 8: Verify compatibility and close M01.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_scale_transform.py tests/unit_tests/test_falcon_kernel.py tests/unit_tests/test_falcon_config_registry.py'
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pre-commit run --files torchtitan/experiments/falcon/falcon.py torchtitan/experiments/falcon/model.py torchtitan/experiments/falcon/config_registry.py tests/unit_tests/test_falcon_scale_transform.py tests/unit_tests/test_falcon_kernel.py tests/unit_tests/test_falcon_config_registry.py torchtitan/experiments/falcon/HARNESS.md torchtitan/experiments/falcon/README.md'
  ```

  Gate: any M2/M4 output, adjusted-state, or gradient mismatch blocks every M
  training ticket and returns to this task.

### Task 3: B10 -- Build Uncontaminated Addition Evaluation and One Smoke Attempt

**Files:**

- Modify: `torchtitan/experiments/falcon/addition.py`
- Create: `experiments/falcon/addition_runner.py`
- Modify: `experiments/falcon/run.sh`
- Create: `tests/unit_tests/test_falcon_addition_protocol.py`
- Modify: `experiments/falcon/README.md`

**Interfaces:**

- Consumes: `write_native_attempt_bundle(results_root, record) -> Path` and
  historical addition encoding/masking behavior.
- Produces: immutable `AdditionSplitManifest`, commutation-aware
  `problem_identity(width, a, b)`, `OnlineAdditionStream`, and
  `evaluate_addition_suffixes(...) -> AdditionMetrics`.

- [ ] **Step 1: Write red tests for three disjoint manifests.**

  Each draw contains eight identities at every width 1-24. Draws are mutually
  disjoint under `(width, min(a,b), max(a,b))`; orientations and SHA-256
  manifest digests are deterministic. ID widths are 1-16 and OOD 17-24.

- [ ] **Step 2: Implement manifest generation and strict validation.**

  Persist width, operands, orientation, prompt, target, draw ID, schema version,
  and digest. Reject duplicates within/across draws, wrong width, wrong target,
  or a digest mismatch.

- [ ] **Step 3: Write red tests for the online training stream.**

  For matched seeds, sample only widths 1-16 and reject both orientations of
  every identity registered in all three evaluation draws.

- [ ] **Step 4: Implement suffix metrics and their strata.**

  Return exact suffix and token accuracy as micro and width-macro aggregates,
  least-significant-first output-position accuracy, and outgoing-carry-count
  accuracy. Masks exclude prompts and padding. Carry count includes the
  most-significant operand column's outgoing carry.

- [ ] **Step 5: Implement one-arm/one-seed runner ownership.**

  Reject multiple arms or seeds in one invocation. Write a native attempt for
  success and failure with split/data/source/config/checkpoint lineage and
  artifact digests.

- [ ] **Step 6: Run the A0 smoke path.**

  ```bash
  experiments/falcon/run.sh addition --arm A0 --seed 0 --steps 4 --dry-run
  ```

  Gate: no more than 0.01 summed B200-hours if CUDA is used; the attempt must
  validate with `claim_label="smoke"`.

- [ ] **Step 7: Verify and close B10.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_addition.py tests/unit_tests/test_falcon_addition_protocol.py tests/unit_tests/test_falcon_evidence.py'
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
  ```

### Task 4: B11 -- Build Fixed-Region Read-Only LM Checkpoint Evaluation

**Files:**

- Create: `torchtitan/experiments/falcon/lm_eval.py`
- Create: `experiments/falcon/checkpoint_eval.py`
- Modify: `experiments/falcon/run.sh`
- Create: `tests/unit_tests/test_falcon_lm_eval.py`
- Modify: `experiments/falcon/README.md`

**Interfaces:**

- Consumes: native attempt writer, nanoGPT val-bin reader, and the existing
  A0/A1/A5 `step-20000` checkpoints.
- Produces: `RegionRegistry`, `validate_region_registry`,
  `aggregate_region_ce`, and one-checkpoint evaluation attempts.

- [ ] **Step 1: Freeze three nonoverlapping validation regions.**

  Record source path/digest, token offsets/counts, registry version/digest, and
  token-weighted aggregation. Reject overlaps, out-of-range offsets, duplicate
  IDs, or source digest drift.

- [ ] **Step 2: Write red weighting and lineage tests.**

  Pin `aggregate_ce = sum(region_loss_sum) / sum(region_token_count)` and
  `ppl = exp(aggregate_ce)`. Reject a checkpoint digest mismatch or a region
  result from another registry.

- [ ] **Step 3: Implement immutable checkpoint restore.**

  Evaluate from the historical checkpoint path while writing all logs,
  temporary state, and evidence to a distinct attempt directory. Snapshot the
  historical file metadata/digests before and after and require no mutation.

- [ ] **Step 4: Evaluate the A0 hero checkpoint twice.**

  ```bash
  experiments/falcon/run.sh checkpoint-eval --arm A0 --checkpoint experiments/falcon/results/hero/hero_20k/A0_seed0/checkpoint/step-20000
  ```

  The two attempts share the logical evaluation identity but have distinct
  attempt IDs. Region loss sums/counts/CE and aggregate values must be bitwise
  identical in the declared deterministic environment.

- [ ] **Step 5: Verify and close B11.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_lm_eval.py tests/unit_tests/test_falcon_evidence.py'
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
  ```

### Task 5: M02 -- Build the Observable Single-Attempt Mechanism Runner

**Files:**

- Create: `torchtitan/experiments/falcon/mechanism.py`
- Modify: `torchtitan/experiments/falcon/falcon.py`
- Modify: `torchtitan/experiments/falcon/model.py`
- Create: `experiments/falcon/mechanism_runner.py`
- Modify: `experiments/falcon/run.sh`
- Create: `tests/unit_tests/test_falcon_mechanism.py`
- Modify: `experiments/falcon/README.md`

**Interfaces:**

- Consumes: M0-M4 materialization, native attempt writer, and fixed probe
  batches.
- Produces: one arm/seed attempt per invocation and `MechanismDiagnostics`
  summaries by layer/head.

- [ ] **Step 1: Write red single-identity and milestone tests.**

  Reject more than one arm/seed. Require screen probes at 0, 100, 500, 1000,
  and 2000; confirmation additionally requires 4000 and 8000.

- [ ] **Step 2: Implement diagnostic observations without persistent model state.**

  Per valid token/layer/head aggregate count, mean, standard deviation, and
  p05/p50/p95 for pre/post-normalization key energy, beta, actual lambda, eta,
  gamma, `-1/log(gamma)`, raw/canonical state norm, update norm, read norm, and
  clamp fraction. Exclude delayed sentinel step zero from write summaries.

- [ ] **Step 3: Canonicalize M4 state before comparison.**

  Record both raw `S_rms` and `sqrt(head_dim) * S_rms`; label effective QK and
  denominator epsilons and actual lambda explicitly.

- [ ] **Step 4: Prove instrumentation invariance.**

  With identical fp32 inputs/weights/cotangents, diagnostics on versus off must
  preserve outputs and all gradients at `rtol=atol=1e-5`.

- [ ] **Step 5: Run one non-science smoke attempt and validate its bundle.**

  ```bash
  experiments/falcon/run.sh mechanism --arm M0 --seed 0 --steps 4 --dry-run
  ```

- [ ] **Step 6: Verify and close M02.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_mechanism.py tests/unit_tests/test_falcon_scale_transform.py tests/unit_tests/test_falcon_evidence.py'
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
  ```

### Task 6: B12 -- Project GDN Addition Cost and Decide Run or Omit

**Files:**

- Create: `torchtitan/experiments/falcon/promotion.py`
- Modify: `experiments/falcon/campaign_driver.py`
- Create: `tests/unit_tests/test_falcon_promotion.py`
- Modify: `.scratch/falcon-fast-weight-attention/issues/12-gdn-addition-feasibility.md`

**Interfaces:**

- Consumes: B10 final-shape stream/evaluator and native attempt bundles.
- Produces: machine-readable `GdnAdditionDecision` with `run` or
  `performance_omission` and reproducible cost arithmetic.

- [ ] **Step 1: Write red boundary tests.**

  Exactly 2.0 summed B200-hours is eligible; any larger value is omitted.
  Missing warm steady-state timing, init/eval/checkpoint allowance, three-seed
  coverage, preflight time, failure accounting, or retry accounting cannot
  authorize spend.

- [ ] **Step 2: Implement disk and device-time accounting.**

  Record free bytes, projected peak, retained artifacts, reserve, warmup steps,
  steady-state step samples, per-draw evaluation, checkpoint time, preflight,
  failures, and one allowed retry per logical run. Sum device wall time even
  when launches could run in parallel.

- [ ] **Step 3: Run exactly one 100-step GDN preflight at final shape.**

  ```bash
  experiments/falcon/run.sh addition-preflight --arm A1 --seed 0 --steps 100
  ```

- [ ] **Step 4: Validate and freeze the binary decision.**

  A `run` decision authorizes only the B13 GDN matrix. An omission carries the
  exact projection and reason; no partial GDN result enters the science table.

### Task 7: M03 -- Certify the Five-Arm Actual-Shape Preflight

**Files:**

- Modify: `experiments/falcon/campaign_driver.py`
- Modify: `torchtitan/experiments/falcon/promotion.py`
- Modify: `tests/unit_tests/test_falcon_promotion.py`
- Modify: `.scratch/falcon-fwa-mechanism/issues/03-five-arm-actual-shape-preflight.md`

**Interfaces:**

- Consumes: M02 runner, B11 region registry/evaluator, and M01 theorem proof.
- Produces: binary `MechanismScreenEligibility` decision.

- [ ] **Step 1: Re-run the affected-arm fixed-bank overfit gate.**

  Require CE collapse for the learning configurations and failure to learn for
  the `lr=0` control after all mixer edits.

- [ ] **Step 2: Run every M arm for a few steps at the exact science shape.**

  Fix 4 layers, hidden 256, 8 heads, head dimension 32, sequence 512, and
  16,384 tokens/step. Use matched seed/data order; produce forward/backward,
  fixed-region evaluation, checkpoint, diagnostics, and valid attempt evidence.

- [ ] **Step 3: Validate numerical discriminability.**

  Require finite trajectories; M2/M4 same-input identity; and at least one
  post-update full-precision loss or gradient-norm difference above `1e-6` for
  every non-equivalent arm pair.

- [ ] **Step 4: Project the complete 15-run screen.**

  Include the actual-shape preflight, all 15 logical runs, all probes, all
  three-region evaluations, final model-only checkpoints, failures/retries,
  and safety reserve. Reject a projection above one summed B200-hour.

- [ ] **Step 5: Freeze the eligible/ineligible decision.**

  Ineligible means no M04 launch. Fix code/evidence gaps or request a new human
  budget decision; do not shrink the science shape to make the gate pass.

### Task 8: M04 -- Run the 15-Cell, 2,000-Step Mechanism Screen

**Files:**

- Modify: `experiments/falcon/campaign_driver.py`
- Modify: `torchtitan/experiments/falcon/promotion.py`
- Modify: `tests/unit_tests/test_falcon_promotion.py`
- Write generated attempts under: `experiments/falcon/results/mechanism/screen/`
- Modify: `.scratch/falcon-fwa-mechanism/issues/04-two-thousand-step-screen.md`
- Modify: `.scratch/falcon-fwa-mechanism/map.md`
- Modify: `.scratch/falcon-fwa-mechanism/sdd/progress.md`

**Interfaces:**

- Consumes: eligible M03 decision and immutable single-attempt runner.
- Produces: 15 independent attempts plus `stop`, `walk_back`, or
  `promote(contrast, arms)` decision.

- [ ] **Step 1: Revalidate disk, source/data/region digests, and accumulated GPU time.**

- [ ] **Step 2: Launch M0-M4 at seeds 0, 1, and 2 as separate attempts.**

  ```bash
  experiments/falcon/run.sh mechanism-screen --steps 2000 --seeds 0,1,2 --arms M0,M1,M2,M3,M4
  ```

  Stop before launching the next cell if projected plus consumed device time
  can exceed one B200-hour.

- [ ] **Step 3: Compute only the predeclared paired contrasts.**

  - Scale/normalization: M2-M0 and M4-M0.
  - Scale equivalence: M4-M2.
  - Alignment: M1-M0 and M3-M2.

  The decision unit is each seed's mean of paired fixed-region CE differences.
  A directional effect needs all three seed signs to agree and absolute grand
  mean CE at least 0.01.

- [ ] **Step 4: Apply the M2/M4 practical-equivalence gate.**

  Every seed mean must be in `[-0.01, 0.01]` and the grand mean in
  `[-0.005, 0.005]`. A seam failure or training mismatch outside this margin is
  `walk_back`, not evidence for a distinct mechanism.

- [ ] **Step 5: Emit the terminal screen decision.**

  If M2 and M4 are equivalent and both improve over M0 by the directional
  rule, close normalization as parameterization/preconditioning and do not run
  a larger normalization comparison. Promote an alignment contrast only when
  M1-M0 or M3-M2 passes. Promote only the minimum arms needed. Otherwise stop
  with unresolved/null/over-budget/incomplete status.

### Task 9: M05 -- Confirm a Surviving Contrast or Close Without a Run

**Files:**

- Modify: `experiments/falcon/campaign_driver.py`
- Modify: `torchtitan/experiments/falcon/promotion.py`
- Modify: `tests/unit_tests/test_falcon_promotion.py`
- Write generated attempts under: `experiments/falcon/results/mechanism/confirmation/`
- Modify: `.scratch/falcon-fwa-mechanism/issues/05-confirm-or-stop.md`
- Modify: `.scratch/falcon-fwa-mechanism/map.md`
- Modify: `.scratch/falcon-fwa-mechanism/sdd/progress.md`

**Interfaces:**

- Consumes: signed M04 decision and immutable screen bundles.
- Produces: terminal mechanism classification and the external gate for B13.

- [ ] **Step 1: Validate that the requested arms exactly match the M04 promotion.**

  Reject new/replacement arms, changed shapes, data, optimizer, evaluator,
  metrics, thresholds, or seeds.

- [ ] **Step 2: Take the no-run branch when no contrast promoted.**

  Emit a complete terminal decision with zero confirmation GPU time and
  classification `parameterization/preconditioning`, `unresolved`, or
  `implementation_walk_back` as supported by M04.

- [ ] **Step 3: Otherwise run only promoted arms for 8,000 steps at three seeds.**

  Require the same sign at every seed, retention of the screen direction, and
  grand-mean magnitude at least 0.01 CE. Sum all confirmation device time and
  stop at the three-B200-hour ceiling.

- [ ] **Step 4: Classify the final mechanism result.**

  Use exactly one of: `parameterization_preconditioning`, `alignment_effect`,
  `non_replication`, or `implementation_walk_back`. State diagnostics explain
  behavior but cannot override the scale theorem or CE gates.

### Task 10: B13 -- Run Repaired Campaign B Evaluation and Close the Campaign

**Files:**

- Modify: `experiments/falcon/campaign_driver.py`
- Create: `tests/unit_tests/test_falcon_closeout.py`
- Write generated attempts under: `experiments/falcon/results/closeout/`
- Create: `experiments/falcon/results/closeout/REPORT.md` (generated compact report)
- Modify: `.scratch/falcon-fast-weight-attention/issues/13-campaign-b-closeout.md`
- Modify: `.scratch/falcon-fast-weight-attention/map.md`
- Modify: `.scratch/falcon-fast-weight-attention/sdd/progress.md`
- Create: `.scratch/falcon-fast-weight-attention/sdd/task-13-report.md`

**Interfaces:**

- Consumes: B10 addition seam, B11 LM evaluator, B12 GDN decision, terminal M05
  decision, native attempts, and immutable hero checkpoints.
- Produces: validator-backed Campaign B closeout with facts, inferences,
  hypotheses, omissions, and source links.

- [ ] **Step 1: Write red full-matrix validator tests.**

  Require A0/A5 seeds 0-2, all three addition draws per checkpoint, all three LM
  regions for A0/A5/A1 heroes, unique logical runs, matched configs/data order,
  valid claim labels, uncontaminated splits, complete time accounting, at most
  one retry per logical run, and a justified GDN run or omission.

- [ ] **Step 2: Preflight the complete A0/A5 addition matrix.**

  Project preflight, six 2,000-step training runs, all evaluations, probes,
  model-only final checkpoints, failed attempts/retries, and disk reserve. A
  projection above one summed B200-hour requires a new human decision and
  launches nothing.

- [ ] **Step 3: Train A0 and A5 addition cells.**

  ```bash
  experiments/falcon/run.sh addition-matrix --arms A0,A5 --seeds 0,1,2 --steps 2000
  ```

  Use matched deterministic online streams that exclude every evaluation
  identity. The final checkpoint is predeclared; evaluation cannot select it.

- [ ] **Step 4: Run or omit GDN exactly as B12 decided.**

  If authorized, use the same seeds, steps, streams, draws, and metrics. If
  omitted, include the machine-readable performance omission verbatim.

- [ ] **Step 5: Evaluate all final addition checkpoints on all three draws.**

  Report exact suffix, suffix token, width-macro, output-position, and
  carry-count metrics. Label teacher-forced evaluation explicitly. If all OOD
  exact-suffix values are zero, close with `no_transfer`; this is a completed
  negative result, not an incomplete gate.

- [ ] **Step 6: Evaluate A0, A5, and A1 hero checkpoints on the same LM regions.**

  A5 passes local LM no-regression only when aggregate A5 CE is no more than
  0.01 worse than aggregate A1 CE. Otherwise record `lm_regression`. Preserve
  A0 as the softmax reference; do not reinterpret single-seed heroes as
  replicated training.

- [ ] **Step 7: Validate budgets, coverage, claims, and provenance.**

  Reject any missing seed/region/draw, duplicate logical run, contaminated
  identity, config mismatch, excess device time, excess retry, invalid claim,
  mutated historical artifact, or unexplained omission.

- [ ] **Step 8: Publish the closeout and terminate the current research line.**

  Separate facts, inferences, hypotheses, and omissions. Link every number to
  a native attempt/import record. State that this is local declared-scale
  evidence, not a paper reproduction. The M05 result may recommend a new
  paper-alignment design gate, but B13 does not authorize it.

### Task 11: Whole-Campaign Verification and Handoff

**Files:**

- Verify all changed Falcon source, tests, docs, tickets, maps, and reports.

**Interfaces:**

- Consumes: resolved B10-B13 and M01-M05 reports.
- Produces: a reproducible final handoff with no hidden active ticket.

- [ ] **Step 1: Run canonical legacy and native-evidence validation.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m torchtitan.experiments.falcon.evidence verify-ledger'
  ```

- [ ] **Step 2: Run the complete Falcon suite.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
  ```

- [ ] **Step 3: Run changed-file lint and whitespace validation.**

  ```bash
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pre-commit run --all-files'
  git diff --check
  ```

- [ ] **Step 4: Reconcile every generated number to an attempt/import ID.**

  Require the campaign validator to reproduce the final tables from bundles,
  the maps to name the terminal decisions, and both progress ledgers to have no
  falsely active or silently blocked ticket.

## Scheduling and Compute Order

The safe implementation order is B09 baseline -> M01 -> B10 -> B11 -> M02 ->
B12 -> M03 -> M04 -> M05 -> B13. M01, B10, and B11 are logically independent
after B09, but implement them sequentially because they share Falcon code and
require isolated review gates. No science GPU work occurs before M01, B10,
B11, M02, and the applicable validator tests are green.

The first material science spend is M04. B12 spends only the declared 100-step
GDN feasibility preflight. B13's addition matrix waits for the terminal M05
decision so the final Campaign B report can incorporate the mechanism result
without reopening its frozen hero.

## Predeclared Research Decision Tree

```text
M2/M4 deterministic seam fails
  -> implementation walk-back; no screen

Actual-shape preflight incomplete, non-finite, collapsed, over disk, or >1 B200h
  -> ineligible; no screen

2k screen M2/M4 outside practical margin
  -> implementation walk-back; no new "L2 geometry" claim

2k M2 ~= M4 and both beat M0 under directional rule
  -> A5 advantage is parameterization/preconditioning; stop normalization branch

M1-M0 or M3-M2 passes all-three-signs + |mean CE| >= 0.01
  -> confirm only that alignment contrast at 8k

No contrast passes
  -> M05 no-run closeout

8k fails to retain sign or |mean CE| >= 0.01
  -> non-replication

Campaign B repaired addition OOD exact suffix remains zero
  -> completed `no_transfer` closeout

Only after M05+B13 terminal reports
  -> decide whether a new, separately specified paper-alignment campaign is warranted
```

## Expected End State

This plan ends with two honest results, not another hero: (1) a mechanism
classification of the A5 normalization/alignment behavior under matched seeds
and fixed regions, and (2) a corrected Campaign B transfer/no-regression table
whose addition examples are truly held out. Any Falcon-3A or exact paper-family
work starts as a new design/spec decision after these closeouts.
