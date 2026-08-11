# Follow-Up Experiment Plan

This plan turns the Countdown pilot and the benchmark landscape into a staged
experiment program. The goal is to test whether verified scaffold behavior can
be compressed into policy behavior, then determine how far that transfer holds
across reasoning and later agentic tasks.

The pilot result is promising but not yet broad enough to justify benchmark
expansion claims. The reliable finding is that LoRA SFT from verified Countdown
traces improved Qwen3-1.7B on a non-overlapping OOD target-range split, with the
`clean` arm strongest. The same-regime dev and IID splits overlapped train and
must be repaired before they support held-out claims.

## Design Principles

- Repair split hygiene before adding new benchmark families.
- Keep all real runs inside the bwrap rootfs.
- Treat best-of-N sampling as the first scaffold; tool-assisted scaffolds are a
  separate condition.
- Preserve upstream benchmark semantics rather than replacing them with generic
  judge scores.
- Label every result as model plus scaffold plus training arm plus harness plus
  environment, not as a model-only score.
- Prefer exact verifiers until the registry and reporting contract are stable.

## Experiment Phases

| Phase | Primary question | Task family | Arms | Promotion gate |
| --- | --- | --- | --- | --- |
| 0 | Was the Countdown pilot signal real after split repair? | Countdown | base, raw, clean | clean held-out dev/IID/OOD results with no train overlap |
| 1 | What mechanism helped in Countdown? | Countdown ablations | raw, clean, formatting, optional negative-filtered | failure modes improve without pass@32 collapse |
| 2 | Does the method transfer to adjacent exact-verifier reasoning? | synthetic reasoning | base, raw, clean, formatting | held-out pass@1 lift across at least two split draws |
| 3 | Does the method transfer to public reasoning distributions? | GSM-style and small MATH | base plus best 2-3 arms | replicated lift with stable verifier validity |
| 4 | Can external agentic harnesses run hermetically? | Harbor/Terminal-Bench and tau2-bench smokes | base first | rootfs compatibility and artifact ingestion only |
| 5 | Does the trained policy help agentic tasks? | pinned agentic suites | base plus best adapter | labeled harness results with upstream metrics |

Phase 4 is not a scientific benchmark phase. It only proves that the bwrap
rootfs, harness pins, artifact paths, and ingestion scripts work.

## Phase 0: Countdown Split Repair

Build the clean evidence base before any new benchmark claims.

Current implementation status:

- `generate-pool` accepts `--exclude-problems` so later splits can avoid earlier
  split problem keys even when the seed is reused.
- `validate-splits` writes a split registry JSON and fails when any `Problem Key`
  appears in more than one split.
- `run_collect.sh` passes train exclusions into dev, train/dev exclusions into
  IID, and train/dev/IID exclusions into OOD.
- `run_full_pilot.sh` runs split validation after collection for reduced and
  full modes.
- The current stale full-run artifacts fail validation with 1000 overlapping
  keys, which matches the known pilot caveat.
- A rootfs smoke under `results/split_registry_smoke/` generated disjoint train,
  dev, IID, and OOD problem files and passed `validate-splits`.

Required implementation:

- canonical split registry under the Countdown experiment data root;
- global `Problem Key` de-duplication across train, dev, IID, and OOD;
- split-specific seeds or deterministic non-overlapping draw ranges;
- preflight command that fails if any overlap is present;
- rerun of base and exported adapters on clean dev/IID/OOD splits.

Required report tables:

- split sizes and overlap validation hashes;
- pass@1, pass@2, pass@4, pass@8, pass@16, pass@32;
- easy, elicitable, and unreached buckets;
- base-elicitable subset metrics;
- verifier failure breakdown;
- representative wins, regressions, and unchanged examples.

Promotion gate:

- no train/dev/IID exact problem-key overlap;
- at least one trained arm improves clean held-out pass@1 by 5 absolute points;
- pass@32 drops by no more than 2 absolute points, unless the report explains a
  deliberate sampling-efficiency tradeoff.

## Phase 1: Countdown Mechanism Ablations

Keep the task fixed and test why the pilot worked.

Arms:

- `raw`: train on all verified traces that pass the verifier.
- `clean`: train on normalized verified traces.
- `formatting`: emphasize valid `FINAL:` production and answer extraction.
- `negative-filtered`: optional arm that excludes brittle, overlong, or
  low-margin traces.

Primary analysis:

- format validity rate;
- missing-final failures;
- unavailable operand failures;
- wrong-final failures;
- non-exact division failures;
- invalid intermediate failures;
- output length and reasoning length;
- pass@k curve shape.

Promotion gate:

- one arm shows a clear mechanism-specific improvement;
- the result replicates across at least two training seeds or split draws;
- the best arm is simple enough to port to the reasoning lane.

## Phase 2: Synthetic Exact-Verifier Reasoning

Use local generated tasks to preserve Countdown's exact-verifier discipline while
expanding beyond arithmetic expression search.

Candidate first tasks:

- multi-step arithmetic word problems with generated rationales and exact
  numeric answers;
- constraint satisfaction puzzles with unique final answers;
- symbolic transformation tasks with deterministic normalization;
- small grid or table reasoning tasks with exact structured outputs.

The first task should be cheap to generate, cheap to verify, and close enough to
Countdown that failures are interpretable.

Required registry additions:

- task generator commit or source hash;
- split manifest hash;
- verifier name and version;
- output format contract;
- scaffold rollout budget;
- training arm list;
- artifact roots.

Promotion gate:

- at least two clean split draws;
- held-out pass@1 lift over base;
- stable or improved verifier validity rate;
- representative examples showing genuine reasoning transfer, not only format
  compliance.

## Phase 3: Public Reasoning Benchmarks

After synthetic transfer is visible, add small public reasoning distributions.

Initial order:

1. GSM8K or GSM-style subset with parsed final numeric answers.
2. Small MATH subset with answer normalization.
3. Optional AIME-style integer-answer subset after the verifier and reporting
   loop are stable.

Design constraints:

- no tools in the first condition;
- no benchmark-specific prompt hacks unless reported as a separate condition;
- keep public train/dev/test provenance in the registry;
- report contamination and split provenance assumptions explicitly;
- keep answer normalization code test-covered.

Promotion gate:

- replicated held-out lift on at least one public reasoning distribution;
- no loss of validity from malformed final answers;
- evidence that improvement is not limited to Countdown-specific notation.

## Phase 4: Agentic Harness Feasibility

This phase is about infrastructure, not capability claims.

Harbor/Terminal-Bench smoke:

- install and run through the bwrap rootfs;
- pin Harbor and Terminal-Bench revisions;
- execute one tiny task or dry-run equivalent;
- ingest final score, logs, task metadata, and environment metadata;
- record container image or task environment digest when available.

tau2-bench smoke:

- install and run through the bwrap rootfs;
- pin tau2-bench revision and domain/task subset;
- run one tiny interaction or dry-run equivalent;
- ingest trajectory, final state score, and harness metadata.

TorchTitan owns rootfs launch, registry entries, artifact paths, and result
ingestion. Harbor, Terminal-Bench, and tau2 own task execution and scoring.

Promotion gate:

- rootfs run is reproducible from a single command;
- artifacts land under the registry-declared root;
- scores and trajectories can be rendered into the shared report format;
- no model-performance claim is made from the smoke.

## Phase 5: Agentic Pilot

Run only after the reasoning lane shows replicated held-out gains.

First scientific agentic pilot:

- choose one small pinned Harbor/Terminal-Bench subset and one small tau2-bench
  subset;
- evaluate base first;
- evaluate only the best reasoning adapter unless the registry can carry a full
  arm matrix cleanly;
- report benchmark, harness, model, adapter, tools, and rootfs version as one
  evaluation object.

Required outputs:

- upstream metric;
- environment errors;
- timeout rate;
- token use;
- wall-clock time;
- action or tool-call count;
- retry count;
- trajectory references;
- representative successful and failed tasks.

Promotion gate:

- adapter improves a task-relevant metric without increasing environment or
  policy failures;
- failures are classifiable enough to motivate the next scaffold or training
  arm;
- the result survives at least one rerun or matched task subset.

## Initial Ticket Order

1. Implement Countdown split registry and overlap validator.
2. Regenerate clean Countdown splits and rerun base plus exported adapters.
3. Add Countdown failure-mode aggregation and examples to the report.
4. Add the formatting arm and test its verifier-validity effect.
5. Add a synthetic exact-verifier reasoning task with registry metadata.
6. Add GSM-style answer normalization and a tiny fixture-backed smoke.
7. Add rootfs-managed Harbor/Terminal-Bench smoke ingestion.
8. Add rootfs-managed tau2-bench smoke ingestion.

Do not start agentic benchmark claims until tickets 1-6 have produced clean
held-out evidence.

