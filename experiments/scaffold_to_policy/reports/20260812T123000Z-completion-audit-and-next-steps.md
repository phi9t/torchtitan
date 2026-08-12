# Scaffold-To-Policy Completion Audit And Next Steps

Audit timestamp: 2026-08-12T12:30:00Z

This audit maps the current `experiments/scaffold_to_policy/spec.md` contract to
artifacts that exist in the checkout. It is intentionally conservative: a smoke,
fixture, preflight, or blocker probe is not treated as a benchmark capability
result.

## Executive Status

Overall status: incomplete, with substantial reasoning and coding plumbing
landed.

Completed or materially advanced:

- Countdown clean and formatting transfer replications ran under the bwrap
  rootfs and produced reports.
- Local exact-verifier reasoning transfer ran on `modular_sequences` and showed
  a positive small-run signal.
- Public no-tool reasoning smokes exist for GSM8K, MATH algebra, AIME, and
  ARC-AGI-2-style exact grids.
- Public executable coding smokes exist for HumanEval, MBPP, and
  BigCodeBench-Hard.
- External harness dry-run, package preflight, tau2 scorer-ingestion, and
  tau2 upstream execution-probe blocker evidence, and Terminal-Bench/Harbor
  Docker-boundary probes exist.

Not complete:

- GPQA Diamond still requires Hugging Face gated dataset credentials.
- Terminal-Bench/Harbor full execution is blocked by missing Docker or an
  equivalent Harbor backend inside the hermetic rootfs.
- tau2-bench full agent execution is incomplete; package, loader, fixture
  scorer ingestion, and upstream `tau2 run` launch have cleared, but the first
  upstream execution probe produced one infra error and zero evaluated tasks.
- Public benchmark smokes are small calibration runs and are not leaderboard
  claims.
- The registry/reporting surface still needs broader latest-row selection,
  stronger fresh/reused stage surfacing, and more offline dataset-loader
  hardening.

## User Story Audit

| Spec stories | Status | Evidence | Remaining gap |
| --- | --- | --- | --- |
| 1-8 run identity, manifests, provenance, environment | Partial | Countdown report inputs, scaffold smoke report inputs, rootfs shell entrypoints, external harness metadata | Manifest stage freshness is stronger in Countdown than in the general scaffold lane; canonical latest-row selection is still not generalized. |
| 9-14 strict format, failure modes, pass@k, buckets | Mostly complete | Countdown reports; arithmetic, modular, GSM, MATH, ARC, and coding summaries | Coding tasks do not have strict final-format metrics because their output contract is executable code rather than `FINAL:` answers. |
| 15-18 Countdown champion, formatting arm, replication, sweeps | Complete for current checkpoint | Clean-arm rank/size sweep and formatting replication reports under `experiments/countdown_search_distill/reports/` | Further promotion should use repeated seeds and larger target tasks, not this audit alone. |
| 19-20 bwrap rootfs and entrypoints | Mostly complete | All current real Python/GPU benchmark scripts re-exec through `scripts/rootfs/enter_rootfs.sh` | Docker-backed Harbor execution still lacks a hermetic backend. |
| 21-22 local generated reasoning tasks | Complete for initial lane | `arithmetic_words`, `modular_sequences`, and modular transfer reports | Larger modular transfer replication is still needed before adapter-science claims. |
| 23-25 public no-tool reasoning semantics | Mostly complete | GSM8K, MATH, AIME, ARC-AGI-2 scripts and reports | GPQA Diamond is blocked by auth; all public runs are too small for public benchmark claims. |
| 26-29 external harness boundaries and labels | Partial | `external_harness` module, dry-run/preflight/tau2/Terminal-Bench reports, tau2 execution-probe ingestion | Full Terminal-Bench/Harbor and successful tau2 task evaluation remain incomplete. |
| 30 generated artifact hygiene | Complete for checked tree | Generated results/data roots are ignored; current tracked edits are code/docs/tests only | Continue to avoid committing runtime results. |
| 31-36 future-agent spec, gates, caveats, examples, exact verifiers, upstream metrics | Mostly complete | `spec.md`, README, reports with examples, exact verifiers, external harness notes | Need a shared registry abstraction for all scaffold lanes rather than per-lane report builders. |

## Harder Reasoning And Coding Status

ARC-AGI-2 exact-grid smoke:

- Run ID: `20260812T090000Z-arc-agi2-public-vllm-smoke`
- Source: `https://github.com/arcprize/ARC-AGI-2.git` at
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`
- Status: completed smoke, not benchmark claim
- Result: dev pass@1 `0.000`, dev pass@2 `0.500`; OOD pass@1/pass@2 `0.000`
- Main failure: final-grid JSON format misses and wrong grids
- Follow-up: the larger `chat` calibration reached dev pass@4 `0.125`; the
  completed `strict_chat` calibration reached dev/OOD pass@1/pass@4 `0.000`.
  `strict_chat` removed missing-final failures but mostly produced wrong grids,
  so it is not a better ARC default.

BigCodeBench-Hard executable smoke:

- Run ID: `20260812T091500Z-bigcodebench-hard-public-vllm-smoke`
- Source: `bigcode/bigcodebench-hard`, split `v0.1.4`
- Status: completed smoke after rootfs dependency repair, not benchmark claim
- Result: dev/OOD pass@1/pass@2 `0.000`
- Main failure after repair: benchmark-test assertion failures
- Infra finding: the first run exposed missing rootfs `matplotlib`; that was an
  environment failure discovered too late, during rollout scoring.
- Follow-up completed: `20260812T173000Z-bigcodebench-hard-public-vllm-expanded`
  expanded to 8 dev and 8 OOD tasks with 4 rollouts per problem. Both selected
  splits passed canonical preflight 8/8 after rootfs dependency repair, and the
  model reached dev/OOD pass@1/pass@4 `0.000`. All 64 sampled candidates failed
  released unit tests as assertion failures.

## New Hardening Landed From This Audit

This pass adds a canonical-solution preflight for executable coding splits:

- `preflight-coding-style-canonical` loads imported coding problems and runs
  each released canonical solution through the same executable verifier used for
  model candidates.
- The preflight writes a JSON artifact with per-problem success, error,
  returncode, stdout, stderr, and aggregate failure breakdown.
- `build-coding-style-report-input` accepts optional split preflight artifacts
  and validates that they are present, count-matched, and selected.
- `run_bigcodebench_hard_public_vllm_smoke.sh` now runs the canonical preflight
  for dev and OOD splits before vLLM generation and includes the artifacts in
  the report input.

This turns missing Python packages, broken released tests, and canonical
solution import failures into early infrastructure failures instead of
post-generation scoring surprises.

## Blocker Backlog

1. GPQA Diamond:
   - Blocker: gated Hugging Face dataset auth.
   - Required next step: run the existing rootfs command with an authenticated
     HF token and record the exact dataset revision and provenance.

2. Terminal-Bench/Harbor:
   - Progress: the bwrap rootfs now has an explicit opt-in host Docker and
     Compose passthrough. Rootfs checks showed Docker client/server `26.1.4`
     and Docker Compose `v2.27.1`.
   - Current blocker: Harbor reaches Docker Compose container creation, but the
     latest probe records one RuntimeError, zero evaluated trials, and a missing
     verifier bind source path under the trial directory.
   - Required next step: diagnose Harbor's verifier-mount lifecycle for the
     pinned Terminal-Bench 2.1 `task.toml` corpus, then rerun
     `harbor run --path <taskdir> ... --env docker --agent oracle` and require
     Harbor `result.json` to show nonzero trials with zero errors.

3. tau2-bench:
   - Blocker: the first upstream `tau2 run` execution probe recorded
     `termination_reason: infrastructure_error`, zero evaluated tasks, and
     `DummyUser.__init__() got an unexpected keyword argument 'tools'`.
   - Required next step: add a minimal local model-provider or deterministic
     benchmark-agent execution path that produces a non-fixture tau2 trajectory
     and scores it with tau2's official evaluator.

4. Registry/reporting:
   - Blocker: scaffold lanes still have per-task report builders rather than a
     single general registry.
   - Required next step: factor the common split/report/preflight checks into a
     shared registry module after the current task-specific behaviors stabilize.

## Promotion Guidance

The next benchmark expansion should remain reasoning-first:

1. Improve BigCodeBench-Hard prompting or add a repair pass, then rerun the same
   preflight-clean 16-task slice before expanding further.
2. Expand ARC-AGI-2 from smoke to a larger calibration slice once the final-grid
   prompt/parser failure is characterized.
3. Add GPQA Diamond only after HF auth is available, preserving no-tool multiple
   choice semantics.
4. Move to Terminal-Bench/Harbor and tau2 full execution only after their
   environment blockers are resolved; until then, report them as harness
   readiness work only.
