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
- External harness dry-run, package preflight, tau2 scorer-ingestion, a
  successful deterministic tau2 upstream execution probe, tau2 noop baseline, and
  Terminal-Bench/Harbor Docker-boundary plus non-oracle `nop` baseline probes
  exist.

Not complete:

- GPQA Diamond still requires Hugging Face gated dataset credentials.
- Terminal-Bench/Harbor oracle execution and the `nop` non-oracle baseline now
  work through the bwrap rootfs with explicit host Docker passthrough and
  host-path repo binding, but no model or scaffold-policy agent has been run.
- tau2-bench full model or learned-policy execution is incomplete; package,
  loader, fixture scorer ingestion, deterministic oracle-style execution, and
  noop baseline execution have cleared, but no Qwen3 or scaffold-policy tau2
  agent has been run.
- Public benchmark smokes are small calibration runs and are not leaderboard
  claims.
- The registry/reporting surface still needs broader latest-row selection and
  more offline dataset-loader hardening. Shared report-input factoring and
  fresh/reused artifact surfacing now cover the main exact-verifier scaffold
  lanes.

## User Story Audit

| Spec stories | Status | Evidence | Remaining gap |
| --- | --- | --- | --- |
| 1-8 run identity, manifests, provenance, environment | Partial | Countdown report inputs, shared scaffold report inputs, rootfs shell entrypoints, external harness metadata, report artifact details | Manifest stage freshness is stronger in Countdown than in the general scaffold lane; canonical latest-row selection is still not generalized. |
| 9-14 strict format, failure modes, pass@k, buckets | Mostly complete | Countdown reports; arithmetic, modular, GSM, MATH, ARC, and coding summaries | Coding tasks do not have strict final-format metrics because their output contract is executable code rather than `FINAL:` answers. |
| 15-18 Countdown champion, formatting arm, replication, sweeps | Complete for current checkpoint | Clean-arm rank/size sweep and formatting replication reports under `experiments/countdown_search_distill/reports/` | Further promotion should use repeated seeds and larger target tasks, not this audit alone. |
| 19-20 bwrap rootfs and entrypoints | Mostly complete | All current real Python/GPU benchmark scripts re-exec through `scripts/rootfs/enter_rootfs.sh`; Harbor can run through opt-in host Docker passthrough | Harbor still depends on host Docker passthrough rather than a fully rootfs-contained backend. |
| 21-22 local generated reasoning tasks | Complete for initial lane | `arithmetic_words`, `modular_sequences`, and modular transfer reports | Larger modular transfer replication is still needed before adapter-science claims. |
| 23-25 public no-tool reasoning semantics | Mostly complete | GSM8K, MATH, AIME, ARC-AGI-2 scripts and reports | GPQA Diamond is blocked by auth; all public runs are too small for public benchmark claims. |
| 26-29 external harness boundaries and labels | Partial | `external_harness` module, dry-run/preflight/tau2/Terminal-Bench reports, tau2 execution-probe ingestion | Terminal-Bench/Harbor `nop` baseline and tau2 noop baseline execution are green; model or learned-policy execution remains incomplete. |
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
- Follow-up completed: the `contract_chat` prompt condition now has parser and
  runner support plus consistent `GPU_MEMORY_UTILIZATION` forwarding into
  vLLM. Run `20260812T235500Z-bigcodebench-hard-contract-chat-rerun` completed
  a 2 dev / 2 OOD slice after memory became available at
  `GPU_MEMORY_UTILIZATION=0.24`. Both splits passed canonical preflight, and
  both reached pass@1/pass@4 `0.000`; all 16 sampled candidates failed released
  unit tests as assertion failures.
- Follow-up partially blocked: the expanded 8 dev / 8 OOD `contract_chat`
  attempt selected all canonical solutions after increasing
  `TIMEOUT_SECONDS=30`. The dev half completed with pass@1/pass@4 `0.000` over
  8 problems and 32 candidates. The OOD half did not complete because shared GPU
  memory changed after a successful preflight: vLLM later saw only 6.6 GiB free
  versus 21.4 GiB requested at `GPU_MEMORY_UTILIZATION=0.12`. This is an
  infrastructure-blocked partial expanded result, not a completed expanded
  score.

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

This pass also adds machine-readable artifact freshness for the hard public
reasoning/coding report inputs:

- `report_artifacts.describe_artifact` records path, existence, size, sha256,
  mtime, and run binding for report inputs.
- `report_artifacts.build_report_input` now owns the common report shell:
  split-registry validation, summary count matching, optional preflight count
  checks, artifact details, freshness summaries, and the
  `artifact_provenance_labeled` check.
- `arithmetic_words`, `gsm_style`, `math_style`, `multiple_choice`,
  `arc_grid`, `coding_style`, and `modular_sequences` now use that shared
  builder while retaining task-owned verifier metadata and analysis sections.
- Freshness is conservative: artifacts are labeled `fresh` only when the run ID
  appears in the artifact path or JSON payload; otherwise they are labeled
  `reused_or_unscoped` rather than failed.

## Blocker Backlog

1. GPQA Diamond:
   - Blocker: gated Hugging Face dataset auth.
   - Required next step: run the existing rootfs command with an authenticated
     HF token and record the exact dataset revision and provenance.

2. Terminal-Bench/Harbor:
   - Progress: the bwrap rootfs now has explicit opt-in host Docker and Compose
     passthrough plus a host-path repo alias for Docker bind mounts.
   - Oracle result: `20260812T190000Z-terminal-bench-harbor-hostpath`
     completed one pinned `headless-terminal` oracle trial with Harbor
     `n_trials=1`, `n_errors=0`, mean metric `1.0`, verifier reward `1.0`,
     and `task_execution_probes_succeeded=true`.
   - Baseline result:
     `20260812T111500Z-terminal-bench-harbor-nop-baseline` completed one
     non-oracle Harbor `nop` trial with `n_trials=1`, `n_errors=0`, no trial
     exceptions, mean metric `0.0`, `task_execution_probes_completed=true`,
     and `task_execution_probes_succeeded=false`.
   - Required next step: replace `nop` with a bounded model or scaffold-policy
     agent while preserving Harbor's released task execution and verifier
     semantics.

3. tau2-bench:
   - Progress: `20260812T105500Z-tau2-deterministic-execution-probe` registers
     a non-solo deterministic tau2 agent and static user before calling
     upstream `tau2.cli.main`; tau2 writes an official `results.json` with
     `num_evaluated=1`, `num_infra_errors=0`, and scaffold report
     `task_execution_probes_succeeded=true`.
   - Baseline result: `20260812T113000Z-tau2-noop-baseline` runs
     `torchtitan_noop_agent` through the same upstream tau2 runner and evaluator
     and records `num_evaluated=1`, `num_infra_errors=0`, average reward `0.0`,
     `task_execution_probes_completed=true`, and
     `task_execution_probes_succeeded=false`.
   - Current blocker: neither path is Qwen3, an adapter, or a learned scaffold
     policy.
   - Required next step: replace the noop or oracle behavior with a bounded
     model or policy agent while preserving tau2's released task state, runner,
     and evaluator semantics.

4. Registry/reporting:
   - Progress: the main exact-verifier scaffold lanes now share report-input
     split checks, optional preflight checks, artifact hashes, and freshness
     labels through `report_artifacts.build_report_input`.
   - Remaining gap: canonical latest-row selection is still not generalized,
     and external-harness report inputs are separate because their semantics are
     not split-summary based.
   - Required next step: add latest-run row selection and decide whether the
     external-harness report shape should reuse only artifact provenance helpers
     or a separate harness-specific shared builder.

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
