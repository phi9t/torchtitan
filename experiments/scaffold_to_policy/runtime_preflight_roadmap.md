# Runtime Contract And Preflight Roadmap

This document turns the current runtime doctor into an executable contract for
all scaffold-to-policy experiment lanes. The goal is that every real result can
be traced back to a rootfs-managed preflight artifact that proves the required
runtime, model assets, harnesses, and benchmark semantics were available before
expensive work started.

## Current State

The first runtime contract doctor is available through:

```bash
experiments/scaffold_to_policy/run_runtime_doctor.sh
```

It re-enters the bwrap rootfs through `scripts/rootfs/enter_rootfs.sh`, sets the
repo-local Python and Hugging Face environment, and writes:

```text
experiments/scaffold_to_policy/results/runtime_doctor/manifests/runtime_doctor_<run_id>.json
```

The doctor currently checks:

- rootfs activation through `TORCHTITAN_IN_ROOTFS=1`;
- required Python imports: `torch`, `vllm`, `datasets`, `transformers`,
  `spmd_types`;
- local Qwen3 model assets with config, tokenizer, and safetensors files;
- CUDA visibility and minimum GPU count;
- CUDA memory-query success;
- selected-device vLLM memory headroom;
- repo-visible `HF_HOME` and `HF_HUB_CACHE`;
- optional external harness executables on `PATH`.

The doctor JSON has one clause per contract item, a top-level `selected` flag,
and `summary.failed` for triage. With the default `--require-selected`, failed
required clauses stop the command.

## Contract Boundary

The runtime contract applies to real setup, generation, training, evaluation,
external harness execution, and benchmark scoring. Those paths must run through
the bwrap rootfs and must produce a doctor artifact before they produce a score,
checkpoint, trajectory, or report input.

The contract does not need to gate cheap host-side editing, static documentation
work, shell syntax checks, or unit tests that intentionally exercise parser-only
behavior. If a command touches model execution, GPU state, external benchmark
harnesses, or benchmark scoring, it must use the contract.

## Lane Profiles

The generic doctor should stay small. Lane-specific requirements should be
expressed as profile wrappers or profile arguments that feed the same doctor and
then add benchmark-semantic checks.

| Profile | Runtime requirements | Extra checks |
| --- | --- | --- |
| `reasoning` | rootfs, packages, model assets, CUDA, vLLM memory, HF cache | dataset cache, prompt format, answer parser, metric config |
| `coding` | rootfs, packages, model assets, CUDA/vLLM when model-backed, HF cache | `git`, `patch`, test runner, timeout policy, sandbox temp dirs |
| `terminal_bench` | rootfs, packages, optional model/vLLM, HF cache | Harbor or Terminal-Bench executable, container backend, task images, official verifier |
| `tau2` | rootfs, packages, optional model/vLLM, HF cache | tau2 harness, simulator domain, policy files, state reset, final-state metric |
| `docs_only` | rootfs when invoking repo Python, otherwise no model/CUDA requirement | benchmark manifest readability and report schema only |

The profile name should appear in every preflight artifact and result manifest.
Disabling CUDA, model assets, or vLLM memory is allowed only when the profile
explicitly does not execute a model.

## Runner Integration

Every real runner should call the doctor before expensive stages. The first
integration target is the scaffold-to-policy shell runners:

- `run_full_pilot.sh`;
- `run_harder_reasoning.sh`;
- `run_coding_benchmarks.sh`;
- Terminal-Bench and Harbor wrappers;
- tau2 wrappers.

The runner should record the doctor artifact path in its stage manifest and in
the final report input. A failed doctor should be reported as
`runtime_contract_failure`, not as a benchmark score of zero.

Recommended runner flow:

```text
enter rootfs
setup repo-local environment
write run manifest
run runtime doctor
run benchmark-specific preflight
run generation, training, evaluation, or harness execution
write result manifest and report input
```

## Unified Preflight Entrypoint

Add a single command for the state of the machine:

```bash
experiments/scaffold_to_policy/run_preflight.sh
```

It should run the generic runtime doctor plus lane-specific preflights selected
by environment variables or arguments. Its output should answer which lanes are
executable now.

Example output shape:

```json
{
  "schema_version": 1,
  "kind": "scaffold_to_policy_preflight",
  "run_id": "20260813T-preflight",
  "rootfs_active": true,
  "lanes": {
    "runtime": {
      "selected": true,
      "doctor_artifact": "experiments/scaffold_to_policy/results/preflight/manifests/runtime_doctor_20260813T-preflight.json"
    },
    "reasoning": {
      "selected": true,
      "preflight_artifact": "experiments/scaffold_to_policy/results/preflight/manifests/reasoning_preflight_20260813T-preflight.json"
    },
    "coding": {
      "selected": false,
      "failure_type": "harness_unavailable",
      "reason": "missing terminal-bench task image"
    },
    "tau2": {
      "selected": false,
      "failure_type": "harness_unavailable",
      "reason": "tau2 harness checkout not configured"
    }
  }
}
```

Future agents should run this command before proposing large experiment runs.
The command is also the right place to print short actionable blockers while
keeping the detailed evidence in JSON.

## Benchmark-Specific Preflights

Runtime checks prove that execution is possible. Benchmark-specific preflights
prove that a score would preserve the benchmark's intended semantics.

### Reasoning Benchmarks

For AIME, GPQA, ARC-AGI, MMLU-Pro, GSM-style, arithmetic words, and modular
sequences:

- verify dataset or fixture cache presence;
- record public/private split provenance;
- check prompt variant and tool policy;
- check answer parser and normalization version;
- run a tiny parser/verifier fixture;
- record whether the benchmark is no-tools, vLLM-only, or scaffolded.

### Coding Benchmarks

For LiveCodeBench, BigCodeBench, SciCode, and HumanEval-style tasks:

- verify task manifest hash and subset selection;
- verify test runner executable and timeout policy;
- verify sandbox temp directories and write permissions;
- run a tiny canonical-output or fixture task;
- record whether network access is disabled, allowed, or irrelevant;
- record language/runtime versions used by the executor.

### Terminal-Bench And Harbor

For Terminal-Bench via Harbor:

- verify Harbor and Terminal-Bench revisions;
- verify container backend availability;
- verify selected task images or compose files;
- verify official final-state verifier availability;
- run a dry-run or one tiny task that exercises artifact ingestion;
- record container image digests when available.

### tau2

For tau2-bench:

- verify harness revision and domain/task subset;
- verify policy files and simulator data;
- verify reset behavior before each task;
- run one deterministic interaction or dry-run;
- ingest trajectory, final state, and score through the shared report schema.

## Result Manifest Requirements

Every real run should write a manifest that makes the result impossible to
confuse with smoke plumbing or blocked work.

Required fields:

```yaml
schema_version: 1
kind: scaffold_to_policy_result_manifest
run_id: exact run id
profile: reasoning | coding | terminal_bench | tau2 | docs_only
benchmark:
  name: benchmark name
  task_release: exact tag, dataset revision, or fixture hash
  evaluator_commit_or_version: exact evaluator identity
  metric: official metric or local exact verifier name
harness:
  name: wrapper or external harness
  command: argv used for execution
  version_or_commit: exact harness identity
runtime:
  rootfs: scripts/rootfs/enter_rootfs.sh
  doctor_artifact: path to runtime doctor JSON
  preflight_artifact: path to benchmark preflight JSON
model:
  base_model: local path or checkpoint id
  adapter: adapter path or null
sampling:
  temperature: value
  top_p: value
  num_rollouts: value
  max_new_tokens: value
result_status: real_score | smoke_only | blocked | docs_only | inconclusive
failure_type: null or taxonomy value
artifacts:
  summaries: paths
  evaluations: paths
  trajectories: paths
  examples: paths
```

The score table should include only `result_status=real_score` rows unless the
report is explicitly about infrastructure.

## Failure Taxonomy

Preflights and runs should classify failures with stable names:

- `runtime_contract_failure`: rootfs, imports, model assets, CUDA, vLLM memory,
  or HF cache failed;
- `missing_assets`: benchmark data, model files, task images, or verifier files
  are absent;
- `harness_unavailable`: external runner, executable, container backend, or
  service is missing;
- `benchmark_manifest_failure`: task manifest, split provenance, or version pin
  is invalid;
- `model_execution_failure`: model loading, generation, or training failed;
- `metric_execution_failure`: verifier, evaluator, or score aggregation failed;
- `score_regression`: run succeeded but failed a configured promotion gate;
- `inconclusive`: run produced artifacts but not enough valid evidence for a
  claim.

Failures in this taxonomy are not benchmark scores. Reports should list them in
blocker tables and keep them separate from pass rates or accuracies.

## Implementation Order

1. Wire `run_runtime_doctor.sh` into every existing real-model scaffold runner.
2. Add `run_preflight.sh` with `runtime` and `reasoning` profiles.
3. Add reasoning preflights for AIME, GPQA, ARC-AGI, and the local synthetic
   exact-verifier tasks.
4. Add result-manifest fields for doctor and preflight artifact paths.
5. Add coding preflights for LiveCodeBench, BigCodeBench, SciCode, and
   HumanEval-style runs.
6. Add Terminal-Bench/Harbor preflight without claiming model performance.
7. Add tau2 preflight without claiming model performance.
8. Teach report generation to separate real scores, smoke-only results, blocked
   work, and inconclusive runs.

The next implementation batch should do items 1-4. That will make the reasoning
lane hermetic enough to run harder benchmarks while keeping coding and agentic
harness expansion honest.
