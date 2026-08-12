# Scaffold-To-Policy Completion Audit Refresh

Audit timestamp: `2026-08-12T21:10:00Z`

Objective audited: implement `experiments/scaffold_to_policy/spec.md` to the
end and ensure the experiments are all carried out under the hermetic bwrap
rootfs discipline.

## Concrete Completion Criteria

The objective is complete only if all of the following are true:

- the scaffold-to-policy registry/reporting contract exists and is exercised by
  rootfs-managed run families;
- Countdown follow-up evidence includes the champion arm, formatting arm,
  replication, and rank/size sweep called for in the spec;
- local exact-verifier reasoning runs precede public benchmark expansion;
- public no-tool reasoning and coding lanes have reportable artifacts that
  preserve benchmark semantics and do not use generic judges;
- Terminal-Bench/Harbor and tau2-bench enter as external-harness smokes with
  upstream scoring boundaries preserved;
- GPQA Diamond either runs with authenticated access or is explicitly blocked
  by a current rootfs artifact proving no benchmark-preserving access exists;
- generated data, checkpoints, caches, rollouts, and result trees remain out of
  git;
- rootfs validation and shell syntax checks cover the current runner/source
  changes.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Use bwrap rootfs for real Python, GPU generation/evaluation, and external harness work | `run_common.sh` rootfs re-entry is used by hard-lane shell runners; fresh GPQA stage manifest records `rootfs_active=true`; MMLU-Pro and BigCodeBench-Hard reports record vLLM inside rootfs; Harbor/tau2 report inputs record `all_rootfs_selected=true` | Covered for current executed lanes |
| Run-scoped manifests and report inputs | Fresh GPQA manifest `experiments/scaffold_to_policy/results/gpqa_public_vllm_workstatus/manifests/20260812T212000Z-gpqa-auth-workstatus.jsonl` records stage command, rootfs state, timing, return code, roots, and `work_status=executed`; MMLU-Pro and BigCodeBench-Hard report inputs under their run-scoped result roots | Covered for hard lanes |
| Split registries and no-overlap checks | `experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_16x16_rollouts4_ood32/split_registry.json` has 16 dev, 16 OOD, `selected=true`, and no overlaps; BigCodeBench-Hard free-GPU split registry has 8 dev, 8 OOD, `selected=true`, and no overlaps | Covered for latest reasoning/coding lanes |
| Artifact provenance and blocker reporting | Latest report inputs have `artifact_provenance_labeled=true`; GPQA blocker report has `blocker_artifacts_present=true` and `benchmark_execution_completed=false` | Covered |
| Countdown champion, formatting arm, replication, and rank/size sweep | `spec.md` lists the Countdown reports: clean split final, formatting arm, base-elicitable examples, replication/rank-size sweep, and formatting replication | Covered for checkpoint |
| Local exact-verifier reasoning before public benchmarks | Reports exist for arithmetic words, modular sequences, modular transfer, GSM-style local fixtures, and vLLM smokes under `experiments/scaffold_to_policy/reports/` | Covered |
| Public no-tool reasoning expansion | GSM8K, MATH, AIME, ARC-AGI-2, MMLU-Pro reports exist; latest MMLU-Pro run `20260812T203000Z-mmlu-pro-16x16-rollouts4-ood32` preserved exact final-letter scoring | Covered as calibration, not leaderboard claims |
| GPQA Diamond lane | Fresh rootfs run `20260812T212000Z-gpqa-auth-workstatus` failed at `datasets.exceptions.DatasetNotFoundError` for gated dataset `Idavidrein/gpqa`; report input records no task execution or score | Blocked by missing HF auth or authorized raw cache |
| Public coding expansion | HumanEval, MBPP, LiveCodeBench, and BigCodeBench-Hard reports exist; latest BigCodeBench-Hard free-GPU run passed canonical preflight and completed model evaluation | Covered as executable/public-test smokes and hard negatives |
| Terminal-Bench/Harbor harness smoke | `experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier/manifests/report_input_20260812Tqwen-headless-terminal-official-verifier.json` records installed preflight, rootfs selection, one completed official verifier trial, and unsuccessful task reward | Covered as model-policy execution plumbing, not solved task |
| tau2-bench harness smoke | `experiments/scaffold_to_policy/results/tau2_qwen_model_policy/manifests/report_input_20260812Ttau2-qwen-model-policy.json` records installed preflight, rootfs selection, one completed upstream mock-domain simulation, and unsuccessful task reward | Covered as model-policy execution plumbing, not solved task |
| Preserve upstream semantics and avoid generic judges | Reports label no-tool, public-test-only, oracle/scripted/model-policy, and blocker conditions; exact verifiers or upstream harness scorers are used | Covered |
| Do not commit generated data/results/checkpoints/caches | Current git status shows only pre-existing `.claude/CLAUDE.md` modified; generated `experiments/scaffold_to_policy/data/` and `results/` are ignored | Covered |
| Current validation | `bash -n` on GPQA/MMLU-Pro/BigCodeBench-Hard runners and wizard passed; rootfs unit test `python -m pytest tests/unit_tests/test_scaffold_to_policy.py -q` passed with 80 tests and 14 warnings; `git diff --check` passed before the latest audit doc edit | Covered before audit doc edit; rerun required after committing this doc |

## Fresh GPQA Evidence

Credential/cache inspection found:

```text
host HF_TOKEN=
repo token file missing
rootfs HF_TOKEN=
rootfs repo token file missing
HF_HOME token file missing at /token
```

Only synthetic GPQA-shaped offline-cache smoke files were present under
`experiments/scaffold_to_policy/data/gpqa_offline_cache_smoke/raw/`; those do
not satisfy the benchmark-preserving GPQA lane.

Fresh command:

```bash
RUN_ID=20260812T212000Z-gpqa-auth-workstatus \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_public_vllm_workstatus \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_public_vllm_workstatus \
DEV_PROBLEMS=1 OOD_PROBLEMS=1 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

Observed rootfs error:

```text
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated
dataset on the Hub. You must be authenticated to access it.
```

The runner exited cleanly after writing:

```text
experiments/scaffold_to_policy/results/gpqa_public_vllm_workstatus/manifests/report_input_20260812T212000Z-gpqa-auth-workstatus.json
```

The report input records:

```json
{
  "checks": {
    "artifact_provenance_labeled": true,
    "benchmark_execution_completed": false,
    "blocker_artifacts_present": true
  },
  "run": {
    "lane": "reasoning",
    "scaffold": {
      "type": "gpqa_import_blocker",
      "budget": 0
    },
    "task": "multiple_choice"
  }
}
```

The command-stage manifest also records the new explicit stage work-status
field:

```json
[
  {
    "stage": "import_dev",
    "return_code": 1,
    "rootfs_active": true,
    "work_status": "executed"
  },
  {
    "stage": "write_import_blocker_report_input",
    "return_code": 0,
    "rootfs_active": true,
    "work_status": "executed"
  }
]
```

## Latest Hard-Lane Evidence

The latest MMLU-Pro 16/16 calibration report input:

```text
experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_rollouts4_ood32/manifests/report_input_20260812T203000Z-mmlu-pro-16x16-rollouts4-ood32.json
```

Key metrics:

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 16 | 64 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | easy 8, elicitable 0, unreached 8 |
| OOD | 16 | 64 | 0.5625 | 0.6250 | 0.6875 | 0.6875 | easy 9, elicitable 2, unreached 5 |

The latest BigCodeBench-Hard `contract_chat` free-GPU report input:

```text
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_freegpu/manifests/report_input_20260812T200000Z-bigcodebench-hard-contract-chat-8x8-freegpu.json
```

Key metrics:

| Split | Problems | Rollouts | pass@1 | pass@8 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| OOD | 8 | 64 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Canonical preflight passed for all 16 BigCodeBench-Hard selected tasks before
model scoring.

## Audit Result

The objective is not complete. The implemented scaffold-to-policy ladder is
otherwise carried through to reportable rootfs-managed smokes, calibrations,
hard negatives, or explicit blocker artifacts, but GPQA Diamond remains blocked
by external access. No current Hugging Face token or authorized raw GPQA cache
is available inside the rootfs-visible environment, and the fresh GPQA run
proves no benchmark task execution or model score was produced.

Do not mark the active goal complete until one of these is done:

1. configure authenticated Hugging Face access inside the bwrap rootfs and run
   `run_gpqa_public_vllm_smoke.sh` against `Idavidrein/gpqa` `gpqa_diamond`; or
2. provide authorized raw GPQA JSONL rows and run the same entrypoint with
   `OFFLINE=1`, `DEV_RAW_CACHE=...`, and `OOD_RAW_CACHE=...`.

The helper for the required human-provided access path is:

```bash
experiments/scaffold_to_policy/setup_gpqa_access_wizard.sh
```
