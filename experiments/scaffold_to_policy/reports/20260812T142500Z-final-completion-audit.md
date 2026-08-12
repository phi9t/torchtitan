# Scaffold-To-Policy Final Completion Audit

Audit timestamp: `2026-08-12T14:25:00Z`

Objective audited: implement `experiments/scaffold_to_policy/spec.md` to the
end and ensure the experiments are all carried out.

## Status

Overall status: blocked on external access, with the implemented experiment
ladder otherwise carried through to reportable smokes, calibrations, hard
negatives, or explicit blocker artifacts.

This audit does not mark the program complete. The GPQA Diamond lane is still
not runnable in the current hermetic rootfs because `HF_TOKEN` is not
configured and the dataset is gated on Hugging Face. The refreshed blocker run
below confirms this with current artifacts.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Mandatory bwrap rootfs for real Python, GPU, training, evaluation, and external harness work | Runners re-enter through `scripts/rootfs/enter_rootfs.sh`; focused validation used rootfs Python; Harbor and tau2 model-policy reports record `all_rootfs_selected=true` | Covered for current runs |
| Run-scoped manifests, split registries, report inputs, artifact provenance, and blocker reporting | `run_common.sh`, `report_artifacts`, `write-blocker-report-input`, stage manifests, latest-report indexing, and report inputs under each result root | Covered for hard lanes; older smokes have thinner traces |
| Countdown champion, formatting arm, replication, rank/size sweep, and reports | Countdown reports under `experiments/countdown_search_distill/reports/`; clean and formatting replications are recorded in `spec.md` | Covered for checkpoint |
| Local exact-verifier reasoning before public benchmarks | Arithmetic words, modular sequences, and modular transfer reports under `experiments/scaffold_to_policy/reports/` | Covered |
| Public no-tool reasoning expansion | GSM8K, MATH algebra, AIME, ARC-AGI-2, and MMLU-Pro reports; exact or benchmark-specific verifiers preserved | Covered as small calibration runs, not leaderboard claims |
| GPQA Diamond gate | Fresh rootfs run `20260812T142500Z-gpqa-auth-final-audit` wrote `experiments/scaffold_to_policy/results/gpqa_public_vllm_final_audit/manifests/report_input_20260812T142500Z-gpqa-auth-final-audit.json` with `gpqa_import_available=false` | Blocked by gated dataset auth |
| Public coding expansion | HumanEval, MBPP, BigCodeBench-Hard, and LiveCodeBench public-test reports | Covered as executable/public-test smokes and hard negatives |
| BigCodeBench-Hard harder contract-chat condition | `20260812Tbigcode-hard-contract-chat-8x8-timeout30-rerun` report input has passing canonical preflights, dev/OOD summaries, and pass@k 0.0 on both splits | Covered as hard-negative model result |
| Terminal-Bench/Harbor harness smoke | Oracle, nop, scripted, and Qwen model-agent reports exist; Qwen official verifier run completed one trial with reward 0.0 | Covered as harness/model-policy execution, not solved task |
| tau2-bench harness smoke | Scorer, deterministic, noop, and Qwen model-policy reports exist; Qwen run completed one official mock-domain simulation with reward 0.0 | Covered as harness/model-policy execution, not solved task |
| Preserve upstream semantics and avoid generic judges | Reports label no-tool, public-test-only, oracle/scripted/model-policy, and blocker conditions separately; exact verifiers and external harness scorers are used | Covered |
| Do not commit generated data/results/checkpoints/caches | Generated result/data roots remain ignored; tracked artifacts are code, scripts, tests, docs, and reports | Covered |

## Fresh GPQA Blocker Refresh

Command:

```bash
RUN_ID=20260812T142500Z-gpqa-auth-final-audit \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_public_vllm_final_audit \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_public_vllm_final_audit \
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
experiments/scaffold_to_policy/results/gpqa_public_vllm_final_audit/manifests/report_input_20260812T142500Z-gpqa-auth-final-audit.json
```

The report input records:

```json
{
  "checks": {
    "gpqa_import_available": false
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

No GPQA task execution or model score was produced.

## Current Evidence Summary

| Lane | Best current evidence | Interpretation |
| --- | --- | --- |
| Countdown | Clean and formatting replications plus rank/size sweep reports | Reference scaffold-to-policy signal exists |
| Modular sequences | Expanded transfer report with dev/OOD pass@1 lift | Local exact-verifier transfer gate cleared |
| GSM8K / MATH / AIME | Rootfs no-tool vLLM reports | Public reasoning smokes/calibrations completed |
| ARC-AGI-2 | Chat, strict-chat, packed-chat reports | Context and final-format blockers characterized; exact-grid solve not achieved |
| MMLU-Pro | `20260812Tmmlu-pro-8x8-rollouts4-promptfix` | Ten-choice multiple-choice lane works; small calibration only |
| GPQA Diamond | `20260812T142500Z-gpqa-auth-final-audit` blocker | Not runnable without HF auth or authorized raw cache |
| HumanEval / MBPP | Executable-code smoke reports | Coding harness path validated on small slices |
| BigCodeBench-Hard | `20260812Tbigcode-hard-contract-chat-8x8-timeout30-rerun` | Hard-negative 8x8 coding result with released tests |
| LiveCodeBench | `20260812Tlivecodebench-public-vllm-smoke` | Public-test-only contest-code smoke, not official score |
| Terminal-Bench / Harbor | `20260812Tqwen-headless-terminal-official-verifier` | Official verifier reached; Qwen task reward 0.0 |
| tau2-bench | `20260812Ttau2-qwen-model-policy` | Official tau2 runner reached; Qwen task reward 0.0 |

## Remaining Work

1. Configure authenticated Hugging Face access inside the bwrap rootfs, then run
   `run_gpqa_public_vllm_smoke.sh` against `Idavidrein/gpqa` `gpqa_diamond`.
2. If no live Hugging Face access is allowed, provide an authorized raw GPQA
   cache and rerun the importer with the offline raw-cache path.
3. Treat all existing public benchmark numbers as calibration slices unless a
   later run expands sample size, pins the complete evaluation object, and
   preserves the upstream metric.

Until item 1 or 2 is completed, the objective remains blocked rather than
achieved.
