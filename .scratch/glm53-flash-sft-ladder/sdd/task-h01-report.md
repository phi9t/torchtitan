# H01 report: HDFS-strict input enforcement

Status: PASS for owned/idle surfaces. No commit was made. No GPU command was
run.

Surface: repo-local experiment orchestration for the GLM-5.3-Flash SFT ladder
and cross-campaign input provenance. Core training, online RL, and active ticket
files were not edited.

## Landed Changes

- Added `experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`.
  - Host-side bash only.
  - Requires each checked root to have `.staged.json` with `hdfs://` source URI.
  - Requires local `MANIFEST.jsonl`.
  - Verifies local manifest md5 against `.staged.json`.
  - Verifies every manifest entry has matching size and md5.
  - `--quick` caches successful md5 checks in `.verified.json`, keyed by file
    mtime and size.
- Added `experiments/glm53_flash_sft_ladder/tests/test_hdfs_strict.sh`.
- Updated `experiments/glm53_flash_sft_ladder/stage.sh` to keep the HDFS
  `MANIFEST.jsonl` beside `.staged.json` in the staged local root.
- Wired `hdfs_strict.sh --quick` into the owned idle runners:
  - `experiments/glm53_flash_sft_ladder/run_rung0.sh`
  - `experiments/glm53_flash_sft_ladder/run_rung.sh`
- Backfilled local `MANIFEST.jsonl` files for the canonical `/data05` staged
  input roots from their recorded HDFS source URIs.
- Published and restaged the old T07b evidence input copies so they now have
  HDFS provenance and strict local markers.

## Canonical Input Cache Audit

All canonical training/eval/parity input roots currently staged under
`/data05/philip.yang/glm53-ladder-cache` passed `hdfs_strict.sh --quick`.

| Input root | HDFS source | Manifest md5 | Files | Status |
| --- | --- | --- | ---: | --- |
| `/data05/philip.yang/glm53-ladder-cache/datasets/countdown-sft/v1` | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | `66f71a353be516264fcc02fc08116ddb` | 8 | strict PASS |
| `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B` | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca` | `441136422c0f62fc97fb83ccf3212b6a` | 10 | strict PASS |
| `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-1.7B` | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `501ad940034aaede84d8af89613ae65c` | 12 | strict PASS |
| `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-8B` | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218` | `6ba6a83877bc9d0f826c547c5d75c291` | 15 | strict PASS |
| `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-30B-A3B` | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-30B-A3B/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | `31dc7602093b6a1bac1e7c6f9d9ec140` | 26 | strict PASS |
| `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16` | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538` | `365857f8b1d7cb0841992fff91240ccf` | 130 | strict PASS |

Notes:

- Child directories under a staged dataset root, such as `train/` and `dev/`,
  are manifest paths inside the parent staged root; they are not standalone
  inputs and do not get their own `.staged.json`.
- `/data05/philip.yang/glm53-ladder-cache/{ingest-tmp,publish-tmp,stage-tmp,results,evidence-large}`
  contains transient publish/stage files, local results, or evidence outputs.
  Those are not launch input roots.

## Runner and Config Input Audit

| Surface | Input path or default | Classification | Action |
| --- | --- | --- | --- |
| `run_rung0.sh` | Countdown v1 plus Qwen3 0.6B and 1.7B roots mirrored into `experiments/glm53_flash_sft_ladder/cache/rung0-inputs` | HDFS-staged canonical inputs | Wired strict check into `stage`, `mirror`, and all launch gates |
| `run_rung.sh` | Countdown v1 plus selected Qwen3 root mirrored into `experiments/glm53_flash_sft_ladder/cache/<model>-inputs` | HDFS-staged canonical inputs | Wired strict check into `stage`, `mirror`, and all launch gates |
| `run_rung0.sh` / `run_rung.sh` resume gate | step-62 checkpoint published to HDFS, then staged back locally | HDFS-staged resume input after publish | Wired strict check immediately after checkpoint staging |
| `run_rung4_glm.sh` | `/data05/.../datasets/countdown-sft/v1` and `/data05/.../model-weights/zai-org/GLM-5.3-Flash-BF16` via `/project/formal-cache` | HDFS-staged canonical inputs | Do not edit: T08d owns it. Patch below |
| `run_rung4_glm.sh` resume gate | step-62 checkpoint published to HDFS, then staged back locally | HDFS-staged resume input after publish | Do not edit: T08d owns it. Patch below |
| `torchtitan/experiments/glm53_flash_sft_ladder/config_registry.py` | defaults under `/data05/${USER}/glm53-ladder-cache` | HDFS-staged if canonical roots above are present | Direct module launches must call strict checker before rootfs |
| `torchtitan/experiments/glm53_flash/config_registry.py` | Countdown defaults under `/project/formal-cache/datasets/countdown-sft/v1` and `/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16` | HDFS-staged through `TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/...` | Runner/launch command must call strict checker first |
| `torchtitan/experiments/glm53_flash/config_registry.py` debug defaults | `./tests/assets/tokenizer`, synthetic data | Not HDFS-staged if used as a training/parity input | Use only for non-campaign synthetic tests, or ingest/stage and override path |
| `torchtitan/experiments/glm53_flash_4gpu_stream/config_registry.py` | inherits GLM countdown data/model defaults | HDFS-staged only if caller checks canonical roots | T09 direct launches need the preflight snippet below |
| `experiments/glm53_arch_speedrun/run_arm.sh` | FineWeb GPT-2 bins under `experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1` | HDFS-published, but local root currently lacks `.staged.json` | Do not edit: S06 owns it. Patch below |
| `experiments/glm53_arch_ladder/run_rung.sh` | same FineWeb GPT-2 staged root | HDFS-published, but local root currently lacks `.staged.json` | Do not edit: S07 owns it. Patch below |
| `experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh` | FineWeb-Edu staged data | Wrapper currently writes `MANIFEST.jsonl` and `_COMPLETE`, not `.staged.json` | Do not edit: S08 owns it. Patch below |
| Arch speedrun and arch ladder config registries | fallback `experiments/modded_nanogpt_b200/.../data/fineweb10B` | Not HDFS-staged | Runner patch below removes the local fallback from launch input provenance |

## Remediated Evidence Inputs

The old T07b evidence input copies under
`.scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/` had legacy
`staged.json` files but no strict marker or local HDFS manifest. I published
those local input subsets to HDFS and staged them back:

| Local dir | HDFS source | Manifest md5 | Files | Status |
| --- | --- | --- | ---: | --- |
| `.scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/Qwen3-0.6B-tokenizer` | `hdfs://harunava/user/tiger/waystone/torchtitan/evidence/glm53-ladder/h01/t07b/inputs/Qwen3-0.6B-tokenizer` | `ad490e19d0492a6cffa2ef4841a78475` | 5 | strict PASS |
| `.scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/countdown-sft-v1` | `hdfs://harunava/user/tiger/waystone/torchtitan/evidence/glm53-ladder/h01/t07b/inputs/countdown-sft-v1` | `d4dda21d68e83c34ab3c0d80c6c0b962` | 2 | strict PASS |

No separate local-only GLM evidence input tree was found. T08c large tensors
under `/data05/philip.yang/glm53-ladder-cache/evidence-large/task-08c` are
evidence outputs, not training/eval input roots. GLM/T09 reports consume the
canonical Countdown and GLM model roots listed above.

## Patches For Active Ticket Files

These were not applied because the H01 brief marks the files/directories as
owned by active tickets.

### T08d: `experiments/glm53_flash_sft_ladder/run_rung4_glm.sh`

```diff
diff --git a/experiments/glm53_flash_sft_ladder/run_rung4_glm.sh b/experiments/glm53_flash_sft_ladder/run_rung4_glm.sh
--- a/experiments/glm53_flash_sft_ladder/run_rung4_glm.sh
+++ b/experiments/glm53_flash_sft_ladder/run_rung4_glm.sh
@@
 check_inputs() {
   [[ -f "${HOST_DATA_ROOT}/train/formatting.jsonl" ]] || {
     printf 'missing staged dataset: %s\n' "${HOST_DATA_ROOT}" >&2
     exit 1
   }
   [[ -f "${HOST_MODEL_ROOT}/.staged.json" ]] || {
     printf 'missing staged GLM checkpoint marker: %s/.staged.json\n' "${HOST_MODEL_ROOT}" >&2
     exit 1
   }
+  "${script_dir}/lib/hdfs_strict.sh" --quick "${HOST_DATA_ROOT}" "${HOST_MODEL_ROOT}"
 }
@@
   "${script_dir}/stage.sh" \
     "hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/step62" \
     "${RESULT_ROOT}/train/seed42_resume_from62/checkpoint/step-62" \
     | tee "${LOG_DIR}/stage_resume_step62.log"
+  "${script_dir}/lib/hdfs_strict.sh" --quick "${RESULT_ROOT}/train/seed42_resume_from62/checkpoint/step-62"
   train_one seed42_resume_from62 42 1 94 62
@@
-  export) export_adapters ;;
-  eval) eval_gate ;;
+  export) check_inputs; export_adapters ;;
+  eval) check_inputs; eval_gate ;;
```

### S06: `experiments/glm53_arch_speedrun/run_arm.sh`

```diff
diff --git a/experiments/glm53_arch_speedrun/run_arm.sh b/experiments/glm53_arch_speedrun/run_arm.sh
--- a/experiments/glm53_arch_speedrun/run_arm.sh
+++ b/experiments/glm53_arch_speedrun/run_arm.sh
@@
 stage_dataset_from_hdfs() {
-  local manifest_tmp rel size md5 hdfs_file local_file
   ensure_waystone_loaded
   if [[ -z "$hdfs_data_uri" ]]; then
     hdfs_data_uri="$(ws_child_uri "$data_rel")"
   fi
-  mkdir -p "$host_staged_data_dir"
-  manifest_tmp="$(mktemp)"
-  "${WAYSTONE_BIN}" --error-format json \
-    --command-timeout-secs "$WAYSTONE_METADATA_TIMEOUT_SECS" \
-    cat "$(ws_join_uri "$hdfs_data_uri" "MANIFEST.jsonl")" > "$manifest_tmp"
-  while IFS= read -r line; do
-    rel="$(jq -r '.file' <<<"$line")"
-    size="$(jq -r '.size' <<<"$line")"
-    md5="$(jq -r '.md5' <<<"$line")"
-    hdfs_file="$(ws_join_uri "$hdfs_data_uri" "$rel")"
-    local_file="${host_staged_data_dir}/${rel}"
-    if [[ -f "$local_file" ]] && [[ "$(ws_file_size_bytes "$local_file")" == "$size" ]] && [[ "$(ws_md5_file "$local_file")" == "$md5" ]]; then
-      continue
-    fi
-    WS_EXPECTED_SIZE="$size" WS_EXPECTED_MD5="$md5" ws_get_file "$hdfs_file" "$local_file"
-  done < "$manifest_tmp"
-  cp "$manifest_tmp" "${host_staged_data_dir}/MANIFEST.jsonl"
-  rm -f "$manifest_tmp"
+  "${repo_root}/experiments/glm53_flash_sft_ladder/stage.sh" \
+    --parallel "${GLM53_SPEEDRUN_STAGE_PARALLEL:-2}" \
+    "$hdfs_data_uri" "$host_staged_data_dir"
 }
 
-if [[ "${GLM53_SPEEDRUN_SKIP_PUBLISH:-0}" == "1" && -f "${host_staged_data_dir}/MANIFEST.jsonl" ]]; then
-  : "${hdfs_data_uri:=local:${staged_data_rel}}"
-else
-  publish_dataset_if_needed
-  stage_dataset_from_hdfs
-fi
+publish_dataset_if_needed
+stage_dataset_from_hdfs
+"${repo_root}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" --quick "$host_staged_data_dir"
@@
-  "data_rootfs_dir": "${rootfs_staged_data_dir}",
+  "data_rootfs_dir": "${rootfs_staged_data_dir}",
+  "data_manifest_md5": "$(jq -r '.manifest_md5' "${host_staged_data_dir}/.staged.json")",
   "ngpu": ${ngpu}
 }
```

This keeps `GLM53_SPEEDRUN_SKIP_PUBLISH=1` meaningful for run-output publishing
later in the file, but no longer lets a local `MANIFEST.jsonl` downgrade data
provenance to `local:...`.

### S07: `experiments/glm53_arch_ladder/run_rung.sh`

```diff
diff --git a/experiments/glm53_arch_ladder/run_rung.sh b/experiments/glm53_arch_ladder/run_rung.sh
--- a/experiments/glm53_arch_ladder/run_rung.sh
+++ b/experiments/glm53_arch_ladder/run_rung.sh
@@
 stage_dataset_from_hdfs() {
-  local manifest_tmp rel size md5 hdfs_file local_file
   ensure_waystone_loaded
   if [[ -z "$hdfs_data_uri" ]]; then
     hdfs_data_uri="$(ws_child_uri "$data_rel")"
   fi
-  mkdir -p "$host_staged_data_dir"
-  manifest_tmp="$(mktemp)"
-  "${WAYSTONE_BIN}" --error-format json \
-    --command-timeout-secs "$WAYSTONE_METADATA_TIMEOUT_SECS" \
-    cat "$(ws_join_uri "$hdfs_data_uri" "MANIFEST.jsonl")" > "$manifest_tmp"
-  while IFS= read -r line; do
-    rel="$(jq -r '.file' <<<"$line")"
-    size="$(jq -r '.size' <<<"$line")"
-    md5="$(jq -r '.md5' <<<"$line")"
-    hdfs_file="$(ws_join_uri "$hdfs_data_uri" "$rel")"
-    local_file="${host_staged_data_dir}/${rel}"
-    if [[ -f "$local_file" ]] && [[ "$(ws_file_size_bytes "$local_file")" == "$size" ]] && [[ "$(ws_md5_file "$local_file")" == "$md5" ]]; then
-      continue
-    fi
-    WS_EXPECTED_SIZE="$size" WS_EXPECTED_MD5="$md5" ws_get_file "$hdfs_file" "$local_file"
-  done < "$manifest_tmp"
-  cp "$manifest_tmp" "${host_staged_data_dir}/MANIFEST.jsonl"
-  rm -f "$manifest_tmp"
+  "${repo_root}/experiments/glm53_flash_sft_ladder/stage.sh" \
+    --parallel "${GLM53_ARCH_LADDER_STAGE_PARALLEL:-2}" \
+    "$hdfs_data_uri" "$host_staged_data_dir"
 }
 
-if [[ -f "${host_staged_data_dir}/MANIFEST.jsonl" ]]; then
-  hdfs_data_uri="${hdfs_data_uri:-local:${staged_data_rel}}"
-else
-  stage_dataset_from_hdfs
-fi
+stage_dataset_from_hdfs
+"${repo_root}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" --quick "$host_staged_data_dir"
@@
-  "data_rootfs_dir": "${rootfs_staged_data_dir}",
+  "data_rootfs_dir": "${rootfs_staged_data_dir}",
+  "data_manifest_md5": "$(jq -r '.manifest_md5' "${host_staged_data_dir}/.staged.json")",
   "ngpu": 1
 }
```

### S08: `experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh`

```diff
diff --git a/experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh b/experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh
--- a/experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh
+++ b/experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh
@@
 "${WAYSTONE_BIN}" --error-format json \
   --command-timeout-secs "${WAYSTONE_METADATA_TIMEOUT_SECS}" \
   cat "$(ws_join_uri "${hdfs_uri}" "MANIFEST.jsonl")" > "${manifest_tmp}"
+manifest_md5="$(ws_md5_file "${manifest_tmp}")"
+file_count="$(wc -l < "${manifest_tmp}" | tr -d '[:space:]')"
@@
 cp "${manifest_tmp}" "${local_dir}/MANIFEST.jsonl"
 rm -f "${manifest_tmp}"
 "${WAYSTONE_BIN}" --error-format json \
   --command-timeout-secs "${WAYSTONE_METADATA_TIMEOUT_SECS}" \
   cat "$(ws_join_uri "${hdfs_uri}" "_COMPLETE")" > "${local_dir}/_COMPLETE"
+jq -n \
+  --arg source_uri "${hdfs_uri}" \
+  --arg manifest_md5 "${manifest_md5}" \
+  --arg staged_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
+  --argjson file_count "${file_count}" \
+  '{source_uri:$source_uri, manifest_md5:$manifest_md5, file_count:$file_count, staged_at:$staged_at}' \
+  > "${local_dir}/.staged.json"
 
 printf 'LOCAL_DIR=%s\n' "${local_dir}"
+printf 'STAGED_JSON=%s\n' "${local_dir}/.staged.json"
```

### T09 direct-launch preflight snippet

The T09 stream package has no host runner in this checkout; its configs inherit
the GLM countdown data/model env paths. Add this before each direct rootfs
training/parity/eval command that uses `torchtitan.experiments.glm53_flash_4gpu_stream`:

```bash
experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh --quick \
  /data05/philip.yang/glm53-ladder-cache/datasets/countdown-sft/v1 \
  /data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
```

For commands that stage or resume from a checkpoint, also strict-check the
checkpoint directory immediately after `stage.sh` returns and before rootfs
launch.

## Verification

Commands run:

```bash
bash -n \
  experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh \
  experiments/glm53_flash_sft_ladder/tests/test_hdfs_strict.sh \
  experiments/glm53_flash_sft_ladder/stage.sh \
  experiments/glm53_flash_sft_ladder/run_rung0.sh \
  experiments/glm53_flash_sft_ladder/run_rung.sh
```

Result: pass.

```bash
experiments/glm53_flash_sft_ladder/tests/test_hdfs_strict.sh
```

Result: `hdfs_strict tests passed`.

```bash
experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh --quick \
  /data05/philip.yang/glm53-ladder-cache/datasets/countdown-sft/v1 \
  /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B \
  /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-1.7B \
  /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-8B \
  /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-30B-A3B \
  /data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
```

Result: pass for all six canonical input roots. The first GLM pass computed
md5s for 130 files; the second cached quick pass over all six roots completed
successfully and printed each HDFS URI plus manifest md5.

```bash
experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh --quick \
  .scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/Qwen3-0.6B-tokenizer \
  .scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/countdown-sft-v1
```

Result: pass for both repaired evidence input dirs.

No GPU tests or training jobs were run for H01.
