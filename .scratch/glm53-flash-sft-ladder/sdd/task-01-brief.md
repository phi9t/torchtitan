# T01 brief: waystone storage wrappers + dataset v1 publish

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/spec.md` (Storage contract), `.claude/CLAUDE.md`
(rootfs rules), `/data02/home/philip.yang/workspace/waystone/AGENTS.md`,
`/data02/home/philip.yang/workspace/waystone/docs/shared-hdfs-storage.md`,
`/data02/home/philip.yang/workspace/waystone/docs/hdfs-cli-contract.md`.

## Requirements
Create `experiments/glm53_flash_sft_ladder/` (runner dir; add `experiments/glm53_flash_sft_ladder/results/`
and `experiments/glm53_flash_sft_ladder/cache/` to `.gitignore`) with bash scripts:

1. `lib/waystone.sh` (sourced): `WAYSTONE_BIN` default `/data02/home/philip.yang/workspace/waystone/scripts/waystone`;
   functions `ws_child_uri <relpath>` (uses `waystone storage-prefix --child`), `ws_layout` ,
   `ws_put_file <local> <uri>` (`--verify-md5 put`, retries 3x with backoff, timeout per call scaled to size),
   `ws_get_file <uri> <local>` (`get`, then verify MD5 against a sidecar or manifest),
   `ws_exists <uri>`, `ws_ls <uri>`. Do NOT pass `--refresh-token` (it prompts for a password);
   auth is maintained by cron `~/.local/bin/waystone-auth-keepalive.sh`. Fail loudly (nonzero, clear message) on timeout.
   Host-side waystone CLI is allowed (it's a Rust binary = orchestration). md5 via `md5sum` on host is fine.
2. `hf_ingest.sh --repo <org/name> --revision <rev|main> [--include <glob>]...`: resolve the revision to a
   commit sha (curl the HF API `https://huggingface.co/api/models/<repo>/revision/<rev>`), list files from the
   API `siblings`, then for each file: download to a temp dir under `${GLM_LADDER_CACHE:-/data05/philip.yang/glm53-ladder-cache}/ingest-tmp`
   with `curl -L --fail --retry 5 -C -` from `https://huggingface.co/<repo>/resolve/<sha>/<file>`, md5 it, `ws_put_file` to
   `model-weights/<org>/<name>/<sha>/<file>` (child of the waystone torchtitan project prefix, i.e. `torchtitan/model-weights/...`),
   append `{file, size, md5, sha256-if-LFS-from-API}` to a local `MANIFEST.jsonl`, delete the local file.
   Skip files already present on HDFS with matching size (idempotent / resumable). Finally upload `MANIFEST.jsonl`
   and a `_COMPLETE` marker. Local footprint must be at most ~2 files at a time. Optional `--parallel N` (default 2).
   Prints the HDFS URI. Log throughput.
3. `stage.sh <hdfs-uri-dir> <local-dir>`: fetch MANIFEST.jsonl, require `_COMPLETE`, get every file not already present
   locally with the right md5, verify md5s, write `<local-dir>/.staged.json` with source URI + manifest md5. Parallel option.
4. `publish.sh <local-dir> <relpath-under-torchtitan>`: put every file recursively with md5 verification, write and
   upload a MANIFEST.jsonl + `_COMPLETE`. Used for checkpoints, adapters, eval outputs, run bundles.
5. `publish_dataset_v1.sh`: builds `countdown-sft/v1` from
   `experiments/countdown_search_distill/sweeps/formatting_replication/formatting_seed43_train2000_rank16/data/`:
   `train/formatting.jsonl`, `train/problems.jsonl`, `{dev,iid_test,ood_test,calibration}/problems.jsonl`,
   `split_registry.json`, plus a `README.md` (provenance: source dir, row counts, md5s, date) into a local temp
   dir, then `publish.sh` to `torchtitan/datasets/countdown-sft/v1`. Actually run it.
6. `README.md` in the runner dir documenting the commands and the contract.

Also run `hf_ingest.sh --repo Qwen/Qwen3-0.6B --revision main` for real and then `stage.sh` it into
`/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B` to prove the round trip. If `/data05/philip.yang`
does not exist or is not writable, create it or pick another dir on /data05 and report it.

## Exclusions
No Python for business logic outside the rootfs (none should be needed; bash + curl + jq + waystone CLI on host).
If `jq` is missing on host, use it via `scripts/rootfs/enter_rootfs.sh -- jq ...` or parse with grep/sed carefully.
No core torchtitan changes. No commit. Do not touch other experiments' files.

## Verification
- `bash -n` all scripts; shellcheck if available.
- Real run outputs: dataset v1 URI listing, Qwen3-0.6B ingest URI listing + `_COMPLETE`, staged dir md5 check passes.
- Re-running `hf_ingest.sh` for Qwen3-0.6B is a no-op (skips everything).

## Report
Write `.scratch/glm53-flash-sft-ladder/sdd/task-01-report.md`: files created, exact commands run, HDFS URIs,
throughput (MB/s for HF download and HDFS put), any failures, claim_label `smoke`. No commit.
