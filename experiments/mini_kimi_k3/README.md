# Mini Kimi K3 Replication Scaffold

This directory is the repo-local runner, manifest, report, and artifact home
for a TorchTitan Mini Kimi K3 replication effort. It is a scaffold for the
first tracer bullet only. It is not a training claim, an architecture-fidelity
claim, or a reproduction of the Vizuara Mini Kimi K3 r1 result.

The current goal is tiny-plumbing-first:

1. Define the repo-local experiment boundary and rootfs entrypoint.
2. Add reusable experiment-owned code under
   `torchtitan/experiments/mini_kimi_k3/`.
3. Add tiny synthetic or pretokenized fixtures before touching real data.
4. Keep full launch gated on model fidelity, corpus provenance, and run-attempt
   evidence.
5. Treat any later GPU run as a separate authorized step with an immutable
   run-attempt bundle.

## Rootfs Requirement

All Python, modeling, tokenization, validation, data preparation, training,
evaluation, and log parsing for this experiment must run through the TorchTitan
rootfs:

```bash
experiments/mini_kimi_k3/run.sh --mode tiny-plumbing
```

The runner resolves the checkout root from any current directory and re-enters
through `scripts/rootfs/enter_rootfs.sh` when `TORCHTITAN_IN_ROOTFS=1` is not
present. Unknown or omitted commands fail closed and do not start training or
evaluation.

The supported operational commands today are evidence initialization, the
full-manifest template initialization, the fixture-scale training smoke, and
the readiness preflight:

```bash
experiments/mini_kimi_k3/run.sh init-evidence \
  --results-root experiments/mini_kimi_k3/results \
  --run-id mini-kimi-k3-r1-local \
  --attempt-id attempt-001 \
  --mode full

experiments/mini_kimi_k3/run.sh init-manifest \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source fineweb-edu \
  --dataset-id HuggingFaceFW/fineweb-edu \
  --snapshot <snapshot-or-commit> \
  --license <license-or-policy> \
  --source-provenance-report reports/fineweb-edu-provenance.json \
  --source-provenance-sha256 <source-provenance-report-fingerprint> \
  --tokenizer-sha256 <k3-tokenizer-fingerprint> \
  --tokenizer-asset-report experiments/mini_kimi_k3/assets/kimi-k3-tokenizer.json \
  --decontamination-report reports/decontamination.json \
  --decontamination-sha256 <decontamination-report-fingerprint>

experiments/mini_kimi_k3/run.sh register-shards \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source fineweb-edu \
  --shard experiments/mini_kimi_k3/data/tokens/fineweb-edu/part-00000.bin

experiments/mini_kimi_k3/run.sh import-shards \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source fineweb-edu \
  --source-dir <stage1-volume-export>/tokens/fineweb-edu \
  --link-mode symlink

experiments/mini_kimi_k3/run.sh build-local-shards \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source fineweb-edu \
  --input <local-text-or-token-jsonl> \
  --input-format jsonl-tokens

experiments/mini_kimi_k3/run.sh prepare-stage1-local \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --reports-dir experiments/mini_kimi_k3/reports \
  --source fineweb-edu \
  --dataset-id <stable-local-or-hf-dataset-id> \
  --snapshot <snapshot-or-commit> \
  --license <license-or-policy> \
  --tokenizer-sha256 <k3-tokenizer-fingerprint> \
  --input <already-local-jsonl-or-parquet> \
  --input-format jsonl-tokens \
  --decontamination-report reports/decontamination.json

experiments/mini_kimi_k3/run.sh write-stage1-input-manifest \
  --output experiments/mini_kimi_k3/reports/source-inputs.json \
  --input-format jsonl-tokens \
  --input <already-local-jsonl-or-parquet> \
  [--input <another-local-file>]

TORCHTITAN_ROOTFS_NETWORK=networked experiments/mini_kimi_k3/run.sh materialize-stage1-local \
  --source-resolution-report experiments/mini_kimi_k3/reports/stage1-source-resolution.json \
  --corpus-plan-report experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --output-root experiments/mini_kimi_k3/data/stage1-local-inputs \
  --report experiments/mini_kimi_k3/reports/stage1-local-materialize.json \
  --allow-download

experiments/mini_kimi_k3/run.sh plan-r1-corpus \
  --mix-plan experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara/data/mix_plan.json \
  --report experiments/mini_kimi_k3/reports/r1-corpus-plan.json

TORCHTITAN_ROOTFS_NETWORK=networked experiments/mini_kimi_k3/run.sh probe-stage1-sources \
  --source-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara \
  --report experiments/mini_kimi_k3/reports/stage1-source-resolution.json

TORCHTITAN_ROOTFS_NETWORK=networked experiments/mini_kimi_k3/run.sh stage1-remote-readiness \
  --env-file .scratch/mini-kimi-k3-replication/.env \
  --report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json

TORCHTITAN_ROOTFS_NETWORK=networked experiments/mini_kimi_k3/run.sh run-stage1-remote \
  --source-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara \
  --readiness-report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json \
  --env-file .scratch/mini-kimi-k3-replication/.env \
  --step probe \
  --report experiments/mini_kimi_k3/reports/stage1-remote-probe.json

experiments/mini_kimi_k3/run.sh audit-r1-corpus-plan \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --report experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json

experiments/mini_kimi_k3/run.sh corpus-disk-readiness \
  --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --target-path experiments/mini_kimi_k3/data/tokens \
  --report experiments/mini_kimi_k3/reports/corpus-disk-readiness.json \
  --overhead-fraction 0.20

experiments/mini_kimi_k3/run.sh decontaminate-local \
  --benchmark <local-benchmark-text-jsonl> \
  --input <local-source-text-jsonl> \
  --output experiments/mini_kimi_k3/data/clean/fineweb-edu.jsonl \
  --report experiments/mini_kimi_k3/reports/decontamination.json \
  --source fineweb-edu

experiments/mini_kimi_k3/run.sh tiny-smoke \
  --token-manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --report experiments/mini_kimi_k3/results/tiny_smoke.json \
  --seq-len 16 \
  --batch-size 2 \
  --steps 1

experiments/mini_kimi_k3/run.sh r1-training-smoke \
  --token-manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --report experiments/mini_kimi_k3/results/r1_training_smoke.json \
  --seq-len 4096 \
  --steps 1

experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness \
  --report experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json \
  --min-free-gpu-memory-mib 12000

experiments/mini_kimi_k3/run.sh preflight \
  --mode tiny-plumbing \
  --report experiments/mini_kimi_k3/results/preflight.json \
  --tiny-smoke-report experiments/mini_kimi_k3/results/tiny_smoke.json \
  --evidence-results-root experiments/mini_kimi_k3/results \
  --run-id mini-kimi-k3-r1-local \
  --attempt-id attempt-001 \
  --token-manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --seq-len 4096
```

It writes a JSON report and exits `21` while required launch gates are missing.
In `--mode full`, the corpus gate requires at least the r1 target budget of 5B
assigned tokens plus Kimi K3 tokenizer identity, tokenizer fingerprint,
completed decontamination evidence, and source provenance. The
`init-manifest` command creates only the metadata skeleton with empty shard
lists; it is not full-run evidence until actual raw `uint32` shards are present
and listed. `register-shards` appends shard names and records byte size,
`uint32` token count, and SHA-256 for each file. `import-shards` handles the
first-party Stage 1 export shape, validating every `.bin` shard against its
same-stem `.json` sidecar before copying or symlinking it under
`<tokens-dir>/<source>/` and updating manifest metadata. `build-local-shards`
is a local producer for already-downloaded text, JSONL, or Parquet inputs; it
writes the same raw little-endian `uint32` shard and sidecar format, then
registers the shards in the manifest. `prepare-stage1-local` wraps that local
producer, creates or updates the manifest source, and writes a source
provenance report with input and shard hashes. It still consumes only
already-local files. `materialize-stage1-local` is the non-Modal acquisition
path for public Hugging Face parquet inputs: it consumes the resolved source
report and r1 corpus plan, requires `--allow-download`, writes one
`mini_kimi_k3_source_input_manifest` per planned source, and does not tokenize,
decontaminate, or register shards by itself. It is an input-materialization
step for `prepare-stage1-local`, not launch evidence. If
`--allow-placeholder-decontamination` is used, full preflight will continue to
reject the manifest until reviewed decontamination evidence is supplied. When
`tokenizer.asset_report` is present, full preflight verifies that the report is
a Mini-K3 tokenizer asset report and that its aggregate SHA-256 matches the
manifest tokenizer fingerprint. When `decontamination.report` is a local file,
full preflight recomputes its SHA-256 and compares it with the manifest
decontamination fingerprint. A corpus source can also carry
`provenance.report` plus `provenance.sha256`; for local reports, full preflight
recomputes the file hash and treats a matching report as the source provenance
proof. Full preflight also recomputes shard metadata before accepting the
corpus gate. `plan-r1-corpus` writes the source-level 5B acquisition target;
`audit-r1-corpus-plan` compares that plan with the current manifest and exits
`21` until every planned source has enough registered tokens. The audit report
is an acquisition/sharding checklist, not launch evidence by itself.
`corpus-disk-readiness` checks the filesystem that will hold raw `uint32`
shards against the r1 token budget plus configurable overhead, and exits `21`
while free space is insufficient. It does not create, download, tokenize, or
register corpus data. Use `--mode tiny-plumbing` for small fixture manifests.

## Claim Ladder

- **Scaffold:** rootfs-governed entrypoint and experiment-owned package exist.
- **Token-shard gate:** tiny `uint32` shards can be read, sharded by rank, and
  resumed through the experiment-owned loader.
- **Tiny plumbing:** synthetic or tiny `uint32` shards flow through a
  Mini-K3-shaped CPU training smoke with token loading, forward, backward,
  and one or more AdamW optimizer steps.
- **Recipe contract:** source-derived r1 launch budget, sequence length,
  optimizer-step count, WSD schedule fractions, optimizer policy, checkpoint
  cadence, and dtype are encoded in a tested contract.
- **Methodology reproduction:** r1 scale, token accounting, WSD schedule,
  checkpoint/resume, and evaluation semantics match the source recipe.
- **Architecture fidelity:** KDA, MLA, SITU, MoE routing, shared experts,
  noaux_tc bias, and parameter counts are verified against an oracle.
- **Training evidence:** the 5B-token r1 attempt has immutable manifests,
  checkpoint lineage, eval summaries, throughput, and cost evidence.

Smoke runs only prove plumbing. They do not prove reward improvement,
convergence, throughput, or reproduction quality.

## Commands

The current runner is a guardrail, not a launcher. It supports:

```bash
experiments/mini_kimi_k3/run.sh init-evidence \
  --results-root experiments/mini_kimi_k3/results \
  --run-id mini-kimi-k3-r1-local \
  --attempt-id attempt-001 \
  --mode full

experiments/mini_kimi_k3/run.sh init-manifest \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source <source-key> \
  --dataset-id <dataset-id> \
  --snapshot <snapshot-or-commit> \
  --license <license-or-policy> \
  [--source-provenance-report <report-path-or-uri>] \
  [--source-provenance-sha256 <report-fingerprint>] \
  --tokenizer-sha256 <k3-tokenizer-fingerprint> \
  [--tokenizer-asset-report experiments/mini_kimi_k3/assets/kimi-k3-tokenizer.json] \
  --decontamination-report <report-path-or-uri> \
  --decontamination-sha256 <report-fingerprint>

experiments/mini_kimi_k3/run.sh install-tokenizer-assets \
  --source-tokenizer-dir <first-party-tokenizer-dir> \
  --output-dir experiments/mini_kimi_k3/assets/kimi-k3-tokenizer \
  --fingerprint-report experiments/mini_kimi_k3/assets/kimi-k3-tokenizer.json

experiments/mini_kimi_k3/run.sh compare-forward-oracle \
  --first-party-logits <first-party-logits.pt> \
  --torchtitan-logits <torchtitan-logits.pt> \
  --report experiments/mini_kimi_k3/results/forward_oracle.json \
  --model-flavor r1 \
  --max-abs-diff 1e-5 \
  --max-rel-diff 1e-5

experiments/mini_kimi_k3/run.sh trace-forward-oracle \
  --source-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara \
  --first-party-state-dict .cache/trae-mini-k3-trainer/first_party_r1_state.pt \
  --output .cache/trae-mini-k3-trainer/forward_trace_r1_candidate_cuda.json \
  --input-ids 1,2,3,4 \
  --device cuda:7

experiments/mini_kimi_k3/run.sh dump-torchtitan-logits \
  --config mini_kimi_k3_tiny_plumbing \
  --output experiments/mini_kimi_k3/results/torchtitan_tiny_logits.pt \
  --input-ids 1,2,3,4 \
  --seed 1234

experiments/mini_kimi_k3/run.sh dump-torchtitan-logits \
  --config mini_kimi_k3_r1_contract \
  --output .cache/trae-mini-k3-trainer/torchtitan_r1_logits.pt \
  --input-ids 1,2,3,4 \
  --seed 1234 \
  --allow-unverified-r1

MINI_KIMI_K3_ORACLE_PYTHON=.cache/trae-mini-k3-trainer/oracle-venv/bin/python \
experiments/mini_kimi_k3/run.sh dump-first-party-logits \
  --source-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara \
  --output .cache/trae-mini-k3-trainer/first_party_r1_logits.pt \
  --input-ids 1,2,3,4 \
  --seed 1234 \
  --device cuda:0 \
  --prepare-for-training

experiments/mini_kimi_k3/run.sh register-shards \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source <source-key> \
  --shard experiments/mini_kimi_k3/data/tokens/<source-key>/<shard>.bin \
  [--shard experiments/mini_kimi_k3/data/tokens/<source-key>/<another-shard>.bin]

experiments/mini_kimi_k3/run.sh import-shards \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source <source-key> \
  --source-dir <stage1-volume-export>/tokens/<source-key> \
  [--link-mode copy|symlink]

experiments/mini_kimi_k3/run.sh build-local-shards \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --source <source-key> \
  --input <local-input-file> \
  [--input <another-local-input-file>] \
  [--input-format text|jsonl-text|jsonl-tokens|parquet-text|parquet-tokens] \
  [--text-field text] \
  [--tokens-field tokens] \
  [--tokenizer-dir experiments/mini_kimi_k3/assets/kimi-k3-tokenizer] \
  [--shard-tokens 100000000]

experiments/mini_kimi_k3/run.sh prepare-stage1-local \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --reports-dir experiments/mini_kimi_k3/reports \
  --source <source-key> \
  --dataset-id <dataset-id> \
  --snapshot <snapshot-or-commit> \
  --license <license-or-policy> \
  --tokenizer-sha256 <k3-tokenizer-fingerprint> \
  [--input-manifest <source-inputs.json>] \
  [--input <local-input-file>] \
  [--input <another-local-input-file>] \
  [--input-format text|jsonl-text|jsonl-tokens|parquet-text|parquet-tokens] \
  [--decontamination-report <reviewed-decontamination-report.json>] \
  [--decontamination-sha256 <expected-report-sha256>] \
  [--allow-placeholder-decontamination]

experiments/mini_kimi_k3/run.sh write-stage1-input-manifest \
  --output <source-inputs.json> \
  --input-format text|jsonl-text|jsonl-tokens|parquet-text|parquet-tokens \
  --input <local-input-file> \
  [--input <another-local-input-file>] \
  [--text-field text] \
  [--tokens-field tokens]

experiments/mini_kimi_k3/run.sh plan-r1-corpus \
  --mix-plan experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara/data/mix_plan.json \
  --report experiments/mini_kimi_k3/reports/r1-corpus-plan.json

experiments/mini_kimi_k3/run.sh audit-r1-corpus-plan \
  --manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --report experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json

experiments/mini_kimi_k3/run.sh corpus-disk-readiness \
  --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json \
  --target-path experiments/mini_kimi_k3/data/tokens \
  --report experiments/mini_kimi_k3/reports/corpus-disk-readiness.json \
  [--overhead-fraction 0.20]

experiments/mini_kimi_k3/run.sh inspect-stage1-inputs \
  --input-manifest <source-inputs.json> \
  --report experiments/mini_kimi_k3/reports/source-inputs-inspection.json \
  [--min-total-tokens 5000000000] \
  [--input-format text|jsonl-text|jsonl-tokens|parquet-text|parquet-tokens] \
  [--text-field text] \
  [--tokens-field tokens] \
  [--tokenizer-dir experiments/mini_kimi_k3/assets/kimi-k3-tokenizer]

experiments/mini_kimi_k3/run.sh decontaminate-local \
  --benchmark <local-benchmark-text-file> \
  [--benchmark <another-local-benchmark-text-file>] \
  --input <local-source-text-file> \
  [--input <another-local-source-text-file>] \
  --output <clean-jsonl-output> \
  --report <decontamination-report.json> \
  --source <source-key> \
  [--benchmark-format text|jsonl-text|parquet-text] \
  [--input-format text|jsonl-text|parquet-text] \
  [--text-field text]

experiments/mini_kimi_k3/run.sh preflight \
  --mode full \
  --report <path> \
  [--tiny-smoke-report <tiny-smoke-report.json>] \
  [--r1-training-smoke-report <r1-training-smoke-report.json>] \
  [--r1-corpus-plan-audit-report <r1-corpus-plan-audit.json>] \
  [--corpus-disk-readiness-report <corpus-disk-readiness.json>] \
  [--stage1-source-resolution-report <stage1-source-resolution.json>] \
  [--stage1-input-inspection-report <stage1-input-inspection.json>] \
  [--stage1-remote-readiness-report <stage1-remote-readiness.json>] \
  [--evidence-results-root <results-root> --run-id <run-id> --attempt-id <attempt-id>] \
  [--oracle-root <first-party-checkout> --oracle-r1-config <r1-config.json>] \
  [--forward-oracle-report <forward-oracle-report.json>] \
  [--launch-backend-report <launch-backend-report.json>] \
  [--token-manifest <manifest.json> --tokens-dir <tokens-dir> --seq-len 4096]

experiments/mini_kimi_k3/run.sh tiny-smoke \
  --token-manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --report experiments/mini_kimi_k3/results/tiny_smoke.json \
  --seq-len 16 \
  --batch-size 2 \
  --steps 1

experiments/mini_kimi_k3/run.sh r1-training-smoke \
  --token-manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --report experiments/mini_kimi_k3/results/r1_training_smoke.json \
  --seq-len 4096 \
  --steps 1

experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness \
  --report experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json \
  [--min-free-gpu-memory-mib 12000]

experiments/mini_kimi_k3/run.sh launch \
  --results-root experiments/mini_kimi_k3/results \
  --run-id mini-kimi-k3-r1-local \
  --attempt-id attempt-001 \
  [--tiny-smoke-report experiments/mini_kimi_k3/results/tiny_smoke.json] \
  [--r1-training-smoke-report <r1-training-smoke-report.json>] \
  [--r1-corpus-plan-audit-report <r1-corpus-plan-audit.json>] \
  [--corpus-disk-readiness-report <corpus-disk-readiness.json>] \
  [--stage1-source-resolution-report <stage1-source-resolution.json>] \
  [--stage1-input-inspection-report <stage1-input-inspection.json>] \
  [--stage1-remote-readiness-report <stage1-remote-readiness.json>] \
  [--oracle-root <first-party-checkout> --oracle-r1-config <r1-config.json>] \
  [--forward-oracle-report <forward-oracle-report.json>] \
  [--launch-backend-report <launch-backend-report.json>] \
  --token-manifest experiments/mini_kimi_k3/data/manifest.json \
  --tokens-dir experiments/mini_kimi_k3/data/tokens \
  --seq-len 4096

experiments/mini_kimi_k3/run.sh completion-audit \
  --preflight-report experiments/mini_kimi_k3/results/preflight.json \
  --results-root experiments/mini_kimi_k3/results \
  --run-id mini-kimi-k3-r1-local \
  --attempt-id attempt-001 \
  --report experiments/mini_kimi_k3/reports/completion-audit.json
```

Likely future commands should stay behind this rootfs-aware entrypoint, for
example:

```bash
experiments/mini_kimi_k3/run.sh tiny-smoke --config mini_kimi_k3_tiny
```

The `launch` command records a preflight stage in the initialized run-attempt
bundle and returns `21` with a blocked outcome while any full-mode gate is
blocked. Once preflight is ready, it starts `run_train.sh` with
`MODULE=mini_kimi_k3`, `CONFIG=mini_kimi_k3_r1_contract`, `NGPU=1`, the
validated token manifest, and the attempt train dump folder.

Oracle paths are optional and must be visible from inside the TorchTitan rootfs.
They compare the exported TorchTitan Mini-K3 contract with first-party
KimiLinearConfig JSON files. First-party r1 logits require the Vizuara-pinned
oracle environment (`transformers==4.57.6`, compatible `kernels`, and
`fla-core`); use `MINI_KIMI_K3_ORACLE_PYTHON` to point the runner at that
ignored oracle venv. Use `--prepare-for-training` for scratch-initialized
first-party r1 logits so the reference `dt_bias` and trainable router patches
are applied before eval. Generate the forward-oracle report with
`compare-forward-oracle` from saved finite first-party and TorchTitan logits.
`dump-torchtitan-logits` can produce a TorchTitan logits payload for the
`mini_kimi_k3_tiny_plumbing` config. The r1 contract config remains guarded:
it refuses by default and emits r1 candidate logits only with the explicit
`--allow-unverified-r1` flag. Those payloads are labeled
`fidelity_status=unverified_candidate` and do not make the model trainable.
For diagnostic narrowing, `dump-first-party-logits` can also emit the exact CPU
state dict from the logits-producing first-party model with
`--state-dict-output`. Then `dump-torchtitan-logits
--align-first-party-weights --first-party-state-dict <path>` loads that saved
state into the TorchTitan candidate, records the alignment report in the logits
payload, and labels it `fidelity_status=aligned_unverified_candidate`. The
current aligned-state payload maps 8,764 TorchTitan parameters and
1,023,206,628 elements with no missing, extra, or shape-mismatched parameters;
the tied first-party `lm_head.weight` entry is verified against
`model.embed_tokens.weight` and skipped because TorchTitan stores the tied
weight once.
The current strongest r1 candidate comparison is recorded at
`.cache/trae-mini-k3-trainer/forward_oracle_r1_candidate_cuda.json` with
`status=pass`, `max_abs_diff=0.0`, and `max_rel_diff=0.0`. This exact-state
CUDA/FLa-backed diagnostic proves that the TorchTitan r1 candidate can
reproduce the prepared first-party forward on the sampled prompt. Pair it with
a passing `review-launch-backend` report for model fidelity, and with a
separate r1 `run_train.sh` smoke report before treating
`mini_kimi_k3_r1_contract` as launchable. The remaining hard blockers for the
full r1 run are Trainer-path smoke evidence and a real 5B-token corpus gate
with tokenizer, decontamination, shard, and source-provenance evidence.
Use `trace-forward-oracle` to produce an OOM-safe layer-by-layer diagnostic: it
keeps the full models on CPU, moves one matching layer at a time to the target
device, and records both isolated layer diffs under exact first-party hidden
states and recursive full-forward drift.
A passing report must have `kind=mini_kimi_k3_forward_oracle`,
`schema_version=1`, `status=pass`, `model_flavor=r1`, and numeric `results`
within declared `tolerances`.

## Implemented Prerequisites

- `torchtitan.experiments.mini_kimi_k3.token_shards.TokenShardLoader` reads raw
  little-endian `uint32` shards through a simple manifest, assigns shards by
  `paths[rank::world_size]`, accounts for emitted tokens, and supports
  checkpoint state round trips.
- `tests/unit_tests/test_mini_kimi_k3_token_shards.py` covers the current loader
  contract with temporary tiny shards.
- `torchtitan.experiments.mini_kimi_k3.recipe` records the r1 launch-recipe
  contract: 5B target tokens, 4096 sequence length, 4 local batch, 8 gradient
  accumulation steps, 131,072 tokens per optimizer step, 38,147 optimizer
  steps, WSD schedule fractions, AdamW betas/epsilon/weight-decay policy,
  router exclusion, 1-D no-decay policy, checkpoint cadence, and bf16.
- `torchtitan.experiments.mini_kimi_k3.activation` implements the SITU gate/up
  activation used by Mini-K3 MLPs.
- `torchtitan.experiments.mini_kimi_k3.router` implements noaux_tc top-k
  routing with non-grad correction-bias state, training load accounting, and
  the bias balancer update.
- `torchtitan.experiments.mini_kimi_k3.moe` implements a differentiable sparse
  MoE dispatch block and SITU MLP primitive for the future full model.
- `torchtitan.experiments.mini_kimi_k3.kda` implements the KDA `dt_bias`
  scratch-training initializer, slow CPU recurrence reference, and a
  CPU-reference `MiniK3DeltaAttention` module.
- `torchtitan.experiments.mini_kimi_k3.mla` implements a CPU-friendly eager MLA
  primitive with Mini-K3's projection layout, RoPE over the rotary q/k slice,
  asymmetric value padding/slicing, and output gate.
- `torchtitan.experiments.mini_kimi_k3.model` assembles these primitives into a
  tiny forward-capable decoder/causal-LM wrapper for CPU plumbing tests.
- `torchtitan.experiments.mini_kimi_k3.config_registry` registers
  `mini_kimi_k3_tiny_plumbing`, a TorchTitan `Trainer.Config` that can run a
  one-step fake-backend training smoke on tiny uint32 token shards.
- `experiments.mini_kimi_k3.training_smoke` runs a rootfs-only fixture-scale
  CPU smoke that reads token shards, builds the tiny decoder, computes
  next-token loss, backpropagates, performs AdamW steps, and writes a compact
  JSON report.
- `experiments.mini_kimi_k3.r1_training_smoke` launches `run_train.sh` with
  `MODULE=mini_kimi_k3`, `CONFIG=mini_kimi_k3_r1_contract`,
  `COMM_MODE=fake_backend`, one GPU rank, one optimizer step, and the supplied
  real token manifest. It writes the r1 Trainer-path smoke report consumed by
  full-mode preflight.
- `experiments.mini_kimi_k3.preflight` writes a machine-readable readiness
  report, validates an optional token-shard manifest through the loader,
  validates optional tiny-smoke evidence for tiny-plumbing mode, rejects
  undersized or under-documented full-run corpora, and blocks launch until
  model/config, corpus, and evidence gates exist.
- `experiments.mini_kimi_k3.init_manifest` initializes the full corpus manifest
  skeleton with tokenizer, decontamination, and source provenance fields. It
  leaves shard lists empty so the corpus gate remains blocked until real token
  shards are populated.
- `experiments.mini_kimi_k3.install_tokenizer_assets` copies the required
  first-party tokenizer files into the experiment asset directory and writes a
  deterministic aggregate SHA-256 report for the manifest tokenizer field.
- `experiments.mini_kimi_k3.compare_forward_oracle` compares saved first-party
  and TorchTitan logits, then writes the forward-oracle report consumed by
  full-mode preflight.
- `experiments.mini_kimi_k3.trace_forward_oracle` writes a layer trace from
  exact first-party state without keeping both full r1 models resident on GPU.
- `experiments.mini_kimi_k3.dump_torchtitan_logits` builds a Mini-K3
  TorchTitan config through `ConfigManager`, emits tiny-plumbing logits for a
  supplied token row, and preserves the r1 contract's default fail-closed
  backend guard. It can emit an explicitly labeled, unverified r1 candidate
  logits payload only when `--allow-unverified-r1` is supplied, with optional
  in-process first-party weight alignment for oracle diagnostics.
- `experiments.mini_kimi_k3.register_shards` registers raw `uint32` shard files
  that already live under `data/tokens/<source>/`, recording each relative shard
  path, byte count, token count, and SHA-256. Full preflight verifies this
  metadata against the actual shard bytes.
- `experiments.mini_kimi_k3.build_local_shards` converts already-local `text`,
  `jsonl-text`, `jsonl-tokens`, `parquet-text`, or `parquet-tokens` inputs into
  first-party-compatible raw `uint32` shards. Text modes require installed Kimi
  tokenizer assets; token JSONL or Parquet is useful for fixture and
  exported-token ingestion.
- `experiments.mini_kimi_k3.prepare_stage1_local` creates or updates the local
  Stage 1 manifest around already-local inputs and writes source provenance
  evidence. Placeholder decontamination is explicitly marked as pending and is
  rejected by full-mode preflight. For large local corpora, prefer
  `--input-manifest` with `kind=mini_kimi_k3_source_input_manifest`,
  `schema_version=1`, a shared `input_format`, and an `inputs` list containing
  each local path plus an expected SHA-256.
- `experiments.mini_kimi_k3.write_stage1_input_manifest` writes that source
  input manifest from explicit local file paths, recording each resolved path,
  byte count, and SHA-256. It refuses to overwrite an existing manifest; run
  `inspect-stage1-inputs` afterward to validate schema and count token volume.
- `experiments.mini_kimi_k3.plan_r1_corpus` scales the first-party 55B-token
  Stage 1 `mix_plan.json` to TorchTitan's 5B-token r1 launch target and writes
  per-source token and estimated file budgets. It is an acquisition plan, not
  corpus evidence; full preflight still requires actual token shards and
  launch-grade decontamination/provenance reports.
- `experiments.mini_kimi_k3.probe_stage1_sources` runs or summarizes the
  first-party Stage 1 Hugging Face metadata probe and writes a stable source
  resolution report. Use `TORCHTITAN_ROOTFS_NETWORK=networked` when running the
  live probe, because the default rootfs network mode is offline. A ready source
  resolution report proves only that the selected repositories, prefixes, and
  fallbacks are resolvable; it does not download data, build the decontamination
  index, tokenize shards, or satisfy the full corpus gate.
- `experiments.mini_kimi_k3.inspect_stage1_inputs` validates a
  `mini_kimi_k3_source_input_manifest` and estimates per-file and total
  document/token counts without writing shards. Use it before a large local
  Stage 1 build to catch stale hashes, missing files, field mismatches, token
  vocabulary overflows, and insufficient local token volume. Pass
  `--min-total-tokens 5000000000` to make the dry run fail closed when the
  local input list cannot meet the r1 corpus target.
- `experiments.mini_kimi_k3.r1_smoke_gpu_readiness` records the visible GPU
  memory state for the r1 Trainer-smoke gate and exits nonzero until at least
  one GPU meets the configured free-memory threshold. It does not launch
  training.
- `experiments.mini_kimi_k3.decontaminate_local` filters already-local text
  inputs through the first-party 13-gram benchmark index logic and writes a
  scoped report plus cleaned JSONL. It only covers benchmark files supplied on
  the command line; it is not a substitute for resolving the full benchmark
  suite from remote datasets.
- `experiments.mini_kimi_k3.stage1_remote_readiness` runs non-ingesting
  readiness checks for the remote Stage 1 path: Hugging Face DNS inside the
  rootfs, Modal CLI availability through `uv tool run --from modal`, and Modal
  authentication. It exits nonzero while any check is blocked and does not
  start remote jobs. Pass `--env-file .scratch/mini-kimi-k3-replication/.env`
  to provide ignored Modal credentials without writing token values into the
  report. Use `.scratch/mini-kimi-k3-replication/.env.example` as the local
  template for the required `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and
  `MODAL_PROFILE` keys; keep the real `.env` file out of git.
- `experiments.mini_kimi_k3.run_stage1_remote` runs one allow-listed first-party
  Modal Stage 1 entrypoint after a ready `stage1-remote-readiness` report. Safe
  steps are `probe`, `gate_tokenizer`, `build_index`, `dry_run`, and `report`.
  The spendful detached `ingest_all` step requires `--allow-spend` and should
  only be used after the readiness report, source resolution, tokenizer gate,
  decontamination index, and dry-run output have been reviewed.
- `experiments.mini_kimi_k3.completion_audit` inspects the full preflight
  report plus the immutable run-attempt outcome/events and exits nonzero unless
  full preflight is ready, every corpus-readiness checklist item is covered, the
  guarded launch completed, and a train stage succeeded with a real measurement.
  The full preflight report and guarded-attempt outcome must both use schema
  version `1`; older or malformed schemas fail closed rather than completing the
  launch checklist.
  The training measurement and train-stage success checks must agree on a
  train `stage_invocation_id` listed in `outcome.json.stage_invocation_ids`,
  and `outcome.json.run_gate.has_real_measurement` must agree with the real
  training measurement. The coordinator event stream must be present and valid
  JSONL; missing, malformed, or non-object event rows fail closed for
  train-stage and real-measurement checks. Stale outcome metadata or mixed
  coordinator events cannot complete a different attempt.
  The corpus checklist breaks out the r1 corpus-plan audit, raw-shard disk
  capacity, local Stage 1 input inspection, source resolution, and remote
  readiness so missing acquisition evidence is visible before any launch. The
  report also includes `success_criteria`, `missing`, `covered`, and
  `not_required` arrays for direct machine-readable launch-status checks.
  Blocked items may include `next_actions[].guidance` with the exact evidence
  needed for the next operator step: `blocked_preflight_requirements` for the
  top-level preflight blocker, `target_tokens`, `available_tokens`, and
  `deficit_tokens` plus per-source `source_deficits` for corpus shortfall
  blockers, `r1_training_smoke_report` for the Trainer-smoke blockers, and
  `env_file_template` plus required Modal keys for the remote Stage 1 readiness
  blocker. Post-preflight launch blockers include `attempt_outcome_report`,
  `execution_outcome`, and `measurement` from the immutable attempt outcome,
  plus `event_stream`, `terminal_events`, and `event_stream_errors` from the
  coordinator event log.
  The CLI also prints a human-readable `next actions:` section on blocked runs;
  each action includes a compact `guidance:` line when structured guidance is
  available.
- `experiments.mini_kimi_k3.evidence` initializes the immutable
  `runs/<run-id>/<attempt-id>/manifest.json` bundle used by preflight and later
  launch stages.

The r1 Mini-K3 `Trainer.Config` and guarded launch train stage are implemented,
but full training remains blocked until full-mode preflight accepts a real
5B-token manifest, r1 Trainer-path smoke evidence, and the launch starts
through `run.sh launch`. Full-mode preflight requires
`--r1-corpus-plan-audit-report` so the source-level 5B token plan and deficit
state are explicit, and `--stage1-source-resolution-report` so Stage 1 corpus
and benchmark sources are resolved before launch. Pass
`--corpus-disk-readiness-report` when the raw-shard target filesystem capacity
report should be embedded and validated before launch. Pass
`--stage1-remote-readiness-report` when the remote Stage 1
Modal/Hugging Face readiness report should be embedded and validated before
launch; if the manifest already proves a complete local Stage 1 or imported
first-party shard route, the remote readiness evidence is recorded as
`not_required` instead of blocking launch. Pass
`--stage1-input-inspection-report` when the manifest uses a local Stage 1 input
route; full preflight and completion audit require that inspection report to
pass with enough local input tokens before the local route counts as
launch-ready evidence. The exact-state CUDA forward-oracle diagnostic and launch-backend review
can prove the model-fidelity side of readiness; they are not Trainer execution,
corpus, convergence, throughput, or evaluation evidence.

## Generated Artifacts

Keep generated data, checkpoints, rollouts, caches, benchmark outputs, and full
run bundles out of git. Commit source, tests, registries, compact summaries,
and reviewed reports. Use stable run IDs and preserve source, tokenizer, corpus,
checkpoint, evaluation, and rootfs provenance before making any training claim.
