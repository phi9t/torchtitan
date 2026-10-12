# FineWeb data preparation and manifest

Type: task
Status: resolved
Blocked by: -

## Requirement

Add the data-preparation slice for the Modded NanoGPT B200 benchmark. The
implementation must preserve upstream train and validation token streams for
competition-comparable claims.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec.

## Scope

Allowed:

- wrap upstream `data/cached_fineweb10B.py`;
- add a rootfs-aware `prepare_data.sh` wrapper;
- add a smoke-size data path that is clearly classified as `smoke`;
- write checksum manifests under the experiment `results/` or `data/` area;
- record fresh versus reused data status.

Excluded:

- no tokenizer or data-pipeline change for competition-comparable runs;
- no host-side Python or host-side dataset download code;
- no full training run;
- no data files tracked in git.

## Verification Evidence

- A smoke manifest generated from a cheap data-preparation path, or a blocker
  with the exact missing dependency/network condition.
- Manifest records `schema_version`, dataset identity, token budget, source
  commit, command, environment, fresh/reused status, shard paths, sizes,
  checksums, total bytes, and shard count.
  - `python -m pytest tests/unit_tests/test_modded_nanogpt_b200_prepare_data.py -q`
    passed for manifest fields, SHA256s, atomic writes, missing-shard failure,
    and full-manifest shape validation.
  - Clean-context manifest SHA hardening fixed the manifest's top-level SHA
    state: `prepare_data.build_manifest` now writes `verified_sha: true` when
    it computes shard `sha256` entries. The root cause was that manifests had
    per-shard hashes but no top-level `verified_sha`, while parser and
    run-index launch-prerequisite logic read parsed manifest SHA evidence from
    that top-level field. TDD evidence for the prepare-data/preflight/summary
    SHA slice recorded RED `3 failed, 45 passed`, GREEN
    `48 passed in 0.15s`, clean-context verification `48 passed in 0.08s`,
    `py_compile` passing for `prepare_data.py`, `preflight.py`, and
    `summarize.py`, and `git diff --check` passing for scoped code/tests.
    A test-only follow-up added coverage and recorded `47 passed in 0.15s`
    plus `git diff --check` passing; no production code change was needed.
- Full manifest records exactly 10 `.bin` shards, total bytes `2000010240`,
  pinned upstream commit `ecbb586296d3dac36fd206211f25d63bad4a6b35`, and
  complete SHA256 entries.
  - `experiments/modded_nanogpt_b200/prepare_data.sh --source experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa --data-dir experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/data/fineweb10B --output experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T000342Z/data_manifest.json --token-budget 900M --freshness reused --skip-upstream-command`
    ran through the rootfs wrapper and wrote a schema-v1 full manifest from
    existing shards. The manifest records `token_budget=900M`, `num_files=10`,
    `total_bytes=2000010240`, source commit
    `ecbb586296d3dac36fd206211f25d63bad4a6b35`, and SHA256 entries for all 10
    shards. Freshness is `reused`; no upstream data download command was run.
- Manifest writes are atomic and do not leave a partial file on interruption.
- Wrapper invocation from the host re-enters rootfs before Python runs.
  - `experiments/modded_nanogpt_b200/prepare_data.sh` re-enters
    `scripts/rootfs/enter_rootfs.sh` before invoking data helper Python.
