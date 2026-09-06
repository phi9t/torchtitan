# Mini Kimi K3 Stage 1 Remote Readiness

Date: 2026-08-22T01:09Z

Purpose: record the safe, non-ingesting checks for the first-party Stage 1
corpus path before any Modal job or large Hugging Face download is authorized.

## Current Status

- Source metadata resolution: ready when the rootfs is entered with
  `TORCHTITAN_ROOTFS_NETWORK=networked`.
- Modal CLI availability: available through `uv tool run --from modal modal`.
- Modal authentication: not available in the rootfs environment.
- Full Stage 1 corpus: not present locally.
- Full Mini-K3 launch: blocked.

## Evidence

### Machine-Readable Readiness Report

Command:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh stage1-remote-readiness --report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json'
```

If Modal credentials are stored in the ignored operator file, prefer passing it
directly instead of sourcing it in the shell:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh stage1-remote-readiness --env-file .scratch/mini-kimi-k3-replication/.env --report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json'
```

Result:

- The command exited `21` and wrote
  `experiments/mini_kimi_k3/reports/stage1-remote-readiness.json`.
- Report status is `blocked`.
- `checks.huggingface_dns.status` is `pass`.
- `checks.modal_cli.status` is `pass`, with Modal client version `1.5.4`.
- `checks.modal_auth.status` is `blocked`, with detail `Token missing`.
- `rootfs.network_mode` is `networked`.

### Hugging Face Source Probe

Command:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan/experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara && python data/probe.py'
```

Result:

- `data/resolved.json` was rewritten with 8 resolved benchmarks and 6 resolved
  corpus sources.
- `experiments/mini_kimi_k3/reports/stage1-source-resolution.json` summarizes
  that result with `status=ready`, `unresolved_benchmarks=[]`, and
  `unresolved_corpora=[]`.
- Selected corpus sources:
  - `fineweb-edu`: `HuggingFaceFW/fineweb-edu`, `sample-100BT`,
    `sample/100BT/`
  - `web-diverse`: `HuggingFaceFW/fineweb`, `sample-100BT`, `sample/100BT/`
  - `finemath`: `HuggingFaceTB/finemath`, `finemath-4plus`,
    `finemath-4plus/`
  - `open-web-math`: `open-web-math/open-web-math`, `data/`
  - `code-python`: `codeparrot/github-code-clean`, `Python-all`, `data/`,
    filtered to `language=Python`
  - `cosmopedia`: `HuggingFaceTB/cosmopedia-v2`, `cosmopedia-v2/`

This proves only repository and prefix resolution. It does not download corpus
data, build the benchmark decontamination index, tokenize documents, or create
launch-grade shards.

### Rootfs Network Mode

Default rootfs mode is offline. In that mode `/etc/resolv.conf` is empty and
`curl -I https://huggingface.co` fails with:

```text
Could not resolve host: huggingface.co
```

The Stage 1 metadata probe must therefore use:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- ...
```

### Modal CLI

Command:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && uv tool run --from modal modal --version'
```

Result:

```text
modal client version: 1.5.4
```

`modal` is not installed in the system rootfs environment. `uv pip install
modal` fails because repo `.venv/bin/python3` is a stale symlink. `python -m pip
install modal` is blocked by the externally-managed system environment.
`uv tool run --from modal modal ...` is the verified least-invasive Modal CLI
path.

### Modal Authentication

Command:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && uv tool run --from modal modal token info'
```

Result:

```text
Token missing. Could not authenticate client.
```

No `MODAL_*` token environment variables were visible in the rootfs environment.
The active profile name is `default`, but it is not authenticated.

## Next Authorized Remote Steps

These commands are not safe to run automatically in the current session because
they require Modal authentication and may submit remote jobs or spend money.

1. Authenticate Modal in the same rootfs-visible state used by the runner.
2. Re-run the metadata probe through Modal if the intended ingest workers must
   use a Modal `hf-secret` rather than anonymous Hugging Face access:

   ```bash
   TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan/experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara && uv tool run --from modal modal run data/stage1.py::probe'
   ```

3. Only after source resolution is reviewed with the production credentials,
   run the first-party Stage 1 gates in order:
   - `modal run data/stage1.py::gate_tokenizer`
   - `modal run data/stage1.py::build_index`
   - `modal run data/stage1.py::dry_run`
   - `modal run --detach data/stage1.py::ingest_all`
   - `modal run data/stage1.py::report`

4. Export or mount the resulting `/data/tokens/<source>/*.bin` and same-stem
   `.json` sidecars, then import each source with
   `experiments/mini_kimi_k3/run.sh import-shards`.

5. Re-run `audit-r1-corpus-plan` and full preflight before any guarded launch.
