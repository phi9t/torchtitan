# Falcon experiment runners

Rootfs-managed commands for the Falcon fast-weight mixer. Reusable Python
lives in `torchtitan/experiments/falcon/`.

```bash
experiments/falcon/run.sh overfit
experiments/falcon/run.sh train
experiments/falcon/run.sh addition --arm A0 --seed 0 --steps 4 --dry-run
experiments/falcon/run.sh mechanism --arm M0 --seed 0 --steps 4 --dry-run
experiments/falcon/run.sh checkpoint-eval --arm A0 --checkpoint experiments/falcon/results/hero/hero_20k/A0_seed0/checkpoint/step-20000
```

`overfit` is the Mercor Step-3 analog: a tiny Falcon-1A LM must memorize a
fixed 8-sequence bank. If that fails, do not start a larger run.

`addition` owns one B10 matrix cell per invocation: exactly one arm and one
seed. It builds the immutable three-draw addition split manifest, uses an
online ID-width stream that excludes every held-out identity under commutation,
and writes a native attempt bundle for both success and argument/validation
failure. Dry runs are smoke claims and do not create checkpoints.

`mechanism` owns one M02 matrix cell per invocation: exactly one M arm and one
seed. Screen attempts require observation probes at optimizer steps 0, 100,
500, 1000, and 2000; confirmation attempts add 4000 and 8000. Dry runs are
non-science smoke attempts and write only step 0 plus final-step observations.
Diagnostic JSON records per-layer/per-head summary statistics for key energy,
beta, actual lambda, eta, gamma, state/update/read norms, clamp fraction, and
the M4 canonicalized state norm relation. Native bundles land under
`experiments/falcon/results/mechanism/evidence/runs/`; raw diagnostics land
under `experiments/falcon/results/mechanism/artifacts/`.

## Evidence

Each future Falcon science launch must write one native attempt bundle through
`torchtitan.experiments.falcon.evidence.write_native_attempt_bundle`. A logical
workload, arm, and seed identify a training run, with canonical ID
`falcon-<workload>-<arm>-seed<seed>`; every restart writes a distinct attempt
under that run. The bundle records the run and attempt identity, declared lane
and claim, process/rank/mesh/device identity, clocks, phases, step facts, data
and checkpoint lineage, artifact index, and outcome.

Campaign B predates that contract. Regenerate its compact, hash-linked legacy
ledger inside the rootfs; this reads source artifacts and writes no checkpoints
or copied raw evidence:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && \
  python -m torchtitan.experiments.falcon.evidence import-legacy'
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && \
  python -m torchtitan.experiments.falcon.evidence verify-ledger'
```

The ledger is immutable and idempotent: regeneration verifies existing source
digests and rejects a changed or corrupt source instead of rewriting history.
It distinguishes raw sources, copied files, derived summaries, and associated
diagnostics/checkpoints; checkpoint directories are indexed but not copied or
content-hashed.

The imported Campaign-B ledger carries catalog ID `falcon_pre_b09_v1`.
The `verify-ledger` command always invokes the caller-owned known-catalog
verifier, which requires that exact marker and regenerates the full catalog from
the repository's fixed importer, artifact inventory, and immutable source
artifacts before accepting stored facts or classifications. Ad hoc fixture
ledgers remain supported through `verify_legacy_ledger`, but that explicitly
limited API performs only structural, selector, binding, and source-digest
checks. It rejects catalog-bearing ledgers with an instruction to use
`verify_known_pre_b09_ledger`.

## Checkpoint Evaluation

`checkpoint-eval` is the B11 fixed-region LM evaluation path for historical
hero checkpoints. It restores exactly one checkpoint in read-only mode, writes
all logs, temporary state, and JSON results under
`experiments/falcon/results/checkpoint_eval/<run_id>/<attempt_id>/`, and writes
the native attempt bundle under `experiments/falcon/results/evidence/runs/`.

The logical eval identity is stable for one arm:
`falcon-b11-checkpoint-eval-<arm>-seed0`. Every launch gets a distinct
`attempt_id`, so repeated evaluations of the same checkpoint can coexist while
remaining comparable.

The scorer freezes three nonoverlapping spans of `fineweb_val_000000.bin` in
`torchtitan.experiments.falcon.lm_eval.RegionRegistry`. The registry records the
source path, source SHA-256 digest, source token count, region offsets/counts,
registry version, and registry digest. Region results must carry that digest;
`aggregate_region_ce` rejects results from any other registry and computes:

```text
aggregate_ce = sum(region_loss_sum) / sum(region_token_count)
ppl = exp(aggregate_ce)
```

Before and after restore/eval, the runner snapshots every historical
checkpoint file's metadata and SHA-256 digest. The command fails if the caller
provides a mismatched `--expected-checkpoint-digest` or if any checkpoint file
is mutated during evaluation.
