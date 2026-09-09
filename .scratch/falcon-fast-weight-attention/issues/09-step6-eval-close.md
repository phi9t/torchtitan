# 09 - Version Falcon evidence and import legacy runs

Type: task
Status: resolved
Blocked by: none
Parent: ../spec.md

## Decomposition history

The research owner approved decomposing the former broad Step-6 closeout on
2026-09-07. Its evaluation, transfer, no-regression, claim-label, and written
closeout intent is preserved in ticket 13. Ticket 09 is now the evidence gate
that blocks every new science run.

## What to build

Make every existing Falcon result discoverable through one regenerable,
versioned evidence ledger while giving future Falcon runs a validated native
attempt-bundle contract.

## Acceptance criteria

- [x] A legacy import record has a deterministic import ID, immutable source
  location and digest, nullable source-declared run and attempt IDs, explicit
  inferred linkage, schema version, available configuration/environment/timing
  facts, and reasons for every unavailable required field.
- [x] A native Falcon attempt bundle requires nonempty logical run and attempt
  IDs plus lane, mode, arm, claim label, evidence tier, environment class,
  process/rank/mesh/device identity, clocks, steps, phases, data/checkpoint
  lineage, artifact index, and outcome.
- [x] One arm and seed is one logical training run; restarts are attempts of
  that run, and matrix orchestration cannot collapse several experiments into
  one evidence identity.
- [x] The importer covers valid, smoke, invalid, incomplete, and omitted Falcon
  evidence while distinguishing raw artifacts from copied or derived tables.
- [x] Importing is idempotent, deduplicates derived artifacts, detects changed
  or corrupted sources, and never copies large checkpoints or fabricates
  provenance.
- [x] The regenerated ledger reproduces the audited Campaign B inventory and
  every row resolves to its immutable source artifact.
- [x] Focused rootfs tests cover both record contracts, conflict/corruption
  failures, idempotence, and the known legacy fixtures; the Falcon suite and
  changed-file lint pass.

## Exclusions

- No training, checkpoint mutation, artifact deletion, or remote export.
- No rewriting historical JSON to look like native evidence.

## Answer

Resolved 2026-09-08 after a targeted trust-boundary rerun, independent review,
and finalizer acceptance. The canonical `falcon_pre_b09_v1` ledger contains 55
pre-B09 records, including the separately checkable 52-row Campaign B subtotal
(`valid=20`, `smoke=23`, `invalid=2`, `incomplete=2`, `omitted=5`), plus 83
artifact-index entries and 33 immutable source paths. Its SHA-256 is
`4c7e150629f5595e7e4600a82ad2129f870eec277d1f78f624bb0ef7d4b57cd8`.

The known-catalog verifier now owns its trust policy in code, rebuilds the
complete expected payload from the fixed catalog and source bytes, and fails
closed when the catalog marker is missing, null, wrong, unknown, or deleted
after coordinated payload/binding edits. Generic ad hoc verification is a
separate API and cannot verify catalog-bearing ledgers. The native writer keeps
one arm and seed as one logical run and publishes distinct immutable attempts.

Fresh rootfs verification: 31 focused evidence tests passed, the complete
Falcon suite passed 100 tests (17 known warnings), canonical CLI verification
returned 55, and changed-file lint plus `git diff --check` passed. Finalizer:
`.superpowers/sdd/09-step6-eval-close/task-09-rerun-finalizer.md`. No GPU was
used and historical raw evidence was not changed or copied.
