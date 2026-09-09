# 12 - Decide GDN addition feasibility

Type: task
Status: complete
Blocked by: 10
Parent: ../spec.md

## What to build

Run the predeclared 100-step GDN addition preflight and issue a reproducible
spend-or-omit decision for the replicated Campaign B control.

## Acceptance criteria

- [x] The preflight uses the final addition shape, batch, online stream,
  evaluation registry, precision, and native attempt-bundle contract.
- [x] Cost projection uses post-warmup steady-state step timing and separately
  accounts for initialization, three 2,000-step training seeds, all evaluation
  draws, checkpoint writing, failed attempts, and the preflight itself.
- [x] GPU-hours are the sum of device wall time; parallel execution does not
  discount the projection.
- [x] Disk preflight records free space, projected peak bytes, retained
  artifacts, and safety reserve before the GPU launch.
- [x] A machine-readable decision authorizes the full GDN matrix only when its
  projected total is at most two B200 GPU-hours; otherwise it records a
  performance-based omission with no partial science claim.
- [x] Focused rootfs tests prove that exactly two projected B200-hours is
  eligible, any larger value is omitted, and missing steady-state timing,
  incomplete matrix coverage, or invalid retry accounting cannot authorize
  spend.
- [x] The preflight attempt, projection inputs, arithmetic, and decision are
  independently reproducible from the evidence bundle.

## Exclusions

- No full 2,000-step GDN addition matrix.
- No GDN kernel optimization or change to the two-B200-hour ceiling.

## B12 execution record

Machine-readable status artifact:
`../evidence/b12-gdn-addition-decision.json`.

Implemented:

- `torchtitan.experiments.falcon.promotion` defines the B12 accounting
  boundary and machine-readable `gdn_addition_decision` payload.
- `experiments/falcon/campaign_driver.py` exposes only the B12
  `addition-preflight` command and computes disk, preflight, steady-state,
  eval-draw, checkpoint, failure, and one-retry-per-logical-run accounting.
- `experiments/falcon/run.sh addition-preflight --arm A1 --seed 0 --steps 100`
  routes through the repo-local rootfs wrapper.
- The final-shape addition evaluator overflow was fixed by preserving large
  OOD operands as Python integers instead of `torch.long` tensors.

Verification:

- Focused rootfs tests:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m py_compile experiments/falcon/campaign_driver.py torchtitan/experiments/falcon/promotion.py torchtitan/experiments/falcon/addition.py && pytest -q tests/unit_tests/test_falcon_promotion.py tests/unit_tests/test_falcon_addition_protocol.py tests/unit_tests/test_falcon_addition.py tests/unit_tests/test_falcon_evidence.py'`
  -> 59 passed.
- Full Falcon rootfs tests:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'`
  -> 151 passed, 17 warnings.

Preflight outcome:

- Authorized command attempted:
  `experiments/falcon/run.sh addition-preflight --arm A1 --seed 0 --steps 100`.
- Attempt 1 failed before producing timing evidence:
  `ValueError: Overflow when unpacking long long`.
- Root cause: final-shape OOD eval operands are up to 24 digits and exceeded
  `torch.long` in `AdditionDataset.collate`.
- Fix: operand metadata now remains Python integers; regression test added.
- Attempt 2 failed after train/eval before artifact publication:
  `RuntimeError: Parent directory experiments/falcon/results/addition_preflight/b12-gdn-addition-preflight-A1-seed0-5ced20eed164 does not exist.`
- Root cause: checkpoint timing path called `torch.save` before creating the
  per-attempt artifact directory.
- Fix: checkpoint parent directory is created before save.

Published preflight:

- Artifact directory:
  `experiments/falcon/results/addition_preflight/b12-gdn-addition-preflight-A1-seed0-fc582fdc0e1f`.
- Native attempt bundle:
  `experiments/falcon/results/evidence/runs/falcon-b12-gdn-addition-preflight-A1-seed0/b12-gdn-addition-preflight-A1-seed0-fc582fdc0e1f`.
- Split manifest digest:
  `15a78d922fd042d03f682c73e448befffeb85127fcd8bc32e2ebac4f838c1cb7`.

Decision:

`performance_omission`. The measured projection is
`0.31080121978043784` summed B200-hours, below the two-hour compute ceiling, but
the disk gate fails: projected peak bytes plus the retained artifact accounting
and 5 GiB safety reserve require `7516192768` bytes while the preflight recorded
`7475847168` free bytes. Therefore B12 does not authorize a B13 GDN addition
matrix and makes no partial science claim for that omitted matrix.
