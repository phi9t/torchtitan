# Ticket 03 brief

Read first: `.scratch/falcon-fast-weight-attention/issues/03-step1-harness-contract.md`
Parent spec: `.scratch/falcon-fast-weight-attention/spec.md`

## Fit

Mercor Step 1 remainder. Falcon-1/1A recurrent kernel already exists.
Add an explicit alignment switch so same-step writes are an ablation,
never the default.

## Do

TDD. Write failing tests first. Run them in rootfs:

```
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
```

1. Add `alignment: Literal["delayed", "same_step"] = "delayed"` to
   `falcon_recurrent_forward` in `torchtitan/experiments/falcon/falcon.py`.
2. Delayed (default): `t=0` no-op; later `x = phi(k_{t-1})`, target `v_t`.
3. Same-step: `x = phi(k_t)`, target `v_t` at every t including 0
   (still read-after-write). `eta` computed normally at t=0.
4. Thread `alignment` through `FalconConfig` / `FalconMixer`.
5. Tests:
   - delayed first write uses k_0 with v_1 (existing test may already cover)
   - same-step first write uses k_0 with v_0
   - reject unknown alignment
   - `FalconRepeatDataLoader` labels are `bank[:, 1:]` not a roll
6. Harness note: `torchtitan/experiments/falcon/HARNESS.md` listing
   silent-failure modes (same-step default, circular labels, later
   missing doc-boundary reset).
7. Overfit gate must stay green on delayed default.

## Do not

- Commit, push, or PR
- Chunk-parallel / FineWeb / addition / Mini-K3 / qwen3_5 edits
- Change overfit defaults to same-step

## Report

Write `.scratch/falcon-fast-weight-attention/sdd/task-03-report.md`
with status DONE/DONE_WITH_CONCERNS/BLOCKED, files changed, test
command + output summary. Do not commit.
