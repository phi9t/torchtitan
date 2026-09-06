# Falcon-1 / 1A recurrent kernel

Type: task
Status: resolved
Blocked by:
Parent: ../spec.md

## Requirements

- Public function `falcon_recurrent_forward` over `q_BLNK`, `k_BLNK`,
  `v_BLNV`, `beta_BLN`, `lambda_BLN`.
- Variant `falcon1a` writes `eta * x v^T`. Variant `falcon1` writes
  `eta * x r^T` with `r = v - S^T x`.
- `t = 0` uses `x = 0`, `eta = 0`; later steps use `x = phi(k_{t-1})`.
- Read after write: `o_t = S_t^T phi(q_t)`.
- Optional QK-RMSNorm; clamp `gamma = 1 - eta lambda` to stay positive.

## Verification

- Worked-example unit tests fail before the kernel exists, then pass.
- Falcon-1 and Falcon-1A diverge on the second write when `S` is nonzero.

## Answer

`falcon_recurrent_forward` is in `torchtitan/experiments/falcon/falcon.py`.
`tests/unit_tests/test_falcon_kernel.py` covers delayed pairing, the first-step
no-op, and residual vs direct writes.
