# Falcon Fast Weight Attention

Experiment-owned Falcon-1 / Falcon-1A mixer from
[arXiv:2608.27763](https://arxiv.org/abs/2608.27763). Optional TorchTitan
code; one-way dependency `experiments -> core`.

The first gate is Mercor Step-3 learnability: a tiny Falcon LM must overfit a
fixed 8-sequence bank. If cross-entropy does not collapse, the mixer or
harness is broken. This is not FineWeb quality evidence.

## Layout

- `falcon.py` — CPU recurrent Falcon-1 / Falcon-1A
- `model.py` — tiny decoder + causal LM
- `mechanism.py` — stateless per-call mechanism diagnostics
- `repeat_data.py` — repeating sequence bank
- `overfit.py` — synchronous overfit report
- `config_registry.py` — `falcon_tiny_overfit` Trainer config

## Mechanism Arms

`apply_mechanism_arm(config, arm_id)` materializes the M01 mechanism arms
without changing the legacy A-arm campaign mapping:

- `M0`: delayed RMS Falcon-1A.
- `M1`: same-step RMS Falcon-1A.
- `M2`: delayed L2 Falcon-1A.
- `M3`: same-step L2 Falcon-1A.
- `M4`: delayed RMS Falcon-1A with `scale_compensation="rms_to_l2"`.

All M arms force `mixer="falcon"` and `variant="falcon1a"`.

## Mechanism Diagnostics

`MechanismDiagnostics` is caller-owned and is passed into
`FalconForCausalLM.forward(..., mechanism_diagnostics=...)`. The model does not
register diagnostic buffers, parameters, or persistent attributes. For each
Falcon layer/head it aggregates count, mean, standard deviation, and p05/p50/p95
over valid token observations for pre/post-normalization key energy, beta,
actual lambda, eta, gamma, `-1/log(gamma)`, raw and canonical state norm, update
norm, read norm, and clamp fraction. The delayed sentinel step zero is excluded
from write summaries.

M4 records the effective QK epsilon, effective NLMS denominator epsilon, the
lambda scale actually applied, raw `S_rms`, and `sqrt(head_dim) * S_rms` as the
canonicalized state norm used for comparison with the L2 arms.

## Scale Compensation

`FalconConfig` has separate `qk_norm_eps` and `nlms_denom_eps` fields, both
defaulting to `1e-6`. `scale_compensation="rms_to_l2"` is valid only for the
Falcon mixer with `phi="rms"`. At head dimension `d`, it resolves QK epsilon to
`base / d`, multiplies the post-softplus lambda by `d`, and resolves the NLMS
epsilon to `base * d`. It does not scale beta, values, `eps_gamma`, projection
weights, or the output.

The M2/M4 identity gate covers the kernel theorem, adjusted final-state
relation, mixer gradients, full-model logits/loss/parameter gradients, and a
short deterministic optimizer trajectory:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_scale_transform.py'
```

Launch the Trainer config with:

```bash
experiments/falcon/run.sh overfit
```

or:

```bash
MODULE=falcon CONFIG=falcon_tiny_overfit NGPU=1 COMM_MODE=fake_backend ./run_train.sh
```
