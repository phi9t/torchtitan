# 01 - Prove the RMS/L2 scale transform

Type: task
Status: ready-for-agent
Blocked by: none
Parent: ../spec.md

## What to build

Make the complete RMS-to-L2 coordinate transform an explicit Falcon mixer knob
and prove that the compensated RMS formulation represents the same computation
without changing any existing arm's defaults.

## Acceptance criteria

- [ ] QK-normalization epsilon and NLMS-denominator epsilon have separate
  ownership while all existing configurations preserve their current values
  and behavior.
- [ ] For head dimension `d`, the compensated arm uses RMS normalization
  epsilon divided by `d`, actual lambda multiplied by `d`, and denominator
  epsilon multiplied by `d` relative to the L2 arm.
- [ ] The high-level scale-compensation knob materializes M4 without exposing a
  second ad hoc configuration path.
- [ ] Kernel tests compare outputs, backward gradients, and
  `sqrt(d) * S_rms` against `S_l2` at the actual science head dimension with
  fp32 relative and absolute tolerances of `1e-5`.
- [ ] Full-mixer and short deterministic trajectory guards establish matching
  logits, full-precision loss, and gradients before M4 is eligible for science
  training.
- [ ] Invalid epsilon or compensation combinations fail with actionable user
  errors, and existing recurrent/masked-parallel identity tests remain green.
- [ ] Development is red-green inside the rootfs; focused tests, the complete
  Falcon suite, and changed-file lint pass.

## Exclusions

- No GPU science run, paper `ctxeta` implementation, or Falcon-3A work.
- No change to existing RMS/L2 defaults outside the explicit M4 arm.
