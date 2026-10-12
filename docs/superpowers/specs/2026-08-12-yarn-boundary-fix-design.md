# YaRN Correction-Boundary Fix Design

Date: 2026-08-12
Status: Approved approach

## Problem

The shared `torchtitan.models.common.rope._yarn_inv_freq` helper rejects a
valid YaRN correction range when the extrapolation cutoff clamps to zero.
The deterministic reproducer is:

```bash
python -c 'from torchtitan.models.common.rope import _yarn_inv_freq; _yarn_inv_freq(128, 10000.0, 40.0, 32.0, 1.0, 64, True)'
```

It fails with:

```text
AssertionError: Invalid YaRN params: 0 < 0 < 17 < 127
```

This prevents the Helion RoPE kernel test fixture from constructing its
PyTorch reference module.

## Root cause

Commit `51c197c86` unified the `ComplexRoPE` and `CosSinRoPE` YaRN
implementations. The extracted helper retained a strict interior-range
assertion from the prior cos/sin path while also adopting the DeepSeek-style
cutoff clamp:

```text
low = max(low, 0)
```

The two rules conflict. For the failing parameters, the unbounded cutoff is
negative and correctly clamps to `0`, after which `0 < low` rejects it.

The prior complex implementation, the official DeepSeek implementation, and
the Hugging Face YaRN implementation accept endpoint cutoffs. The official
formula also protects a collapsed `low == high` ramp by adding a small positive
width. Swapping `beta_fast` and `beta_slow` produces a genuinely reversed range
and remains invalid.

## Behavior contract

The shared helper will:

1. continue to compute and truncate the correction cutoffs as it does now;
2. accept a cutoff clamped to `0` or the upper supported endpoint;
3. preserve the official `low == high` singularity guard by increasing the
   local upper cutoff by `0.001` before constructing the ramp;
4. raise `ValueError` when the resulting cutoff order is reversed; and
5. preserve bitwise output for existing configurations whose cutoffs already
   satisfy `low < high`.

The function remains a single source of truth for both complex and cos/sin
RoPE. This change does not alter checkpoint or state-dict structure.

## Tests

Add a focused CPU test module under `tests/unit_tests/` so the regression does
not depend on CUDA, Helion, Qwen3.5, or the optional
`flash-linear-attention` package.

The tests will cover:

- the exact `low=0, high=17` regression and compare the returned frequencies
  with the reference ramp;
- a collapsed cutoff range, asserting finite output; and
- reversed beta cutoffs, asserting `ValueError`.

TDD order is mandatory: add and run the endpoint test against the unchanged
helper, observe the expected assertion failure, then implement the minimal
range handling and run the focused tests green. Finally re-run the original
Helion test and the changed-file hooks.

## Alternatives rejected

- Changing the Helion fixture to a longer original context hides the shared
  helper regression.
- Merely changing `<` to `<=` allows the endpoint but leaves the official
  collapsed-ramp case dividing by zero.
- Removing all validation would make reversed cutoffs silently produce an
  unintended ramp.
