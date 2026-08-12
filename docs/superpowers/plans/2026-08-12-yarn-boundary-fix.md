# YaRN Correction-Boundary Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared YaRN inverse-frequency helper accept valid endpoint and collapsed correction ranges while rejecting reversed ranges.

**Architecture:** Keep `_yarn_inv_freq` as the single computation seam for complex and cos/sin RoPE. Replace its contradictory strict-interior assertion with the official collapsed-ramp guard plus an explicit `ValueError` for reversed cutoffs, and lock the behavior down in a dependency-light CPU unit test.

**Tech Stack:** Python 3.11, PyTorch, `unittest`, pytest, repository pre-commit hooks

## Global Constraints

- Follow red-green TDD: production code may change only after the new focused test file has been run and failed for the expected YaRN range reasons.
- Preserve the existing cutoff calculation, truncation, clamp, interpolation formula, dtype, and output shape.
- Accept cutoff endpoints, including `low == 0`.
- For `low == high`, increase only the local upper cutoff by `0.001` before division, matching the official DeepSeek and Hugging Face ramp guard.
- For `low > high`, raise `ValueError`; invalid user configuration must not use `assert`.
- Existing configurations with `low < high` must retain bitwise-identical output.
- Add no dependency on CUDA, Helion, Qwen3.5, or `flash-linear-attention` to the regression test.
- Do not modify checkpoint or state-dict behavior.
- Keep the fix limited to the shared helper and its focused tests; no adjacent RoPE refactor.

---

## File structure

- Create `tests/unit_tests/test_yarn_rope.py`: dependency-light CPU behavior tests for endpoint, collapsed, and reversed YaRN correction ranges.
- Modify `torchtitan/models/common/rope.py`: replace the contradictory cutoff assertion with explicit reversed-range validation and the collapsed-ramp guard.

### Task 1: Fix YaRN correction-range boundary handling

**Files:**
- Create: `tests/unit_tests/test_yarn_rope.py`
- Modify: `torchtitan/models/common/rope.py:69`
- Verify: `tests/unit_tests/test_helion_rope.py`

**Interfaces:**
- Consumes: `_yarn_inv_freq(dim: int, base: float, rope_factor: float, beta_fast: float, beta_slow: float, original_seq_len: int, truncate: bool) -> torch.Tensor`.
- Produces: the same signature and tensor contract, accepting endpoint/collapsed ranges and raising `ValueError` for reversed cutoffs.

- [ ] **Step 1: Add the focused CPU regression tests**

Create `tests/unit_tests/test_yarn_rope.py` with this exact content:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.rope import _yarn_inv_freq


class TestYaRNInvFreq(unittest.TestCase):
    def test_zero_clamped_cutoff_matches_reference_values(self):
        actual = _yarn_inv_freq(
            dim=128,
            base=10_000.0,
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=64,
            truncate=True,
        )

        selected = actual[torch.tensor([0, 1, 8, 16, 17, 63])]
        expected = torch.tensor(
            [
                1.0,
                0.8162987232208252,
                0.1711350381374359,
                0.008235293440520763,
                0.0021649107802659273,
                2.8869549169030506e-06,
            ]
        )
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    def test_collapsed_cutoff_produces_finite_frequencies(self):
        actual = _yarn_inv_freq(
            dim=128,
            base=10_000.0,
            rope_factor=40.0,
            beta_fast=1.0,
            beta_slow=1.0,
            original_seq_len=64,
            truncate=False,
        )

        self.assertEqual(actual.shape, (64,))
        self.assertTrue(torch.isfinite(actual).all())

    def test_reversed_cutoff_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "reversed YaRN correction range"):
            _yarn_inv_freq(
                dim=128,
                base=10_000.0,
                rope_factor=40.0,
                beta_fast=1.0,
                beta_slow=32.0,
                original_seq_len=64,
                truncate=True,
            )
```

Mutation check:

- Restoring `0 < low` makes the endpoint test fail.
- Omitting the singularity guard makes the collapsed test produce nonfinite values.
- Omitting reversed-order validation makes the error test fail.

- [ ] **Step 2: Run the tests and verify the red state**

Run:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pytest \
  -q tests/unit_tests/test_yarn_rope.py
```

Expected: three failed tests. The endpoint and collapsed tests fail at the
current strict range assertion; the reversed-range test receives
`AssertionError` instead of `ValueError`. Collection must succeed without
optional model packages.

- [ ] **Step 3: Implement the minimal range handling**

In `torchtitan/models/common/rope.py`, replace:

```python
    assert (
        0 < low < high < dim - 1
    ), f"Invalid YaRN params: 0 < {low} < {high} < {dim - 1}"
```

with:

```python
    if low > high:
        raise ValueError(f"reversed YaRN correction range: {low=} must be <= {high=}")
    if low == high:
        high += 0.001
```

Do not change the cutoff calculation or the ramp/interpolation expression.

- [ ] **Step 4: Run the focused tests and verify the green state**

Run:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pytest \
  -q tests/unit_tests/test_yarn_rope.py
```

Expected: `3 passed` with no warnings.

- [ ] **Step 5: Re-run the original failing feedback loop**

Run:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pytest -q \
  tests/unit_tests/test_helion_rope.py::TestHelionRoPEKernel::test_backward_custom_op_opcheck_with_noncontiguous_grads
```

Expected: the test passes rather than failing while constructing
`ComplexRoPE` with `low=0, high=17`.

- [ ] **Step 6: Run the owning focused suite**

Run:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pytest -q \
  tests/unit_tests/test_yarn_rope.py \
  tests/unit_tests/test_helion_rope.py
```

Expected: all focused common/Helion RoPE tests pass. CUDA/Helion-dependent tests
may be skipped only when the environment reports the missing capability; the
CPU regression tests must run.

- [ ] **Step 7: Run changed-file hooks and inspect the diff**

Run:

```bash
/data02/home/philip.yang/.local/bin/pre-commit run --files \
  torchtitan/models/common/rope.py tests/unit_tests/test_yarn_rope.py
git diff --check -- torchtitan/models/common/rope.py tests/unit_tests/test_yarn_rope.py
git diff --stat
git diff -- torchtitan/models/common/rope.py tests/unit_tests/test_yarn_rope.py
```

Expected: all hooks pass, `git diff --check` is silent, and the implementation
diff contains only the shared helper and focused test file.

- [ ] **Step 8: Commit the tested fix**

Run:

```bash
git add -- torchtitan/models/common/rope.py tests/unit_tests/test_yarn_rope.py
git diff --cached --name-status
git commit -m "Fix YaRN correction range boundaries"
```

Expected: the staged diff contains only the two implementation paths and the
commit succeeds.
