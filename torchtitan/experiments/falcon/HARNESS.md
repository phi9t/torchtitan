# Falcon harness contract and silent-failure modes

Mercor Step 1 for Falcon fast-weight attention is mathematical, not
infrastructural: the thing the Trainer optimizes must be the paper's mixer on
the labels we think we are training. Each bug below produces a plausible loss
curve while training a different model, so each has a guarding test.

## Write pairing: `alignment`

`falcon_recurrent_forward(..., alignment=...)` selects the write pairing:

- `delayed` (default, paper): `(x_t, y_t) = (phi(k_{t-1}), v_t)`, with `x_0 = 0`
  and `eta_0 = 0`. `t = 0` is a no-op write; the first real write pairs
  `phi(k_0)` with `v_1`. Read is after write (`o_t = S_t^T phi(q_t)`).
- `same_step` (ablation only): `(x_t, y_t) = (phi(k_t), v_t)` at every `t`
  including `t = 0`, still read-after-write, with `eta_0` computed normally.
  The first write pairs `phi(k_0)` with `v_0`.

Same-step is a different paper. It is an explicit ablation switch, never a
silent default. `alignment` threads through `FalconConfig` and `FalconMixer`;
the config default is `delayed`. An unknown alignment raises `ValueError`.

Tests: `test_delayed_alignment_first_write_uses_k0_with_v1`,
`test_same_step_alignment_first_write_uses_k0_with_v0`,
`test_falcon_recurrent_forward_rejects_unknown_alignment`,
`test_falcon_config_alignment_defaults_to_delayed`,
`test_falcon_mixer_threads_same_step_alignment_to_the_kernel`.

## Silent-failure modes

1. **Same-step write as the default.** If `alignment` silently defaulted to
   `same_step`, the model would leak `v_t` into its own step-`t` read and learn
   a copy rule instead of the delayed association the paper claims. Guarded by
   the `delayed` default and the mixer threading test.
2. **Circular labels (roll, not shift).** Next-token labels are `bank[:, 1:]`
   (a shift with an appended target token), never `torch.roll`. A roll wraps the
   first token in as the label for the last position, teaching a wrap-around
   target that does not exist. Guarded by
   `test_repeat_dataloader_labels_are_a_shift_not_a_circular_roll`.
3. **Missing document-boundary state reset (later).** When document packing is
   introduced, `S` must reset at document boundaries (as Qwen3.5 GDN does via
   `cu_seqlens`). Without a reset, associations bleed across unrelated
   documents. Not implemented in this campaign step; the tiny overfit bank is
   one document per row, so no reset is needed yet. This is a known future
   requirement, not a current bug.

## Other Step 1 invariants (already held)

- Default `phi` is QK-RMSNorm; L2 is an ablation, not a silent default.
- Values are not normalized unless an ablation enables VNorm.
- Trainer batch is `{input: tokens[:, :-1], positions}`, `labels = tokens[:, 1:]`
  with no text encode/decode on the token path.
- Falcon-1 uses a residual (delta) write; Falcon-1A writes `v_t` directly.

## Re-run the overfit gate after any harness change

The Step 3 learnability gate must stay green on the `delayed` default after any
mixer or alignment edit:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
```

A new write rule that cannot overfit the fixed bank is a broken Step 1, not a
reason to start FineWeb.
