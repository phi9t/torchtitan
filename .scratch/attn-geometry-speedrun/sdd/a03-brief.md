# A03 brief: Family A kernel library (spec.md Revision 3) -- code + tests only, no training

Read `spec.md` (Revisions 2 and 3; the Family A table is binding) and
`torchtitan/experiments/attn_geometry_speedrun/attention_geometry.py` (existing anchor/dot/cosine/gauss).
Another worker (A01b, then A02) owns `attention_geometry.py`, the runner and `variant/train_gpt.py`: DO NOT edit them.
Write only NEW files:
- `torchtitan/experiments/attn_geometry_speedrun/kernel_family_a.py`
- `tests/unit_tests/test_attn_geometry_family_a.py`

Implement, per arm, one function returning attention output [B, T, H, D] for causal attention:
- A4 key-norm penalty (alpha, beta >= 0 learnable per head via softplus/exp) and A5 diagonal anisotropic Gaussian
  (diagonal M >= 0 per head, with a declared constraint against absorption into W_Q/W_K -- e.g. trace(M) fixed per head
  -- documented in the docstring) through augmented SDPA (reuse the gauss augmentation pattern; pad head dim to 8).
- A7, A8 (3/2 and 5/2), A9, A10 (R=3 components, pi via softmax over logits), A11, A12 (p in {1,2,4}), A13, A14 through
  `torch.nn.attention.flex_attention` with a causal block mask and a score_mod that returns log k computed from the
  scaled dot product plus per-token ||q||^2 / ||k||^2 captured tensors. fp32 inside score_mod. Clamp r^2 >= 0 and use
  sqrt(r^2 + eps) where r appears; log(|s| + eps) for A14; -inf only where k is exactly 0.
- A6 L1 Laplace: explicit chunked op (query chunks so peak memory is bounded), marked side-table.
- A shared `explicit_kernel_attention(kernel_fn, ...)` reference that materializes k(q_i, k_j), causal-masks, normalizes
  by the row sum (not softmax), for tests.

Tests (CUDA if available, else CPU at small T): every arm vs the explicit reference at T=1024 (fwd allclose with stated
tolerances, fp32 and bf16 inputs), finite backward at T=1024 including large-norm inputs; A4 beta=alpha/2 == gauss, A4
beta=0 == dot; A10 with one component == A3; A12 p=1 at antipode gives zero weight without NaN; A13 positivity; A14
s and -s give equal weight; flex compile works (torch.compile of flex_attention) for each arm. Report fwd+bwd time per arm
vs SDPA at the A01 shape (B=8, T=1024, H=6, D=64) on one GPU.

GPU: only one that has no other process (check nvidia-smi; GPUs 0-6 and 7 are busy with other workers right now -- if
none is free, run CPU-only tests and mark CUDA timings NOT RUN; retry once later). Never touch processes you did not
start. Rootfs for all Python/CUDA. Do not commit.
Report `sdd/a03-report.md`: per-arm implementation tier, test table, timing table, any arm whose numerics are unsafe.
