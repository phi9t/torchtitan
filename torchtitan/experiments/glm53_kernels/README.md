# GLM-5.3 Kernel Package Convention

GLM kernel work stays experiment-owned until a candidate is profiled, proven,
and general enough to move into core.

The convention is:

- Triton kernels are Python JIT code in `torchtitan/experiments/glm53_kernels/`.
- CuTe DSL kernels are Python JIT code in this package and use the rootfs
  `cutlass` / `cutlass.cute` packages.
- C++/CUDA `torch.library` ops live under `native/` and are built into wheels
  by `experiments/glm53_native/build_native.sh`.
- Every op needs fake/meta coverage, a torch.compile smoke, forward and
  backward numerics against a high-precision oracle and eager same-dtype
  baseline, plus a microbench template.

Current examples:

- `triton_examples.triton_scale_add`: runnable Triton `torch.library` example.
- `cute_examples.cute_scale_add_convention`: CuTe DSL package/import convention
  check. It is intentionally marked `PASS_WITH_LIMITATIONS` until a real CuTe
  candidate kernel lands.
- `glm53_kernel_examples.scale_add`: C++/CUDA `torch.library` wheel example.

Candidate kernels to evaluate before implementing custom versions:

- fused mHC Sinkhorn/HcPost;
- fused indexer top-k plus IndexPool;
- sparse MLA gather attention, checking FlashMLA sparse first;
- KDA, checking `fla` first;
- MoE permute for the selected dispatcher.

Run:

```bash
experiments/glm53_native/run_kernel_checks.sh
```

