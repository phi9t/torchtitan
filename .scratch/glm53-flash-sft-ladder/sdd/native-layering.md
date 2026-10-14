# Decision: three-layer native/runtime setup for GLM work (user-approved 2026-10-14 ~19:55Z)
- L0 rootfs image (TorchTitan Insula builder `scripts/rootfs/build_rootfs.sh`): system libs + toolchain only (CUDA 13
  toolkit matching torch, NVSHMEM if a system install is needed, rdma/ibverbs/numa, cmake/ninja/ccache, CUTLASS headers,
  Nsight CLIs). Changes rarely. Published as an immutable store entry; never edit the live legacy rootfs in place.
- L1 Python env: declared lock + sync + verify (house pattern: `experiments/modded_nanogpt_b200/runtime/`), torch nightly
  pinned and bumped deliberately; records the L0 store id it was synced on.
- L2 native artifacts: fast-changing native code -- DeepEP (pinned upstream commit + our patch queue) and our C++/CUDA ops --
  built by ONE `build_native.sh` inside the rootfs into wheels keyed by (source hash incl. patches, torch version, CUDA
  version, arch sm_100). Wheels are published to HDFS via waystone with md5 + `_COMPLETE` and staged/verified like any
  other input (HDFS-strict rule). Dev loop: editable/ninja/ccache builds. Every run records the native build keys it used.
- Triton / CuTe DSL kernels are Python (JIT) and live in the repo; no build layer.
- Reuse upstream kernels first (fla KDA, FlashMLA sparse / DSA, DeepGEMM, Dist-MoE); write custom kernels only for
  profiled hot spots (likely: fused mHC Sinkhorn/HcPost, fused indexer top-k + IndexPool, MoE permute for the chosen
  dispatcher).
- Rung 4 does not wait on DeepEP: grouped experts (T17) + all-to-all EP (T13) carry it; DeepEP is a measured optimization.
