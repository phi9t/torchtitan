# R01 brief: declared, reproducible `glm53_train` rootfs env with DeepEP v2 (+HybridEP) and MXFP8 for B200 -- PRIORITY 1
Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit. Read `env-facts.md`
(MANDATORY), `scripts/rootfs/{enter_rootfs.sh,runtime_env.sh,build_rootfs.sh,verify_runtime_env.py}`,
`experiments/modded_nanogpt_b200/runtime/{pyproject.toml,requirements.direct.txt,requirements.lock,sync_python_env.sh,
verify_runtime.py,mise.toml}` (the house pattern for a declared rootfs env), `.github/scripts/install_deepep_v2.sh`
(DeepEP v2 pinned 01dc3aaac82068020353dce2c302e38153c0bfaa, `from deep_ep import ElasticBuffer`),
`.ci/docker/common/install_deepep.sh` (hybrid-ep branch pinned f725d29699f5bda9ba789456bb9579af69844685),
`torchtitan/distributed/deepep/{deepep.py,hybridep.py}`, `torchtitan/models/common/token_dispatcher.py`
(DeepEPTokenDispatcher L820, HybridEPTokenDispatcher L955), `torchtitan/models/common/dist_moe/README.md`,
the PyTorch blog https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/
(MXFP8 grouped GEMMs via torchao + DeepEP on B200), and https://github.com/deepseek-ai/DeepEP (README requirements, SM100).

Problem: the `glm53_train` env (state under `.cache/torchtitan-rootfs/glm53_train`, venv `/project/venvs/b200-runtime`,
torch 2.16.0.dev20261013+cu132 layered over the rootfs system site-packages) was assembled by hand; nothing declares or
reproduces it, and DeepEP is not installed. A previous worker (T18, stopped) may have left a partial DeepEP build -- inspect,
do not trust it.

Deliver `experiments/glm53_flash_sft_ladder/runtime/` following the modded_nanogpt_b200 pattern:
1. `requirements.direct.txt` + hashed `requirements.lock` (uv pip compile) capturing what the env actually needs today:
   the exact torch nightly (+ matching triton), repo `requirements.txt` pins (spmd_types, attn_gym, torch_remat, ...), fla /
   KDA kernels, torchao (MXFP8 MoE training per the blog), tensorboard, safetensors, tiktoken, etc. Freeze the current env
   first (`uv pip freeze`) to `evidence/r01/freeze-before.txt` so nothing regresses.
2. `sync_python_env.sh`: re-enters the rootfs, creates/syncs the venv from the lock (system-site-packages as today if
   required), then builds DeepEP v2 at the pinned commit for SM100 (`TORCH_CUDA_ARCH_LIST=10.0`, `CUDA_HOME` inside the
   rootfs, `--no-build-isolation`, never altering torch), and the hybrid-ep branch build for HybridEP if it supports SM100
   (if the two cannot coexist, say so and pick DeepEP v2 by default with a flag for hybrid-ep). NVSHMEM from pip wheels
   (nvidia-nvshmem for CUDA 13) if DeepEP needs it; rdma-core/ibverbs: if missing from the rootfs and only needed for
   inter-node RDMA, document and skip; if needed even intranode, add them to `scripts/rootfs/build_rootfs.sh`'s package
   list as a PROPOSED patch (do NOT rebuild or modify the live rootfs at scripts/rootfs/rootfs -- jobs are running on it).
   Pin sources by commit + record wheel sha256s in `runtime/deepep.lock.json`. Use `TORCHTITAN_ROOTFS_NETWORK=networked`.
   Sync into a NEW env name first (`TORCHTITAN_ROOTFS_ENV=glm53_ep`) so live jobs on `glm53_train` are untouched.
3. `verify_runtime.py` (run inside the rootfs) checking: torch version/CUDA arch 10.0, `from deep_ep import ElasticBuffer`,
   (hybrid_ep import if built), torchao MXFP8 grouped GEMM import + a tiny numeric check on 1 GPU, fla KDA import,
   NVSHMEM init; prints a JSON receipt. Then on GPUs: DeepEP's own intranode test on 2 and 8 GPUs (only when free; else 2)
   and TorchTitan's `qwen3_moe_deepep` (EP=4) for a few steps vs the all-to-all backend (docstring: step-1 bitwise).
   Respect co-run limits: a fallback 8-GPU GLM job may be live (~136-155 GiB/GPU); <= 30 GiB per process, never kill it or
   foreign processes; skip/defer 8-GPU tests if the GPUs are busy and say so.
4. Docs: `experiments/glm53_flash_sft_ladder/runtime/README.md` (how to build/verify/use `glm53_ep`, versions, caveats),
   update `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` with the new env name and launch line.
Report `sdd/task-r01-report.md` with versions, pins, receipts, test results, and anything blocked (with the exact patch).
