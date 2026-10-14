# R01 brief (rev 2): DeepEP baked into the TorchTitan Insula rootfs via scripts/rootfs/build_rootfs.sh -- PRIORITY 1
Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit. Read `env-facts.md`
(MANDATORY), `scripts/rootfs/{build_rootfs.sh,enter_rootfs.sh,runtime_env.sh,rootfs_target.sh,verify_runtime_env.py}`,
`tests/unit_tests/test_rootfs_build_store_shell.py`, `.github/scripts/install_deepep_v2.sh` (DeepEP v2 pinned
01dc3aaac82068020353dce2c302e38153c0bfaa; `from deep_ep import ElasticBuffer`), `.ci/docker/common/install_deepep.sh`
(hybrid-ep branch f725d29699f5bda9ba789456bb9579af69844685), `torchtitan/distributed/deepep/{deepep.py,hybridep.py}`,
`torchtitan/models/common/token_dispatcher.py` (DeepEPTokenDispatcher L820, HybridEPTokenDispatcher L955),
https://github.com/deepseek-ai/DeepEP (requirements, SM100), and the PyTorch blog
https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/
(torchao MXFP8 grouped GEMM + DeepEP on B200 in TorchTitan).

User decision: DeepEP must live in the rootfs image built by TorchTitan's Insula builder (`scripts/rootfs/build_rootfs.sh`),
not in a hand-assembled venv. Context:
- Current rootfs: `scripts/rootfs/rootfs` (legacy, store_id legacy-rootfs), built from the inline Dockerfile in
  build_rootfs.sh on `ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime`; it already has libibverbs-dev,
  librdmacm-dev, libnuma-dev, build-essential, cmake, ninja, uv, mise, and pip nvcc cu13 at CUDA_HOME=/opt/cuda-synth.
  Docker 26.1.4 is available on the host (image `torchtitan-rootfs:local` exists).
- System torch is 2.13 (vLLM 0.27 eval depends on it). Training uses torch nightly `2.16.0.dev20261013+cu132` from the
  hand-made `glm53_train` venv layered over system site-packages (`.cache/torchtitan-rootfs/glm53_train`). DeepEP must be
  compiled against the torch it runs with, so it cannot go into the 2.13 system python.
- Live jobs (GLM 8-GPU run, speedrun, ladder) run on the legacy rootfs and `glm53_train`; do NOT modify, rebuild or
  re-export `scripts/rootfs/rootfs`, and do not touch the `glm53_train` venv.

Do:
1. Extend `build_rootfs.sh` with an opt-in training variant (e.g. `--variant ep`, own default tag
   `torchtitan-rootfs:ep`) that adds Dockerfile layers on top of the base image: a training python environment with the
   pinned torch nightly + matching triton (choose: system python replaced only in this variant, or `/opt/venvs/train`
   baked in the image -- justify; the launch path must make that torch the one used), repo `requirements.txt` pins,
   torchao (MXFP8 MoE per the blog), fla/KDA kernels, tensorboard/safetensors/tiktoken as the GLM + speedrun code needs
   (derive from `uv pip freeze` of `glm53_train` -> save to `evidence/r01/glm53_train-freeze.txt`), NVSHMEM (pip
   wheel for CUDA 13 if DeepEP v2 needs it), and DeepEP v2 at the pinned commit built with `TORCH_CUDA_ARCH_LIST=10.0`
   (+ `9.0` optional), `--no-build-isolation`; HybridEP (hybrid-ep branch) only if it builds for SM100 and coexists --
   otherwise document. Pin every source by commit/version/sha256 in a checked-in lock file next to the builder (e.g.
   `scripts/rootfs/ep_variant.lock.json`). The default (non-variant) build output must stay byte-for-byte the same recipe.
   Add a build-time smoke `python -c "import torch, deep_ep; from deep_ep import ElasticBuffer; print(torch.__version__)"`.
2. Tests: extend `tests/unit_tests/test_rootfs_build_store_shell.py` (or add a sibling) for the new flag/argument
   parsing and that the default path is unchanged; ShellCheck the script.
3. Build it and publish with `--store /data05/philip.yang/torchtitan-rootfs-store` (check disk: /data05 ~187 GB free;
   clean only your own temp layers). Confirm `selected.json` and the content dir.
4. Wire selection: document (and, if needed, minimally extend `enter_rootfs.sh`/`runtime_env.sh`) how to enter the EP
   rootfs (`--rootfs-store ...` or `--rootfs ...`) with an env name (e.g. `TORCHTITAN_ROOTFS_ENV=glm53_ep`) whose venv
   does not shadow the image's torch/deep_ep. Existing default entry must behave exactly as before.
5. Verify inside the EP rootfs: torch version/CUDA arch 10.0, `ElasticBuffer`, torchao MXFP8 grouped GEMM tiny numeric
   check (1 GPU), fla KDA import, NVSHMEM init; DeepEP's intranode test on 2 GPUs (8 only if free); TorchTitan
   `qwen3_moe_deepep` (EP=4) a few steps vs the all-to-all backend (recipe: step-1 bitwise); GLM experiment tests
   (`torchtitan/experiments/glm53_flash/tests`, `glm53_flash_ep/tests`) pass. Co-run limits: <= 30 GiB per process,
   never kill the live GLM job or foreign processes; skip/defer GPU tests if busy and say so.
6. Docs: a `scripts/rootfs/README` section (or the rootfs doc that exists) on the EP variant: build, publish, select,
   verify, versions; update `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` with the EP rootfs launch line.
Report `sdd/task-r01-report.md`: pins, build log summary, image/store ids and sizes, verify receipts, test results, and
any proposed follow-up (e.g. migrating the default rootfs to a store).
