# R01 brief (rev 3): L0 + L1 of the native layering -- declared GLM training env on the Insula rootfs
Read `native-layering.md` (binding decision), `env-facts.md` (MANDATORY), `scripts/rootfs/{build_rootfs.sh,enter_rootfs.sh,
runtime_env.sh,rootfs_target.sh,verify_runtime_env.py}`, `tests/unit_tests/test_rootfs_build_store_shell.py`,
`experiments/modded_nanogpt_b200/runtime/{pyproject.toml,requirements.direct.txt,requirements.lock,sync_python_env.sh,
verify_runtime.py,mise.toml}`. A previous rev captured `uv pip freeze` of `glm53_train` (torch 2.16.0.dev20261013+cu132,
triton 3.9.0+gitfc8d31a5, fla-core 0.5.2, nvidia-nvshmem-cu13 3.7.2, ...) -- find it under
`.scratch/glm53-flash-sft-ladder/evidence/r01/` or redo it. Do NOT commit. Do not modify the live legacy rootfs
(`scripts/rootfs/rootfs`) or the live `glm53_train` venv; jobs are running on both.
L0 (image): audit what L1 + L2 need at system level for building DeepEP v2 and C++/CUDA extensions for sm_100 against
torch 2.16 cu132: full CUDA 13.x toolkit headers/libs matching torch's CUDA (the image has pip nvcc at /opt/cuda-synth --
check whether cuda_runtime/cublas/nvrtc headers, nvcc version and libcuda stubs suffice for DeepEP and cpp_extension),
NVSHMEM (pip wheel in L1 vs system), ibverbs/rdmacm/numa (already present), ninja/cmake (present), ccache, CUTLASS
headers, Nsight Compute/Systems CLIs. ONLY if something is missing, add it to `build_rootfs.sh` behind an opt-in flag
(e.g. `--variant native`) with versions pinned in a checked-in lock; keep the default recipe byte-identical; add/extend
the shell test; build and publish to a store (`--store /data05/philip.yang/torchtitan-rootfs-store`, check disk) and
document selection via `enter_rootfs.sh --rootfs-store`. If nothing is missing, say so and change nothing in L0.
L1 (env): `experiments/glm53_flash_sft_ladder/runtime/` (or a shared `experiments/glm53_runtime/` -- justify) with
`requirements.direct.txt`, hashed `requirements.lock`, `sync_python_env.sh` (re-enters rootfs, builds the venv from the lock,
records the L0 store id / rootfs digest), `verify_runtime.py` (torch/CUDA/arch, triton, fla KDA import, torchao MXFP8
import, NVSHMEM wheel presence; JSON receipt). Sync into a NEW env name `glm53_ep` and verify it reproduces `glm53_train`
(diff the freezes; run `pytest torchtitan/experiments/glm53_flash/tests` and a 2-step GLM debugmodel train in `glm53_ep`).
Docs: runtime README + env-facts update with the `glm53_ep` launch line. Report `sdd/task-r01-report.md`.
