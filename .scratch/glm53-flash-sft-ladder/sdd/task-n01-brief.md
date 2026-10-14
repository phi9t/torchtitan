# N01 brief (rev 2): L2 native-artifact layer -- build_native.sh, DeepEP as first package, GLM kernel package convention
Blocked by: R01 (L1 env `glm53_ep`, and L0 store entry if R01 added one). Read `native-layering.md` (binding), R01 report,
`env-facts.md` (MANDATORY: HDFS-strict inputs via waystone), `.github/scripts/install_deepep_v2.sh` (DeepEP v2 pinned
01dc3aaac82068020353dce2c302e38153c0bfaa), `.ci/docker/common/install_deepep.sh` (hybrid-ep f725d29...),
`torchtitan/distributed/deepep/`, `torchtitan/models/common/token_dispatcher.py` (DeepEPTokenDispatcher L820),
https://github.com/deepseek-ai/DeepEP, the B200 MXFP8+DeepEP TorchTitan blog, `.claude/skills/kernel-numerics-verifier`,
`experiments/glm53_flash_sft_ladder/{lib/waystone.sh,publish.sh,stage.sh,lib/hdfs_strict.sh}`. Do NOT commit.
1. `native/` layout (e.g. `experiments/glm53_native/`): `packages.lock.json` listing each native package (name, upstream
   repo + pinned commit or in-repo path, patch files under `patches/<pkg>/`, build args e.g. TORCH_CUDA_ARCH_LIST=10.0),
   and `build_native.sh <pkg>|all [--dev]` that runs inside the rootfs (`glm53_ep` env): fetch source at the pin (networked),
   apply patches, build a wheel with `--no-build-isolation` against the env's torch, compute the build key
   sha256(source tree + patches + torch version + CUDA version + arch + build args), write `<pkg>-<key>.json` receipt,
   publish wheel + receipt to HDFS via waystone (`torchtitan/native-wheels/<pkg>/<key>/`, md5 + `_COMPLETE`), and an
   `install_native.sh` that stages from HDFS, verifies with hdfs_strict.sh, and installs into the env. `--dev` does an
   editable/ninja build with ccache for iteration (not publishable). ccache dir under the env state, not the image.
2. DeepEP v2 as the first package (hybrid-ep only if it builds for sm_100 and coexists; else document). Verify:
   `from deep_ep import ElasticBuffer`; DeepEP intranode test on 2 GPUs (8 only if free); TorchTitan `qwen3_moe_deepep`
   (EP=4) few steps vs all-to-all (step-1 bitwise per the recipe docstring). Co-run: <= 30 GiB/process; never kill the live
   GLM job or foreign processes.
3. Kernel package convention `torchtitan/experiments/glm53_kernels/`: Triton + CuTe DSL kernels as Python; C++/CUDA ops
   registered via `torch.library` (fake/meta impls, works under torch.compile + FSDP/DTensor) built through build_native.sh;
   one trivial example op per path end-to-end; numerics harness (oracle + eager baseline, fwd+bwd) and a microbench
   template; `experiments/glm53_native/run_kernel_checks.sh`. README listing candidate kernels (do not implement):
   fused mHC Sinkhorn/HcPost, fused indexer top-k + IndexPool, sparse MLA gather attention (check FlashMLA sparse first),
   KDA (check fla first), MoE permute for the chosen dispatcher.
4. Run artifacts: a helper that writes the installed native build keys into attempt.json for training runners.
Report `sdd/task-n01-report.md` with keys, HDFS URIs, test results.
