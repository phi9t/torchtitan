# R01b brief: make `glm53_ep` reproduce `glm53_train` exactly (then it is L1-done)
Read `native-layering.md`, `env-facts.md` (MANDATORY), `task-r01-report.md` (blockers: `fla.ops.kda` and
`torchao.prototype.moe_training` imports fail in glm53_ep with "Triton autotune key contains BT, but the kernel argument list
omits BT"; flash-attn CUDA ext skipped), `experiments/glm53_flash_sft_ladder/runtime/`. Do NOT commit. Never touch the live
`glm53_train` venv or legacy rootfs; never kill other jobs.
Key fact: the SAME nominal versions (torch 2.16.0.dev20261013+cu132, triton 3.9.0+gitfc8d31a5, fla-core 0.5.2, torchao)
import and train fine in `glm53_train` (all GLM work so far ran there, incl. KDA). So glm53_ep differs from glm53_train.
1. Diff effective import resolution in both envs (inside each, same rootfs where possible): for torch, triton, fla, torchao,
   flash_attn, attn_gym, torch_remat, spmd_types print `__version__`, `__file__`, dist-info RECORD hashes; `sys.path` order;
   system-site-packages layering (glm53_train was a venv over the rootfs system python which has torch 2.13/vLLM/triton);
   env vars (TRITON_*, PYTHONPATH); and whether glm53_train loads triton from a different location/build. Find the exact
   difference that makes the autotune-key validation fire.
2. Fix the L1 lock/sync so glm53_ep reproduces glm53_train's working resolution (pin the exact wheels/builds glm53_train
   uses, by sha256; if glm53_train relies on a locally built or system-provided package, capture that as a declared build
   step or L2 native package). Re-verify: `verify_runtime.py` green, `pytest torchtitan/experiments/glm53_flash/tests`,
   `glm53_flash_ep/tests`, `glm53_arch_speedrun` tests, and a 2-step GLM debugmodel train in glm53_ep on the native store
   rootfs; compare `uv pip freeze` of both envs (expect identical or explained diffs).
3. flash-attn: is it actually used by any GLM/speedrun path? If not, drop it from glm53_ep; if yes, make it an L2 package.
Report `sdd/task-r01b-report.md`; update the runtime README and env-facts.
