# Ledger: glm53-flash-sft-ladder

- T00 rootfs dep sync (orchestrator): complete -- requirements.txt pins installed; torch-nightly training env `glm53_train`; see sdd/env-facts.md
- T01 waystone wrappers + dataset v1: complete (traecli) -- countdown-sft/v1 + Qwen3-0.6B on HDFS, stage round trip MD5-verified; claim smoke
- T02 rung 0 Qwen3 0.6B/1.7B: complete -- PASS all gates; resume bitwise; dev pass@1 base->adapter 0.6B 0.004->0.254/0.326, 1.7B 0.024->0.370/0.228 (replicated_eval)
- T03 HF ingest to HDFS (+ kept local copy): complete 10:23Z -- Qwen3-0.6B/1.7B/8B/30B-A3B and GLM-5.3-Flash-BF16 (a5b45eb) on HDFS with _COMPLETE + local staged copies on /data05
- T04 mHC HcPost: complete -- CONFIRMED; fixed in 96760e290 (sum over source axis), DSv4 debug trains
- T05 rung 1 Qwen3 8B: complete -- PASS; parity top-1 1.0; resume PASS; dev pass@1 base 0.028 -> 0.474/0.488 (replicated_eval)
- T06 rung 2 Qwen3 30B-A3B MoE: in-progress (traecli)
- T07a GLM-5.3-Flash model seam: complete (bf5b4e8c5) -- 11 oracle tests, debug+LoRA 2-GPU smoke; gaps: EP dispatch, TP/CP, real-weight parity (-> T07c)
- T07b rung 3 mini hybrid learnability: complete -- overfit gate PASS (indexer + dense ablation, lr=0 flat, curves differ); seq 4k/8k finite; full-size plan 86.9 GiB/GPU @8 (analytical); found: LoRA path does not freeze base (-> T08 req 0)
- T07c real-weight parity: complete -- fp32 slices 0-3 (1e-5) and 20-27 (top-1 1.0, KL 2e-4) PASS, selection masks exact; key coverage 0 unknown; GLM chat/mask data path; OPEN: bf16 20-27 top-1 0.43 (-> T08 req 0b)
- T08 rung 4 attempt 1: BLOCKED -- LoRA freeze fixed+tested; full HF load materializes full model per rank (host RAM incident 12:56Z); bf16 slice drift persists on countdown prompts. -> T08b (sharded loader + trainable-only ckpt + run), T08c (bf16 diag)
- T08c bf16 diag: complete -- TT-bf16 and HF-bf16 equally far from fp32 truth (top-1 ~0.03-0.09 on the truncated mid slice); not a TT bug; fp32 parity remains the identity gate

## 14:25Z
- T08b PARTIAL -> committed db23be429: rank-local sharded HF load (21.7 s, rank RSS 3.75 GiB), trainable-only checkpoint, full 8xB200 one-step LoRA probe (loss 1.676, 131.6 GiB/GPU). claim: smoke.
- T08d launched: full rung-4 run (seed 42 + resume via HDFS + seed 43), eval base vs 2 adapters, publish. Gated on all 8 GPUs >= 150 GiB free; foreign root proc observed on GPU 3 (~54 GB) at 14:17Z.
- 14:35Z user asked for all three 4-GPU directions: T09a weight streaming (FSDP2 CPU offload + rank-local loader), T09b FP8 frozen base quantized at load, T09c bf16 overhead cuts. Launched concurrently in separate packages; full-depth launches serialized by flock, yield to T08d.
- User: next step after T08d/T09 is an expert-parallelism deep dive; capture all work in docs. Launched T10 (campaign doc `docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md`, TODO placeholders for in-flight tickets; refresh after T06/T08d/T09 land) and T11 (EP deep dive phase 1: analysis + design doc `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, fake-backend only; phase 2 GPU work queued after T08d/T09).
- ~15:05Z T11 EP phase 1 COMPLETE (EP does not cut static frozen-base memory at fixed world size; shrinks largest unshard; routed experts are not LoRA targets). T12 (EP phase 2 tickets 1-3, fake backend) launched. Schedule table added to ladder doc.
- ~15:20Z T09a PARTIAL (committed fc412ef58): stream package + truncated 3-step smoke; step-1 loss identical but grad norm 1.7734 vs 1.8828 -> T09a2 root-causes, then full depth via lock (T09a gave up on held lock).
- 15:17Z T08d starved by T09c full-depth lean job (2,3,6,7) + T06/T09 probes. Started priority_hold.sh: takes glm-fullsize lock when free, holds until T08d report/exit, so T09 full-depth legs queue behind rung 4. T12 DONE (9124de431); T13-16 briefs ready, queued after T08d.
- ~15:50Z T09c DONE (committed): full-depth bf16 4-GPU fits (174.7/178.4 GiB) via split FSDP units + no prefetch; 6.3 min/step, timed out at step 6 -> negative readiness. Throughput is the next 4-GPU problem.
- ~16:45Z T09b DONE (committed): FP8 frozen base fits full-depth on 4 GPUs (~116 GiB reserved, static 74.5 GiB), 2 steps in 45 min (dequant+F.linear, not FP8 GEMM). Truncated loss drift 0.0004 -> 0.029 by step 3 vs bf16.
- 4-GPU synthesis: all three directions FIT (lean 174.7 GiB tight; FP8 ~116 GiB; stream ~68 GiB GPU on truncated) but all are too slow (lean 6.3 min/step, FP8 ~20 min/step, stream pending T09a2). Candidate next ticket (after T08d frees GPUs): T09d = FP8 storage + normal FSDP prefetch + fused dequant/FP8 GEMM (torch._scaled_mm blockwise), target < 1 min/step.
- 17:25Z T08d attempt-02 seed42 full launched ~16:58 then died 17:17: 'CUDA-capable device(s) is/are busy or unavailable' -- GPUs 2,3 were switched to Exclusive_Process compute mode with foreign root sonic profiling procs on them. Memory-free gate is insufficient under exclusive mode. attempt-03 waiting at gate. Nothing to kill (foreign).
- ~17:35Z T06 rung 2 PASS (1e707e0b6): dev pass@1 base 0.006 -> armA 0.440/0.476 (replicated), armB 0.600 (1 seed); resume exact; router no collapse; HF parity smoke only (top-1 0.912, max |dlogit| 3.66 bf16). Doc TODO(T06) to fill in next doc refresh.
- ~18:05Z user: all training uses HDFS strictly. env-facts gains HDFS-STRICT INPUT RULE; H01 launched (hdfs_strict.sh guard + audit + patches for in-flight tickets).
- 18:10Z user clarified: pasted Primus text was a wrong paste; HDFS access remains via waystone only (added to env-facts).
- 17:50Z T08d gate requires ZERO GPU processes; our speedrun jobs starve it. Started priority_pause.sh: SIGSTOPs S06/S07/S09 traecli workers while T08d gates, SIGCONT when GLM is up.
- ~18:20Z H01 DONE (committed): hdfs_strict.sh + test; wired into run_rung0/run_rung; all /data05 canonical inputs strict-PASS; T07b evidence inputs re-ingested to HDFS.
  PENDING PATCHES (apply when owner ticket lands; text in task-h01-report.md "Patches For Active Ticket Files"):
  T08d run_rung4_glm.sh; S06 experiments/glm53_arch_speedrun/run_arm.sh; S07 experiments/glm53_arch_ladder/run_rung.sh;
  T09 direct-launch preflight snippet. S08 patch applied. Backfilling .staged.json for FineWeb + FineWeb-Edu data dirs.
- 18:40Z T08d seed42 full leg running since ~18:00Z but ~250 s/step (6 steps/25 min; loss 1.678 -> 0.715). py-spy: ranks
  CPU-bound in GlmExperts.forward per-expert Python loop (nonzero/where host syncs). Likely also explains T09c/T09b
  slowness (correcting earlier "prefetch/dequant" attribution). T17 launched: grouped-GEMM experts behind
  GLM53_FLASH_EXPERTS_IMPL (default loop). Decision pending T17: keep T08d legs on loop path (~6.6 h/leg) or restart with
  grouped after parity. Current seed42 leg serves as the loop-path reference curve.
- 18:50Z user: "do something like DeepEP / smarter EP, not FSDP for experts". EP moved onto rung-4 critical path.
  Launched T13 (EP truncated parity + timing, grouped local experts) and T18 (DeepEP-class dispatch eval: DeepEP SM100
  support, all-to-all vs DeepEP vs Dist-MoE microbench at GLM shapes, dispatcher adapter). T16 (Dist-MoE) queued after T13.
  T17 (grouped experts) continues as the local-compute prerequisite. T08d loop-path run kept as fallback.
  priority_pause.sh restarted to also cover task-13..18 workers.
- ~19:00Z user: TorchTitan already integrates DeepEP v2 (DeepEPTokenDispatcher) + B200 MXFP8/DeepEP blog. T18 relaunched: install pinned DeepEP v2 for SM100 in glm53_train, qwen3_moe_deepep smoke, GLM EP with DeepEPTokenDispatcher, benchmark vs all-to-all/FSDP.
- ~19:10Z user: first priority = rootfs has DeepEP done properly, following TorchTitan + the B200 MXFP8/DeepEP blog. Stopped
  T18 (had begun an ad hoc DeepEP build). Launched R01: declared reproducible env (lock + sync + verify) as new env
  `glm53_ep` with DeepEP v2 @01dc3aa (SM100), hybrid-ep if supported, torchao MXFP8, NVSHMEM; no live-rootfs rebuild;
  insula repo untouched (its AGENTS.md excludes ML framework builds). T18 (GLM on DeepEPTokenDispatcher) re-queued after R01.
- ~19:25Z user: DeepEP must be in the rootfs image via torchtitan's Insula builder (build_rootfs.sh). R01 rev2 relaunched: build_rootfs.sh --variant ep (torch nightly + DeepEP v2 SM100 + torchao MXFP8 + fla + NVSHMEM), publish to store /data05/philip.yang/torchtitan-rootfs-store, select per-run; legacy rootfs/glm53_train untouched.
- ~19:45Z T17 DONE (23e39799e): grouped experts 39.5x on a full-width MoE layer; parity fp32 <=7e-8. T08d loop-path seed42
  at step 8 / 44 min (loss 0.289). Plan: stop T08d, relaunch rung 4 as T08e on grouped path (brief written). Stopping our
  own T08d processes was blocked by the auto-mode permission check -> awaiting user OK.
- User: GLM work will need many custom kernels -> N01 brief (native toolchain in the EP rootfs variant, kernel package
  convention with torch.library ops, numerics/bench harness), blocked by R01.
- ~19:55Z user approved three-layer native setup (`sdd/native-layering.md`). R01 rev3 (L0 audit + L1 declared env
  `glm53_ep`) relaunched; N01 rev2 (L2 build_native.sh + DeepEP first + kernel package convention + wheels on HDFS) blocked
  by R01; T18 (GLM on DeepEPTokenDispatcher benchmark) blocked by N01. T08e still awaits user OK to stop T08d.
- ~20:00Z user: "kill the slow run first; get EP to a good state before a big MoE run". Stopped T08d (worker, runner,
  8 GLM ranks; loop path reached step 8-10, loss 0.289 at step 8 -- kept as reference curve) and T09a2 (worker + its
  full-depth 4-GPU stream leg, loop path). priority_hold/priority_pause exited and released. Rung 4 (T08e brief ready) and
  4-GPU re-measurement are ON HOLD until EP is in a good state: T13 (all-to-all EP parity/timing), R01 -> N01 (DeepEP
  wheel) -> T18 (DeepEP vs all-to-all), T16 (Dist-MoE).
- ~19:20Z T13 DONE (3aa4f8048): first-8-layer EP4 on 8 GPUs 5.3 s/step (vs ~110 s loop FSDP), 21.6 GiB/rank; step-1 loss
  delta 8e-5; grad norm +3.4% (same symptom as T09a/b). T13b launched: per-param grad dumps to root-cause, fair grouped
  FSDP baseline, EP8, 20-step curves, full-depth EP8 timing probe (<= 15 steps). T16 (Dist-MoE) waits for T13b.
- ~19:35Z K01 DONE (kernel correct, slower than masked dense; attention is not the bottleneck). K02 launched: profile-first indexer hot-spot fix (target A4/A5 >= 0.5x B0).
- ~19:45Z R01 DONE partial (committed): native L0 store entry + glm53_ep declared env; glm53_ep import fails (fla/torchao Triton autotune-key) though glm53_train works -> R01b launched to reproduce glm53_train exactly. N01 waits on R01b.
- ~20:00Z R01b DONE (committed): glm53_ep green. N01 launched (L2: DeepEP v2, then torchao from source for MXFP8).
- ~20:15Z N01 partial (f20d2d9db): L2 layer + kernel convention done; DeepEP/torchao blocked by mixed CUDA pip toolchain (nvvm/crt 13.4 vs nvcc/ptxas 13.2). N01b launched to pin one CUDA release in native L0, rebuild store entry, then build DeepEP + torchao.
