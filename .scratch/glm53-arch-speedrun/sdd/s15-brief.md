# S15 brief: close-out -- results doc, HDFS bundle index, campaign claim
Blocked by: S09, S14 (and S10-S13 reports). Rewrite `docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md`
results sections from the reports (no TODOs left; every number cites its report), add an HDFS index of every run bundle
and dataset with md5 manifests, a 'what transfers to full-scale GLM-5.3-Flash and what does not' section with honest claim
labels, and open questions for the next campaign. Update `map.md` and `progress.md`. `s15-report.md`. No GPU.

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.
