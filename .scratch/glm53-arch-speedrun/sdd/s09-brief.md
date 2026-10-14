# S09 brief: ladder epoch runs -- L0-L4 x 2 seeds, one full epoch of FineWeb-Edu sample-10BT
Blocked by: S07 (ladder configs + per-rung Step-3 gate PASS), S08 (FineWeb-Edu 10BT on HDFS), H01 (hdfs_strict.sh).
Goal: spec 'Epoch rule' + 'Architecture ladder'. Each rung L0 (GPT-2/B0), L1 Qwen3 dense, L2 Qwen3-MoE, L3 mini-Kimi-K3,
L4 mini-GLM-5.3-Flash trains exactly one epoch over the S08 train shards (all tokens once, deterministic order, no
repeats; record the exact token count), seeds {0,1}, with the S07 frozen configs (no knob changes except LR schedule
length scaled to the epoch: same peak LR, warmup/cooldown fractions from S05). Eval every ~250M tokens on the FineWeb-Edu
val shard AND the FineWeb val shard (S01). Checkpoint every ~1B tokens to HDFS (resumable; a preempted run resumes from
HDFS, not local). Use 1-4 GPUs per run with the same global batch when free; at most 6 GPUs total for S09.
Report `s09-report.md`: per-rung curves, final val CE (both sets) mean+/-std, tokens/s, GPU-hours, deltas vs L0 and
previous rung with the 2x seed-std rule, HDFS URIs; fill TODO(S09) in the ablation doc. Claim: `representative_training`
at 10B tokens for 124M-scale models.

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.
