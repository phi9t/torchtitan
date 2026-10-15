# T08g brief: close out T08f (rung 4) -- sampled eval, EP-path agreement, HDFS publish, report

Read first: `task-08f-brief.md` (gates), `progress.md` (entries from ~04:40Z and ~16:10Z on 10-15), `spec.md`
(done-definition). Run dir: `experiments/glm53_flash_sft_ladder/results/t08f-20261014T213158Z/attempt-01` ($R).
Training gates 1-4 are done (seed42/seed43 94 steps, exact HDFS resume at 62, PEFT adapters in $R/adapters/). The first
EP eval was invalid (generator sampled hidden states; fixed in e2e8af1b6; invalid outputs in $R/eval_invalid_skip_lm_head).
The greedy reference gate is green: `torchtitan/experiments/glm53_flash/reference_eval.py`, launcher
$R/logs/reference_eval.sh, results $R/eval_reference/dev/{base,seed42,seed43}: pass@1 0.38 / 0.84 / 0.87.

Decided (do not re-litigate): the plain-model reference evaluator is the eval of record; the EP generator is an
agreement check. vLLM cannot load glm5_next (record $R/eval/vllm_probe.json as the reason).

Steps:
1. Gate 5 sampled eval with reference_eval.py on GPUs 0-6 only (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`; GPU 7 belongs to
   another worker until it is idle): base, seed42, seed43; first 100 dev problems x 4 samples, temperature 0.8, top-p
   0.95, max 192 new tokens, seed 804, chat template. Choose --batch-size and --layer-split to fit 7 GPUs (6 sparse layers
   + batch 25 peaked ~143 GB/GPU on 8 GPUs). Output $R/eval_reference_sampled/dev/<leg>. Report pass@1 and strict pass@1
   with bootstrap 95% CIs (problem-level resampling) and paired deltas vs base; `replicated_eval` only if both seeds beat
   base with CIs excluding 0. Keep the base format-limit caveat (base writes long reasoning, hits 192 tokens).
2. EP-path agreement: once GPU 7 has no process (check with nvidia-smi; never signal anything you did not start), run
   the fixed `generate.py` (EP8, grouped experts, the T08f config) greedy on the first 20 dev problems for seed42 and
   base; compare per-problem success and completion text with the greedy reference outputs. Expect bf16 drift, not
   identity: report exact-match rate, first-divergence token positions, and pass@1 on the subset. Flag a disagreement
   in pass@1 > 2 problems as a bug to investigate. If GPU 7 is not free by the time everything else is done, report NOT RUN.
3. Gate 6: publish adapters, both eval trees (greedy + sampled + EP agreement), logs, TensorBoard, configs, manifests to
   HDFS `torchtitan/runs/glm53-ladder/rung4/<run_id>/<attempt_id>/` via the existing waystone helpers (host side), with
   md5 manifest + `_COMPLETE`, verified get. Large checkpoints already on HDFS from the resume gate: reference, do not
   re-upload.
4. Report `sdd/task-08g-report.md` (and fill `task-08f-report.md` as the rung-4 summary): every T08f gate PASS/FAIL/NOT RUN
   with evidence, HDFS URIs, throughput/memory from the training legs, and the spec.md done-definition checklist.
   Compact summaries (summary.json/md, manifests, rollouts <= 200 KB each) to `evidence/t08f-reference-eval-sampled/`.
Rootfs for all Python/CUDA (TORCHTITAN_ROOTFS_ENV per the T08f launcher). Do not commit.
