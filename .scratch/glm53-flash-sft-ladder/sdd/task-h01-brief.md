# H01 brief: enforce the HDFS-strict input rule across both campaigns
Read `env-facts.md` (esp. "HDFS-STRICT INPUT RULE", binding), `experiments/glm53_flash_sft_ladder/{lib/waystone.sh,stage.sh,
publish.sh,hf_ingest.sh,README.md}`, and the runners: `experiments/glm53_flash_sft_ladder/{run_rung0.sh,run_rung.sh,
run_rung4_glm.sh}`, `experiments/glm53_arch_speedrun/{run_arm.sh,scheduler.sh}`, `experiments/glm53_arch_ladder/*`
(may be in progress by S07), and the GLM / T09 / EP config registries' env-path defaults. Do NOT commit. No GPU use.
Do not edit files owned by running tickets (T08d: glm53_flash/*, run_rung4_glm.sh; T09a2: glm53_flash_4gpu_stream;
S06: glm53_arch_speedrun/*; S07: glm53_arch_ladder; S08: data_fineweb_edu) -- instead write exact patches for them into
your report so I can apply them when each ticket lands.
1. Add `experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh` (host-side bash; md5 via `md5sum`, no Python):
   `hdfs_strict.sh [--quick] <local_dir>...` -> for each dir require `.staged.json` (source HDFS URI) and a local copy of
   the HDFS `MANIFEST.jsonl`; verify every manifest file exists with matching size (+ md5 unless --quick, where md5 is
   verified once and cached in `.verified.json` keyed by mtime+size); print the HDFS URI + manifest md5 per dir; exit
   non-zero on any mismatch or missing marker. Add a small bash test under `experiments/glm53_flash_sft_ladder/tests/`.
2. Audit: list every input path each runner / config default / recent run used (datasets, weights, tokenizers,
   checkpoints), classify HDFS-staged vs not, and for each non-HDFS input either ingest it to HDFS + stage it (e.g. the
   Qwen3-0.6B tokenizer copy under `.scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/`, GLM evidence inputs), or
   name the patch. Check that `.staged.json` exists for every dir under /data05/philip.yang/glm53-ladder-cache and the
   arch-speedrun FineWeb data dir.
3. Wire `hdfs_strict.sh --quick` into runners you own/that are idle: `run_rung0.sh`, `run_rung.sh` (T06 done).
Report `.scratch/glm53-flash-sft-ladder/sdd/task-h01-report.md` with the audit table and ready-to-apply patches.
