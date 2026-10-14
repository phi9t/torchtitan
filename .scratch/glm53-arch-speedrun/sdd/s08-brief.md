# S08 brief: FineWeb-Edu sample-10BT -> GPT-2 token shards on HDFS (data for the epoch runs)
Read `../spec.md` ("Epoch rule"), `s01-report.md` (FineWeb .bin format + HDFS publish pattern),
`.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY: rootfs for all Python, networked mode for downloads,
waystone HDFS publish with md5), `experiments/glm53_flash_sft_ladder/{hf_ingest.sh,publish.sh,lib/waystone.sh}`,
`experiments/modded_nanogpt_b200/{prepare_data.sh,fetch_upstream.sh}`. Do NOT commit. No GPU needed.
You own `experiments/glm53_arch_speedrun/data_fineweb_edu/` (new) and `torchtitan/experiments/glm53_arch_speedrun/data_prep/` (new).
1. Download `HuggingFaceFW/fineweb-edu` config `sample-10BT` at a pinned dataset revision (record the commit sha) into
   /data05/philip.yang/glm53-ladder-cache/datasets-raw/ (check free space first; ~30 GB parquet). Inside the rootfs with
   `TORCHTITAN_ROOTFS_NETWORK=networked`. Optionally mirror the raw parquet to HDFS too.
2. Tokenize with GPT-2 tiktoken into the exact modded-nanogpt/S01 `.bin` shard format (same header magic/version, uint16
   tokens, EOT `<|endoftext|>` before each document, 100M tokens per shard), multiprocessing. First shard (100M tokens) is
   `fineweb_edu_val_000000.bin`; the rest `fineweb_edu_train_*.bin`. Deterministic document order; record total tokens.
3. Verify with S01's bin reader (header, token count, a decode spot-check); write MANIFEST.jsonl with md5 per shard.
4. Publish to HDFS `torchtitan/datasets/fineweb-edu-10BT-gpt2/<hf-rev-short>-v1` with `_COMPLETE` and md5 verification,
   then stage it back locally to `experiments/glm53_arch_speedrun/results/data/` path convention S01 uses (check disk).
   Delete raw parquet locally after a successful HDFS publish if disk is tight (keep the HDFS copy if you mirrored it).
Report `.scratch/glm53-arch-speedrun/sdd/s08-report.md`: revision, document/token counts, shard count, HDFS URI, local
path, md5 manifest location, timings, and the exact reader config fields the epoch runs should use.
