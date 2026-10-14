# FineWeb-Edu sample-10BT GPT-2 data

This directory holds host wrappers for S08 data preparation. Python work runs
inside `scripts/rootfs/enter_rootfs.sh`; use networked rootfs mode for the
Hugging Face download.

```bash
TORCHTITAN_ROOTFS_NETWORK=networked experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh download \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh tokenize \
  --raw-dir /data05/philip.yang/glm53-ladder-cache/datasets-raw/fineweb-edu/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --output-dir /data05/philip.yang/glm53-ladder-cache/datasets-tokenized/fineweb-edu-10BT-gpt2/87f0914-v1

experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh verify \
  --data-dir /data05/philip.yang/glm53-ladder-cache/datasets-tokenized/fineweb-edu-10BT-gpt2/87f0914-v1
```

