# Falcon Step 4 ablation table (claim_label=representative_small + replicated_eval)

## LM arms (held-out val on fineweb_val_000000.bin)

| arm | mixer | variant | alignment | phi | seed | steps | final_loss | val_ce | val_ppl | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | gdn | falcon1a | delayed | rms | 0 | 8000 | 4.6746 | 4.7973 | 121.19 | True |

## Addition transfer (separate small train per mixer, teacher-forced)

| arm | mixer | variant | alignment | phi | seed | ID acc | OOD acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
