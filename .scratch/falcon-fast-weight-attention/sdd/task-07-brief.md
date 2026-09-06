# Ticket 07 brief

Read first:

1. `.scratch/falcon-fast-weight-attention/issues/07-step4-ablations.md`
2. `.scratch/falcon-fast-weight-attention/sdd/task-06-report.md`
3. `.scratch/falcon-fast-weight-attention/spec.md` Step 4 arms A0-A5

Ticket 06 is done. Host has 8 idle B200s and **44G free**. Do not enable
checkpoints. Do not download data. Do not commit.

## Close 06 leftovers first (TDD)

1. Thread `normalize_qk` / `phi` (`"rms"` default vs `"l2"`) from
   `FalconConfig` into `FalconMixer` so A5 is a real arm. Kernel already
   accepts `normalize_qk`.
2. Optional but preferred: GDN `q` scale by `K**-0.5` after L2, matching
   `kda_reference_forward`. Keep GDN in-file; do not import Mini-K3.

Existing `test_falcon_*.py` must stay green (60+).

## Arms (matched: FineWeb bins, seq 512, GBS 32, 16384 tok/step)

| ID | Knob |
| --- | --- |
| A0 | mixer=softmax |
| A1 | mixer=gdn |
| A2 | mixer=falcon variant=falcon1a alignment=delayed phi=rms |
| A3 | mixer=falcon variant=falcon1 alignment=delayed phi=rms |
| A4 | mixer=falcon variant=falcon1a alignment=same_step phi=rms |
| A5 | mixer=falcon variant=falcon1a alignment=delayed phi=l2 |

Skip A6/A7.

## Run plan

1. Few-step dry-run of all 6 arms in rootfs (2-4 steps). Finite loss. Smoke only.
2. Then the table: **2 seeds**, **8000 steps** (floor 4000 if a run is
   unstable). Use **one B200** sequential (do not hog all 8). Rootfs
   `enter_rootfs.sh`. `checkpoint.enable=False`. Compact JSON/markdown
   table under `experiments/falcon/results/` (gitignored).
3. Held-out CE/PPL on `fineweb_val_000000.bin` (or a bounded val window).
4. Separate small addition trains per mixer (N=16 ID, M=24 OOD), teacher-forced
   suffix acc. Keep these cheap vs the LM arms.
5. Freeze one mixer+alignment+phi triple for ticket 08. Deltas inside seed
   noise are ties. Record the table even if Falcon only wins on addition.

## Claims

- Dry-run: `smoke`
- Table: `representative_small` + `replicated_eval`
- Do **not** call this 130M / 50B.

## Do not

Commit. Hero budgets (08). Falcon-2/3. Import qwen3_5 / mini_kimi_k3.
Fill the 44G disk. Change knobs after seeing results.

## Report

`.scratch/falcon-fast-weight-attention/sdd/task-07-report.md`
plus `## Answer` on the issue and winner in `map.md`.
