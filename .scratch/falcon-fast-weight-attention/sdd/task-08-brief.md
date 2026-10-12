# Ticket 08 brief — Step 5 local hero

Read first: `spec.md` Step 5, `issues/08-step5-local-hero.md`,
`.claude/skills/running-experiment-campaigns/SKILL.md` + `workflow.md`.

## Requirements

- One 20k-step (328M tokens) run each of frozen A5, Transformer A0, and
  GDN A1. Same tokens/step (16,384). Do not search knobs.
- Single node, three GPUs in parallel, rootfs for Python/CUDA.
- Pause/resume: full train-state checkpoints (`enable=True`,
  `interval=500`, `keep_latest_k=2`, `last_save_model_only=False`).
  `PAUSE` file or SIGINT/SIGTERM finishes the current step, writes a
  resumable checkpoint, and exits. Re-run the same command to resume.
- Fail loud if free disk is below 5G.
- `claim_label=representative_training`. Not Table 1 / 49.2B.

## Exclusions

- New mixer variants, retuning, context parallel, promotion into
  `torchtitan/models/`, science fetch, commit unless asked.

## Blockers

- 05 resolved (FineWeb 10B GPT-2 bins, 16,384 tok/step).
- 07 resolved; winner frozen A5 (`falcon1a` / delayed / `l2`).

## Verification

- `pytest -q tests/unit_tests/test_falcon_hero_resume.py` in rootfs.
- Dry-run three arms a few steps, then launch 20k.
- Three run-attempt reports (or a written skip) + finite losses +
  checkpoints.

## Report

`sdd/task-08-report.md` + issue Answer + `map.md` Frontier.
