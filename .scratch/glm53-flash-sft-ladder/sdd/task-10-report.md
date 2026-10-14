# T10 report: GLM-5.3-Flash SFT ladder campaign documentation

Status: COMPLETE. No commit was made. No GPU work was run.

Surface: documentation only.

## Output

Wrote the first full campaign engineering record:

- `docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md`

The document covers the required goal/done definition, storage contract,
environment contract, ladder table, GLM integration seam, numerics, memory,
incidents/lessons, reproduction commands, and open items.

## TODO placeholders left

The living doc intentionally leaves these in-flight-ticket placeholders:

| Placeholder | Reason |
| --- | --- |
| `TODO(T06)` | Qwen3 30B-A3B rung 2 report is not present in the current SDD set. |
| `TODO(T08d)` | Full GLM-5.3-Flash rung 4 run, resume, eval, and publish are explicitly in flight. |
| `TODO(T09a/b/c)` | The three 4-GPU GLM directions are launched/planned but have no reports in the current SDD set. |
| `TODO(T11)` | Expert-parallelism deep dive is a separate in-flight design ticket. |

No results were guessed for those placeholders.

## Source policy

Claims were sourced to the campaign spec/map/progress files, SDD briefs and
reports, runner scripts, GLM experiment files, the architecture research doc, and
the commit window `git log --oneline 96760e290^..HEAD`.

## Verification

Run after writing:

```bash
! LC_ALL=C rg -n "[^\\x00-\\x7F]" \
  docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md \
  .scratch/glm53-flash-sft-ladder/sdd/task-10-report.md

rg -n "TODO\\(T06\\)|TODO\\(T08d\\)|TODO\\(T09a/b/c\\)|TODO\\(T11\\)" \
  docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md \
  .scratch/glm53-flash-sft-ladder/sdd/task-10-report.md

rg -n "## Goal and Done Definition|## Storage Contract|## Environment|## Ladder Status|## GLM Integration Seam|## Numerics|## Memory|## Incidents and Lessons|## Open Items and Next Steps|## How to Reproduce" \
  docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md

git diff --check -- \
  docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md \
  .scratch/glm53-flash-sft-ladder/sdd/task-10-report.md
```

