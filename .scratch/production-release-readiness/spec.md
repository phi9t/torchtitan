# TorchTitan Production Release Readiness

Status: approved
Path: gated
Parent: ultron:.scratch/production-release-readiness/issues/04-qualify-torchtitan.md

## Outcome

Produce a sanitized `ultron/mainline` candidate that independently implements
the pinned Vaso v1 contract and proves TorchTitan's native build, tests, and
multi-B200 debug-training boundary inside bwrap.

The Ultron parent effort keeps rootfs implementation in `needs-info` until
Vaso's linked `02-freeze-rootfs-spec-v1` ticket resolves and its digest is
recorded here.

## Current state

TorchTitan already has a substantial builder and bwrap launcher, CUDA-driver
projection, content-store selection, and rootfs tests. Its remaining contract
gaps include mutable image/tool inputs and rootfs identity that does not cover
the full locked recipe and output. The primary checkout and additional
worktrees contain substantial unrelated training work that requires explicit
disposition. The redacted baseline also found machine/network paths and commit
metadata requiring review.

## Rootfs adaptation

TorchTitan owns its materializer, content store, selection, launcher, schemas,
fixtures, and evidence. Replace floating inputs with immutable digests; bind
identity to the complete recipe, dependencies, and filesystem content; enforce
read-only root, repo-local writable state, environment filtering, explicit
network mode, and compatible B200 projection. Existing rootfs entrypoints may
remain when they pass the pinned Vaso conformance behavior without importing
Vaso or Ultron.

## Approved test seam

Use materialization manifests, selected-rootfs validation, bwrap plans, launcher
results, and run-attempt receipts as public seams. Native acceptance includes
focused rootfs tests, the owning unit suites, and a rootfs-wrapped multi-B200
debug training run with distributed checks and immutable evidence.

Before that run, a human must approve an exact qualification contract naming
the command, immutable configuration/model/data fixtures, GPU count and
topology, steps or duration, correctness and tolerance thresholds, any claimed
performance floor, negative cases, timeout/resource budget, receipt schema,
and signing verification. The resulting child receipt is signed and binds all
of those fields to the exact commit and rootfs identity.

## Publication

Preserve reviewed public upstream ancestry. Reconstruct admissible local work
with approved public identity, classify every dirty worktree, and resolve all
PII, machine, network, proprietary, generated-artifact, license, and metadata
findings. `origin` must be the `phi9t` fork; `upstream` is fetch-only.

## Acceptance criteria

- The complete rootfs input and output closure is immutable and verified.
- Clean bootstrap, tamper, environment, writable-path, network, and B200 driver
  failure tests pass.
- Native tests and the human-approved exact multi-B200 contract pass inside
  bwrap and produce an independently verified signed receipt.
- All local work has a disposition and all release-relevant work is integrated.
- Exact history/tree desensitization has no unresolved finding.
- Independent review finds no unresolved blocking defect.

## Out of scope

- Publishing unrelated experimental work, packages, containers, or model
  artifacts; qualifying non-B200 hardware; depending on Vaso or Ultron.

## Authority

Approved tickets may change local files and run tests in repo-local worktrees.
Commits, branch moves, remotes, pushes, tags, releases, deletion, and merges
require separate authorization.
