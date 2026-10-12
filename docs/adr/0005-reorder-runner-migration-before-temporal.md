# Reorder Runner Migration Before Temporal, Split F3

Given the state of the runtime-preflight implementation after waves F0-F3, the
remaining plan in `experiments/scaffold_to_policy/runtime_preflight_roadmap.md`
Section 19 is revised in two ways.

## Reorder: runner migration before Temporal durability

The original order put Temporal durability (old F4) before runner migration
(old F5). But the typed lifecycle (F1) and profile doctor (F2) are built and
tested while no runner uses them yet: roughly two dozen runners still write the
prototype manifest directly through `run_common.sh`, and only `run_preflight.sh`
touches the new path. Wrapping runners in durable Activities before a single
runner speaks the lifecycle would wrap an unproven path.

We therefore migrate one host-testable reference runner
(`run_arithmetic_words_smoke.sh`, which uses fixture rollouts and needs no GPU
or vLLM) to the begin/stage/finish lifecycle plus the `host_static` preflight
first, proving the foundation end to end. Temporal durability follows, wrapping
a lifecycle path already exercised by a real runner. Runner migration is now F4
and Temporal durability is F5; F6 is unchanged.

## Split F3 into F3a (landed) and F3b (on-device)

F3 combined a host-testable identity model with destructive filesystem and
GPU-gated capability work. Only the host-testable slice can be proven without a
rootfs/GPU session, so F3 is split. F3a is the landed content-addressed
identity, immutable store, atomic selection with rollback and quarantine, and
the fail-closed selection resolver, all covered by host-only unit and shell
tests. F3b defers the destructive `build_rootfs.sh`/`enter_rootfs.sh` rewrite,
read-only mounts, environment and capability policy, garbage collection, and
the Section 8.5 acquisition migration to a rootfs/GPU integration session,
because those change destructive paths and distributed defaults.

## Consequences

Host-testable progress continues without an on-device session. The Foundation
Promotion Gate (Section 20) still requires F3b before expensive replicated
research, since recorded and validated runtime capabilities on the live path
remain a prerequisite; the reorder does not weaken that gate, it only sequences
the host-provable work first.
