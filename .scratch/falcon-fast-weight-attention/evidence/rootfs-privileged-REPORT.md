# Perf tools inside the bwrap rootfs, without Docker (2026-09-06)

Claim label: `smoke` (tool plumbing). Not evidence about Falcon.

Supersedes the docker-privileged BPF path in `../rootfs_all_tools/REPORT.md`.
Every attach tool now loads from inside `enter_rootfs.sh`. Docker is no
longer part of any tracing path.

## Why the unprivileged sandbox could never load BPF

The earlier conclusion "BPF cannot load inside bwrap" was correct about the
default sandbox but wrong about bwrap. Three facts, measured on this host:

| Fact | Value |
| --- | --- |
| `kernel.unprivileged_bpf_disabled` | `1` (bpf() requires CAP_BPF or CAP_SYS_ADMIN) |
| `kernel.perf_event_paranoid` | `2` (no kernel-wide perf events unprivileged) |
| `CapEff` in the default rootfs | `0000000000000000` |

`bpf()` and the tracing `perf_event_open()` paths check capabilities against
the **initial** user namespace. BPF objects are global kernel resources and
are not user-namespace aware, so a capability minted inside a nested user
namespace is never accepted. `bwrap --unshare-all` creates exactly such a
nested namespace and additionally clears the bounding set, so no amount of
binding, `--cap-add`, or `/sys` exposure can help there.

The fix is therefore not a mount, it is a namespace decision: start bwrap
from root, do **not** unshare the user namespace, and keep only the
capabilities the tracing tools test for. Confirmation that the sandbox is in
the initial user namespace:

```
# enter_rootfs.sh --privileged
/proc/self/ns/user -> user:[4026531837]
/proc/1/ns/user    -> user:[4026531837]     # identical: init userns
CapEff: 000001ffffffffff
```

`bwrap` drops every capability by default, so `--cap-add` is an allowlist,
not an escalation: `CAP_BPF`, `CAP_PERFMON`, `CAP_SYS_ADMIN`,
`CAP_SYS_PTRACE`, `CAP_SYSLOG`.

## Second, independent bug: zero-byte library placeholders

`perf` and `bpftool` failed inside the rootfs for an unrelated reason that
looked like a sandbox problem. Two layered causes:

1. The rootfs `/usr/bin/perf` and `/usr/sbin/bpftool` are distro wrappers
   that dispatch on `uname -r`. This is a `5.15.152.bsk.9` vendor kernel with
   no matching `linux-tools` package, so they print
   `WARNING: perf not found for kernel 5.15.152.bsk.9`.
2. Binding the host binaries over them then fails with
   `libunwind-x86_64.so.8: file too short`, because the image ships
   **zero-byte** files where host-only libraries belong. `[[ -e ]]` finds
   them, so they were never noticed; they are `0` bytes against a real host
   `libbfd-2.40-system.so` of 1,514,448 bytes.

`enter_rootfs.sh` now derives the needed set from `ldd` of each bound host
binary and fills in only the paths where the rootfs copy is empty or absent,
so it tracks a host toolchain change and cannot shadow a working rootfs
library. As a side effect `perf` also works in the default sandbox now.

## Rejected alternative: keep the sandbox unprivileged

Ambient capabilities do survive a drop back to the invoking uid, so
`setpriv --reuid=1018 --ambient-caps=+cap_19,+cap_21,+cap_34,+cap_38,+cap_39`
inside a root bwrap yields `CapEff: 000000c400280000` as uid 1018. This was
rejected: `bpftrace` refuses to start with
`ERROR: bpftrace currently only supports running as the root user`, and BCC
fails on kprobe detach. The blocker is a userspace policy check, not the
kernel, so the sandbox runs as root and ownership is handed back afterward
instead.

## Results, all from inside the rootfs

`scripts/rootfs/enter_rootfs.sh --privileged`. No Docker.

| Tool | Result |
| --- | --- |
| `bpftrace` profile + `ustack` | pid-filtered user stacks, e.g. `@[python3]: 297` |
| `bpftrace` `tracepoint:sched:sched_switch` | per-`next_comm` counts |
| BCC `cachestat` | 581,757 hits / 0 misses, 100.00% |
| BCC `runqlat` | full run-queue latency histogram |
| BCC `profile -p` | on-CPU stacks for the target pid |
| BCC `offcputime` | off-CPU stacks with kernel + user frames |
| BCC `biolatency` | block-I/O histogram |
| `perf --version` | `6.6.40.ga4d39821c034` (host binary, host libs) |
| `perf stat -e sched:*,syscalls:*` | tracepoint counters against a live pid |
| `perf record -g -p PID` | 298-496 `cpu-clock` samples, report renders |
| `bpftool prog show` / `feature probe` | lists loaded programs; JIT enabled |
| `dcgmi dmon` | per-GPU rows across all 8 B200 |
| `nvidia-smi`, `nsys` 2025.6.3, `py-spy dump` | unchanged, still work |

End-to-end through the skill runner:

```
.claude/skills/attaching-live-training/scripts/attach_live.sh \
  --pid $PID --seconds 5 --privileged --tracepoints
```

17 artifacts, every one owned by the invoking user, target still `R` on
detach: `bpftrace_ustack.txt`, `bpftrace_switches.txt`, `offcputime.txt`,
`runqlat.txt`, `cachestat.txt`, `tracepoints.txt`, `perf.data`,
`perf_report.txt`, `pyspy_python.*`, `dcgm.txt`, `gpu.txt`, `proc.txt`,
`threads.tsv`, `pid.txt`, `alive.txt`.

## Ownership

The sandbox runs as real root, so it would otherwise leave root-owned files
in the checkout. Each owner cleans up its own directory: `enter_rootfs.sh`
chowns the runtime state root back after bwrap exits, and `attach_live.sh`
chowns its `--out` directory. Neither uses `exec`, because both need to run
that step as the invoking user.

## Boundaries

- Default training is untouched: still `--unshare-all`, `CapEff=0`, no
  capabilities, no sbin on PATH. Verified by
  `test_enter_rootfs_default_mode_grants_no_capabilities`.
- `--privileged` needs root or passwordless `sudo -n`, and fails with a
  message that says why rather than silently degrading.
- Use it for read-only diagnostics only, never to run training.
- Not sandbox limitations, unchanged: hardware `cycles` is unsupported on
  this vendor kernel (`cpu-clock` software sampling works), and py-spy
  `--native` gives `UNW_EBADREG`. Do not wrap host `nsys` around
  `enter_rootfs.sh`.

## Tests

`pytest tests/unit_tests/test_rootfs_bwrap_plan.py` plus the launcher and
scaffold suites: **62 passed, 1 skipped**. New cases pin the initial user
namespace, the capability allowlist, kernel-header and sbin exposure, the
empty-library fill rule, and the unprivileged default.
