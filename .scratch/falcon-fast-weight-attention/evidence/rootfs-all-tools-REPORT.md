# In-rootfs tool proof (2026-09-06)

Everything below was launched with `scripts/rootfs/enter_rootfs.sh --profile`
(or the default sandbox for Nsight). Claim: `smoke`. Not ticket 08.

`--profile` keeps host PIDs, binds Docker, and shares net so DCGM
host-engine and the privileged helper work. Default training stays
`--unshare-all` + offline.

BPF cannot load in the unprivileged user namespace. The helper
`run_privileged.sh` is still *from inside the rootfs*: it uses the
bound Docker socket, `docker run --privileged --pid=host`, and chroots
the rootfs (or `--host` for distro `perf`/`bpftool`).

## Works

| Tool | How | Evidence |
| --- | --- | --- |
| nsys | default rootfs (`/sys` + CUDA bind) | `nsys_in_rootfs/a5_20.nsys-rep` cutlass sgemm |
| ncu | default rootfs | `ncu_sgemm.ncu-rep` (`cutlass3x_sm100_simt_sgemm_*`, 9 passes) |
| compute-sanitizer | default rootfs | `compute_sanitizer.txt` 0 errors |
| nvidia-smi / dcgmi dmon | default or profile | prior V1 + discovery of 8× B200 |
| dcgmi health | `--profile` (needs host-engine / net) | set PCIe+memory; check **Healthy** |
| dcgmi stats job | `--profile` | `rootfs_smoke` 6.61 s window, Healthy, 0 Xid |
| py-spy | `--share-pid` / profile | prior V1/V3 |
| memray launch | default | `py_profilers/memray.bin` |
| memray attach | profile + rootfs `gdb` | `memray_attach.bin` gdb `SUCCESS` |
| Scalene `--gpu` | default | `scalene_gpu.json` |
| Fil | default | prior toy flamegraph (no cgroup `memory.high` on this host) |
| tracemalloc | default | `tracemalloc.json` peak 1.6 MiB |
| bpftrace | `--profile` + helper | `BEGIN` ok; `profile:hz:49`; `sched:sched_switch` |
| BCC `cachestat` | helper + host `/lib/modules` + headers | 98–100% HITRATIO, 2.7 TiB cached |
| BCC `runqlat` | helper | histogram, most mass 0–7 us |
| BCC `profile` | helper | kernel stacks |
| `perf` tracepoints | helper `--host` | `sched_switch` / `sys_enter_*` |
| `bpftool prog show` | helper `--host` | lists loaded programs |

## Still not a tool failure

| Item | Reality |
| --- | --- |
| Unprivileged `bpf()` in bwrap | `kernel.unprivileged_bpf_disabled=1` and CapEff=0. Helper is required. |
| Host `nsys` around `enter_rootfs.sh` | Still cannot inject. Use in-rootfs nsys. |
| py-spy `--native` | `UNW_EBADREG` on this CPU. Python-only works. |
| hardware `cycles` | `<not supported>` even as container-root. Use `cpu-clock`. |
| Fil `memory.high` | This host has no cgroup v2 `memory.high`. Fil still writes a flamegraph. |
| DCGM job process attribution | Short gemm window showed 0 processes; health/Xid path works. Start the watch before a longer job. |

Rebuild of the rootfs wipes apt/pip extras. Reinstall:

```
.claude/skills/attaching-live-training/scripts/install_ebpf_tools.sh
.claude/skills/attaching-live-training/scripts/install_python_profilers.sh
```
