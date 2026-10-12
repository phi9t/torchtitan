# Live-attach command reference

NVIDIA, Nsight, DCGM, Python profilers, bpftrace, BCC, and `perf`
tracepoints all run from inside `enter_rootfs.sh`. `attach_live.sh`
re-enters with `--profile`, or with `--privileged` when given
`--privileged`. BPF loads natively under `enter_rootfs.sh --privileged`
(root bwrap in the initial user namespace, `CAP_BPF` retained); no Docker.
Substitute `$PID` (host pid) and `$OUT`. Bound every probe. The process
must still be `R` or `S` when you detach.

Catalog of NVIDIA / eBPF / Python tools, attach class, and TODOs:
[tools.md](tools.md).

## Privilege ladder

```bash
# What this login can do without sudo
sysctl kernel.unprivileged_bpf_disabled
cat /proc/sys/kernel/perf_event_paranoid   # 2 = userspace only
cat /proc/sys/kernel/yama/ptrace_scope
id   # docker group? sudo group?
command -v perf bpftool nvidia-smi dcgmi
docker ps >/dev/null && echo docker_ok
```

| Rung | When it works | What you get |
| --- | --- | --- |
| Own pid + `perf_event_paranoid=2` | You own the trainer | `cpu-clock:u` samples, `/proc`, NVML |
| `enter_rootfs.sh --privileged` | `sudo -n` works | BCC, bpftrace, `perf` tracepoints, `bpftool` |
| `docker` group + `--privileged --pid=host` | `docker ps` works, no sudo | Same, through the older helper |
| Unprivileged BPF | `=1` here | Nothing. Do not wait for it. |

`--privileged` is preferred over the docker rung: fewer moving parts, and
the tool runs in the rootfs with the checkout mounted. It needs root only
because `bpf()` resolves capabilities in the initial user namespace.

Hardware `cycles` may stay `<not supported>` even as root. Fall back to
`cpu-clock`.

## Host PID under bwrap

```bash
# Host python pid, not the inner pid
ps -o pid,ppid,cmd -p $(pgrep -n -f 'python -m .*train')
awk '/^NSpid|^NStgid|^Pid|^PPid/' /proc/$PID/status
# NSpid: <host> <inner>   -> attach <host>
```

## Step 2: USE + GPU (no eBPF)

```bash
uptime
dmesg | tail
vmstat 1 5
mpstat -P ALL 1 3
pidstat -p $PID 1 5
iostat -xz 1 3
free -m

nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
nvidia-smi dmon -s pucvmet -c 5
dcgmi dmon -e 150,155,203,204,252 -c 5
# 150 temp, 155 power, 203 GPU util, 204 mem util, 252 FB used
# Field 230 is lifetime Xid count, not "this job caused an Xid"
```

Do not enable DCGM profiling fields (`1002` SM active, `1003` occupancy)
during a comparative run. They conflict with Nsight. Pause first:
`dcgmi profile --pause`.

## Step 3: no-sudo own-process sample

```bash
# /proc snapshot
cat /proc/$PID/status /proc/$PID/io /proc/$PID/wchan /proc/$PID/syscall
ls -l /proc/$PID/fd

# 3s thread delta (who actually burns a core)
# see scripts/attach_live.sh

# Software on-CPU profile. cycles:u often unsupported.
perf record --no-buildid --no-buildid-cache -F 99 -e cpu-clock:u \
  --call-graph fp -p $PID -o $OUT/perf.data -- sleep 12
perf report -i $OUT/perf.data --stdio --no-children -n --percent-limit 1
```

`read_bytes` vs `rchar` in `/proc/$PID/io`: disk vs page cache.

## Step 5a: tracepoints without sudo (docker group)

```bash
docker run --rm --privileged --pid=host --net=host -v /:/host \
  alpine:3.20 chroot /host /usr/bin/perf stat --no-big-num \
  -e sched:sched_switch,sched:sched_wakeup \
  -e syscalls:sys_enter_read,syscalls:sys_enter_ioctl,syscalls:sys_enter_futex \
  -p $PID -- sleep 8

docker run --rm --privileged --pid=host --net=host \
  -v /:/host -v "$OUT:$OUT" \
  alpine:3.20 chroot /host /usr/bin/perf record --no-buildid \
  -e sched:sched_switch --call-graph fp -p $PID -o $OUT/sched.data -- sleep 10

docker run --rm --privileged --pid=host --net=host \
  -v /:/host -v "$OUT:$OUT" \
  alpine:3.20 chroot /host /usr/bin/perf report -i $OUT/sched.data \
  --stdio --no-children -n --percent-limit 2
```

`sys_enter_ioctl=0` plus `cuStreamSynchronize` on-CPU is the spin-wait
signature. Helper-thread `futex`/`poll` switches are NCCL/CUDA daemons
going idle.

## Step 5b: BCC on the host (needs CAP_BPF)

Install on the **host** (`bpfcc-tools`), not in the rootfs.

```bash
# On-CPU stacks (the CUDA-spin tool)
sudo /usr/share/bcc/tools/profile -p $PID -fd -F 49 15

# Off-CPU stacks (disk, futex, page fault). Misses CUDA spin.
sudo /usr/share/bcc/tools/offcputime -p $PID -f 15

sudo /usr/share/bcc/tools/runqlat -p $PID 1 10
sudo /usr/share/bcc/tools/cachestat 1 10
sudo /usr/share/bcc/tools/biolatency -D 1 10
# opensnoop only for a few seconds
sudo /usr/share/bcc/tools/opensnoop -p $PID
```

Skip always-on `fileslower` / `biosnoop`. `tcpretrans` only when
NCCL uses the host network; collectives themselves are Flight Recorder /
NCCL RAS.

## Stop-job / launch-wrap / inject (not the default path)

Nsight Systems, Nsight Compute, DCGM EUD, `nccl-tests`, SuperBench,
`strace`, `gdb`, `py-spy dump`, memray attach, Scalene wrap, Fil wrap.
After the job is stopped, or a **throwaway** pid, or the user accepts
the pause. See [tools.md](tools.md).
