#!/usr/bin/env bash
# Non-intrusive attach for a live trainer. Never signals the target.
# Re-enters the bwrap rootfs so NVIDIA, Python, perf, and BPF probes all run
# from inside the sandbox.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

PID=""
SECONDS_N=12
OUT=""
TRACEPOINTS=0
PRIVILEGED=0

usage() {
  cat <<'EOF'
Usage: attach_live.sh --pid HOST_PID [--out DIR] [--seconds N]
                     [--tracepoints] [--privileged]

Sample only. Does not stop, ptrace-pause, or strace the job.

Re-enters the rootfs with --profile by default. --privileged re-enters with
enter_rootfs.sh --privileged instead, which needs passwordless sudo but runs
bpftrace/BCC/perf tracepoints natively in the sandbox and adds BPF probes.
--tracepoints collects tracepoint counters, natively under --privileged or
through the docker helper otherwise.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid) PID="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --seconds) SECONDS_N="${2:?}"; shift 2 ;;
    --tracepoints) TRACEPOINTS=1; shift ;;
    --privileged) PRIVILEGED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PID" || ! -d "/proc/$PID" ]]; then
  echo "need a live --pid" >&2
  exit 2
fi

if [[ -z "$OUT" ]]; then
  OUT="${PWD}/experiments/falcon/results/live_attach_$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUT"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  out_in_rootfs="$OUT"
  case "$OUT" in
    "${REPO_ROOT}"/*) out_in_rootfs="/workspace/torchtitan/${OUT#"${REPO_ROOT}"/}" ;;
  esac
  extra=()
  if [[ "$TRACEPOINTS" -eq 1 ]]; then
    extra+=(--tracepoints)
  fi
  enter_mode=--profile
  if [[ "$PRIVILEGED" -eq 1 ]]; then
    enter_mode=--privileged
    extra+=(--privileged)
  fi
  # Not exec: a privileged sandbox writes into OUT as root, so ownership has
  # to be handed back here, where we are still the invoking user.
  status=0
  "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" "$enter_mode" -- \
    /bin/bash -lc \
    'cd /workspace/torchtitan && exec .claude/skills/attaching-live-training/scripts/attach_live.sh "$@"' \
    attach_live.sh --pid "$PID" --out "$out_in_rootfs" --seconds "$SECONDS_N" \
    "${extra[@]}" || status=$?
  if [[ "$PRIVILEGED" -eq 1 ]]; then
    sudo -n chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true
  fi
  exit "$status"
fi
echo "$PID" > "$OUT/pid.txt"
tr '\0' ' ' < "/proc/$PID/cmdline" >> "$OUT/pid.txt"
echo >> "$OUT/pid.txt"

alive() { [[ -d "/proc/$PID" ]]; }

{
  date -u
  echo "=== status ==="
  awk '/^Name|^State|^Pid|^PPid|^NSpid|^Threads|^VmRSS|^voluntary|^nonvoluntary/' "/proc/$PID/status"
  echo "=== io ==="
  cat "/proc/$PID/io"
  echo "=== wchan ==="
  cat "/proc/$PID/wchan"; echo
  echo "=== syscall ==="
  cat "/proc/$PID/syscall"; echo
  echo "=== cpuset ==="
  cat "/proc/$PID/cpuset"
} > "$OUT/proc.txt"

python3 - "$PID" "$OUT/threads.tsv" <<'PY'
import glob, os, sys
pid, out = sys.argv[1], sys.argv[2]
rows = ["tid\tstate\tcomm\twchan\tutime\tstime\tprocessor"]
for tdir in sorted(glob.glob(f"/proc/{pid}/task/*")):
    tid = os.path.basename(tdir)
    try:
        comm = open(f"{tdir}/comm").read().strip()
        stat = open(f"{tdir}/stat").read()
        rest = stat[stat.rfind(")") + 2 :].split()
        wchan = open(f"{tdir}/wchan").read().strip()
        cpu = rest[36] if len(rest) > 36 else "?"
        rows.append(f"{tid}\t{rest[0]}\t{comm}\t{wchan}\t{rest[11]}\t{rest[12]}\t{cpu}")
    except OSError:
        continue
open(out, "w").write("\n".join(rows) + "\n")
PY

{
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv
  echo
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
  echo
  nvidia-smi dmon -s pucvmet -c 3
} > "$OUT/gpu.txt" 2>&1 || true

if command -v dcgmi >/dev/null; then
  timeout 8 dcgmi dmon -e 150,155,203,204,252 -c 3 > "$OUT/dcgm.txt" 2>&1 || true
fi

if command -v perf >/dev/null && alive; then
  perf record --no-buildid --no-buildid-cache -F 99 -e cpu-clock:u \
    --call-graph fp -p "$PID" -o "$OUT/perf.data" -- sleep "$SECONDS_N" \
    > "$OUT/perf_record.txt" 2>&1 || true
  if [[ -s "$OUT/perf.data" ]]; then
    perf report -i "$OUT/perf.data" --stdio --no-children -n --percent-limit 1 \
      > "$OUT/perf_report.txt" 2>&1 || true
  fi
fi

if command -v py-spy >/dev/null && alive; then
  py-spy record --pid "$PID" --rate 50 --duration 5 --format raw \
    -o "$OUT/pyspy_python.raw" > "$OUT/pyspy_python.txt" 2>&1 || true
fi

if [[ "$PRIVILEGED" -eq 1 ]] && alive; then
  # BPF loads natively here, so take the off-CPU and run-queue evidence that
  # sampling profilers cannot give: where the job blocks, not where it spins.
  if command -v bpftrace >/dev/null; then
    bpftrace -e "profile:hz:99 /pid == $PID/ { @[ustack] = count(); }
                 interval:s:${SECONDS_N} { exit(); }" \
      > "$OUT/bpftrace_ustack.txt" 2>&1 || true
    bpftrace -e "tracepoint:sched:sched_switch /args->prev_pid == $PID/ { @ = count(); }
                 interval:s:${SECONDS_N} { exit(); }" \
      > "$OUT/bpftrace_switches.txt" 2>&1 || true
  fi
  if command -v offcputime-bpfcc >/dev/null; then
    offcputime-bpfcc -p "$PID" "$SECONDS_N" \
      > "$OUT/offcputime.txt" 2>&1 || true
  fi
  if command -v runqlat-bpfcc >/dev/null; then
    runqlat-bpfcc "$SECONDS_N" 1 > "$OUT/runqlat.txt" 2>&1 || true
  fi
  if command -v cachestat-bpfcc >/dev/null; then
    cachestat-bpfcc 1 "$SECONDS_N" > "$OUT/cachestat.txt" 2>&1 || true
  fi
fi

if [[ "$TRACEPOINTS" -eq 1 ]] && [[ -x "${SCRIPT_DIR}/run_privileged.sh" ]]; then
  "${SCRIPT_DIR}/run_privileged.sh" --host /usr/bin/perf stat --no-big-num \
    -e sched:sched_switch,sched:sched_wakeup \
    -e syscalls:sys_enter_read,syscalls:sys_enter_ioctl,syscalls:sys_enter_futex \
    -p "$PID" -- sleep 8 > "$OUT/tracepoints.txt" 2>&1 || true
fi

if alive; then
  echo "still_alive $(awk '/^State/{print $2}' "/proc/$PID/status")" | tee "$OUT/alive.txt"
else
  echo "target_exited_during_attach" | tee "$OUT/alive.txt"
  exit 1
fi

echo "wrote $OUT"
