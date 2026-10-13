#!/usr/bin/env bash
# =============================================================================
# STEP 4: hardware counters for every kernel, collected with Nsight Compute (ncu).
#
#   scripts/profile.sh [N]          default N = 4096 (the Makefile passes PROFILE_N)
#
# Writes to results/ncu/:
#   sgemm_N<N>.ncu-rep   full report; open it in the Nsight Compute GUI (ncu-ui) to explore
#   raw_N<N>.csv         every collected metric, one row per kernel launch (ncu --page raw)
#   details_N<N>.txt     human-readable section output (ncu --page details)
#   summary.csv/.md      the handful of numbers this project discusses (scripts/ncu_summary.py)
#
# Reading GPU performance counters needs permission. Without it ncu prints ERR_NVGPUCTRPERM and
# this script exits with status 3 (the README explains the one-time fix).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-4096}"
NCU="${NCU:-}"
PYTHON="${PYTHON:-python3}"
OUT=results/ncu

[[ -n "$NCU" ]] || NCU="$(command -v ncu || true)"
if [[ -z "$NCU" || ! -x "$NCU" ]]; then
  echo "profile.sh: Nsight Compute (ncu) not found; skipping profiling." >&2
  exit 2
fi
[[ -x bin/sgemm ]] || { echo "profile.sh: bin/sgemm is missing, run 'make' first." >&2; exit 1; }
mkdir -p "$OUT"

# 1) Permission check: profile one tiny launch and look for the permission error.
set +e
"$NCU" --metrics gpu__time_duration.sum bin/sgemm --profile --sizes 256 --kernels 1_naive \
  > "$OUT/permission_check.log" 2>&1
rc=$?
set -e
if grep -q "ERR_NVGPUCTRPERM" "$OUT/permission_check.log"; then
  cat >&2 <<'EOF'
profile.sh: no permission to read GPU performance counters (ERR_NVGPUCTRPERM).
One-time fix, then reboot:
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u -k all
  sudo reboot
EOF
  exit 3
fi
if [[ $rc -ne 0 ]]; then
  echo "profile.sh: ncu failed (exit $rc); see $OUT/permission_check.log" >&2
  exit 1
fi
rm -f "$OUT/permission_check.log"

# 2) The metrics this project uses, on top of the standard sections.
METRICS=(
  gpu__time_duration.sum                                  # kernel duration
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum     # FP32 work actually executed:
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum     #   FLOPs = 2*FFMA + FMUL + FADD
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
  dram__bytes_read.sum                                    # DRAM traffic -> arithmetic intensity
  dram__bytes_write.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed      # achieved DRAM bandwidth, % of peak
  lts__t_bytes.sum                                        # L2 traffic
  sm__throughput.avg.pct_of_peak_sustained_elapsed        # compute "speed of light", % of peak
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active  # FMA pipe busy %
  smsp__issue_active.avg.pct_of_peak_sustained_active     # % of cycles a warp instruction issued
  sm__warps_active.avg.pct_of_peak_sustained_active       # ACHIEVED occupancy
  sm__maximum_warps_per_active_cycle_pct                  # THEORETICAL occupancy
  launch__registers_per_thread
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum         # global loads: sectors per request
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum          #   (4 = perfectly coalesced fp32 warp)
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  # shared-memory bank conflicts
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
)
metric_list=$(printf '%s\n' "${METRICS[@]}" | sed 's/[[:space:]]*#.*//' | grep -v '^$' | paste -sd, -)

SECTIONS=(SpeedOfLight SpeedOfLight_RooflineChart LaunchStats Occupancy MemoryWorkloadAnalysis
          ComputeWorkloadAnalysis WarpStateStats)
section_args=()
for s in "${SECTIONS[@]}"; do section_args+=(--section "$s"); done

echo "profile.sh: profiling every kernel at N=$N with $(basename "$NCU") (this replays each kernel many times)..."
# --clock-control none: let the GPU boost exactly as it does in the benchmark (ncu's default
# would lock the clocks to the base clock).
"$NCU" --clock-control none "${section_args[@]}" --metrics "$metric_list" \
  --export "$OUT/sgemm_N$N" --force-overwrite \
  bin/sgemm --profile --sizes "$N" > "$OUT/ncu_log_N$N.txt" 2>&1

"$NCU" --import "$OUT/sgemm_N$N.ncu-rep" --csv --page raw --print-units base > "$OUT/raw_N$N.csv"
"$NCU" --import "$OUT/sgemm_N$N.ncu-rep" --page details > "$OUT/details_N$N.txt"
"$PYTHON" scripts/ncu_summary.py "$OUT/raw_N$N.csv" "$N"
