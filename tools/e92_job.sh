#!/usr/bin/env bash
# Run one E92 job: check the machine, pin to the allowed cores, cap memory.
#   tools/e92_job.sh MAX_GB THREADS -- command ...
# THREADS sets the BLAS/OpenMP thread counts. Cores 10 and 26 are never used.
set -euo pipefail
cap="$1"; threads="$2"; shift 3
offline="$(cat /sys/devices/system/cpu/offline)"
mce="$(journalctl -k -b 0 | grep -cE 'MC[0-9]+_STATUS|Machine Check: .*Bank [0-9]+' || true)"
if [[ "$offline" != "10,26" || "$mce" != "1" ]]; then
  echo "STOP: offline=$offline machine_checks=$mce" >&2
  exit 99
fi
cd "$(dirname "$0")/.."
blas="${E92_BLAS_THREADS:-$threads}"  # nutpie runs chains as threads: give it 1 BLAS thread each
export OPENBLAS_NUM_THREADS="$blas" OMP_NUM_THREADS="$blas" MKL_NUM_THREADS="$blas"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTENSOR_FLAGS="base_compiledir=/home/data/e92_scratch/pytensor"
mkdir -p /home/data/e92_scratch/pytensor
if (( threads == 1 )); then cores="0"; elif (( threads == 2 )); then cores="0-1";
elif (( threads <= 10 )); then cores="0-$((threads - 1))"; else cores="0-9,11-$((threads))"; fi
cores="${E92_CORES:-$cores}"  # an explicit list, e.g. 2-5 for the PyMC audit beside a GPU job
if [[ ",$cores," =~ [,-](10|26)[,-] ]]; then echo "STOP: cores $cores include 10 or 26" >&2; exit 98; fi
echo "[e92_job] offline=$offline mce=$mce cores=$cores cap=${cap}GB $(date -u +%FT%TZ)" >&2
exec taskset -c "$cores" .venv/bin/python tools/memguard.py --max-gb "$cap" -- "$@"
