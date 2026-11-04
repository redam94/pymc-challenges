#!/usr/bin/env bash
# E92 FULL: GPU stages in order, a fresh guarded process per stage (or per LIMIT chunks). Detached:
#   setsid nohup tools/e92_full_driver.sh > /home/data/e92_scratch/full.log 2>&1 &
# Stops on a STOP file (the watcher's), on a machine change (exit 99 from the wrapper), or when a
# stage fails twice in a row. GPU processes use cores 0-1,6-9,11-15 (the PyMC audit has 2-5).
set -uo pipefail
cd "$(dirname "$0")/.."
stop=/home/data/e92_scratch/STOP
declare -A LIMIT=([hessian]=20 [sobol]=64 [design]=8 [design_alt]=8 [fit_B]=30 [failure_runs]=4)
last=""; fails=0
while true; do
  [[ -f $stop ]] && { echo "STOP file present"; exit 1; }
  stage="$(E92_CORES=0-1 tools/e92_job.sh 4 2 -- .venv/bin/python tools/run_e92_full.py next 2>/dev/null | tail -1)"
  [[ -z "$stage" ]] && { echo "could not read the next stage"; exit 1; }
  [[ "$stage" == none ]] && break
  echo "=== $(date -u +%FT%TZ) stage $stage"
  XLA_PYTHON_CLIENT_PREALLOCATE=true E92_CORES=0-1,6-9,11-15 \
    tools/e92_job.sh 16 8 -- .venv/bin/python tools/run_e92_full.py "$stage" ${LIMIT[$stage]:-}
  rc=$?
  if (( rc == 99 || rc == 98 )); then echo "machine check refused the job (rc $rc)"; exit 1; fi
  if (( rc != 0 )); then
    if [[ "$stage" == "$last" ]]; then fails=$((fails + 1)); else fails=1; fi
    echo "stage $stage exited $rc (failure $fails)"
    (( fails >= 2 )) && { echo "stage $stage failed twice: stopping"; exit 1; }
  else fails=0; fi
  last="$stage"
done
echo "FULL driver finished $(date -u +%FT%TZ)"
