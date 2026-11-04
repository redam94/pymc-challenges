#!/usr/bin/env bash
# E92: the declared operable-water rerun of the design (run_e92_full.py design_masked), 8 chunks a
# fresh guarded process, until the stage is marked done. Detached:
#   setsid nohup tools/e92_masked_driver.sh > /home/data/e92_scratch/masked.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
stop=/home/data/e92_scratch/STOP
for attempt in $(seq 1 12); do
  [[ -f $stop ]] && { echo "STOP file present"; exit 1; }
  if grep -q '"stage": "design_masked"' results/E92/stages.json; then break; fi
  echo "=== $(date -u +%FT%TZ) design_masked, process $attempt"
  XLA_PYTHON_CLIENT_PREALLOCATE=true E92_CORES=0-1,6-9,11-15 \
    tools/e92_job.sh 16 8 -- .venv/bin/python tools/run_e92_full.py design_masked 8
  rc=$?
  if (( rc == 99 || rc == 98 )); then echo "machine check refused the job (rc $rc)"; exit 1; fi
  (( rc != 0 )) && echo "exited $rc"
done
echo "masked driver finished $(date -u +%FT%TZ)"
