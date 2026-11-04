#!/usr/bin/env bash
# E92 FULL: the PyMC audit beside the GPU stages, on cores 2-5 only. Detached:
#   setsid nohup tools/e92_audit_driver.sh > /home/data/e92_scratch/audit.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
stop=/home/data/e92_scratch/STOP
job() { [[ -f $stop ]] && { echo "STOP file present"; exit 1; }; tools/e92_job.sh "$@"; }
done_stage() { .venv/bin/python -c "from pymc_challenges.bigrun import Store; import sys; sys.exit(0 if Store('E92').done('$1') else 1)"; }
for n in 800 2000; do
  done_stage "pymc_audit_$n" && continue
  if ! done_stage "pymc_audit_laplace_$n"; then
    E92_CORES=2-5 job 6 4 -- .venv/bin/python tools/run_e92_audit.py laplace "$n" || exit 1
  fi
  pids=()
  for k in 0 1 2 3; do
    if [[ -f "results/E92/chunks/pymc_audit_$n/$(printf %08d $k).npz.sha256" ]]; then continue; fi
    E92_BLAS_THREADS=1 E92_CORES=$((2 + k)) job 5 1 -- .venv/bin/python tools/run_e92_audit.py chain "$n" "$k" \
      > "/home/data/e92_scratch/audit_${n}_chain$k.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  (( fail )) && { echo "a chain failed at n=$n"; exit 1; }
  E92_CORES=2-5 job 4 4 -- .venv/bin/python tools/run_e92_audit.py summary "$n" || exit 1
done
echo "audit driver finished $(date -u +%FT%TZ)"
