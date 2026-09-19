"""Run a command under a hard memory cap (whole process tree), and report its peak.

    uv run python tools/memguard.py --max-gb 3 -- python tools/build.py E13
    uv run python tools/memguard.py --max-gb 3 -- python my_prototype.py

macOS does not enforce ``ulimit -v``, so this polls the resident memory of the command and
all its descendants (sampler worker processes, Jupyter kernels) and kills the whole tree if
the total crosses the cap. Exit code 137 means "killed for memory"; otherwise the command's
own exit code is passed through.

Why it exists: PyMC's multiprocess samplers (PyMC NUTS, PGBART, SMC) copy the model into
one worker per chain, so four chains can use four times the memory of one. On a machine with
8 GB to spare, a few of those at once is enough to run out.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import psutil


def tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def kill_tree(proc: psutil.Process) -> None:
    # Descendants first (a Jupyter kernel starts its own session, so a group kill alone would
    # miss it), then the command's whole process group to catch anything spawned meanwhile.
    try:
        procs = [*proc.children(recursive=True), proc]
    except psutil.NoSuchProcess:
        procs = []
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    psutil.wait_procs(procs, timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-gb", type=float, default=3.0, help="cap on total resident memory (default 3)")
    parser.add_argument("--interval", type=float, default=0.25, help="seconds between checks")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command to run")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    cap = args.max_gb * 1e9
    child = subprocess.Popen(command, start_new_session=True)  # own process group
    proc = psutil.Process(child.pid)
    peak = 0
    while child.poll() is None:
        try:
            rss = tree_rss(proc)
        except psutil.NoSuchProcess:
            break
        peak = max(peak, rss)
        if rss > cap:
            kill_tree(proc)
            print(
                f"\n[memguard] KILLED: process tree reached {rss / 1e9:.2f} GB, cap is {args.max_gb:g} GB. "
                "Reduce cores/chains, subsample, or store fewer variables.",
                file=sys.stderr,
                flush=True,
            )
            return 137
        time.sleep(args.interval)
    print(f"[memguard] peak memory {peak / 1e9:.2f} GB (cap {args.max_gb:g} GB)", file=sys.stderr, flush=True)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
