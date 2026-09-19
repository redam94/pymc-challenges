"""Execute a notebook source cell by cell and report the kernel's memory after each cell.

    uv run python tools/memguard.py --max-gb 3.5 -- python tools/memprofile_nb.py notebook_src/examples/E03_glm_model_comparison.py

Prints one line per code cell (memory after the cell, the change, seconds, first line of the
cell) and then the cells that grew memory the most. Memory is the resident size of the whole
kernel process tree, so sampler worker processes are included. Nothing is written to
``notebooks/``. For challenge sources the solution variant is profiled.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jupytext
import psutil
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def tree_rss(pid: int) -> float:
    try:
        proc = psutil.Process(pid)
        procs = [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1e9


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    nb = jupytext.read(src)
    nb.cells = [c for c in nb.cells if "task" not in c.metadata.get("tags", [])]
    out_dir = ROOT / "notebooks" / ("solutions" if src.parent.name == "challenges" else "examples")
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(out_dir)}})

    rows = []
    with client.setup_kernel():
        kernel_pid = client.km.provisioner.process.pid
        previous = tree_rss(kernel_pid)
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            start = time.time()

            client.execute_cell(cell, index)
            now = tree_rss(kernel_pid)
            first = next((line for line in cell.source.splitlines() if line.strip()), "")[:70]
            rows.append((index, now, now - previous, time.time() - start, first))
            print(f"cell {index:3d}  {now:5.2f} GB  {now - previous:+6.2f}  {time.time() - start:6.1f}s  {first}", flush=True)
            previous = now

    print("\nLargest increases:")
    for index, now, delta, seconds, first in sorted(rows, key=lambda r: -r[2])[:8]:
        print(f"  cell {index:3d}  {delta:+6.2f} GB -> {now:5.2f} GB   {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
