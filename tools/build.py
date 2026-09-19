"""Build the notebooks in ``notebooks/`` from the jupytext sources in ``notebook_src/``.

    uv run python tools/build.py                 # build everything (executes examples + solutions)
    uv run python tools/build.py C01 E02         # only sources whose filename starts with these ids
    uv run python tools/build.py --no-execute    # convert only, do not run anything

Each challenge has ONE source file, from which two notebooks are produced:

    cells tagged "solution"  -> only in notebooks/solutions/<name>_solution.ipynb (executed)
    cells tagged "task"      -> only in notebooks/challenges/<name>.ipynb (left unexecuted)
    untagged cells           -> in both

Examples have no tags and are simply converted and executed.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook_src"
OUT = ROOT / "notebooks"

# Executing a solution must not count as the user having used hints or passed checks.
os.environ["PYMC_CHALLENGES_NO_PROGRESS"] = "1"

KERNELSPEC = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}


def _tags(cell) -> set[str]:
    return set(cell.metadata.get("tags", []))


def _variant(nb, drop_tag: str):
    out = copy.deepcopy(nb)
    out.cells = [c for c in out.cells if drop_tag not in _tags(c)]
    out.metadata = {"kernelspec": KERNELSPEC}
    return out


def _strip_progress_bars(nb) -> None:
    """PyMC's live progress bars leave a bare ``Output()`` repr behind once executed headlessly."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = [
                o for o in cell.outputs
                if not (o.output_type == "display_data" and o.get("data", {}).get("text/plain") == "Output()")
            ]


def _execute(nb, cwd: Path, label: str) -> None:
    start = time.time()
    cwd.mkdir(parents=True, exist_ok=True)
    print(f"    executing {label} ...", flush=True)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute()
    _strip_progress_bars(nb)
    print(f"    done in {time.time() - start:.0f}s", flush=True)


def _write(nb, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    print(f"    wrote {path.relative_to(ROOT)}")


def build(src: Path, execute: bool) -> None:
    print(f"[{src.relative_to(ROOT)}]")
    nb = jupytext.read(src)
    kind = src.parent.name
    if kind == "examples":
        out = _variant(nb, drop_tag="task")
        if execute:
            _execute(out, OUT / "examples", src.stem)
        _write(out, OUT / "examples" / f"{src.stem}.ipynb")
    elif kind == "challenges":
        _write(_variant(nb, drop_tag="solution"), OUT / "challenges" / f"{src.stem}.ipynb")
        solution = _variant(nb, drop_tag="task")
        if execute:
            _execute(solution, OUT / "solutions", f"{src.stem}_solution")
        _write(solution, OUT / "solutions" / f"{src.stem}_solution.ipynb")
    else:
        raise ValueError(f"Unexpected source location: {src}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="filename prefixes to build, e.g. C01 E02 (default: all)")
    parser.add_argument("--no-execute", action="store_true", help="convert only")
    args = parser.parse_args()

    sources = sorted(SRC.glob("*/*.py"))
    if args.ids:
        sources = [s for s in sources if any(s.name.startswith(i) for i in args.ids)]
    if not sources:
        print("No matching sources.")
        return 1
    failed = []
    for src in sources:
        try:
            build(src, execute=not args.no_execute)
        except Exception as exc:  # keep going so one broken notebook does not block the rest
            failed.append(src.name)
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[-1500:]}")
    if failed:
        print(f"\nFailed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
