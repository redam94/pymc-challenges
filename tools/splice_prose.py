"""Put edited prose into an executed example notebook without running it again.

Reads the jupytext source and the executed notebook in ``notebooks/examples/``. Every code
cell must be unchanged, in the same order; markdown cells are taken from the source and code
cells (with their outputs) from the executed notebook. Refuses if any code cell differs, so
only prose can change this way.

    python tools/splice_prose.py E88            # check and write
    python tools/splice_prose.py E88 --check    # check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def splice(prefix: str, write: bool) -> int:
    (src,) = sorted((ROOT / "notebook_src" / "examples").glob(f"{prefix}_*.py"))
    out = ROOT / "notebooks" / "examples" / f"{src.stem}.ipynb"
    new = jupytext.read(src)
    old = nbformat.read(out, as_version=4)
    new_code = [c for c in new.cells if c.cell_type == "code"]
    old_code = [c for c in old.cells if c.cell_type == "code"]
    if len(new_code) != len(old_code):
        print(f"{prefix}: {len(new_code)} code cells in the source, {len(old_code)} executed")
        return 1
    bad = [i for i, (a, b) in enumerate(zip(new_code, old_code)) if a.source.strip() != b.source.strip()]
    if bad:
        print(f"{prefix}: code cells differ at code index {bad}: rebuild with tools/build.py")
        return 1
    executed = iter(old_code)
    cells = []
    for cell in new.cells:
        if cell.cell_type == "code":
            cells.append(next(executed))
        else:
            cell = nbformat.v4.new_markdown_cell(cell.source)
            cells.append(cell)
    md_old = sum(c.cell_type == "markdown" for c in old.cells)
    md_new = sum(c.cell_type == "markdown" for c in cells)
    old.cells = cells
    nbformat.validate(old)
    print(f"{prefix}: {len(new_code)} code cells kept with outputs, markdown cells {md_old} -> {md_new}")
    if write:
        nbformat.write(old, out)
        print(f"    wrote {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return max(splice(i, write=not args.check) for i in args.ids)


if __name__ == "__main__":
    sys.exit(main())
