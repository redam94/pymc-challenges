"""Put edited prose into an executed example notebook without running it again.

Reads the jupytext source and the built notebooks in ``notebooks/`` (example, or challenge and solution). Every code
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import _variant  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _merge(new, executed_path: Path, label: str, write: bool) -> int:
    old = nbformat.read(executed_path, as_version=4)
    new_code = [c for c in new.cells if c.cell_type == "code"]
    old_code = [c for c in old.cells if c.cell_type == "code"]
    if len(new_code) != len(old_code):
        print(f"{label}: {len(new_code)} code cells in the source, {len(old_code)} executed")
        return 1
    bad = [i for i, (a, b) in enumerate(zip(new_code, old_code)) if a.source.strip() != b.source.strip()]
    if bad:
        print(f"{label}: code cells differ at code index {bad}: rebuild with tools/build.py")
        return 1
    executed = iter(old_code)
    cells = [next(executed) if c.cell_type == "code" else c for c in new.cells]
    md_old = sum(c.cell_type == "markdown" for c in old.cells)
    md_new = sum(c.cell_type == "markdown" for c in cells)
    old.cells = cells
    nbformat.validate(old)
    print(f"{label}: {len(new_code)} code cells kept with outputs, markdown cells {md_old} -> {md_new}")
    if write:
        nbformat.write(old, executed_path)
        print(f"    wrote {executed_path.relative_to(ROOT)}")
    return 0


def splice(prefix: str, write: bool) -> int:
    (src,) = sorted((ROOT / "notebook_src").glob(f"*/{prefix}_*.py"))
    nb = jupytext.read(src)
    out = ROOT / "notebooks"
    if src.parent.name == "examples":
        return _merge(_variant(nb, drop_tag="task"), out / "examples" / f"{src.stem}.ipynb", prefix, write)
    # challenges: the challenge notebook is never executed, so it is simply rewritten
    rc = _merge(_variant(nb, drop_tag="solution"), out / "challenges" / f"{src.stem}.ipynb", f"{prefix} challenge", write)
    return max(rc, _merge(_variant(nb, drop_tag="task"), out / "solutions" / f"{src.stem}_solution.ipynb",
                          f"{prefix} solution", write))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return max(splice(i, write=not args.check) for i in args.ids)


if __name__ == "__main__":
    sys.exit(main())
