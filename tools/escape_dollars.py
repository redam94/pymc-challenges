"""Escape currency dollar signs in the markdown cells of the jupytext sources.

Jupyter and the site render $...$ as maths, so "$481M against $434M" becomes italic run-together
text. This script finds dollar signs that start an amount of money and writes them as \\$. It
pairs unescaped dollars the way MathJax does (after removing code spans and $$...$$ blocks) and
treats an opening dollar as currency when the "maths" it opens starts with an amount followed by
words, or reads as prose.

    python tools/escape_dollars.py            # list what would change
    python tools/escape_dollars.py --write    # change the sources
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AMOUNT = re.compile(
    r"\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|M|B|bn|m|tn|million|billion|trillion|thousand))?"
    r"(?=\*{0,2}[,.;:!?)]?(?:\s+(?:(?:a|[A-Za-z]{2,})\b(?![_^{])|[(±])|[-–]\d|/(?:yr|year|week|month|day|hour)\b))"
)
SUFFIXED = re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:k|M|B|bn|tn)(?=[\s,.;:)*])")
PROSE = re.compile(r"(?<![\\a-zA-Z])[A-Za-z]{3,}\s+[A-Za-z]{3,}(?![a-zA-Z])")


def _strip_tex(s: str) -> str:
    s = re.sub(r"\\(?:text|mathrm|textrm|operatorname|mathbf|mbox)\s*\{[^{}]*\}", " ", s)
    return re.sub(r"\\[A-Za-z]+", " ", s)


def _mask(text: str) -> str:
    """Blank out code spans and display maths so their dollars are ignored."""
    out = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group()), text)
    return re.sub(r"\$\$.*?\$\$", lambda m: " " * len(m.group()), out, flags=re.S)


def currency_positions(text: str) -> list[int]:
    masked = _mask(text).replace("\n# ", "\n  ").replace("\n#", "\n ")  # comment prefixes are not text
    dollars = [m.start() for m in re.finditer(r"(?<!\\)\$", masked)]
    found, i = [], 0
    while i < len(dollars):
        a = dollars[i]
        b = dollars[i + 1] if i + 1 < len(dollars) else None
        after = masked[a + 1 :]
        if b is None:
            if re.match(r"\d", after):
                found.append(a)
            i += 1
            continue
        content = masked[a + 1 : b]
        money = bool(re.match(r"\d", content)) and (
            AMOUNT.match(content) is not None
            or SUFFIXED.match(content) is not None
            or PROSE.search(_strip_tex(content)) is not None
        )
        if money:
            found.append(a)
            i += 1  # b may open real maths or be another amount: pair again from b
        else:
            i += 2
    return found


def markdown_blocks(lines: list[str]):
    """Yield (start, end) line ranges of markdown cells in a percent-format source."""
    start = None
    for k, line in enumerate(lines + ["# %%"]):
        if line.startswith("# %%"):
            if start is not None:
                yield start, k
            start = k + 1 if line.startswith("# %% [markdown]") else None


def process(path: Path, write: bool) -> int:
    lines = path.read_text().split("\n")
    changes = 0
    for s, e in markdown_blocks(lines):
        block = "\n".join(lines[s:e])
        pos = currency_positions(block)
        if not pos:
            continue
        for p in pos:
            ls = block.rfind("\n", 0, p) + 1
            le = block.find("\n", p)
            line = block[ls : le if le >= 0 else None]
            col = p - ls
            print(f"{path.relative_to(ROOT)}:{s + block.count(chr(10), 0, p) + 1}: {line[max(0, col - 40):col + 50]}")
        for p in reversed(pos):
            block = block[:p] + "\\" + block[p:]
        lines[s:e] = block.split("\n")
        changes += len(pos)
    if write and changes:
        path.write_text("\n".join(lines))
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    paths = [Path(p).resolve() for p in args.paths] or sorted((ROOT / "notebook_src").glob("*/*.py"))
    total = sum(process(p, args.write) for p in paths)
    print(f"{total} currency dollar signs {'escaped' if args.write else 'found'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
