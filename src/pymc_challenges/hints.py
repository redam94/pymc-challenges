"""Opt-in, tiered hints and self-checks for the challenge notebooks.

    from pymc_challenges import Hints
    h = Hints("C01")

    h.tasks()                  # what is there to do?
    h.hint("task2")            # reveal the *next* hint for task2 (level 1, then 2, then 3)
    h.hint("task2", level=3)   # jump straight to a level
    h.check("task2", slope=-0.25)   # compare your number with the reference solution
    h.progress()               # hints used / checks passed

Hint levels are always the same:

    1. nudge     -- a question or concept pointing you in the right direction
    2. approach  -- the modelling idea spelled out, plus the relevant PyMC API
    3. skeleton  -- code structure with the key blanks left for you

Nothing is shown until you ask. Hints used are remembered in ``.progress/`` so you
can be honest with yourself about how much help a challenge needed.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

import yaml

LEVEL_NAMES = {1: "nudge", 2: "approach", 3: "skeleton"}
PROGRESS_DIR = Path(__file__).resolve().parents[2] / ".progress"


def _show(markdown: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display

        if get_ipython() is not None:
            display(Markdown(markdown))
            return
    except ImportError:
        pass
    print(markdown)


class Hints:
    def __init__(self, challenge: str):
        self.challenge = challenge
        spec_file = resources.files("pymc_challenges") / "hints" / f"{challenge}.yaml"
        if not spec_file.is_file():
            raise FileNotFoundError(f"No hints file for challenge {challenge!r}")
        self._spec = yaml.safe_load(spec_file.read_text())
        self._tasks: dict = self._spec["tasks"]
        self._progress_file = PROGRESS_DIR / f"{challenge}.json"
        self._progress = {"hints": {}, "checks": {}}
        if self._progress_file.exists() and not os.environ.get("PYMC_CHALLENGES_NO_PROGRESS"):
            self._progress.update(json.loads(self._progress_file.read_text()))

    # ------------------------------------------------------------------ utils
    def _save(self) -> None:
        if os.environ.get("PYMC_CHALLENGES_NO_PROGRESS"):  # set by tools/build.py
            return
        PROGRESS_DIR.mkdir(exist_ok=True)
        self._progress_file.write_text(json.dumps(self._progress, indent=2))

    def _task(self, task: str) -> dict:
        if task not in self._tasks:
            raise KeyError(f"Unknown task {task!r}. Tasks: {list(self._tasks)}")
        return self._tasks[task]

    # -------------------------------------------------------------------- api
    def tasks(self) -> None:
        """List the tasks of this challenge and how much help is available."""
        lines = [f"**{self.challenge} - {self._spec.get('title', '')}**", ""]
        for name, t in self._tasks.items():
            n_checks = len(t.get("checks", {}))
            extra = f", {n_checks} check(s)" if n_checks else ""
            lines.append(f"- `{name}` - {t.get('title', '')} ({len(t.get('hints', []))} hints{extra})")
        _show("\n".join(lines))

    def hint(self, task: str, level: int | None = None) -> None:
        """Reveal a hint. Without ``level`` you get the next one you have not seen."""
        hints = self._task(task).get("hints", [])
        if not hints:
            _show(f"No hints for `{task}` - you are on your own here.")
            return
        seen = self._progress["hints"].get(task, 0)
        if level is None:
            level = min(seen + 1, len(hints))
        if not 1 <= level <= len(hints):
            raise ValueError(f"`{task}` has hint levels 1..{len(hints)}")
        self._progress["hints"][task] = max(seen, level)
        self._save()
        label = LEVEL_NAMES.get(level, f"level {level}")
        more = "" if level == len(hints) else f"\n\n*{len(hints) - level} more hint(s) available for this task.*"
        _show(f"**Hint {level}/{len(hints)} for `{task}` ({label})**\n\n{hints[level - 1]}{more}")

    def check(self, task: str, **values) -> bool:
        """Compare your numbers with the reference solution, e.g. ``h.check("task1", slope=-0.25)``.

        Ranges are generous (they allow for Monte Carlo error and reasonable prior
        choices) so a pass means "you are in the right place", not "identical model".
        """
        checks = self._task(task).get("checks", {})
        if not checks:
            _show(f"No numeric checks for `{task}`.")
            return True
        if not values:
            wanted = "\n".join(f"- `{k}`: {c['what']}" for k, c in checks.items())
            _show(f"Checks available for `{task}` - pass them as keyword arguments:\n\n{wanted}")
            return False
        lines, all_ok = [], True
        for key, value in values.items():
            if key not in checks:
                raise KeyError(f"No check named {key!r} for {task}. Available: {list(checks)}")
            lo, hi = checks[key]["range"]
            value = float(value)
            ok = lo <= value <= hi
            all_ok &= ok
            self._progress["checks"][f"{task}.{key}"] = ok
            if ok:
                lines.append(f"- PASS `{key}` = {value:.4g} is consistent with the reference solution.")
            else:
                direction = "low" if value < lo else "high"
                lines.append(
                    f"- MISS `{key}` = {value:.4g} looks too {direction}. Expected: {checks[key]['what']}."
                )
        self._save()
        _show("\n".join(lines))
        return all_ok

    def progress(self) -> None:
        """Summarise hints used and checks passed for this challenge."""
        total_hints = sum(len(t.get("hints", [])) for t in self._tasks.values())
        used = sum(self._progress["hints"].values())
        total_checks = sum(len(t.get("checks", {})) for t in self._tasks.values())
        passed = sum(bool(v) for v in self._progress["checks"].values())
        _show(
            f"**{self.challenge}**: {used}/{total_hints} hints used, "
            f"{passed}/{total_checks} checks passed."
        )

    def reset(self) -> None:
        """Forget hints used and checks passed for this challenge."""
        self._progress = {"hints": {}, "checks": {}}
        if self._progress_file.exists():
            self._progress_file.unlink()
