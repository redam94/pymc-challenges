"""Foundation for the large notebooks (E87-E93): readable and partly runnable without the big run.

Run levels, set with ``PYMC_CHALLENGES_RUN``:

    read     nothing is computed: only stored numbers and figures are shown
    replay   the default: stored results are loaded and the analysis after them is re-run
    small    the reduced problem is fitted, within the notebook budget of a reader's laptop
    full     the whole run; also needs ``PYMC_CHALLENGES_CONFIRM_FULL=yes`` and a passing preflight

A notebook starts with::

    level = run_level(needs=dict(ram_gb=40, vram_gb=18, disk_gb=50))
    store = Store("E87")
    if level == "full" and not store.done("warmup"):
        ...
        store.save_array("warmup_positions", positions)
        store.mark_done("warmup")
    for i in store.pending("sweep", n=500):
        store.save_chunk("sweep", i, {"draws": fit(i)})

Results live in ``results/<notebook id>/``. Every write is temp file, fsync, ``os.replace``, so a
crash leaves the old file or the new one and never part of one. Each file has a ``.sha256`` sidecar
that ``sha256sum -c`` can read; sidecars rather than one shared table so that a CPU job pool can
write chunks at once. A crash between a file and its sidecar makes that file fail its checksum,
which a load refuses and ``pending`` recomputes, so it costs work and never gives a wrong answer.

Importing this module does not import JAX or start CUDA; the pilot helpers import JAX when called.
It does load NumPy (so does the package), so set ``OPENBLAS_NUM_THREADS=1`` before
``import pymc_challenges`` in any notebook that runs nutpie: BLAS oversubscription slowed E86's
baseline about 2.7x.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
LEVELS = ("read", "replay", "small", "full")
LABELS = ("measured", "estimated", "extrapolated")
CUDA_HOST_GB = 2.0  # host memory CUDA costs (measured in E86)
JAX_VRAM_FRACTION = 0.75  # JAX's default share of VRAM
TMP_SUFFIX = ".tmp"
STALE_TMP_S = 3600  # a temp file this old belongs to a process that died


class PreflightError(RuntimeError):
    """This machine or configuration cannot do the run; the message says why and what to do."""


class ChecksumError(RuntimeError):
    """A stored file does not match its checksum."""


# Run levels and preflight -----------------------------------------------------------------------


def run_level(needs=None, env=None):
    """The run level. 'full' also needs the opt-in and passes ``preflight(**needs)``, or refuses."""
    env = os.environ if env is None else env
    level = env.get("PYMC_CHALLENGES_RUN", "").strip().lower() or "replay"
    if level not in LEVELS:
        raise ValueError(f"PYMC_CHALLENGES_RUN={level!r} is not one of {', '.join(LEVELS)}")
    if level == "full":
        if env.get("PYMC_CHALLENGES_CONFIRM_FULL") != "yes":
            raise PreflightError("A full run is long and heavy. Set PYMC_CHALLENGES_CONFIRM_FULL=yes "
                                 "to confirm it, or use PYMC_CHALLENGES_RUN=replay or small.")
        if needs is None:
            raise PreflightError("A full run must state its needs: "
                                 "run_level(needs=dict(ram_gb=..., vram_gb=..., disk_gb=...)).")
        preflight(**needs)
    return level


def gpu_present():
    """Can JAX use a GPU here? Looks for the plugin and device files; does not start CUDA."""
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    if platforms and not {"gpu", "cuda"} & set(platforms.split(",")):
        return False
    return (importlib.util.find_spec("jax_cuda13_plugin") is not None
            and bool(glob.glob("/dev/nvidia[0-9]*") or glob.glob("/dev/dxg")))


def gpu_memory():
    """VRAM of the first GPU in GB, or None if unknown. JAX is the last resort: it starts CUDA.

    ``usable_gb`` allows for what JAX will take, by default 75% of the total, whatever is free.
    """
    total = free = None
    source = "nvidia-smi"
    if exe := shutil.which("nvidia-smi"):
        try:
            out = subprocess.run([exe, "--query-gpu=memory.total,memory.free",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, check=True).stdout
            total, free = (float(x) / 1024 for x in out.splitlines()[0].split(","))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            total = free = None
    if total is None and gpu_present():
        import jax

        source = "jax"
        try:
            stats = jax.devices("gpu")[0].memory_stats() or {}
            total = stats["bytes_limit"] / 1024**3
            free = total - stats.get("bytes_in_use", 0) / 1024**3
        except (RuntimeError, KeyError):
            return None
    if total is None:
        return None
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform":
        fraction = 1.0
    else:
        fraction = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", JAX_VRAM_FRACTION))
    return {"total_gb": total, "free_gb": free, "usable_gb": min(free, fraction * total),
            "source": source}


def preflight(ram_gb=0.0, vram_gb=0.0, disk_gb=0.0, disk_path=None):
    """Raise PreflightError listing every shortfall, or return what is available in GB."""
    import psutil

    host_gb = ram_gb + (CUDA_HOST_GB if vram_gb > 0 else 0.0)
    problems = []
    have = {"ram_gb": psutil.virtual_memory().available / 1024**3}
    if have["ram_gb"] < host_gb:
        extra = f" (including {CUDA_HOST_GB:.0f} GB for CUDA)" if vram_gb > 0 else ""
        problems.append(f"RAM: need {host_gb:.1f} GB{extra}, {have['ram_gb']:.1f} GB available")

    path = Path(disk_path or RESULTS_DIR)
    while not path.exists():
        path = path.parent
    have["disk_gb"] = shutil.disk_usage(path).free / 1024**3
    if have["disk_gb"] < disk_gb:
        problems.append(f"disk: need {disk_gb:.1f} GB, {have['disk_gb']:.1f} GB free at {path}")

    if vram_gb > 0:
        vram = gpu_memory() if gpu_present() else None
        have["vram_gb"] = vram["usable_gb"] if vram else 0.0
        if vram is None:
            problems.append(f"GPU: need {vram_gb:.1f} GB of VRAM but JAX has no usable GPU (needs "
                            "`uv sync --extra gpu`, an NVIDIA device, and no JAX_PLATFORMS=cpu)")
        elif vram["usable_gb"] < vram_gb:
            problems.append(f"GPU: need {vram_gb:.1f} GB of VRAM, {vram['usable_gb']:.1f} GB usable "
                            f"({vram['free_gb']:.1f} of {vram['total_gb']:.1f} GB free, and JAX "
                            "takes a fixed share of the total)")
    if problems:
        raise PreflightError("This machine cannot do the full run:\n  " + "\n  ".join(problems)
                             + "\nUse PYMC_CHALLENGES_RUN=replay or small instead.")
    return have


# Atomic files -----------------------------------------------------------------------------------


def sha256_file(path):
    """SHA-256 of a file, for checksums and data fingerprints."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, write):
    """Call ``write(binary_file)`` on a temp file, fsync, rename over ``path``; returns its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=TMP_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)  # mkstemp makes files private
        digest = sha256_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)  # the rename itself must survive a power cut
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest


def _sidecar(path):
    return path.with_name(path.name + ".sha256")


def _save(path, write):
    digest = atomic_write(path, write)
    line = f"{digest}  {path.name}\n".encode()
    atomic_write(_sidecar(path), lambda f: f.write(line))
    return digest


def _intact(path):
    side = _sidecar(path)
    return path.exists() and side.exists() and side.read_text().split()[:1] == [sha256_file(path)]


def _numeric(array):
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise TypeError("Object arrays need pickle, which the store refuses: store numbers.")
    return array


def _json_value(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"{type(x).__name__} is not JSON serialisable")


def _clean(name):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Bad name {name!r}: use letters, digits, '_', '-' and '.'")
    return name


# Results store ----------------------------------------------------------------------------------


class Store:
    """``results/<notebook id>/``: checksummed arrays and JSON, manifest, chunks and stage markers."""

    _reserved = {"manifest", "stages"}

    def __init__(self, notebook_id, root=None):
        self.id = _clean(notebook_id)
        self.path = Path(RESULTS_DIR if root is None else root) / self.id
        if self.path.is_dir():
            for tmp in self.path.rglob(f".*{TMP_SUFFIX}"):
                try:
                    if time.time() - tmp.stat().st_mtime > STALE_TMP_S:
                        tmp.unlink(missing_ok=True)
                except FileNotFoundError:  # another worker in the pool just finished this write
                    pass

    def _file(self, name, suffix):
        if _clean(name) in self._reserved:
            raise ValueError(f"{name!r} is reserved by the store")
        return self.path / f"{name}{suffix}"

    def _checked(self, path, check=True):
        if not path.exists():
            raise FileNotFoundError(f"{path} is not in the store: restore the stored results, or "
                                    "produce them with PYMC_CHALLENGES_RUN=small or full.")
        if check and not _intact(path):
            raise ChecksumError(f"{path} does not match its checksum (corrupt, or the writer died "
                                "before finishing). Re-run the stage that produces it.")
        return path

    def save_array(self, name, array):
        """Store one array as ``.npy``; returns its SHA-256."""
        array = _numeric(array)
        return _save(self._file(name, ".npy"), lambda f: np.save(f, array, allow_pickle=False))

    def load_array(self, name, mmap=False, check=True):
        """Load an array, verifying it unless ``check=False``. ``mmap=True`` keeps it on disk."""
        path = self._checked(self._file(name, ".npy"), check)
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)

    def save_json(self, name, value):
        """Store a JSON value (NumPy numbers and arrays are converted); returns its SHA-256."""
        text = json.dumps(value, indent=2, sort_keys=True, default=_json_value).encode()
        return _save(self._file(name, ".json"), lambda f: f.write(text))

    def load_json(self, name, check=True):
        return json.loads(self._checked(self._file(name, ".json"), check).read_text())

    # Provenance

    def write_manifest(self, *, git_hash, versions, data_fingerprint, hardware, started,
                       finished=None, **extra):
        """Record what produced these results, with the checksums of every stored file."""
        files = {str(p.relative_to(self.path))[:-len(".sha256")]: p.read_text().split()[0]
                 for p in sorted(self.path.rglob("*.sha256")) if p.name != "manifest.json.sha256"}
        manifest = {"notebook": self.id, "git_hash": git_hash, "versions": versions,
                    "data_fingerprint": data_fingerprint, "hardware": hardware, "started": started,
                    "finished": finished, "files": files, **extra}
        text = json.dumps(manifest, indent=2, sort_keys=True, default=_json_value).encode()
        return _save(self.path / "manifest.json", lambda f: f.write(text))

    def manifest(self):
        return json.loads(self._checked(self.path / "manifest.json").read_text())

    def verify(self):
        """Re-hash the store against sidecars and the manifest; returns a list of problems."""
        skip = {"stages.json"}
        problems = [f"{p.relative_to(self.path)}: bad or missing checksum"
                    for p in sorted(self.path.rglob("*"))
                    if p.is_file() and p.suffix != ".sha256" and p.name not in skip
                    and not p.name.endswith(TMP_SUFFIX) and not _intact(p)]
        if _intact(self.path / "manifest.json"):
            for name, digest in self.manifest()["files"].items():
                if not (self.path / name).exists():
                    problems.append(f"{name}: in the manifest but missing")
                elif sha256_file(self.path / name) != digest:
                    problems.append(f"{name}: changed since the manifest was written")
        return problems

    # Chunked work and checkpoints

    def _chunk(self, name, index):
        return self.path / "chunks" / _clean(name) / f"{int(index):08d}.npz"

    def _chunk_ids(self, name):
        return sorted(int(p.stem) for p in self._chunk(name, 0).parent.glob("*.npz"))

    def save_chunk(self, name, index, arrays, keep=None):
        """Store a dict of arrays as chunk ``index``. ``keep=k`` retains only the last k chunks."""
        arrays = {k: _numeric(v) for k, v in arrays.items()}
        digest = _save(self._chunk(name, index), lambda f: np.savez(f, **arrays))
        if keep is not None:
            for old in self._chunk_ids(name)[:-keep]:
                self._chunk(name, old).unlink(missing_ok=True)
                _sidecar(self._chunk(name, old)).unlink(missing_ok=True)
        return digest

    def load_chunk(self, name, index, check=True):
        with np.load(self._checked(self._chunk(name, index), check), allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def pending(self, name, n):
        """Indices in ``range(n)`` still to do. Every finished chunk is re-hashed; bad ones redo."""
        return [i for i in range(n) if not _intact(self._chunk(name, i))]

    def resume(self, name):
        """(index, arrays) of the newest intact chunk, skipping corrupt ones, or (None, None)."""
        for index in reversed(self._chunk_ids(name)):
            if _intact(self._chunk(name, index)):
                return index, self.load_chunk(name, index, check=False)
        return None, None

    def gather(self, name, n, axis=0):
        """Concatenate chunks 0..n-1, key by key. Refuses if any is missing or corrupt."""
        if todo := self.pending(name, n):
            raise FileNotFoundError(f"{name}: {len(todo)} of {n} chunks are not finished, "
                                    f"first {todo[:5]}")
        chunks = [self.load_chunk(name, i, check=False) for i in range(n)]
        return {k: np.concatenate([c[k] for c in chunks], axis=axis) for k in chunks[0]}

    # Stages

    def stages(self):
        """Finished stages, oldest first."""
        path = self.path / "stages.json"
        return json.loads(path.read_text()) if path.exists() else []

    def done(self, stage):
        return any(s["stage"] == stage for s in self.stages())

    def mark_done(self, stage, **info):
        """Record a finished stage. Call it after the stage's results are saved, never before."""
        records = [s for s in self.stages() if s["stage"] != stage]
        records.append({"stage": stage, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        **info})
        text = json.dumps(records, indent=2, default=_json_value).encode()
        atomic_write(self.path / "stages.json", lambda f: f.write(text))

    def next_stage(self, order):
        """The first stage of ``order`` not finished, or None when all are."""
        return next((stage for stage in order if not self.done(stage)), None)


def git_hash(repo=None):
    """HEAD of the repository, with ``-dirty`` if the tree has changes; None outside git."""
    repo = RESULTS_DIR.parent if repo is None else repo
    try:
        return subprocess.run(["git", "-C", str(repo), "describe", "--always", "--dirty",
                               "--abbrev=12"], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def library_versions(*names):
    """Installed versions, read from package metadata so that nothing is imported."""
    return {name: importlib.metadata.version(name) for name in names}


# Banner -----------------------------------------------------------------------------------------

_ROWS = {"wall_time": "Wall time", "peak_ram": "Peak RAM", "peak_vram": "Peak VRAM", "disk": "Disk",
         "download": "Download size", "precision": "Precision", "chains": "Chains",
         "hardware": "Hardware", "build_date": "Build date", "git_hash": "Git hash",
         "smaller_machine": "On a smaller machine"}
_LEVEL_NOTES = {"read": "Nothing was computed here: these are the stored numbers of the full run.",
                "replay": "Loaded from the stored full run: these numbers are not from this machine.",
                "small": "This session runs the small problem: these numbers describe the full run."}


def measured(value, unit=""):
    """A value this repository observed in a run. Plain JSON, so it survives ``save_json``."""
    return {"value": value, "unit": unit, "label": "measured"}


def estimated(value, unit=""):
    """A value judged rather than observed, for example scaled from a pilot."""
    return {"value": value, "unit": unit, "label": "estimated"}


def extrapolated(value, unit=""):
    """A value computed from a model of cost beyond what was run."""
    return {"value": value, "unit": unit, "label": "extrapolated"}


class Banner(str):
    """A Markdown table: renders as a table when it ends a notebook cell, prints as plain text."""

    def _repr_markdown_(self):
        return str(self)


def banner(level=None, **values):
    """The table that opens a large notebook. Every row is required and every value labelled."""
    if missing := [k for k in _ROWS if k not in values]:
        raise TypeError(f"banner needs {', '.join(missing)}")
    if unknown := sorted(set(values) - set(_ROWS)):
        raise TypeError(f"banner does not know {', '.join(unknown)}")
    lines = ["| | Value | Status |", "|---|---|---|"]
    for key, title in _ROWS.items():
        cell = values[key]
        if not isinstance(cell, dict) or cell.get("label") not in LABELS:
            raise TypeError(f"{key} must be labelled: wrap it in measured() or estimated()")
        text = f"{cell['value']} {cell.get('unit', '')}".strip().replace("|", "\\|")
        lines.append(f"| {title} | {text} | {cell['label']} |")
    note = _LEVEL_NOTES.get(level)
    return Banner("\n".join(([note, ""] if note else []) + lines))


# Pilot helpers: JAX is imported inside each one ---------------------------------------------------


def time_gradient(logdensity, q0, chains=(1, 8, 32, 128), repeat=5, jitter=0.1, seed=0, device=None):
    """Time ``vmap(value_and_grad(logdensity))`` at each chain count, from a flat position ``q0``.

    One row per count: compile seconds, best-of-``repeat`` milliseconds, milliseconds per chain, and
    whether all outputs were finite (a NaN path can run at another speed, so distrust those rows).
    Both outputs are returned and waited on, so XLA cannot delete the work. Each count is a new
    compiled program: caches are cleared between counts, including any of the caller's.
    """
    import jax
    import jax.numpy as jnp

    q0 = jnp.asarray(q0)
    if q0.ndim != 1:
        raise ValueError(f"q0 must be one flat position, got shape {q0.shape}")
    device = device or jax.devices()[0]
    rng = np.random.default_rng(seed)
    rows = []
    for n in chains:
        q = np.asarray(q0) + jitter * rng.normal(size=(n, q0.size))
        q = jax.device_put(jnp.asarray(q, q0.dtype), device)
        step = jax.jit(jax.vmap(jax.value_and_grad(logdensity)))
        t0 = time.perf_counter()
        jax.block_until_ready(step(q))
        compile_s = time.perf_counter() - t0
        runs = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = jax.block_until_ready(step(q))
            runs.append(time.perf_counter() - t0)
        rows.append({"chains": n, "compile_s": compile_s, "run_ms": min(runs) * 1e3,
                     "ms_per_chain": min(runs) * 1e3 / n, "dtype": str(q0.dtype),
                     "device": str(device), "finite": all(bool(np.isfinite(np.asarray(leaf)).all())
                                                          for leaf in jax.tree.leaves(out))})
        jax.clear_caches()
    return rows


def vram_high_water(device=None):
    """JAX's peak and current allocation in GB, from the allocator, which sees past preallocation.

    The peak is for the whole process and cannot be reset: measure each configuration in a fresh
    process. On a device without these statistics the numbers are None.
    """
    import jax

    device = device or jax.devices()[0]
    stats = (device.memory_stats() or {}) if device.platform in ("gpu", "cuda") else {}
    gb = {key: stats[name] / 1024**3 if name in stats else None
          for key, name in (("peak_gb", "peak_bytes_in_use"), ("in_use_gb", "bytes_in_use"),
                            ("limit_gb", "bytes_limit"))}
    return {**gb, "device": str(device)}


def precision_check(logdensity32, logdensity64, draws, tol=0.1):
    """float32 minus float64 log density at the same points; a pilot before committing to float32.

    Build each function under its own precision. ``draws`` is (draws, dim); both functions see the
    float32 values, exactly, so the error is arithmetic alone. Only spread matters to HMC, which uses
    energy differences, so ``ok`` tests the sd against ``tol`` and that both stayed finite. Each
    function runs under its own x64 setting, which is restored afterwards, and a function that
    returns the wrong precision raises, since it would otherwise compare a result with itself.

    Passing says the sampler sees accurate densities here. It says nothing about divergences:
    blackjax flags those at an energy error above 1000 or NaN, far beyond rounding of this size.
    """
    import jax

    q32 = np.asarray(draws, np.float32)
    x64 = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        lp32 = jax.vmap(logdensity32)(jax.numpy.asarray(q32))
        lp32 = np.asarray(jax.block_until_ready(lp32))
        jax.config.update("jax_enable_x64", True)
        lp64 = jax.vmap(logdensity64)(jax.numpy.asarray(q32.astype(np.float64)))
        lp64 = np.asarray(jax.block_until_ready(lp64))
    finally:
        jax.config.update("jax_enable_x64", x64)
    if lp32.dtype != np.float32 or lp64.dtype != np.float64:
        raise TypeError(f"Expected float32 and float64 log densities, got {lp32.dtype} and "
                        f"{lp64.dtype}: each function must be built under its own precision")
    finite = np.isfinite(lp32) & np.isfinite(lp64)
    error = lp32[finite].astype(np.float64) - lp64[finite]
    sd = float(np.std(error)) if error.size else np.nan
    return {"n": lp32.size, "compared": error.size, "sd": sd,
            "mean": float(np.mean(error)) if error.size else np.nan,
            "max_abs": float(np.abs(error).max()) if error.size else np.nan,
            "nonfinite32": int(np.sum(~np.isfinite(lp32) & np.isfinite(lp64))),
            "ok": bool(finite.all()) and sd <= tol}


def find_stragglers(logp, hypers=None, threshold=6.0):
    """Flag chains more than ``threshold`` robust sds from the ensemble.

    ``logp`` is (chains,), ``hypers`` (chains, h) on the unconstrained scale. Returns ``bad``, the
    (chains, 1 + h) robust ``z``, ``stuck`` (flagged, and log density far below the ensemble: safe to
    cull in warmup) and ``check_mode`` (flagged without that: it may be a true minor mode, and
    culling would hide it, so look before you cull). Non-finite chains are always stuck.
    """
    logp = np.asarray(logp, float)
    columns = logp[:, None] if hypers is None else np.c_[logp, np.asarray(hypers, float)]
    if len(columns) < 8:
        raise ValueError(f"{len(columns)} chains is too few for a robust spread: need at least 8")
    nonfinite = ~np.isfinite(columns).all(axis=1)
    healthy = columns[~nonfinite]
    if len(healthy) == 0:
        return {"bad": nonfinite, "stuck": nonfinite, "check_mode": ~nonfinite,
                "z": np.full(columns.shape, np.inf), "worst_z": np.inf, "n_bad": len(columns)}
    median = np.median(healthy, axis=0)
    deviation = np.abs(healthy - median)
    # Over half the chains identical gives MAD 0: fall back on the mean deviation (scale 1.2533).
    scale = 1.4826 * np.median(deviation, axis=0)
    scale = np.where(scale > 0, scale, 1.2533 * deviation.mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(columns - median) / scale
    z = np.where(columns == median, 0.0, z)  # 0/0 where every chain agrees
    z = np.where(np.isfinite(columns), z, np.inf)
    bad = (z > threshold).any(axis=1)
    stuck = nonfinite | (bad & (z[:, 0] > threshold) & (logp < median[0]))
    return {"bad": bad, "stuck": stuck, "check_mode": bad & ~stuck, "z": z,
            "worst_z": float(z.max()), "n_bad": int(bad.sum())}


def short_warmup_probe(warmup, start, logdensity, steps=100, n_hyper=0, threshold=6.0, seed=0):
    """Run a short warmup twice to see speed and health before the big run.

    ``warmup(key, start, steps)`` returns ``(positions, scalars)``: positions (chains, dim) and a
    dict of tuned scalars such as step size. Reports both times (the first includes compilation; a
    warmup rebuilt per call compiles in both, so compare them), ``s_per_step``, which is an
    estimate from this short run, ``moved`` (the share of chains that left ``start``: E86's frozen
    MEADS chains showed healthy acceptance and no divergences), and ``find_stragglers`` on the
    result with the first ``n_hyper`` coordinates as hyperparameters.
    """
    import jax

    first, second = jax.random.split(jax.random.key(seed))
    t0 = time.perf_counter()
    jax.block_until_ready(warmup(first, start, steps))
    first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    positions, scalars = jax.block_until_ready(warmup(second, start, steps))
    second_s = time.perf_counter() - t0
    logp = np.asarray(jax.block_until_ready(jax.vmap(logdensity)(positions)))
    positions = np.asarray(positions)
    return {"steps": steps, "first_s": first_s, "second_s": second_s, "s_per_step": second_s / steps,
            "moved": float((positions != np.asarray(start)).any(axis=1).mean()),
            "tuned": {k: float(v) for k, v in scalars.items()},
            "stragglers": find_stragglers(logp, positions[:, :n_hyper] if n_hyper else None,
                                          threshold)}
