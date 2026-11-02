"""Foundation for the large notebooks (E87-E93): readable and partly runnable without the big run.

Run levels, set with ``PYMC_CHALLENGES_RUN``:

    read     nothing is computed: only stored numbers and figures are shown
    replay   the default: stored results are loaded and the analysis after them is re-run
    small    the reduced problem is fitted, within the notebook budget of a reader's laptop
    full     the whole run; also needs ``PYMC_CHALLENGES_CONFIRM_FULL=yes`` and a passing preflight

A notebook starts with::

    level = run_level(needs=dict(ram_gb=40, vram_gb=16, disk_gb=50))
    store = Store("E87")
    if level == "replay":
        store.restore()  # arrays are not in git: says so, and what to do, if they are not here
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

Only small JSON goes in git, so on a fresh clone the manifest and summaries are here and the arrays
are not. ``Store.restore`` brings back a copy from
``~/.cache/pymc-challenges/results/<notebook id>/`` (or any folder), checked against the manifest;
without one it raises ``ResultsMissing``, whose message says what to do. Downloads and other large files also go under ``CACHE_DIR``, not in git.

Importing this module does not import JAX or start CUDA; the pilot helpers import JAX when called.
It does load NumPy (so does the package), so set ``OPENBLAS_NUM_THREADS=1`` before
``import pymc_challenges`` in any notebook that runs nutpie: BLAS oversubscription slowed E86's
baseline about 2.7x.
"""

from __future__ import annotations

import glob
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CACHE_DIR = Path.home() / ".cache" / "pymc-challenges"  # downloads, anything too big for git
RESTORE_DIR = CACHE_DIR / "results"  # where Store.restore looks: <notebook id>/ under here
GPU_LOCK = CACHE_DIR / "gpu.lock"
CPU_OFFLINE = Path("/sys/devices/system/cpu/offline")
LEVELS = ("read", "replay", "small", "full")
LABELS = ("measured", "estimated", "extrapolated")  # the only labels a number may carry
CUDA_HOST_GB = 2.0  # host memory CUDA costs (measured in E86)
JAX_VRAM_FRACTION = 0.75  # JAX's default share of VRAM
TMP_SUFFIX = ".tmp"
STALE_TMP_S = 3600  # a temp file this old belongs to a process that died


class PreflightError(RuntimeError):
    """This machine or configuration cannot do the run; the message says why and what to do."""


class ChecksumError(RuntimeError):
    """A stored file does not match its checksum."""


class ResultsMissing(RuntimeError):
    """Results a level needs are not on this machine; the message says what to do."""


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
        have = preflight(**needs)
        print(f"Preflight: offline CPUs {have['offline_cpus']}, "
              f"machine-check records this boot {have['machine_checks']} (None means unknown)")
    return level


def gpu_present():
    """Can JAX use a GPU here? Looks for the plugin and device files; does not start CUDA."""
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    if platforms and not {"gpu", "cuda"} & set(platforms.split(",")):
        return False
    return (importlib.util.find_spec("jax_cuda13_plugin") is not None
            and bool(glob.glob("/dev/nvidia[0-9]*") or glob.glob("/dev/dxg")))


def _jax_if_loaded():
    """JAX if something has really imported it, else None, without importing it.

    ``sys.modules["jax"]`` alone is not enough: arviz leaves a lazy stub there, and touching any
    attribute of the stub imports all of JAX and maps the CUDA library. A real import has also
    loaded ``jax.numpy``; the stub has not.
    """
    return sys.modules.get("jax") if "jax.numpy" in sys.modules else None


def _nvidia_smi_memory():
    """(total, free, reserved) VRAM of the first GPU in GB as nvidia-smi sees them, or None.

    ``reserved`` is what the driver keeps for itself; 0 if not reported. CUDA's total leaves it out.
    """
    if not (exe := shutil.which("nvidia-smi")):
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.total,memory.free,memory.reserved",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        total, free, *rest = out.splitlines()[0].split(",")
        total, free = float(total) / 1024, float(free) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    try:
        reserved = float(rest[0]) / 1024
    except (ValueError, IndexError):
        reserved = 0.0
    return total, free, reserved


def _jax_gpu_stats(start=False):
    """JAX's allocator statistics for the first GPU, or None. Starts CUDA only if ``start``."""
    if not gpu_present():
        return None
    try:
        if start:
            import jax
        elif (jax := _jax_if_loaded()) is None:
            return None
        else:
            from jax._src.xla_bridge import backends_are_initialized  # no public way to ask

            if not backends_are_initialized():
                return None
        return jax.devices("gpu")[0].memory_stats() or None
    except (ImportError, AttributeError, RuntimeError):
        return None


def gpu_memory():
    """VRAM of the first GPU in GB, or None if unknown. JAX is the last resort: it starts CUDA.

    ``usable_gb`` is what JAX can still give this process. Before CUDA starts that is its share of
    the total less the driver's reserve, by default 75%, or what is free if less. On an RTX 3090 Ti
    that gives JAX's real ``bytes_limit`` of 16.56 GiB (measured), not 75% of 22.49. Once JAX has
    started CUDA, nvidia-smi counts JAX's own pool as taken, so the answer comes from JAX instead
    (``source`` is "jax"): ``bytes_limit`` is already JAX's share, so no fraction is applied, and
    ``free_gb`` is the same as ``usable_gb``. Without nvidia-smi, ``total_gb`` is only JAX's limit,
    not the size of the card.
    """
    smi = _nvidia_smi_memory()
    stats = _jax_gpu_stats(start=smi is None) or {}
    if "bytes_limit" in stats:
        limit = stats["bytes_limit"] / 1024**3
        pool = stats.get("pool_bytes", stats["bytes_limit"]) / 1024**3  # taken from the card so far
        growth = limit - pool
        if smi:
            growth = min(growth, smi[1])
        usable = max(0.0, pool - stats.get("bytes_in_use", 0) / 1024**3 + growth)
        return {"total_gb": smi[0] if smi else limit, "free_gb": usable, "usable_gb": usable,
                "source": "jax"}
    if smi is None:
        return None
    total, free, reserved = smi
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform":
        fraction = 1.0
    else:
        fraction = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", JAX_VRAM_FRACTION))
    return {"total_gb": total, "free_gb": free, "usable_gb": min(free, fraction * (total - reserved)),
            "source": "nvidia-smi"}


def online_core_check():
    """Offline CPUs and this boot's kernel machine-check records, for a run to report.

    ``offline_cpus`` is a sorted list and ``machine_checks`` a count of records: decoded
    ``MC<n>_STATUS`` lines plus raw ``Machine Check: ... Bank <n>`` lines. The raw form matters
    because a record left by a crash is logged about a second into the next boot, before the decoder
    loads, so a count can be above zero at the start of a healthy run. Compare it with the count
    before the job, not with zero. Either value is None where it cannot be read (not Linux, no
    journal access), which is unknown, not zero. A machine with a bad core taken offline by hand
    gets it back on a reboot, so compare the list with what that machine expects: this function
    cannot know that.
    """
    try:
        cpus = []
        for part in CPU_OFFLINE.read_text().strip().split(","):
            first, _, last = part.partition("-")
            if first:
                cpus.extend(range(int(first), int(last or first) + 1))
    except (OSError, ValueError):
        cpus = None
    try:
        out = subprocess.run(["journalctl", "-b", "0", "-k", "--no-pager"], capture_output=True,
                             text=True, errors="replace", timeout=30, check=True).stdout
        lines = out.splitlines()
    except (OSError, subprocess.SubprocessError):
        lines = []
    # A booted kernel always logs something, so no entries means the log could not be read.
    readable = bool(lines) and "No entries" not in lines[0]
    count = sum(bool(re.search(r"MC[0-9]+_STATUS|Machine Check: .*Bank [0-9]+", line)) for line in lines)
    return {"offline_cpus": cpus, "machine_checks": count if readable else None}


@contextlib.contextmanager
def gpu_lock(timeout=None, path=None, poll=1.0):
    """``with gpu_lock():`` holds the machine's one GPU slot, waiting for it if another job has it.

    Raises TimeoutError after ``timeout`` seconds (None waits for ever). The kernel drops the lock
    when the holder exits or dies, so there is no stale lock to clear. It only excludes jobs that
    also take it, and it is not re-entrant: taking it twice in one process waits for itself.
    """
    import fcntl

    path = Path(GPU_LOCK if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    announced = False
    with open(path, "a+") as f:  # "a+", as "w" would wipe the holder's note before the lock is ours
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                f.seek(0)
                holder = f.read().strip() or "an unknown process"
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"GPU lock {path} is held by {holder}, after {timeout} s")
            if not announced:
                print(f"Waiting for the GPU lock {path}, held by {holder}", flush=True)
                announced = True
            time.sleep(poll)
        f.truncate(0)
        f.write(f"pid {os.getpid()}, since {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.flush()
        try:
            yield path
        finally:
            f.truncate(0)
            f.flush()


def preflight(ram_gb=0.0, vram_gb=0.0, disk_gb=0.0, disk_path=None):
    """Raise PreflightError listing every shortfall, or return what is available.

    The return has the GB available, and ``online_core_check``'s ``offline_cpus`` and
    ``machine_checks``, which are reported rather than judged: see that function.
    """
    import psutil

    host_gb = ram_gb + (CUDA_HOST_GB if vram_gb > 0 else 0.0)
    problems = []
    have = {"ram_gb": psutil.virtual_memory().available / 1024**3, **online_core_check()}
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

    def _what_to_do(self):
        return (f"Only small JSON results are in git, so arrays from a full run are not in a fresh "
                f"clone. If you have a copy, put it in {RESTORE_DIR / self.id} and call "
                f"Store({self.id!r}).restore(). If not, PYMC_CHALLENGES_RUN=read shows the stored "
                "numbers and PYMC_CHALLENGES_RUN=small computes the reduced problem.")

    def _checked(self, path, check=True):
        if not path.exists():
            raise ResultsMissing(f"{path.relative_to(self.path)} is not in {self.path}. "
                                 + self._what_to_do())
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

    def missing(self):
        """Files the manifest lists that are not on disk; empty if there is no manifest."""
        if not (self.path / "manifest.json").exists():
            return []
        return [name for name in self.manifest()["files"] if not (self.path / name).exists()]

    def restore(self, source=None):
        """Bring in a copy of this store's files, then raise ResultsMissing if any are still absent.

        ``source`` is a folder laid out like this store, by default under ``RESTORE_DIR``. Files
        that are already intact are left alone, so results in git are never overwritten. A file is
        taken only if it matches the manifest here, or without an entry there its own sidecar; any
        other is listed in ``refused``, so arrays of another run cannot mix with this run's numbers.
        ``stages.json`` has no checksum and is not copied. With no copy nothing is hashed, so this
        is cheap to call at the top of a notebook. Returns the restored and refused names.
        """
        source = Path(RESTORE_DIR / self.id if source is None else source)
        restored, refused = [], []

        def bring(name, digest):
            path, copy = self.path / name, source / name
            if _intact(path) and digest in (None, sha256_file(path)):
                return
            if digest is None and _sidecar(copy).exists():
                digest = next(iter(_sidecar(copy).read_text().split()), None)
            if digest is None or sha256_file(copy) != digest:
                refused.append(name)
                return
            with open(copy, "rb") as f:
                _save(path, lambda out: shutil.copyfileobj(f, out))
            restored.append(name)

        if (source / "manifest.json").is_file():
            bring("manifest.json", None)  # first: it is what the other files are checked against
        has_manifest = (self.path / "manifest.json").exists()
        listed = self.manifest()["files"] if has_manifest else {}
        for copy in sorted(source.rglob("*")) if source.is_dir() else ():
            name = copy.relative_to(source).as_posix()
            if (copy.is_file() and copy.suffix != ".sha256" and not name.endswith(TMP_SUFFIX)
                    and name not in ("manifest.json", "stages.json")):
                bring(name, listed.get(name))
        absent = self.missing() if has_manifest else ["manifest.json"]
        if absent:
            why = f" Refused, as checksums did not match: {refused[:5]}." if refused else ""
            raise ResultsMissing(f"{self.id}: {len(absent)} stored files are not in {self.path}, "
                                 f"first {absent[:5]}.{why} " + self._what_to_do())
        return {"restored": restored, "refused": refused}

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
            raise ResultsMissing(f"{name}: {len(todo)} of {n} chunks are not finished or not here, "
                                 f"first {todo[:5]}. " + self._what_to_do())
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
            raise TypeError(f"{key} must be labelled {', '.join(LABELS)}: "
                            "wrap it in the function of that name")
        text = f"{cell['value']} {cell.get('unit', '')}".strip().replace("|", "\\|")
        lines.append(f"| {title} | {text} | {cell['label']} |")
    note = _LEVEL_NOTES.get(level)
    return Banner("\n".join(([note, ""] if note else []) + lines))


# Pilot helpers: JAX is imported inside each one ---------------------------------------------------


def _leaves(out):
    jax = _jax_if_loaded()
    if jax is not None:
        return jax.tree.leaves(out)
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        return [leaf for item in out for leaf in _leaves(item)]
    return [] if out is None else [out]


def _finite(out):
    for leaf in _leaves(out):
        try:
            array = np.asarray(leaf)
        except TypeError:  # JAX PRNG keys and the like hold no floats
            continue
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            return False
    return True


def time_call(fn, *args, repeat=5, **kwargs):
    """Time ``fn(*args, **kwargs)`` and wait for its result, so that the work itself is measured.

    Returns ``first_s`` (the first call, with any compilation), ``run_ms`` (best of ``repeat`` more
    calls), ``repeat`` and ``finite`` (whether all float outputs were; a NaN path can run at another
    speed). ``fn`` must return its result: unused JAX work can be deleted and a result nobody holds
    cannot be waited on, so a function that returns nothing raises ValueError. JAX is only touched
    if something has imported it, so this also times NumPy code without loading JAX.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")

    def timed():
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if (jax := _jax_if_loaded()) is not None:
            out = jax.block_until_ready(out)
        return out, time.perf_counter() - t0

    out, first_s = timed()
    if not _leaves(out):
        raise ValueError("time_call needs fn to return its result, so that it can be waited on")
    runs = []
    for _ in range(repeat):
        out, seconds = timed()
        runs.append(seconds)
    return {"first_s": first_s, "run_ms": min(runs) * 1e3, "repeat": repeat, "finite": _finite(out)}


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
        timing = time_call(step, q, repeat=repeat)
        rows.append({"chains": n, "compile_s": timing["first_s"], "run_ms": timing["run_ms"],
                     "ms_per_chain": timing["run_ms"] / n, "dtype": str(q0.dtype),
                     "device": str(device), "finite": timing["finite"]})
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
