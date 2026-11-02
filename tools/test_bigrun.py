"""Plain-assert tests of ``pymc_challenges.bigrun``. CPU only, a few seconds, and never touches the
GPU (``nvidia-smi`` is replaced by a fake on PATH).

    JAX_PLATFORMS=cpu .venv/bin/python tools/test_bigrun.py
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from pymc_challenges import bigrun
from pymc_challenges.bigrun import (ChecksumError, PreflightError, ResultsMissing, Store, banner,
                                    estimated, extrapolated, find_stragglers, measured, run_level)


def raises(kind, fn, *args, match="", **kwargs):
    try:
        fn(*args, **kwargs)
    except kind as error:
        assert match in str(error), f"{match!r} not in {error}"
        return error
    raise AssertionError(f"{kind.__name__} not raised")


def test_import_leaves_jax_alone():
    code = "import sys, pymc_challenges.bigrun; sys.exit('jax' in sys.modules or 'jaxlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_run_levels():
    assert run_level(env={}) == "replay"
    assert run_level(env={"PYMC_CHALLENGES_RUN": ""}) == "replay"
    assert run_level(env={"PYMC_CHALLENGES_RUN": " Small "}) == "small"
    assert run_level(env={"PYMC_CHALLENGES_RUN": "read"}) == "read"
    raises(ValueError, run_level, env={"PYMC_CHALLENGES_RUN": "ful"}, match="ful")

    full = {"PYMC_CHALLENGES_RUN": "full"}
    needs = {"ram_gb": 0.0}
    raises(PreflightError, run_level, needs=needs, env=full, match="CONFIRM_FULL")
    confirmed = {**full, "PYMC_CHALLENGES_CONFIRM_FULL": "yes"}
    raises(PreflightError, run_level, env=confirmed, match="needs")
    raises(PreflightError, run_level, needs={"ram_gb": 1e9}, env=confirmed, match="RAM")
    assert run_level(needs=needs, env=confirmed) == "full"


def test_preflight(tmp):
    have = bigrun.preflight(ram_gb=0.001, disk_gb=0.001, disk_path=tmp / "not" / "yet")
    assert have["ram_gb"] > 0 and have["disk_gb"] > 0
    message = str(raises(PreflightError, bigrun.preflight, ram_gb=1e9, disk_gb=1e9, disk_path=tmp))
    assert "RAM" in message and "disk" in message and "replay or small" in message  # every shortfall

    # JAX forced to the CPU means no VRAM, even on a machine with a GPU.
    raises(PreflightError, bigrun.preflight, vram_gb=1.0, match="no usable GPU")

    fake = tmp / "bin"
    fake.mkdir()
    smi = fake / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '24576, 20480'\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    path, present = os.environ["PATH"], bigrun.gpu_present
    os.environ["PATH"] = f"{fake}{os.pathsep}{path}"
    bigrun.gpu_present = lambda: True
    try:
        memory = bigrun.gpu_memory()
        assert memory["total_gb"] == 24 and memory["free_gb"] == 20 and memory["usable_gb"] == 18
        assert bigrun.preflight(vram_gb=17.0)["vram_gb"] == 18
        raises(PreflightError, bigrun.preflight, vram_gb=19.0, match="18.0 GB usable")
        raises(PreflightError, bigrun.preflight, ram_gb=1e9, vram_gb=1.0, match="for CUDA")
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert bigrun.gpu_memory()["usable_gb"] == 20
        del os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]
        smi.write_text("#!/bin/sh\necho '[N/A], [N/A]'\n")  # falls through; JAX has no GPU here
        bigrun.gpu_present = lambda: False
        assert bigrun.gpu_memory() is None
    finally:
        os.environ["PATH"] = path
        bigrun.gpu_present = present


def test_gpu_memory_once_jax_has_started(tmp):
    gib, mib = 1024**3, 1024**2
    fake = tmp / "bin"
    fake.mkdir()
    # This card: 23028 MiB, with JAX's own 75% pool already gone from what nvidia-smi calls free.
    smi = fake / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '23028, 5757'\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    limit = int(0.75 * 23028 * mib)
    path, present, stats = os.environ["PATH"], bigrun.gpu_present, bigrun._jax_gpu_stats
    os.environ["PATH"] = f"{fake}{os.pathsep}{path}"
    bigrun.gpu_present = lambda: True
    try:
        bigrun._jax_gpu_stats = lambda start=False: None  # CUDA not started: nvidia-smi decides
        before = bigrun.gpu_memory()
        assert before["source"] == "nvidia-smi" and abs(before["usable_gb"] - 5757 / 1024) < 1e-9

        # JAX takes 75% of CUDA's total, which leaves out the driver's reserve. RTX 3090 Ti, measured:
        # nvidia-smi 23028 total, 22424 free, 413 reserved MiB; bytes_limit 17785946112 bytes.
        smi.write_text("#!/bin/sh\necho '23028, 22424, 413'\n")
        real = bigrun.gpu_memory()
        assert real["total_gb"] == 23028 / 1024 and real["free_gb"] == 22424 / 1024
        assert abs(real["usable_gb"] - 17785946112 / gib) < 0.001  # 16.56, not 16.87
        smi.write_text("#!/bin/sh\necho '23028, 22424, [N/A]'\n")  # no reserve reported: use the total
        assert bigrun.gpu_memory()["usable_gb"] == 0.75 * 23028 / 1024
        smi.write_text("#!/bin/sh\necho '23028, 5757'\n")

        started = {"bytes_limit": limit, "pool_bytes": limit, "bytes_in_use": gib}
        bigrun._jax_gpu_stats = lambda start=False: started
        memory = bigrun.gpu_memory()
        assert memory["source"] == "jax" and abs(memory["total_gb"] - 23028 / 1024) < 1e-9
        assert abs(memory["usable_gb"] - (limit / gib - 1)) < 1e-9  # not 5.6: the pool is ours
        assert bigrun.preflight(vram_gb=15.0)["vram_gb"] == memory["usable_gb"]
        raises(PreflightError, bigrun.preflight, vram_gb=16.0, match="15.9 GB usable")

        # JAX growing on demand: the part not yet taken from the card can be lost to other programs.
        started = {"bytes_limit": limit, "pool_bytes": 2 * gib, "bytes_in_use": gib}
        assert abs(bigrun.gpu_memory()["usable_gb"] - (1 + 5757 / 1024)) < 1e-9

        # No nvidia-smi: bytes_limit is already JAX's share, so 0.75 must not be applied twice.
        smi.unlink()
        os.environ["PATH"] = str(fake)
        started = {"bytes_limit": limit}
        asked = []
        bigrun._jax_gpu_stats = lambda start=False: asked.append(start) or started
        memory = bigrun.gpu_memory()
        assert asked == [True] and memory["source"] == "jax"
        assert memory["total_gb"] == memory["usable_gb"] == limit / gib
        assert abs(memory["usable_gb"] - 16.866) < 0.001  # the fallback used to say 12.6
        started = {}
        assert bigrun.gpu_memory() is None
    finally:
        os.environ["PATH"] = path
        bigrun.gpu_present, bigrun._jax_gpu_stats = present, stats


def test_online_core_check(tmp):
    fake = tmp / "bin"
    fake.mkdir()
    journal = fake / "journalctl"
    journal.write_text("#!/bin/sh\necho 'kernel: mce: [Hardware Error]: CPU 26: Machine Check'\n"
                       "echo 'kernel: mce: MC1_STATUS[Over|UE|PCC]: 0xbea'\n"
                       "echo 'kernel: mce: MC12_STATUS: 0xb'\necho 'kernel: nothing'\n"
                       "echo 'kernel: mce: [Hardware Error]: CPU 0: Machine Check: 0 Bank 27: faa'\n")
    journal.chmod(journal.stat().st_mode | stat.S_IEXEC)
    offline = tmp / "offline"
    path, cpu_offline = os.environ["PATH"], bigrun.CPU_OFFLINE
    os.environ["PATH"] = f"{fake}{os.pathsep}{path}"
    bigrun.CPU_OFFLINE = offline
    try:
        offline.write_text("10,26\n")
        assert bigrun.online_core_check() == {"offline_cpus": [10, 26], "machine_checks": 3}
        offline.write_text("3-5,9\n")
        assert bigrun.online_core_check()["offline_cpus"] == [3, 4, 5, 9]
        offline.write_text("\n")  # the usual machine: everything online
        assert bigrun.online_core_check()["offline_cpus"] == []
        have = bigrun.preflight()
        assert have["offline_cpus"] == [] and have["machine_checks"] == 3

        printed = subprocess.run([sys.executable, "-c", textwrap.dedent(f"""
            from pathlib import Path
            from pymc_challenges import bigrun
            bigrun.CPU_OFFLINE = Path({str(offline)!r})
            bigrun.run_level(needs={{}}, env={{"PYMC_CHALLENGES_RUN": "full",
                                              "PYMC_CHALLENGES_CONFIRM_FULL": "yes"}})""")],
                                 capture_output=True, text=True, check=True)
        assert "offline CPUs []" in printed.stdout and "this boot 3" in printed.stdout

        # Not knowing is not zero.
        offline.unlink()
        journal.write_text("#!/bin/sh\necho '-- No entries --'\n")
        assert bigrun.online_core_check() == {"offline_cpus": None, "machine_checks": None}
        journal.write_text("#!/bin/sh\nexit 1\n")
        assert bigrun.online_core_check()["machine_checks"] is None
        journal.unlink()
        os.environ["PATH"] = str(fake)
        assert bigrun.online_core_check()["machine_checks"] is None
    finally:
        os.environ["PATH"] = path
        bigrun.CPU_OFFLINE = cpu_offline


def test_gpu_lock(tmp):
    lock = tmp / "deep" / "gpu.lock"
    with bigrun.gpu_lock(timeout=0, path=lock) as held:
        assert held == lock and f"pid {os.getpid()}" in lock.read_text()
    assert lock.read_text() == ""  # released, and the file stays: removing it would race

    code = f"""
        import os, sys
        from pymc_challenges.bigrun import gpu_lock
        with gpu_lock(path={str(lock)!r}):
            print("held", flush=True)
            sys.stdin.read()
            os._exit(9)  # dies holding it, never reaching the release"""
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert holder.stdout.readline().strip() == "held"
    raises(TimeoutError, lambda: bigrun.gpu_lock(timeout=0, path=lock).__enter__(),
           match=f"pid {holder.pid}")
    t0 = time.perf_counter()
    raises(TimeoutError, lambda: bigrun.gpu_lock(timeout=0.2, path=lock, poll=0.05).__enter__())
    assert time.perf_counter() - t0 >= 0.2
    holder.stdin.close()
    assert holder.wait() == 9
    with bigrun.gpu_lock(timeout=5, path=lock):  # the kernel dropped a dead holder's lock
        assert f"pid {os.getpid()}" in lock.read_text()
    assert bigrun.GPU_LOCK.parent == bigrun.CACHE_DIR
    assert bigrun.CACHE_DIR == Path.home() / ".cache" / "pymc-challenges"


def test_restore(tmp):
    provenance = dict(git_hash="abc", versions={}, data_fingerprint="d", hardware={}, started="now")

    def full_run(name, value):
        store = Store(name, root=tmp)
        store.save_array("draws", np.full(4, value))
        store.save_chunk("sweep", 0, {"x": np.full(2, value)})
        store.save_json("summary", {"ess": value})
        store.mark_done("sample")
        store.write_manifest(**provenance)
        return store

    run = full_run("T6", 1.0)
    backup = tmp / "backup"
    shutil.copytree(run.path, backup)
    for path in [*run.path.rglob("*.npy*"), *run.path.rglob("*.npz*")]:  # as .gitignore leaves it
        path.unlink()

    # A fresh clone: JSON is here, arrays are not, and every way in says what to do.
    assert run.load_json("summary") == {"ess": 1.0}
    assert sorted(run.missing()) == ["chunks/sweep/00000000.npz", "draws.npy"]
    for load in (lambda: run.load_array("draws"), lambda: run.load_chunk("sweep", 0),
                 lambda: run.gather("sweep", 1), lambda: run.restore(tmp / "no-copy")):
        error = raises(ResultsMissing, load, match="PYMC_CHALLENGES_RUN=small")
        assert not isinstance(error, FileNotFoundError)
    raises(ResultsMissing, run.restore, tmp / "no-copy", match="2 stored files")
    assert "restore()" in str(raises(ResultsMissing, run.load_array, "draws"))

    # Arrays of another run are refused, though their own sidecars are fine.
    other = full_run("T7", 2.0)
    error = raises(ResultsMissing, run.restore, other.path, match="Refused")
    assert "draws.npy" in str(error) and not (run.path / "draws.npy").exists()

    # A copy corrupted in storage is refused too.
    corrupt = tmp / "corrupt"
    shutil.copytree(backup, corrupt)
    (corrupt / "draws.npy").write_bytes(b"torn")
    raises(ResultsMissing, run.restore, corrupt, match="draws.npy")

    summary = (run.path / "summary.json").read_bytes()
    out = run.restore(backup)
    assert out["restored"] == ["draws.npy"]  # the good file of the corrupt copy was kept
    assert out["refused"] == [] and run.missing() == [] and run.verify() == []
    assert run.load_array("draws").tolist() == [1.0] * 4
    assert (run.path / "summary.json").read_bytes() == summary
    assert run.restore(tmp / "no-copy") == {"restored": [], "refused": []}  # all here: fine
    assert run.restore(backup) == {"restored": [], "refused": []}  # intact files are left alone

    # A local file gone bad is replaced from a good copy.
    (run.path / "draws.npy").write_bytes(b"torn")
    assert run.restore(backup)["restored"] == ["draws.npy"] and run.verify() == []

    # No results at all: the manifest comes from the copy, or the message says it is absent.
    fresh = Store("T8", root=tmp)
    raises(ResultsMissing, fresh.restore, tmp / "no-copy", match="manifest.json")
    assert sorted(fresh.restore(backup)["restored"]) == [
        "chunks/sweep/00000000.npz", "draws.npy", "manifest.json", "summary.json"]
    assert fresh.verify() == [] and fresh.load_chunk("sweep", 0)["x"].tolist() == [1.0, 1.0]
    assert bigrun.RESTORE_DIR == bigrun.CACHE_DIR / "results"
    assert (Store("T9", root=tmp).missing()) == []  # no manifest, nothing known to be missing


def test_atomic_save_and_load(tmp):
    store = Store("T1", root=tmp)
    a = np.arange(12.0, dtype=np.float32).reshape(3, 4)
    digest = store.save_array("draws", a)
    b = store.load_array("draws")
    assert b.dtype == np.float32 and np.array_equal(a, b)
    assert np.array_equal(store.load_array("draws", mmap=True), a)
    assert digest == bigrun.sha256_file(store.path / "draws.npy")
    assert stat.S_IMODE((store.path / "draws.npy").stat().st_mode) == 0o644

    value = {"ess": np.float64(1.5), "n": np.int64(3), "grid": np.arange(2), "path": Path("a")}
    store.save_json("summary", value)
    assert store.load_json("summary") == {"ess": 1.5, "n": 3, "grid": [0, 1], "path": "a"}

    # A writer that dies mid-write leaves the old file and no temp file.
    target = store.path / "draws.npy"
    before = target.read_bytes()
    def dies(f):
        f.write(b"partial")
        raise OSError("disk full")
    raises(OSError, bigrun.atomic_write, target, dies, match="disk full")
    assert target.read_bytes() == before
    assert not list(store.path.glob(f".*{bigrun.TMP_SUFFIX}"))

    # A process killed mid-write leaves a stale temp file, which the next Store removes.
    code = f"""
        import os
        from pymc_challenges.bigrun import atomic_write
        def dies(f):
            f.write(b"partial"); f.flush(); os._exit(9)
        atomic_write({str(target)!r}, dies)"""
    assert subprocess.run([sys.executable, "-c", textwrap.dedent(code)]).returncode == 9
    assert target.read_bytes() == before
    temps = list(store.path.glob(f".*{bigrun.TMP_SUFFIX}"))
    assert len(temps) == 1
    Store("T1", root=tmp)
    assert temps[0].exists()  # could be a live writer
    os.utime(temps[0], (time.time() - 2 * bigrun.STALE_TMP_S,) * 2)
    Store("T1", root=tmp)
    assert not temps[0].exists()
    assert not store.verify()  # the temp file was never a result

    raises(TypeError, store.save_array, "objects", np.array([{}, None], dtype=object))
    raises(ValueError, store.save_array, "../escape", a)
    raises(ValueError, Store, "../escape", root=tmp)
    raises(ValueError, store.save_json, "manifest", {})
    raises(ResultsMissing, store.load_array, "absent", match="PYMC_CHALLENGES_RUN")


def test_checksums(tmp):
    store = Store("T2", root=tmp)
    store.save_array("x", np.arange(100))
    store.save_json("y", {"a": 1})
    assert store.verify() == []
    if shutil.which("sha256sum"):
        subprocess.run(["sha256sum", "-c", "--quiet", "x.npy.sha256"], cwd=store.path, check=True)

    path = store.path / "x.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    raises(ChecksumError, store.load_array, "x", match="checksum")
    assert store.load_array("x", check=False)[-1] != 99  # the escape hatch does not verify
    assert any("x.npy" in p for p in store.verify())

    store.save_array("x", np.arange(100))
    (store.path / "x.npy.sha256").unlink()  # the writer died before the sidecar
    raises(ChecksumError, store.load_array, "x")
    store.save_array("x", np.arange(100))

    provenance = dict(git_hash="abc123", versions={"pymc": "6.3.2"}, data_fingerprint="d" * 64,
                      hardware={"gpu": "RTX 3090 Ti"}, started="2026-11-01T10:00:00Z")
    store.write_manifest(**provenance, finished="2026-11-01T11:00:00Z", note="test")
    manifest = store.manifest()
    assert manifest["git_hash"] == "abc123" and manifest["note"] == "test"
    assert manifest["versions"] == {"pymc": "6.3.2"} and manifest["data_fingerprint"] == "d" * 64
    assert manifest["hardware"] == {"gpu": "RTX 3090 Ti"}
    assert manifest["started"].startswith("2026") and manifest["finished"].endswith("Z")
    assert set(manifest["files"]) == {"x.npy", "y.json"}
    assert store.verify() == []
    raises(TypeError, store.write_manifest, git_hash="abc")  # provenance is not optional
    store.save_array("x", np.zeros(3))
    assert any("changed since" in p for p in store.verify())
    (store.path / "y.json").unlink()
    (store.path / "y.json.sha256").unlink()
    assert any("missing" in p for p in store.verify())
    (store.path / "manifest.json").write_text("torn")
    assert any("manifest.json" in p for p in store.verify())  # reported, does not crash verify

    assert bigrun.git_hash(tmp) is None
    assert bigrun.library_versions("numpy") == {"numpy": np.__version__}


def test_chunks_and_resume(tmp):
    root = str(tmp)
    worker = f"""
        import sys
        import numpy as np
        from pymc_challenges.bigrun import Store
        store = Store("T3", root={root!r})
        for i in store.pending("sweep", 6):
            store.save_chunk("sweep", i, {{"x": np.full((2, 3), float(i)), "i": np.array([i])}})
            print(i, flush=True)
            if i == 2 and len(sys.argv) > 1:
                sys.exit(7)  # a crash partway through"""
    code = textwrap.dedent(worker)
    run = subprocess.run([sys.executable, "-c", code, "crash"], capture_output=True, text=True)
    assert run.returncode == 7 and run.stdout.split() == ["0", "1", "2"], run

    store = Store("T3", root=tmp)
    assert store.pending("sweep", 6) == [3, 4, 5]
    raises(ResultsMissing, store.gather, "sweep", 6, match="3 of 6")

    # A chunk left corrupt by the crash is redone along with the rest.
    with open(store._chunk("sweep", 1), "r+b") as f:
        f.seek(40)
        f.write(b"\xff\xff\xff\xff")
    assert store.pending("sweep", 6) == [1, 3, 4, 5]
    raises(ChecksumError, store.load_chunk, "sweep", 1)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout.split() == ["1", "3", "4", "5"], run

    assert store.pending("sweep", 6) == []
    out = store.gather("sweep", 6)
    assert out["x"].shape == (12, 3) and out["i"].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.array_equal(out["x"][::2, 0], np.arange(6.0))
    assert store.verify() == []

    # A job pool: workers write different chunks of one store at once, each opening the store.
    pool = """
        import sys
        import numpy as np
        from pymc_challenges.bigrun import Store
        for i in range(int(sys.argv[1]), 30, 3):
            Store("T3", root=sys.argv[2]).save_chunk("pool", i, {"x": np.full(1000, float(i))})"""
    workers = [subprocess.Popen([sys.executable, "-c", textwrap.dedent(pool), str(k), root])
               for k in range(3)]
    assert [w.wait() for w in workers] == [0, 0, 0]
    assert store.pending("pool", 30) == []
    assert store.gather("pool", 30)["x"][::1000].tolist() == list(map(float, range(30)))

    # Rolling state: only the last two survive, and a corrupt newest falls back to the previous.
    assert store.resume("state") == (None, None)
    for i in range(5):
        store.save_chunk("state", i, {"q": np.array([i, i + 1.0])}, keep=2)
    assert [p.stem for p in sorted(store._chunk("state", 0).parent.glob("*.npz"))] == ["00000003", "00000004"]
    assert not store._chunk("state", 0).with_suffix(".npz.sha256").exists()
    index, state = store.resume("state")
    assert index == 4 and state["q"].tolist() == [4.0, 5.0]
    store._chunk("state", 4).write_bytes(b"torn")
    index, state = store.resume("state")
    assert index == 3 and state["q"].tolist() == [3.0, 4.0]


def test_stages(tmp):
    order = ["pilot", "warmup", "sample", "summaries"]
    store = Store("T4", root=tmp)
    assert store.next_stage(order) == "pilot" and store.stages() == []
    store.mark_done("pilot", seconds=12.5)
    store.mark_done("warmup", ess=np.float64(3.0))

    again = Store("T4", root=tmp)  # as after a crash: a new process finds the stage on disk
    assert again.done("warmup") and not again.done("sample")
    assert again.next_stage(order) == "sample"
    assert again.stages()[0]["seconds"] == 12.5 and again.stages()[1]["ess"] == 3.0
    again.mark_done("warmup")  # marking twice leaves one record
    assert [s["stage"] for s in again.stages()] == ["pilot", "warmup"]
    for stage in order:
        again.mark_done(stage)
    assert again.next_stage(order) is None
    assert again.verify() == []


def test_banner(tmp):
    values = dict(wall_time=measured(2.1, "h"), peak_ram=measured(38, "GB"),
                  peak_vram=measured(17.5, "GB"), disk=estimated(40, "GB"), download=measured(3.8, "GB"),
                  precision=measured("float32"), chains=measured(1024),
                  hardware=measured("RTX 3090 Ti, 15 cores"), build_date=measured("2026-11-01"),
                  git_hash=measured("abc123"), smaller_machine=extrapolated("12 h | on a 12 GB GPU"))
    table = banner(**values)
    rows = table.splitlines()
    assert len(rows) == 13
    for row, title in zip(rows[2:], bigrun._ROWS.values(), strict=True):
        assert row.startswith(f"| {title} |") and row.split(" | ")[-1].rstrip(" |") in bigrun.LABELS
    assert "| Disk | 40 GB | estimated |" in table and "12 h \\| on" in table
    assert table._repr_markdown_() == str(table)
    assert "not from this machine" in banner(level="replay", **values)

    store = Store("T5", root=tmp)  # measured numbers are stored, not retyped
    store.save_json("banner", values)
    assert banner(**store.load_json("banner")) == table

    raises(TypeError, banner, **{**values, "disk": "40 GB"}, match="disk must be labelled")
    raises(TypeError, banner, **{**values, "disk": {"value": 1, "label": "guess"}}, match="labelled")
    for label, wrap in zip(bigrun.LABELS, (measured, estimated, extrapolated), strict=True):
        assert wrap(1, "h") == {"value": 1, "unit": "h", "label": label}
        assert f"| Disk | 1 h | {label} |" in banner(**{**values, "disk": wrap(1, "h")})
    error = raises(TypeError, banner, **{**values, "disk": {"value": 1, "label": "assumed"}})
    assert all(label in str(error) for label in bigrun.LABELS)  # the message names all three
    raises(TypeError, banner, **{k: v for k, v in values.items() if k != "chains"}, match="chains")
    raises(TypeError, banner, **values, typo=measured(1), match="typo")


def test_time_call_without_jax():
    code = """
        import sys
        import numpy as np
        from pymc_challenges.bigrun import time_call
        a = np.eye(50) * 2.0
        row = time_call(np.linalg.cholesky, a, repeat=2)
        assert row["run_ms"] > 0 and row["first_s"] > 0 and row["repeat"] == 2 and row["finite"]
        assert not time_call(lambda: {"x": [np.array([np.nan])]}, repeat=1)["finite"]
        assert time_call(lambda: "text", repeat=1)["finite"]
        assert time_call(lambda x, y=1: x + y, 1, y=2, repeat=1)["finite"]
        sys.exit("jax" in sys.modules)"""
    assert subprocess.run([sys.executable, "-c", textwrap.dedent(code)]).returncode == 0
    raises(ValueError, bigrun.time_call, lambda: None, match="return its result")
    raises(ValueError, bigrun.time_call, len, "a", repeat=0, match="repeat")


def test_time_call_leaves_a_lazy_jax_stub_alone():
    code = """\
        import importlib.util, sys
        spec = importlib.util.find_spec("jax")
        spec.loader = importlib.util.LazyLoader(spec.loader)
        sys.modules["jax"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["jax"])  # arviz leaves a stub like this one
        import numpy as np
        from pymc_challenges.bigrun import time_call
        assert time_call(np.eye, 3, repeat=1)["finite"]
        sys.exit("jax.numpy" in sys.modules)"""
    assert subprocess.run([sys.executable, "-c", textwrap.dedent(code)]).returncode == 0


def test_find_stragglers():
    rng = np.random.default_rng(1)
    logp = -1000 + rng.normal(size=32)
    hypers = rng.normal(size=(32, 3))
    assert find_stragglers(logp, hypers)["n_bad"] == 0

    stuck = logp.copy()
    stuck[5] = -1400  # far below the ensemble
    out = find_stragglers(stuck, hypers)
    assert out["bad"].tolist() == out["stuck"].tolist() == [i == 5 for i in range(32)]
    assert not out["check_mode"].any() and out["worst_z"] > 6 and out["z"].shape == (32, 4)

    # Same density, a far hyperparameter: perhaps a mode, so it is not for culling.
    mode = hypers.copy()
    mode[9, 2] = 50.0
    out = find_stragglers(logp, mode)
    assert out["check_mode"][9] and not out["stuck"].any()
    high = logp.copy()
    high[3] = -500  # far above the ensemble is not stuck either
    assert find_stragglers(high)["check_mode"][3]

    nan = logp.copy()
    nan[0], nan[1] = np.nan, -np.inf
    out = find_stragglers(nan, hypers)
    assert out["stuck"][:2].all() and out["n_bad"] == 2 and np.isfinite(out["z"][2:]).all()

    # Most chains identical: MAD is 0, where E86's version gave 0/0.
    frozen = np.full(16, -100.0)
    assert find_stragglers(frozen)["n_bad"] == 0
    frozen[4] = -130.0
    out = find_stragglers(frozen)
    assert out["bad"].tolist() == [i == 4 for i in range(16)] and np.isfinite(out["z"]).all()

    allnan = find_stragglers(np.full(8, np.nan))
    assert allnan["stuck"].all()
    raises(ValueError, find_stragglers, np.zeros(4), match="at least 8")


def test_pilots():
    t0 = time.perf_counter()
    import jax
    import jax.numpy as jnp

    assert not bigrun.gpu_present()
    x64 = jax.config.jax_enable_x64
    logdensity = lambda q: -0.5 * jnp.sum(q**2)
    q0 = jnp.ones(5, jnp.float32)

    rows = bigrun.time_gradient(logdensity, q0, chains=(1, 4), repeat=2)
    assert [r["chains"] for r in rows] == [1, 4]
    assert all(r["run_ms"] > 0 and r["compile_s"] > 0 and r["finite"] for r in rows)
    assert rows[1]["ms_per_chain"] == rows[1]["run_ms"] / 4 and rows[0]["dtype"] == "float32"
    assert not bigrun.time_gradient(lambda q: jnp.sum(jnp.sqrt(q - 9)), q0, (2,), 1)[0]["finite"]
    raises(ValueError, bigrun.time_gradient, logdensity, jnp.ones((2, 2)))

    # time_call waits for JAX work, and for pytrees, keys and plain NumPy results alongside it.
    work = jax.jit(lambda x: (x @ x.T, {"key": jax.random.key(0)}))
    t = bigrun.time_call(work, jnp.ones((64, 64)), repeat=2)
    assert t["first_s"] > 0 and t["run_ms"] > 0 and t["finite"]
    assert not bigrun.time_call(lambda x: jnp.log(-x), jnp.ones(3), repeat=1)["finite"]
    assert bigrun.time_call(np.ones, 3, repeat=1)["finite"]
    raises(ValueError, bigrun.time_call, lambda: None)

    # The private JAX call gpu_memory relies on, and that a process on the CPU reports no GPU stats.
    from jax._src.xla_bridge import backends_are_initialized
    assert backends_are_initialized()
    present = bigrun.gpu_present
    bigrun.gpu_present = lambda: True
    try:
        assert bigrun._jax_gpu_stats() is None
    finally:
        bigrun.gpu_present = present

    water = bigrun.vram_high_water()
    assert water["peak_gb"] is None and "cpu" in water["device"].lower()  # never a fake VRAM number

    def lp(q):  # a sum of offsets of very different sizes loses digits in float32
        return jnp.sum(q) + 1e7 + jnp.sum(jnp.cos(q) * 1e-2)

    draws = np.random.default_rng(2).normal(size=(50, 4)) * 1000
    out = bigrun.precision_check(lp, lp, draws)
    assert out["n"] == 50 and out["sd"] > 0.01 and out["max_abs"] >= abs(out["mean"])
    assert out["ok"] == (out["sd"] <= 0.1) and out["nonfinite32"] == 0
    assert not bigrun.precision_check(lp, lp, draws, tol=1e-6)["ok"]
    assert jax.config.jax_enable_x64 == x64

    # float32 overflow in a place float64 is fine: exp(100).
    over = bigrun.precision_check(lambda q: jnp.exp(q[0]), lambda q: jnp.exp(q[0]),
                                  np.array([[100.0], [1.0]]))
    assert over["nonfinite32"] == 1 and over["compared"] == 1 and not over["ok"]
    broken = bigrun.precision_check(lambda q: q[0], lambda q: q[0] * jnp.nan, np.ones((3, 1)))
    assert broken["compared"] == 0 and not broken["ok"]  # nothing compared is not a pass

    # A "float64" function that is really float32 would otherwise compare a result with itself.
    fake64 = lambda q: jnp.sum(q.astype(jnp.float32))
    raises(TypeError, bigrun.precision_check, lp, fake64, draws, match="own precision")
    assert jax.config.jax_enable_x64 == x64

    start = np.random.default_rng(3).normal(size=(16, 3)).astype(np.float32)

    @jax.jit
    def warmup(key, q, steps):
        return q + 0.01 * jax.random.normal(key, q.shape, q.dtype) * steps

    probe = bigrun.short_warmup_probe(lambda key, q, steps: (warmup(key, q, steps), {"step": 0.5}),
                                      start, logdensity, steps=4, n_hyper=2)
    assert probe["moved"] == 1 and probe["tuned"] == {"step": 0.5} and probe["s_per_step"] > 0
    assert probe["stragglers"]["z"].shape == (16, 3)
    frozen = bigrun.short_warmup_probe(lambda key, q, steps: (jnp.asarray(q), {}), start, logdensity)
    assert frozen["moved"] == 0
    print(f"  (pilots took {time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    tests = {k: v for k, v in sorted(globals().items()) if k.startswith("test_")}
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        for name, test in tests.items():
            t0 = time.perf_counter()
            tmp = Path(directory) / name
            tmp.mkdir()
            test(tmp) if test.__code__.co_argcount else test()
            print(f"ok {name} ({time.perf_counter() - t0:.2f} s)")
    print(f"{len(tests)} tests passed in {time.perf_counter() - t_start:.1f} s")
