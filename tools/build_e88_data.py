"""Build the E88 data: full dSTORM recordings of Unc-13 (Mrestani 2022, Zenodo 7328805, CC BY 4.0).

    .venv/bin/python tools/build_e88_data.py selftest          # no network, no real data
    .venv/bin/python tools/build_e88_data.py check             # a few range requests, < 300 KB
    .venv/bin/python tools/build_e88_data.py mask --src DIR    # PNG masks already in DIR -> polygons
    PYMC_CHALLENGES_CONFIRM_FULL=yes .venv/bin/python tools/build_e88_data.py fetch
    PYMC_CHALLENGES_CONFIRM_FULL=yes .venv/bin/python tools/build_e88_data.py stats --workers 14
    .venv/bin/python tools/build_e88_data.py pack

Plan: plans/large/E88.md. Status: selftest, check and mask have been run. fetch, stats and pack have
NOT been run on the real TIFFs (their code paths are covered by selftest on a synthetic file only).

File layout, confirmed by range requests: little-endian classic TIFF, 8-byte header, then for each
frame a 650-byte IFD block followed by 80,000 bytes of uint16 pixels. So frame k starts at
658 + 80650 k and the file is 8 + 15000 x 80650 = 1,209,750,008 bytes. Frame 3000 of recording 7 is
pixel-identical to the first frame of the crop E58 caches, at row 25, column 50.

Raw TIFFs stay outside git in RAW_DIR (default ~/.cache/pymc-challenges/e88, or PYMC_CHALLENGES_RAW).
Only derived summaries go in data/, with the CC BY 4.0 attribution in ATTRIBUTION.

Long runs resume. fetch appends to a .part file and renames after the md5 matches; stats saves each
block of frames as a checksummed chunk under results/E88/ and redoes only the missing ones.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before NumPy: workers must not oversubscribe
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import struct
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pymc_challenges import bigrun

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("PYMC_CHALLENGES_RAW", Path.home() / ".cache/pymc-challenges/e88"))
OUT = ROOT / "data" / "dannhauser2022_e88_summary.npz"
BASE = "https://zenodo.org/api/records/7328805/files"
UA = {"User-Agent": "pymc-challenges"}
ATTRIBUTION = ("Derived from Mrestani (2022), Example raw dSTORM data of Brp (Alexa Fluor532) and Unc-13 "
               "(Alexa Fluor647), Zenodo doi:10.5281/zenodo.7328805, CC BY 4.0. Changed: block "
               "statistics computed, masks converted to polygons.")

N_FRAMES, SIDE = 15000, 200
HEADER, IFD, PIXELS = 8, 650, SIDE * SIDE * 2
STRIDE = IFD + PIXELS
FILE_SIZE = HEADER + N_FRAMES * STRIDE
BLOCK = 500                                     # frames per statistics block
RED_FLOOR, OTHER_CEIL = 150, 80                 # the mask's outline colour

# size and md5 from the Zenodo record, read 2026-11-01. The 532 file is sampled, not fetched by default.
TIFFS = {
    "r7": ("dStorm647_Unc13_7.tif", "dba305d2a4802c5ccf319aac7369330b"),
    "r32": ("dStorm647_Unc13_32.tif", "db08e05711041ace96718614ce83aee9"),
}
MASKS = {"r7": "dStorm647_Unc13_7_mask.png", "r32": "dStorm647_Unc13_32_mask.png"}


def frame_offset(k):
    return HEADER + IFD + STRIDE * k


def ifd_offset(k):
    return HEADER + STRIDE * k


def open_stack(path):
    """The stack as a read-only memmap of (ifd, px); ``stack["px"][k]`` is frame k."""
    if Path(path).stat().st_size != FILE_SIZE:
        raise ValueError(f"{path}: expected {FILE_SIZE} bytes")
    kind = np.dtype([("ifd", f"V{IFD}"), ("px", "<u2", (SIDE, SIDE))])
    return np.memmap(path, dtype=kind, mode="r", offset=HEADER, shape=(N_FRAMES,))


def block_stats(frames):
    """Statistics of one block (T, H, W): E58's photon-transfer inputs, plus the median frame."""
    z = np.asarray(frames, dtype=np.float32)
    diff = np.diff(z, axis=0)
    centre = np.median(diff, axis=0)
    robust_var = (1.4826 * np.median(np.abs(diff - centre), axis=0)) ** 2 / 2
    return {"median": np.median(z, axis=0).astype(np.float32), "diff_var": robust_var.astype(np.float32),
            "max": z.max(axis=0).astype(np.float32), "frame_median": np.median(z, axis=(1, 2))}


def photon_transfer(med, dvar):
    """E58's fit: drop the brightest 10% of pixels, then variance = slope * (median - offset)."""
    med, dvar = med.ravel(), dvar.ravel()
    keep = med < np.percentile(med, 90)
    slope, icpt = np.polyfit(med[keep], dvar[keep], 1)
    return float(slope), float(-icpt / slope)


def _stats_job(job):
    path, block = job
    stack = open_stack(path)
    return block, block_stats(np.asarray(stack["px"][block * BLOCK:(block + 1) * BLOCK]))


def run_stats(store, workers, raw=RAW_DIR, tiffs=TIFFS):
    n = N_FRAMES // BLOCK
    for key, (name, _) in tiffs.items():
        todo = store.pending(f"stats_{key}", n)
        jobs = [(str(Path(raw) / name), b) for b in todo]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for block, arrays in pool.map(_stats_job, jobs):
                store.save_chunk(f"stats_{key}", block, arrays)
        print(f"{key}: {n - len(todo)} blocks were already done, {len(todo)} computed", flush=True)


def pack(store, out=OUT, tiffs=TIFFS, masks=None):
    """Join the chunks into the small file the notebook reads."""
    n = N_FRAMES // BLOCK
    arrays = {"attribution": np.array(ATTRIBUTION), "block_frames": np.array(BLOCK)}
    for key in tiffs:
        s = {k: np.stack([store.load_chunk(f"stats_{key}", i)[k] for i in range(n)])
             for k in ("median", "diff_var", "max")}
        fm = np.concatenate([store.load_chunk(f"stats_{key}", i)["frame_median"] for i in range(n)])
        ptc = np.array([photon_transfer(m, d) for m, d in zip(s["median"], s["diff_var"])])
        arrays |= {f"{key}_median": np.round(s["median"]).astype(np.uint16),
                   f"{key}_max": np.round(s["max"]).astype(np.uint16),
                   f"{key}_frame_median": fm.astype(np.float32),
                   f"{key}_ptc_slope": ptc[:, 0], f"{key}_ptc_offset": ptc[:, 1]}
    arrays |= masks or {}
    bigrun.atomic_write(out, lambda f: np.savez_compressed(f, **arrays))
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


def mask_from_png(path):
    """The authors' mask PNG (a render with a red outline, 10 nm/px) -> a boolean mask of filled regions."""
    from PIL import Image                       # present through matplotlib
    from scipy import ndimage as ndi

    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    red = (rgb[..., 0] > RED_FLOOR) & (rgb[..., 1] < OTHER_CEIL) & (rgb[..., 2] < OTHER_CEIL)
    return ndi.binary_fill_holes(ndi.binary_dilation(red, iterations=3)), red


def mask_to_frame(fine):
    """Share of each camera pixel inside the mask, on the 200 x 200 frame grid."""
    rows = (np.arange(fine.shape[0]) * SIDE / fine.shape[0]).astype(int)
    width = np.bincount(rows, minlength=SIDE)
    total = np.zeros((SIDE, SIDE))
    np.add.at(total, (rows[:, None], rows[None, :]), fine)
    return total / np.outer(width, width)


def run_masks(src):
    out = {}
    for key, name in MASKS.items():
        fine, red = mask_from_png(Path(src) / name)
        share = mask_to_frame(fine)
        out[f"{key}_mask_share"] = share.astype(np.float32)
        print(f"{key}: fine image {fine.shape}, {red.sum()} outline pixels, "
              f"mask covers {(share > 0.5).mean():.3f} of the frame")
    return out


def _get(url, start=None, stop=None):
    headers = dict(UA)
    if start is not None:
        headers["Range"] = f"bytes={start}-{stop - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        data = r.read()
    if start is not None and len(data) != stop - start:
        raise OSError(f"asked for {stop - start} bytes, got {len(data)}")
    return data


def read_ifd(blob, offset):
    """(next IFD offset, {tag: first value}) of the IFD at ``offset`` in ``blob`` (starting there)."""
    count = struct.unpack("<H", blob[:2])[0]
    fmt = {3: "<H", 4: "<I", 11: "<f", 1: "<B"}
    tags = {}
    for i in range(count):
        tag, kind, n = struct.unpack("<HHI", blob[2 + 12 * i:10 + 12 * i])
        if n == 1 and kind in fmt:
            tags[tag] = struct.unpack(fmt[kind], blob[10 + 12 * i:10 + 12 * i + struct.calcsize(fmt[kind])])[0]
    return struct.unpack("<I", blob[2 + 12 * count:6 + 12 * count])[0], tags


def check():
    """Confirm the layout of each TIFF from its first and last IFD. Downloads under 300 KB."""
    for key, (name, _) in TIFFS.items():
        url = f"{BASE}/{name}/content"
        head = _get(url, 0, 8)
        if head[:4] != b"II*\x00" or struct.unpack("<I", head[4:])[0] != HEADER:
            raise ValueError(f"{name}: unexpected header {head!r}")
        for k in (0, N_FRAMES - 1):
            following, tags = read_ifd(_get(url, ifd_offset(k), ifd_offset(k) + IFD), ifd_offset(k))
            want_next = 0 if k == N_FRAMES - 1 else ifd_offset(k + 1)
            if (tags[256], tags[257], tags[258], following) != (SIDE, SIDE, 16, want_next):
                raise ValueError(f"{name} frame {k}: {tags}, next IFD {following}, wanted {want_next}")
            print(f"{key} frame {k}: {tags[256]} x {tags[257]}, {tags[258]} bit, next IFD {following}; "
                  f"exposure {tags[4876]:.4f} s, cycle {tags[4878]:.6f} s")
    crop = np.load(ROOT / "data" / "dannhauser2022_unc13_dstorm_roi.npz")["frames"][0]
    url = f"{BASE}/{TIFFS['r7'][0]}/content"
    frame = np.frombuffer(_get(url, frame_offset(3000), frame_offset(3000) + PIXELS), "<u2")
    same = (frame.reshape(SIDE, SIDE)[25:65, 50:90] == crop).mean()
    print(f"recording 7 frame 3000 against E58's cached crop: {same:.4f} of pixels equal")
    if same != 1:
        raise ValueError("frame offsets do not match the cached crop")


def fetch(key, raw=RAW_DIR):
    """Download one TIFF, resuming from .part. Renamed into place only after the md5 matches."""
    name, md5 = TIFFS[key]
    path, part = Path(raw) / name, Path(raw) / (name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"{name}: already here")
        return path
    url, step = f"{BASE}/{name}/content", 64 * 2**20
    have = part.stat().st_size if part.exists() else 0
    with open(part, "ab") as f:
        while have < FILE_SIZE:
            data = _get(url, have, min(have + step, FILE_SIZE))
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            have += len(data)
            print(f"{name}: {have / FILE_SIZE:.0%}", flush=True)
    digest = hashlib.md5()
    with open(part, "rb") as f:
        while chunk := f.read(2**24):
            digest.update(chunk)
    if digest.hexdigest() != md5:
        part.unlink()
        raise bigrun.ChecksumError(f"{name}: md5 {digest.hexdigest()}, Zenodo says {md5}; removed")
    os.replace(part, path)
    return path


def full_gate(disk_gb=0.0):
    if os.environ.get("PYMC_CHALLENGES_CONFIRM_FULL") != "yes":
        raise bigrun.PreflightError("This touches the full 1.2 GB recordings: "
                                    "set PYMC_CHALLENGES_CONFIRM_FULL=yes to confirm.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    bigrun.preflight(ram_gb=2.0, disk_gb=disk_gb, disk_path=RAW_DIR)


def selftest():
    """The offsets, memmap, chunked stats and pack on a synthetic file, in a temp directory."""
    global N_FRAMES, FILE_SIZE
    saved = N_FRAMES, FILE_SIZE
    N_FRAMES, FILE_SIZE = 2 * BLOCK, HEADER + 2 * BLOCK * STRIDE
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            frames = rng.poisson(50.0, (N_FRAMES, SIDE, SIDE)).astype("<u2") * 14 + 457
            with open(Path(tmp) / "a.tif", "wb") as f:
                f.write(b"II*\x00" + struct.pack("<I", HEADER))
                for frame in frames:
                    f.write(b"\0" * IFD + frame.tobytes())
            stack = open_stack(Path(tmp) / "a.tif")
            assert (stack["px"][7] == frames[7]).all()
            with open(Path(tmp) / "a.tif", "rb") as f:
                f.seek(frame_offset(7))
                assert f.read(PIXELS) == frames[7].tobytes()
            store = bigrun.Store("E88test", root=tmp)
            tiffs = {"r7": ("a.tif", "")}
            run_stats(store, 2, raw=tmp, tiffs=tiffs)
            run_stats(store, 2, raw=tmp, tiffs=tiffs)     # second pass must find nothing to do
            assert store.pending("stats_r7", 2) == []
            out = Path(tmp) / "out.npz"
            pack(store, out=out, tiffs=tiffs)
            with np.load(out) as z:
                assert z["r7_median"].shape == (2, SIDE, SIDE)
                assert abs(z["r7_median"].mean() - (50 * 14 + 457)) < 3
                print(f"synthetic slope {z['r7_ptc_slope'][0]:.1f} (no signal in the variance gradient "
                      "here, so it only proves the code runs)")
    finally:
        N_FRAMES, FILE_SIZE = saved
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=("selftest", "check", "mask", "fetch", "stats", "pack"))
    parser.add_argument("--src", default=str(RAW_DIR), help="folder holding the mask PNGs")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "check":
        check()
    elif args.command == "mask":
        run_masks(args.src)
    elif args.command == "fetch":
        full_gate(disk_gb=2 * FILE_SIZE / 1e9 + 1)
        for key in TIFFS:
            fetch(key)
        for name in MASKS.values():
            if not (RAW_DIR / name).exists():
                data = _get(f"{BASE}/{name}/content")
                bigrun.atomic_write(RAW_DIR / name, lambda f: f.write(data))
    elif args.command == "stats":
        full_gate()
        run_stats(bigrun.Store("E88"), args.workers)
    elif args.command == "pack":
        pack(bigrun.Store("E88"), masks=run_masks(args.src))


if __name__ == "__main__":
    sys.exit(main())
