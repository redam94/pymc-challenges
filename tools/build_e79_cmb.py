"""Build the E79 dataset: flat-sky patches of the Planck 2018 (PR3) SMICA CMB temperature map.

    uv run --no-project --with healpy --with numpy --with truststore \
        python tools/build_e79_cmb.py

healpy is only needed here (pixel geometry and the HEALPix pixel window); the notebook reads the .npz.

The full-resolution maps are large (the SMICA "full" file is 2.0 GB, each half-mission file 0.6 GB,
the common mask 0.2 GB). They are HEALPix NESTED binary tables, so a region of sky is a handful of
contiguous row blocks: we compute which Nside-2048 pixels a patch touches and download only those
rows with HTTP range requests from the IRSA mirror of the Planck Legacy Archive (about 60 MB per
patch in total).

Projection: gnomonic (tangent plane) around the patch centre, N x N pixels of RESO arcmin. Each flat
pixel is the average of SUB x SUB point samples of the HEALPix map (nearest HEALPix pixel), so the
flat pixel carries a square top-hat window on top of the SMICA beam and the HEALPix pixel window.
Array layout: row index increases with galactic latitude b ("origin=lower" in imshow), column index
increases towards DECREASING galactic longitude l (the usual astronomical view of the sky from inside).

* data/planck_smica_patches.npz - for each patch (prefix "lmc_" and "north_"): I (SMICA full-mission
  temperature, uK_CMB), hm1, hm2 (half-mission 1 and 2), inp (SMICA's own inpainted map), mask
  (fraction of each flat pixel's samples that the Planck 2018 common intensity mask keeps; 1 = usable),
  tmask (the same for SMICA's own confidence mask), centre (l, b in degrees); and global: reso_arcmin,
  n, sub, pixwin_2048 (HEALPix Nside-2048 temperature pixel window, ell = 0..4096), beam_fwhm_arcmin.
* data/planck_theory_tt.txt, data/planck_tt_binned.txt - copies of the PLA text files (see data.py).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
IRSA = "https://irsa.ipac.caltech.edu/data/Planck/release_3"
FILES = {
    "full": (f"{IRSA}/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits", 40),
    "hm1": (f"{IRSA}/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_hm1.fits", 12),
    "hm2": (f"{IRSA}/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_hm2.fits", 12),
    "mask": (f"{IRSA}/ancillary-data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits", 4),
}
TEXT = {
    "planck_theory_tt.txt": f"{IRSA}/ancillary-data/cosmoparams/"
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt",
    "planck_tt_binned.txt": f"{IRSA}/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-binned_R3.01.txt",
}
PATCHES = {"lmc": (280.0, -35.0), "north": (60.0, 55.0)}
NSIDE, N, RESO, SUB = 2048, 256, 3.75, 4
PARENT = 16                      # fetch whole Nside-16 parents: 128^2 = 16384 contiguous rows each
UA = {"User-Agent": "pymc-challenges"}

try:  # use the operating system's certificate store (some networks re-sign TLS traffic)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def fetch(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    headers = dict(UA)
    if start is not None:
        headers["Range"] = f"bytes={start}-{stop - 1}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = r.read()
            if start is not None and len(out) != stop - start:
                raise OSError(f"short read {len(out)} of {stop - start}")
            return out
        except OSError as e:  # IRSA occasionally answers 504 on the 2 GB file: retry
            print("  retry", attempt, e)
    raise RuntimeError(url)


def data_offset(url: str) -> int:
    """Byte offset of the first binary-table row: after the primary and extension headers."""
    head = fetch(url, 0, 2880 * 12).decode("latin-1")
    ends = [i for i in range(0, len(head), 80) if head[i:i + 80].rstrip() == "END"]
    return (ends[1] // 2880 + 1) * 2880    # the table starts in the block after the 2nd END card


def patch_pixels(l0: float, b0: float, hp) -> np.ndarray:
    """Nested Nside-2048 pixel index of every sub-sample point, shape (N, SUB, N, SUB)."""
    step = np.deg2rad(RESO / 60) / SUB
    u = (np.arange(N * SUB) - N * SUB / 2 + 0.5) * step
    X = -u[None, :]              # columns: towards decreasing longitude
    Y = u[:, None]               # rows: increasing latitude
    X, Y = np.broadcast_arrays(X, Y)
    lam0, phi0 = np.deg2rad(l0), np.deg2rad(b0)
    rho = np.hypot(X, Y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.arcsin(np.cos(c) * np.sin(phi0) + np.where(rho > 0, Y * np.sin(c) * np.cos(phi0) / rho, 0))
    lam = lam0 + np.arctan2(X * np.sin(c), rho * np.cos(phi0) * np.cos(c) - Y * np.sin(phi0) * np.sin(c))
    pix = hp.ang2pix(NSIDE, np.pi / 2 - phi, lam, nest=True)
    return pix.reshape(N, SUB, N, SUB)


def read_rows(name: str, pix: np.ndarray, columns: list[int]) -> dict[int, np.ndarray]:
    url, width = FILES[name]
    off = data_offset(url)
    per = (NSIDE // PARENT) ** 2
    parents = np.unique(pix // per)
    # merge consecutive parents into single range requests
    runs, start = [], parents[0]
    for a, b in zip(parents[:-1], parents[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, parents[-1]))
    lookup = {c: np.full(pix.shape, np.nan, np.float32) for c in columns}
    for p0, p1 in runs:
        r0, r1 = p0 * per, (p1 + 1) * per
        raw = np.frombuffer(fetch(url, off + r0 * width, off + r1 * width), dtype=">f4")
        raw = raw.reshape(r1 - r0, width // 4)
        sel = (pix >= r0) & (pix < r1)
        for c in columns:
            lookup[c][sel] = raw[pix[sel] - r0, c]
    print(f"  {name}: {len(parents)} parents in {len(runs)} requests")
    return lookup


def main() -> None:
    import healpy as hp

    out = {"reso_arcmin": RESO, "n": N, "sub": SUB, "beam_fwhm_arcmin": 5.0,
           "pixwin_2048": hp.pixwin(NSIDE, lmax=2 * NSIDE).astype(np.float64)}
    for key, (l0, b0) in PATCHES.items():
        print(key, l0, b0)
        pix = patch_pixels(l0, b0, hp)
        full = read_rows("full", pix, [0, 3, 5])        # I_STOKES, TMASK, I_STOKES_INP
        hm1 = read_rows("hm1", pix, [0])
        hm2 = read_rows("hm2", pix, [0])
        mask = read_rows("mask", pix, [0])
        mean = lambda a: a.mean(axis=(1, 3))            # noqa: E731 - average the SUB x SUB samples
        for arr in [full[0], full[5], hm1[0], hm2[0], full[3], mask[0]]:
            assert np.isfinite(arr).all()
        out[f"{key}_I"] = (mean(full[0]) * 1e6).astype(np.float32)
        out[f"{key}_inp"] = (mean(full[5]) * 1e6).astype(np.float32)
        out[f"{key}_hm1"] = (mean(hm1[0]) * 1e6).astype(np.float32)
        out[f"{key}_hm2"] = (mean(hm2[0]) * 1e6).astype(np.float32)
        out[f"{key}_tmask"] = mean(full[3]).astype(np.float32)
        out[f"{key}_mask"] = mean(mask[0]).astype(np.float32)
        out[f"{key}_centre"] = np.array([l0, b0])
        print(f"  masked (common) {1 - (out[f'{key}_mask'] == 1).mean():.3f}, "
              f"rms {out[f'{key}_I'].std():.1f} uK")
    np.savez_compressed(DATA / "planck_smica_patches.npz", **out)
    for fname, url in TEXT.items():
        (DATA / fname).write_bytes(fetch(url))
    print({p.name: p.stat().st_size for p in DATA.glob("planck_*")})


if __name__ == "__main__":
    main()
