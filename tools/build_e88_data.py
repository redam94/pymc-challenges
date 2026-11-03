"""Build the E88 data: full dSTORM recordings of Unc-13 (Mrestani 2022, Zenodo 7328805, CC BY 4.0).

    .venv/bin/python tools/build_e88_data.py selftest          # no network, no real data
    .venv/bin/python tools/build_e88_data.py check             # a few range requests, < 300 KB
    .venv/bin/python tools/build_e88_data.py mask --src DIR    # PNG masks already in DIR -> polygons
    PYMC_CHALLENGES_CONFIRM_FULL=yes .venv/bin/python tools/build_e88_data.py fetch
    PYMC_CHALLENGES_CONFIRM_FULL=yes .venv/bin/python tools/build_e88_data.py stats --workers 14
    .venv/bin/python tools/build_e88_data.py pack              # needs a FULL run's tables: tools/build_e88_pack.py
    .venv/bin/python tools/build_e88_data.py p1                # pilot P1, cached crop, no download

Plan: plans/large/E88.md. Status: every command has been run on the real TIFFs, on 2026-11-02. `pack` now calls
tools/build_e88_pack.py; the `pack` function here is the stats-only version, used by selftest alone.

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
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import erf, expit, gammaln

from pymc_challenges import bigrun

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("PYMC_CHALLENGES_RAW", Path.home() / ".cache/pymc-challenges/e88"))
OUT = ROOT / "data" / "dannhauser2022_e88_summary.npz"
BASE = "https://zenodo.org/api/records/7328805/files"
UA = {"User-Agent": "pymc-challenges"}
ATTRIBUTION = ("Derived from Mrestani (2022), Example raw dSTORM data of Brp (Alexa Fluor532) and Unc-13 "
               "(Alexa Fluor647), Zenodo doi:10.5281/zenodo.7328805, CC BY 4.0, "
               "https://creativecommons.org/licenses/by/4.0/. Paper: Dannhaeuser et al. (2022), Frontiers in Cellular "
               "Neuroscience 16:1074304. Changed: two recordings reduced to a few raw frames and windows, audit patches, "
               "background and blob statistics, the authors' masks on the 200 px grid, and renders, FRC curves, drift "
               "estimates and paired-test quantiles from a local analysis, with simulated lineup panels added. "
               "No endorsement by the authors is implied.")

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


# Pilot P1: E58's classical fitter, copied from notebook_src/examples/E58_storm_localization_microscopy.py
# (a notebook cannot be imported). Keep it in step with E58; the check is that p1 reproduces E58's blob
# count and K counts on the crop.

SQ2, SQ2PI = np.sqrt(2.0), np.sqrt(2 * np.pi)
PATCH = 11
FOUR = np.array([[[0, 0, 0]] * 3, [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0]] * 3])


def pixel_gauss(u, sig, n):
    i = np.arange(n)
    lo = (i - 0.5 - u[..., None]) / sig[..., None]
    hi = (i + 0.5 - u[..., None]) / sig[..., None]
    E = 0.5 * (erf(hi / SQ2) - erf(lo / SQ2))
    glo, ghi = np.exp(-0.5 * lo**2) / SQ2PI, np.exp(-0.5 * hi**2) / SQ2PI
    return E, (glo - ghi) / sig[..., None], (lo * glo - hi * ghi) / sig[..., None]


def model_and_jacobian(th, K, P, jacobian=True):
    B = th.shape[0]
    b, s = np.exp(th[:, 0]), np.exp(th[:, 1])
    mu = np.repeat(b[:, None, None], P, 1).repeat(P, 2)
    J = np.zeros((B, P, P, 2 + 3 * K)) if jacobian else None
    if jacobian:
        J[..., 0] = b[:, None, None]
    for k in range(K):
        x, y, N = th[:, 2 + 3 * k], th[:, 3 + 3 * k], np.exp(th[:, 4 + 3 * k])
        Ex, dEx, sEx = pixel_gauss(x, s, P)
        Ey, dEy, sEy = pixel_gauss(y, s, P)
        img = Ey[:, :, None] * Ex[:, None, :]
        mu = mu + N[:, None, None] * img
        if jacobian:
            Nk = N[:, None, None]
            J[..., 2 + 3 * k] = Nk * Ey[:, :, None] * dEx[:, None, :]
            J[..., 3 + 3 * k] = Nk * dEy[:, :, None] * Ex[:, None, :]
            J[..., 4 + 3 * k] = Nk * img
            J[..., 1] += Nk * s[:, None, None] * (sEy[:, :, None] * Ex[:, None, :]
                                                  + Ey[:, :, None] * sEx[:, None, :])
    mu = mu.reshape(B, -1)
    return (mu, J.reshape(B, P * P, -1)) if jacobian else mu


def detect(z, bg, thr_sd=5.0):
    S = ndi.gaussian_filter(z - bg, sigma=(0, 1.0, 1.0))
    thr = thr_sd * 1.4826 * np.median(np.abs(S - np.median(S)))
    lab, n = ndi.label(S > thr, structure=FOUR)
    return S, lab, n, thr


def blob_patches(z, lab, n, P=PATCH):
    T, H, W = z.shape
    Z, Wm = np.zeros((n, P, P)), np.zeros((n, P, P))
    org, cen = np.zeros((n, 3), int), np.zeros((n, 2))
    st = ndi.generate_binary_structure(2, 1)
    for i, sl in enumerate(ndi.find_objects(lab)):
        t = sl[0].start
        cy, cx = (sl[1].start + sl[1].stop - 1) / 2, (sl[2].start + sl[2].stop - 1) / 2
        r0 = int(np.clip(round(cy) - P // 2, 0, H - P))
        c0 = int(np.clip(round(cx) - P // 2, 0, W - P))
        Z[i] = z[t, r0:r0 + P, c0:c0 + P]
        L = lab[t, r0:r0 + P, c0:c0 + P]
        Wm[i] = ~ndi.binary_dilation((L > 0) & (L != i + 1), st, iterations=2)
        org[i], cen[i] = (t, r0, c0), (cx - c0, cy - r0)
    return Z.reshape(n, -1), Wm.reshape(n, -1), org, cen


class Prior:
    def __init__(self, logb, logs=(0.0, 0.2), logN=(np.log(1500.0), 1.0), centre=None, pos_sd=2.0):
        self.logb, self.logs, self.logN, self.centre, self.pos_sd = logb, logs, logN, centre, pos_sd

    def mean_prec(self, K):
        c = np.zeros((1, 2)) if self.centre is None else np.atleast_2d(self.centre)
        B = c.shape[0]
        m = np.concatenate([np.full((B, 1), self.logb[0]), np.full((B, 1), self.logs[0])]
                           + [np.c_[c, np.full(B, self.logN[0])]] * K, axis=1)
        p = np.array([self.logb[1] ** -2, self.logs[1] ** -2]
                     + [self.pos_sd ** -2, self.pos_sd ** -2, self.logN[1] ** -2] * K)
        return m, p

    def log_norm(self, K):
        return (-0.5 * np.log(2 * np.pi * self.logb[1] ** 2) - 0.5 * np.log(2 * np.pi * self.logs[1] ** 2)
                + K * (-0.5 * np.log(2 * np.pi * self.logN[1] ** 2) - np.log(2 * np.pi * self.pos_sd ** 2)))


def info(Je, wt):
    return np.matmul(np.swapaxes(Je * wt[..., None], 1, 2), Je)


def log_lik(th, z, w, K, P, excess=2.0):
    m = model_and_jacobian(th, K, P, jacobian=False) / excess
    return np.sum(w * (z * np.log(m) - m - gammaln(z + 1)), axis=1)


def fit_batch(th0, z, K, P, prior, w, excess=2.0, iters=40):
    th = th0.copy()
    B, D = th.shape
    pm_, pp = prior.mean_prec(K)
    obj = lambda t: log_lik(t, z, w, K, P, excess) - 0.5 * np.sum(pp * (t - pm_) ** 2, axis=1)
    lp, lam = obj(th), np.full(B, 1e-2)

    def grad_info(t):
        mu, J = model_and_jacobian(t, K, P)
        m, Je = mu / excess, J / excess
        return np.einsum("bpd,bp->bd", Je, w * (z / m - 1.0)), info(Je, w / m)

    for _ in range(iters):
        g, I = grad_info(th)
        g = g - pp * (th - pm_)
        I = I + np.diag(pp)
        damp = lam[:, None, None] * (np.eye(D) * np.diagonal(I, axis1=1, axis2=2)[:, :, None]
                                     + 1e-9 * np.eye(D))
        new = th + np.clip(np.linalg.solve(I + damp, g[..., None])[..., 0], -1.0, 1.0)
        for k in range(K):
            new[:, 2 + 3 * k:4 + 3 * k] = np.clip(new[:, 2 + 3 * k:4 + 3 * k], -0.5, P - 0.5)
        lpn = obj(new)
        ok = np.isfinite(lpn) & (lpn >= lp)
        th[ok], lp[ok] = new[ok], lpn[ok]
        lam = np.clip(np.where(ok, lam * 0.3, lam * 10.0), 1e-6, 1e6)
    return th, lp, grad_info(th)[1] + np.diag(pp)


def safe_sd(I, j):
    v = np.linalg.pinv(I)[:, j, j]
    return np.sqrt(np.where(v > 0, v, np.inf))


def count_emitters(z, P, prior, Kmax, w):
    B = z.shape[0]
    out = []
    th = np.c_[np.log(np.maximum(2 * np.median(z, 1), 1.0)), np.full(B, prior.logs[0])]
    for K in range(Kmax + 1):
        if K == 0:
            starts = [th]
        else:
            prev = out[-1][0]
            resid = ((z - model_and_jacobian(prev, K - 1, P, False) / 2) * w).reshape(B, P, P)
            rs = ndi.uniform_filter(resid, size=(1, 3, 3)).reshape(B, -1)
            yy, xx = np.divmod(rs.argmax(1), P)
            new = np.c_[prev, xx, yy, np.log(np.maximum(36 * rs.max(1), 100.0))]
            starts = [new]
            if K >= 2:
                split = new.copy()
                j = 2 + 3 * prev[:, 4::3].argmax(1)
                rows = np.arange(B)
                split[rows, j] -= 0.5
                split[:, -3] = prev[rows, j] + 0.5
                split[:, -2] = prev[rows, j + 1]
                split[rows, j + 2] -= np.log(2)
                split[:, -1] = prev[rows, j + 2] - np.log(2)
                starts.append(split)
        best = None
        for s0 in starts:
            fitK = fit_batch(s0, z, K, P, prior, w)
            if best is None:
                best = [a.copy() for a in fitK]
            else:
                better = fitK[1] > best[1]
                for a, b_ in zip(best, fitK):
                    a[better] = b_[better]
        thK, lp, I = best
        logZ = lp + prior.log_norm(K) + 0.5 * (2 + 3 * K) * np.log(2 * np.pi) \
            - 0.5 * np.linalg.slogdet(I)[1] + gammaln(K + 1)
        out.append((thK, logZ, I))
    return out


def localisation_table(res, org, K, Kmax):
    """One row per emitter, for a given emitter count K per blob. Columns as in E58's bayes_pipeline."""
    rows = {k: [] for k in ("t", "xy", "sd", "N", "K", "blob")}
    for k_ in range(1, Kmax + 1):
        s = K == k_
        th, I = res[k_][0][s], res[k_][2][s]
        for j in range(k_):
            rows["t"].append(org[s, 0])
            rows["xy"].append(np.c_[th[:, 2 + 3 * j] + org[s, 2], th[:, 3 + 3 * j] + org[s, 1]])
            rows["sd"].append(np.sqrt(0.5 * safe_sd(I, 2 + 3 * j) ** 2 + 0.5 * safe_sd(I, 3 + 3 * j) ** 2))
            rows["N"].append(np.exp(th[:, 4 + 3 * j]))
            rows["K"].append(np.full(s.sum(), k_))
            rows["blob"].append(np.nonzero(s)[0])
    return {k: np.vstack(v) if k == "xy" else np.concatenate(v) for k, v in rows.items()}


def localise(z, logb, Kmax=4):
    """E58's pipeline on a movie of effective counts: tables for the chosen K and for K = 1 everywhere."""
    _, lab, n, _ = detect(z, np.median(z, axis=0))
    Z, Wm, org, cen = blob_patches(z, lab, n)
    res = count_emitters(Z, PATCH, Prior(logb=(logb, 0.5), centre=cen), Kmax, Wm)
    K = np.array([r[1] for r in res]).T.argmax(1)
    return (n, np.bincount(K, minlength=Kmax + 1), localisation_table(res, org, K, Kmax),
            localisation_table(res, org, np.ones(n, int), 1))


# Pilot P1: linking and the scale of z

SD_MAX_NM, EDGE_PX = 30.0, 2.0                  # E58's quality filter; blobs cut by the crop edge
RADIUS_PX, RADII_PX = 1.5, (0.75, 1.0, 1.5, 2.0, 3.0)  # primary radius, fixed before any result was seen
CONTROL_GAP = 200                               # frames: far beyond any continuous blink


def good_rows(loc, side, pixel_nm):
    keep = (loc["sd"] * pixel_nm < SD_MAX_NM) & (loc["xy"].min(1) >= EDGE_PX) \
        & (loc["xy"].max(1) <= side - 1 - EDGE_PX)
    return {k: v[keep] for k, v in loc.items()}


def link_pairs(t, xy, gap, radius):
    """Mutual nearest neighbours between frames f and f + gap, within radius: index arrays (i, j)."""
    frames = {f: np.nonzero(t == f)[0] for f in np.unique(t)}
    out_i, out_j = [], []
    for f, a in frames.items():
        b = frames.get(f + gap)
        if b is None:
            continue
        d = np.hypot(*(xy[a, None] - xy[None, b]).transpose(2, 0, 1))
        near_b, near_a = d.argmin(1), d.argmin(0)
        for ia, ib in enumerate(near_b):
            if near_a[ib] == ia and d[ia, ib] <= radius:
                out_i.append(a[ia])
                out_j.append(b[ib])
    return np.array(out_i, int), np.array(out_j, int)


def tracks(i, j, n):
    """Group pairs that share a localisation: they are not independent, so they resample together."""
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1][i]


def z_scale_nll(p, r2, s2, radius, weight=None):
    """Negative log-likelihood of link displacements inside the disc of ``radius``: a truncated
    2-D normal of per-axis variance c^2 s2, plus a floor a^2 if p has a third value (true links), or
    uniform on the disc (false links)."""
    v = np.exp(2 * p[0]) * s2 + (np.exp(2 * p[2]) if len(p) > 2 else 0.0)
    good = np.exp(-r2 / (2 * v)) / (2 * np.pi * v) / -np.expm1(-radius**2 / (2 * v))
    lp = np.log(expit(p[1]) * good + (1 - expit(p[1])) / (np.pi * radius**2))
    return -np.sum(lp if weight is None else weight * lp)


def fit_z_scale(r2, s2, radius, weight=None, start=(0.1, 2.0)):
    """(c, w): the sd of z among true links, and the share of links that are true."""
    fit = minimize(z_scale_nll, start, (r2, s2, radius, weight), method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-3))
    return np.exp(fit.x[0]), expit(fit.x[1]), fit.x


def fit_floor(d, s2, track, radius, rng, pixel_nm, boots=100):
    """Is the excess a scale on every sd, or a fixed extra per-axis sd in the difference of two frames?"""
    r2 = np.sum(d**2, axis=1)

    def fit(weight, start):
        return minimize(z_scale_nll, start, (r2, s2, radius, weight), method="Nelder-Mead",
                        options=dict(xatol=1e-3, fatol=1e-3, maxiter=600)).x

    x = fit(None, (0.1, 2.0, np.log(0.03)))
    ids, inv = np.unique(track, return_inverse=True)
    draws = np.array([fit(np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv].astype(float), x)
                      for _ in range(boots)])
    q = lambda v: np.quantile(v, [0.025, 0.975]).tolist()
    return dict(c=np.exp(x[0]), c_ci95=q(np.exp(draws[:, 0])), floor_nm=np.exp(x[2]) * pixel_nm,
                floor_ci95_nm=q(np.exp(draws[:, 2]) * pixel_nm), true_share=expit(x[1]))


def z_summary(d, s2, track, radius, rng, boots=400):
    """Mixture estimate of the z sd with a bootstrap over tracks, plus cruder numbers beside it."""
    r2 = np.sum(d**2, axis=1)
    u = d / np.sqrt(s2)[:, None]
    out = dict(pairs=len(d), axis_values=2 * len(d))
    if len(d) < 30:
        return out | dict(note="under 30 pairs: no fit")
    c, w, x = fit_z_scale(r2, s2, radius)
    ids, inv = np.unique(track, return_inverse=True)
    boot = []
    for _ in range(boots):
        draw = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        boot.append(fit_z_scale(r2, s2, radius, draw[inv].astype(float), start=x)[0])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return out | dict(tracks=len(ids), z_sd=c, true_share=w, true_pairs=w * len(d), z_sd_lo=lo, z_sd_hi=hi,
                      halfwidth95=(hi - lo) / 2, boot_se=float(np.std(boot)),
                      z_sd_mad_no_mixture=1.4826 * np.median(np.abs(u)), z_rms_no_mixture=np.sqrt(np.mean(u**2)),
                      mean_dx_px=d[:, 0].mean(), mean_dy_px=d[:, 1].mean(),
                      max_abs_z=np.abs(u).max())


def linked(loc, gap, radius):
    i, j = link_pairs(loc["t"], loc["xy"], gap, radius)
    return (i, j, loc["xy"][j] - loc["xy"][i], loc["sd"][i] ** 2 + loc["sd"][j] ** 2,
            tracks(i, j, len(loc["t"])) if len(i) else np.zeros(0, int))


def simulate_linked_movie(rng, bg_counts, sites, events, follow, sig=1.05, gain=6.93, offset=457.0,
                          scale=13.86, read_sd=5.0):
    """A movie with persistent blinks at resampled real sites, on a flat-in-time real background.

    Each event lasts 1 + geometric frames (stay on with probability ``follow``). Returns effective
    counts and the true emitters (frame, x, y, event). This is E58's forward model: it has no PSF wings,
    no out-of-focus light and no moving background, so it shows what the estimator does when the model
    is right, and nothing about the microscope."""
    T, H = 2000, bg_counts.shape[0]
    mu = np.repeat(2 * bg_counts[None], T, axis=0)
    pick = rng.integers(0, len(sites), events)
    start, length = rng.integers(0, T, events), rng.geometric(1 - follow, events)
    truth = []
    for e in range(events):
        for f in range(start[e], min(start[e] + length[e], T)):
            truth.append((f, sites[pick[e], 0], sites[pick[e], 1], e, sites[pick[e], 2]))
    truth = np.array(truth)
    s = np.full(len(truth), sig)
    Ex, Ey = pixel_gauss(truth[:, 1], s, H)[0], pixel_gauss(truth[:, 2], s, H)[0]
    np.add.at(mu, truth[:, 0].astype(int), truth[:, 4, None, None] * Ey[:, :, None] * Ex[:, None, :])
    n = rng.poisson(mu)
    charge = np.where(n > 0, rng.gamma(np.maximum(n, 1), gain), 0.0)
    adu = np.round(charge + offset + rng.normal(0.0, read_sd, n.shape))
    return np.maximum(np.round((adu - offset) / scale), 0.0), truth


def owners(loc, truth, tol=1.0):
    """The true event behind each localisation (-1: nothing within tol px in that frame)."""
    who, err = np.full(len(loc["t"]), -1), np.full((len(loc["t"]), 2), np.nan)
    for f in np.unique(loc["t"]):
        a, b = np.nonzero(loc["t"] == f)[0], np.nonzero(truth[:, 0] == f)[0]
        if len(b):
            d = loc["xy"][a, None] - truth[b, 1:3][None]
            r = np.hypot(d[..., 0], d[..., 1])
            near = r.argmin(1)
            hit = r[np.arange(len(a)), near] <= tol
            who[a[hit]] = truth[b[near[hit]], 3].astype(int)
            err[a[hit]] = d[np.arange(len(a)), near][hit]
    return who, err


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    return x


def pilot_p1(store, boots=400, seed=88):
    """P1: are there enough linked pairs on the crop, and can the z sd be measured? Writes p1_links."""
    started = time.time()
    raw = np.load(ROOT / "data" / "dannhauser2022_unc13_dstorm_roi.npz")
    frames, pixel_nm = raw["frames"].astype(float), float(raw["pixel_nm"])
    slope, offset = photon_transfer(raw["field_median_adu"], raw["field_diff_var_adu2"])
    z = np.maximum(np.round((frames - offset) / slope), 0.0)
    bg = np.median(z, axis=0)
    logb = np.log(2 * np.median(bg))
    side = z.shape[1]
    rng = np.random.default_rng(seed)

    t0 = time.time()
    blobs, Kcounts, chosen, single = localise(z, logb)
    fit_s = time.time() - t0
    print(f"{blobs} blobs, K counts {Kcounts} ({fit_s:.0f} s); E58 printed 4192 and [15 2022 930 440 785]")
    print(f"photon transfer: slope {slope:.2f}, offset {offset:.1f}", flush=True)

    tables = dict(chosen_K=good_rows(chosen, side, pixel_nm), K1=good_rows(single, side, pixel_nm))
    report = dict(blobs=blobs, K_counts=Kcounts.tolist(), fit_seconds=fit_s, slope=slope, offset=offset,
                  rows_all=dict(chosen_K=len(chosen["t"]), K1=len(single["t"])),
                  rows_filtered={k: len(v["t"]) for k, v in tables.items()},
                  median_sd_nm={k: float(np.median(v["sd"])) * pixel_nm for k, v in tables.items()},
                  filter=dict(sd_max_nm=SD_MAX_NM, edge_px=EDGE_PX), primary_radius_px=RADIUS_PX,
                  control_gap=CONTROL_GAP, bootstrap=dict(boots=boots, unit="track"), seed=seed)

    report["real"] = {}
    for name, loc in tables.items():
        row = {}
        for gap in (1, 2, 3):
            _, _, d, s2, tr = linked(loc, gap, RADIUS_PX)
            row[f"gap{gap}"] = z_summary(d, s2, tr, RADIUS_PX, rng, boots)
            print(name, f"gap {gap}", rounded(row[f"gap{gap}"]), flush=True)
        for radius in RADII_PX:
            _, _, d, s2, tr = linked(loc, 1, radius)
            row[f"gap1_R{radius}"] = z_summary(d, s2, tr, radius, rng, boots // 4)
            control = len(link_pairs(loc["t"], loc["xy"], CONTROL_GAP, radius)[0])
            row[f"gap1_R{radius}"]["control_pairs"] = control
            print(name, f"R {radius}", rounded(row[f"gap1_R{radius}"]), flush=True)
        i, j, d, s2, tr = linked(loc, 1, RADIUS_PX)
        s = np.sqrt(s2)
        cuts = np.quantile(s, [1 / 3, 2 / 3])
        strata = dict(s_low=s <= cuts[0], s_mid=(s > cuts[0]) & (s <= cuts[1]), s_high=s > cuts[1])
        if name == "chosen_K":
            strata |= dict(both_K1=(loc["K"][i] == 1) & (loc["K"][j] == 1),
                           any_K2plus=(loc["K"][i] > 1) | (loc["K"][j] > 1))
        row["strata_gap1"] = {}
        for label, m in strata.items():
            if m.any():
                row["strata_gap1"][label] = z_summary(d[m], s2[m], tr[m], RADIUS_PX, rng, boots // 4) \
                    | dict(s_nm_lo=s[m].min() * pixel_nm, s_nm_hi=s[m].max() * pixel_nm)
                print(name, label, rounded(row["strata_gap1"][label]), flush=True)
        row["floor_gap1"] = fit_floor(d, s2, tr, RADIUS_PX, rng, pixel_nm)
        print(name, "floor", rounded(row["floor_gap1"]), flush=True)
        report["real"][name] = row

    # the estimator on a movie where the truth is known
    loc = tables["chosen_K"]
    sites = np.c_[loc["xy"], loc["N"]]
    follow = 0.55
    events = int(blobs * (1 - follow))
    zs, truth = simulate_linked_movie(rng, bg, sites, events, follow, gain=slope / 2, offset=offset, scale=slope)
    sim_blobs, sim_K, sim_chosen, sim_single = localise(zs, logb)
    report["simulation"] = dict(follow=follow, events=events, blobs=sim_blobs, K_counts=sim_K.tolist())
    for name, table in (("chosen_K", sim_chosen), ("K1", sim_single)):
        sim = good_rows(table, side, pixel_nm)
        who, err = owners(sim, truth)
        i, j, d, s2, tr = linked(sim, 1, RADIUS_PX)
        same = (who[i] == who[j]) & (who[i] >= 0)
        u = d[same] / np.sqrt(s2[same])[:, None]
        hit = who >= 0
        keep = np.abs(u).max(1) < 8                   # true links, without blobs fitted to a neighbour
        row = dict(rows_filtered=len(sim["t"]), unmatched_share=1 - float(hit.mean()),
                   mixture=z_summary(d, s2, tr, RADIUS_PX, rng, boots // 4),
                   floor=fit_floor(d, s2, tr, RADIUS_PX, rng, pixel_nm),
                   true_link_share=float(same.mean()), true_link_pairs=int(same.sum()),
                   z_rms_true_links=float(np.sqrt(np.mean(u**2))),
                   z_rms_true_links_under_8=float(np.sqrt(np.mean(u[keep] ** 2))),
                   z_sd_mad_true_links=float(1.4826 * np.median(np.abs(u))),
                   err_over_sd_rms=float(np.sqrt(np.nanmean((err[hit] / sim["sd"][hit, None]) ** 2))),
                   err_over_sd_mad=float(1.4826 * np.nanmedian(np.abs(err[hit] / sim["sd"][hit, None]))))
        if name == "chosen_K":
            m = (sim["K"][i] == 1) & (sim["K"][j] == 1)
            row["both_K1"] = z_summary(d[m], s2[m], tr[m], RADIUS_PX, rng, boots // 4)
            row["any_K2plus"] = z_summary(d[~m], s2[~m], tr[~m], RADIUS_PX, rng, boots // 4)
        for radius in RADII_PX:
            _, _, dr, s2r, trr = linked(sim, 1, radius)
            row[f"R{radius}"] = z_summary(dr, s2r, trr, radius, rng, boots // 4)
        report["simulation"][name] = row
        print("simulation", name, rounded(row), flush=True)

    primary = report["real"]["chosen_K"]["gap1"]
    verdict = ("no-go" if primary["pairs"] < 300 else
               "go" if primary["pairs"] >= 500 and primary["halfwidth95"] <= 0.05 else "marginal")
    report |= dict(verdict=verdict, seconds=time.time() - started, labels="all measured on the crop")
    store.save_json("p1_links", rounded(report))
    print(f"P1 verdict {verdict}; {report['seconds']:.0f} s", flush=True)
    return report


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
    t = np.array([0, 0, 1, 1, 2, 5])
    xy = np.array([[1.0, 1], [9, 9], [1.2, 1], [30, 30], [1.3, 1.1], [1, 1]])
    i, j = link_pairs(t, xy, 1, 1.5)
    assert sorted(zip(i, j)) == [(0, 2), (2, 4)], "one chain, nothing across the gap or over the radius"
    assert len(set(tracks(i, j, len(t)))) == 1
    rng = np.random.default_rng(1)
    n, radius = 20000, 1.5
    s2 = np.full(n, 0.02)
    good = rng.random(n) < 0.8
    d = rng.normal(0, 1.1 * np.sqrt(s2)[:, None], (n, 2))
    phi, rad = rng.uniform(0, 2 * np.pi, n), radius * np.sqrt(rng.random(n))
    d[~good] = np.c_[rad * np.cos(phi), rad * np.sin(phi)][~good]
    c, w, _ = fit_z_scale(np.sum(d**2, axis=1), s2, radius)
    assert abs(c - 1.1) < 0.02 and abs(w - 0.8) < 0.02, (c, w)
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=("selftest", "check", "mask", "fetch", "stats", "pack", "p1"))
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
    elif args.command == "pack":                 # the notebook's file needs a FULL run's tables: see that script
        import build_e88_pack

        build_e88_pack.main()
    elif args.command == "p1":
        pilot_p1(bigrun.Store("E88"))


if __name__ == "__main__":
    sys.exit(main())
