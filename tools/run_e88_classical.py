"""E88 pilots P2, P2b and P3: the classical pipeline on the full recordings, and the frames decision.

    .venv/bin/python tools/run_e88_classical.py p3
    .venv/bin/python tools/run_e88_classical.py classical --rec r7 --workers 13
    .venv/bin/python tools/run_e88_classical.py p2
    .venv/bin/python tools/run_e88_classical.py p2b --workers 13

The gates and every definition used here are in results/E88/pilot_gates.md, written before these ran.
Chunks go through bigrun.Store("E88"), so a crash redoes one block of 500 frames. Pin the process with
taskset before starting it; workers inherit the mask. The FULL rung reuses these stages.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before NumPy: workers must not oversubscribe
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_e88_data as b  # noqa: E402

from pymc_challenges import bigrun  # noqa: E402

PIXEL_NM = 127.0
SUB, WINDOW = 100, 100                          # sub-block length; half window of the rolling median
CHUNKS = b.N_FRAMES // b.BLOCK
EXPECT_OFFLINE, EXPECT_MC = [10, 26], 1
GRID_N = (2000, 3000, 4000, 5000, 6000, 7500, 9000, 10500, 12000, 13500, 15000)
FRC_BLOCK, DRAWS, RENDER_NM = 50, 200, 10.0
MIN_EVENTS = 7
FIT_BATCH = 64
PRIMARY = "chosen_K"


def guard():
    """Stop if the machine has rebooted (core 10 back) or logged a new machine check."""
    state = bigrun.online_core_check()
    if state["offline_cpus"] != EXPECT_OFFLINE or state["machine_checks"] != EXPECT_MC:
        raise SystemExit(f"SAFETY STOP: {state}, expected offline {EXPECT_OFFLINE}, checks {EXPECT_MC}")
    return state


def calibration(rec, tag=""):
    """Offsets and slopes per block. P3 failed, so its written action applies: per block, per recording.

    tag "line" is the post-hoc sensitivity run: one line per recording, fitted across blocks. See p3_line."""
    store = bigrun.Store("E88")
    if tag == "line":
        line = store.load_json("p3_line")[rec]
        return dict(offset=[line["offset"]] * CHUNKS, slope=[line["slope"]] * CHUNKS)
    rows = store.load_json("p3_calibration")["recordings"][rec]
    return dict(offset=rows["offsets"], slope=rows["slopes"])


def rolling_background(stack, start, offset, slope):
    """Median of the 201 frames centred on a sub-block, clipped at the ends of the movie."""
    centre = start + SUB // 2
    lo, hi = max(0, centre - WINDOW), min(b.N_FRAMES, centre + WINDOW + 1)
    return np.median(np.maximum(np.round((np.asarray(stack["px"][lo:hi], float) - offset) / slope), 0.0), axis=0)


def classical_job(job):
    """One chunk of 500 frames, as five sub-blocks with their own background and threshold."""
    path, block, offset, slope = job
    stack = b.open_stack(path)
    tables = {"chosen": [], "single": []}
    blobs, seen = [], 0
    for s in range(0, b.BLOCK, SUB):
        start = block * b.BLOCK + s
        z = np.maximum(np.round((np.asarray(stack["px"][start:start + SUB], float) - offset) / slope), 0.0)
        bg = rolling_background(stack, start, offset, slope)
        logb = np.log(2 * np.median(bg))
        _, lab, n, _ = b.detect(z, bg)
        if n == 0:
            continue
        Z, Wm, org, cen = b.blob_patches(z, lab, n)
        # Small batches: each blob's fit is independent of its batch, and arrays of a few MB stay in
        # cache. Batches of 2,500 ran 8x slower with 13 workers than one worker alone (memory bandwidth).
        for i in range(0, n, FIT_BATCH):
            s_ = slice(i, i + FIT_BATCH)
            m = len(Z[s_])
            res = b.count_emitters(Z[s_], b.PATCH, b.Prior(logb=(logb, 0.5), centre=cen[s_]), 4, Wm[s_])
            K = np.array([r[1] for r in res]).T.argmax(1)
            for name, table in (("chosen", b.localisation_table(res, org[s_], K, 4)),
                                ("single", b.localisation_table(res, org[s_], np.ones(m, int), 1))):
                table["t"] = table["t"] + start
                table["blob"] = table["blob"] + seen
                tables[name].append(table)
            blobs.append(np.c_[org[s_, 0] + start, cen[s_, 0] + org[s_, 2], cen[s_, 1] + org[s_, 1], K,
                               np.arange(m) + seen])
            seen += m
    out = {"blobs": np.vstack(blobs)}
    for name, parts in tables.items():
        for key in parts[0]:
            out[f"{name}_{key}"] = np.concatenate([p[key] for p in parts])
    return block, out


def run_classical(rec, workers, tag=""):
    store = bigrun.Store("E88")
    state = guard()
    cal = calibration(rec, tag)
    stage = f"classical_{tag}_{rec}" if tag else f"classical_{rec}"
    path = str(b.RAW_DIR / b.TIFFS[rec][0])
    todo = store.pending(stage, CHUNKS)
    print(f"{stage}: {len(todo)} of {CHUNKS} chunks to do, {workers} workers, {state}", flush=True)
    started, last, done = time.time(), time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(classical_job, (path, i, cal["offset"][i], cal["slope"][i])) for i in todo]
        for f in as_completed(futures):
            block, arrays = f.result()
            store.save_chunk(stage, block, arrays)
            done += 1
            print(f"{rec} block {block}: {len(arrays['blobs'])} blobs, {done}/{len(todo)}, "
                  f"{time.time() - started:.0f} s", flush=True)
            if time.time() - last > 30:
                guard()
                last = time.time()
    seconds = time.time() - started
    store.save_json(f"{stage}_timing", dict(seconds=seconds, chunks=len(todo), workers=workers))
    store.mark_done(stage, seconds=seconds)
    print(f"{rec}: finished in {seconds:.0f} s", flush=True)


def load_rec(rec, tag=""):
    """The chosen-K table, the K = 1 table and the detector's blobs, for one recording."""
    all_ = bigrun.Store("E88").gather(f"classical_{tag}_{rec}" if tag else f"classical_{rec}", CHUNKS)
    # blob ids restart per chunk: make them unique with the chunk's frame range
    def table(prefix):
        t = {k: all_[f"{prefix}_{k}"] for k in ("t", "xy", "sd", "N", "K", "blob")}
        t["blob"] = (t["t"] // b.BLOCK) * 10_000_000 + t["blob"]
        return t
    bl = all_["blobs"]
    blobs = dict(t=bl[:, 0].astype(int), xy=bl[:, 1:3], K=bl[:, 3].astype(int),
                 id=(bl[:, 0].astype(int) // b.BLOCK) * 10_000_000 + bl[:, 4].astype(int))
    return table("chosen"), table("single"), blobs


def peak_shift(cc, nm_per_bin):
    """Sub-bin peak of a centred cross-correlation, by a quadratic through the peak on each axis."""
    y, x = np.unravel_index(cc.argmax(), cc.shape)
    def vertex(a, c, e):
        return 0.5 * (a - e) / (a - 2 * c + e)
    dy = vertex(cc[y - 1, x], cc[y, x], cc[y + 1, x])
    dx = vertex(cc[y, x - 1], cc[y, x], cc[y, x + 1])
    return np.array([(x + dx - cc.shape[1] // 2), (y + dy - cc.shape[0] // 2)]) * nm_per_bin


def render(xy, grid=0.25, side=b.SIDE, sigma=1.0):
    n = int(side / grid)
    h = np.histogram2d(xy[:, 1], xy[:, 0], bins=n, range=[[-0.5, side - 0.5]] * 2)[0]
    return ndi.gaussian_filter(h, sigma)


def cross(a, c):
    return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a - a.mean()) * np.conj(np.fft.rfft2(c - c.mean())), s=a.shape))


def drift_blocks(t, xy, block=1000):
    return np.minimum(t // block, b.N_FRAMES // block - 1)


def estimate(xy, blocks, refxy, grid=0.25):
    """Shift of each time block against a reference render, px. Sign: add it to undo the drift."""
    ref = render(refxy, grid)
    nm = grid                                   # result in px
    shifts = []
    for k in range(blocks.max() + 1):
        shifts.append(-peak_shift(cross(render(xy[blocks == k], grid), ref), nm))
    return np.array(shifts)


def correct(t, xy, shifts, block=1000):
    centres = (np.arange(len(shifts)) + 0.5) * block
    return xy + np.c_[np.interp(t, centres, shifts[:, 0]), np.interp(t, centres, shifts[:, 1])]


def p2(tag=""):
    """Drift on recording 7, cross-validated between odd and even frames, and the P2 gate."""
    store = bigrun.Store("E88")
    guard()
    stage = f"classical_{tag}_r7" if tag else "classical_r7"
    suffix = f"_{tag}" if tag else ""
    chosen, single, blobs = load_rec("r7", tag)
    loc = b.good_rows(chosen, b.SIDE, PIXEL_NM)
    t, xy = loc["t"], loc["xy"]
    odd, even = t % 2 == 1, t % 2 == 0
    blocks = drift_blocks(t, xy)
    d_odd = estimate(xy[odd], blocks[odd], xy[even])
    d_even = estimate(xy[even], blocks[even], xy[odd])
    # residual: both parities corrected with the odd estimate; even re-estimated against odd
    fixed = correct(t, xy, d_odd)
    residual = estimate(fixed[even], blocks[even], fixed[odd])
    resid_nm = float(np.sqrt(np.mean(residual**2)) * PIXEL_NM)
    # the same with each block left out of the reference, because blinks persist across frames
    d_lbo = []
    for k in range(blocks.max() + 1):
        out = blocks != k
        d_lbo.append(estimate(xy[~out], blocks[~out] * 0, xy[out])[0])
    d_lbo = np.array(d_lbo)
    report = dict(
        rows_all=len(chosen["t"]), rows_filtered=len(t), blobs=len(blobs["t"]),
        K_counts=np.bincount(blobs["K"], minlength=5).tolist(),
        drift_odd_nm=(d_odd * PIXEL_NM).round(2).tolist(), drift_even_nm=(d_even * PIXEL_NM).round(2).tolist(),
        drift_leave_block_out_nm=(d_lbo * PIXEL_NM).round(2).tolist(),
        range_nm=np.ptp(d_lbo * PIXEL_NM, axis=0).round(2).tolist(),
        odd_even_rms_nm=float(np.sqrt(np.mean((d_odd - d_even) ** 2)) * PIXEL_NM),
        residual_rms_nm=resid_nm,
        residual_shifts_nm=(residual * PIXEL_NM).round(2).tolist(),
        median_sd_nm=float(np.median(loc["sd"]) * PIXEL_NM),
        share_over_12000_adu_rule=float(np.mean(loc["N"] >= 1730)),
        timing=store.load_json(f"{stage}_timing"))
    store.save_array(f"drift{suffix}_r7", d_lbo)
    store.save_json(f"p2_classical{suffix}_r7", report)
    print({k: v for k, v in report.items() if "drift_" not in k and "shifts" not in k}, flush=True)
    print("leave-block-out drift nm\n", np.round(d_lbo * PIXEL_NM, 1), flush=True)


# P2b

def mask_share(rec):
    fine, _ = b.mask_from_png(b.RAW_DIR / b.MASKS[rec])
    return b.mask_to_frame(fine) > 0.5


def in_mask(xy, mask):
    i = np.clip(np.round(xy).astype(int), 0, b.SIDE - 1)
    return mask[i[:, 1], i[:, 0]]


def merge_events(loc):
    """Merge gap-1 links into events: inverse-variance mean, first frame, combined sd. (px, frames)"""
    i, j = b.link_pairs(loc["t"], loc["xy"], 1, b.RADIUS_PX)
    n = len(loc["t"])
    lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[1]
    w = 1.0 / loc["sd"] ** 2
    m = lab.max() + 1
    sw = np.bincount(lab, w, m)
    xy = np.c_[np.bincount(lab, w * loc["xy"][:, 0], m), np.bincount(lab, w * loc["xy"][:, 1], m)] / sw[:, None]
    first = np.full(m, 10**9)
    np.minimum.at(first, lab, loc["t"])
    return dict(t=first, xy=xy, sd=1 / np.sqrt(sw))


def frc_resolution(a, c, pix_nm):
    """FRC of two renders; resolution in nm at the 1/7 crossing, by rings one Fourier bin wide."""
    Fa, Fc = np.fft.rfft2(a), np.fft.rfft2(c)
    ny, nx = a.shape
    qy = np.fft.fftfreq(ny)[:, None]
    qx = np.fft.rfftfreq(nx)[None, :]
    q = np.hypot(qx, qy)
    width = 1.0 / min(ny, nx)
    ring = np.minimum((q / width).astype(int).ravel(), int(0.5 / width))
    num = np.bincount(ring, (Fa * np.conj(Fc)).real.ravel())
    d1 = np.bincount(ring, (np.abs(Fa) ** 2).ravel())
    d2 = np.bincount(ring, (np.abs(Fc) ** 2).ravel())
    frc = num / np.sqrt(np.maximum(d1 * d2, 1e-30))
    frc = np.convolve(frc, np.ones(5) / 5, mode="same")
    start = int(1 / (2000.0 / pix_nm) / width) + 2   # skip the rings below (2 um)^-1
    below = np.nonzero(frc[start:] < 1 / 7)[0]
    if not len(below):
        return np.nan
    k = start + below[0]
    f1, f2 = frc[k - 1], frc[k]
    ring_at = k - 1 + (f1 - 1 / 7) / max(f1 - f2, 1e-12)
    return pix_nm / (ring_at * width)


class Field:
    """Events in the mask on the 10 nm grid of their bounding box."""

    def __init__(self, ev, mask):
        keep = in_mask(ev["xy"], mask)
        self.t, xy, self.sd = ev["t"][keep], ev["xy"][keep] * PIXEL_NM, ev["sd"][keep] * PIXEL_NM
        self.lo = xy.min(0) - 40
        self.shape = tuple((np.ceil((xy.max(0) - self.lo + 40) / RENDER_NM).astype(int))[::-1])
        self.xy = xy
        self.window = np.outer(np.hanning(self.shape[0]), np.hanning(self.shape[1]))
        self.tblock = self.t // FRC_BLOCK

    def bins(self, xy):
        ij = ((xy - self.lo) / RENDER_NM).astype(int)
        ij = np.clip(ij, 0, np.array(self.shape[::-1]) - 1)
        return ij[:, 1] * self.shape[1] + ij[:, 0]

    def hist(self, xy, weight):
        return np.bincount(self.bins(xy), weight, self.shape[0] * self.shape[1]).reshape(self.shape)

    def clusters(self, xy, weight):
        h = self.hist(xy, weight)
        cross_ = ndi.generate_binary_structure(2, 1)
        lab, n = ndi.label(ndi.binary_dilation(h > 0, cross_), structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel(), h.ravel(), n + 1)[1:]
        return int(np.sum(sizes >= MIN_EVENTS)), float(sizes.max() / sizes.sum()) if n else 0.0

    def measure(self, N, rng=None):
        keep = self.t < N
        xy, sd, tb = self.xy[keep], self.sd[keep], self.tblock[keep]
        half = tb % 2
        if rng is None:
            w = np.ones(len(xy))
        else:
            ids = np.unique(tb)
            counts = np.zeros(ids.max() + 1)
            for h in (0, 1):
                mine = ids[ids % 2 == h]
                counts[mine] = np.bincount(rng.integers(0, len(mine), len(mine)), minlength=len(mine))
            w = counts[tb]
            xy = xy + rng.normal(0, 1, xy.shape) * sd[:, None]
        a = self.hist(xy[half == 0], w[half == 0]) * self.window
        c = self.hist(xy[half == 1], w[half == 1]) * self.window
        count, largest = self.clusters(xy, w)
        return frc_resolution(a, c, RENDER_NM), count, largest


FIELDS = {}


def curve_job(job):
    rec, N, draw = job
    field = FIELDS[rec]
    rng = None if draw < 0 else np.random.default_rng([88, int(rec[1:]), N, draw])
    return (rec, N, draw, *field.measure(N, rng))


def overlap(blobs, chosen, mask):
    """Primary overlap number and the secondaries (see pilot_gates.md), for a set of frames."""
    keep_in = in_mask(blobs["xy"], mask)
    order = np.argsort(blobs["t"], kind="stable")
    t, xy, inside = blobs["t"][order], blobs["xy"][order], keep_in[order]
    cuts = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]

    def stats(lo, hi):
        ann, under, n_in = 0, 0, 0
        for a, e in zip(cuts[:-1], cuts[1:]):
            if not lo <= t[a] < hi:
                continue
            ins = np.nonzero(inside[a:e])[0]
            if not len(ins):
                continue
            d = np.hypot(*(xy[a:e][ins, None] - xy[a:e][None]).transpose(2, 0, 1))
            ann += int(((d >= 4) & (d < 8)).sum())
            under += int(((d < 4) & (d > 0)).any(1).sum())
            n_in += len(ins)
        frames = hi - lo
        return dict(blobs_in_mask=n_in, per_frame=n_in / frames, n_4_8=ann / max(n_in, 1),
                    primary=1 - np.exp(-(ann / max(n_in, 1)) / 3), detector_under_4=under / max(n_in, 1))

    out = {"all": stats(0, b.N_FRAMES), "first_2000": stats(0, 2000), "last_5000": stats(10000, b.N_FRAMES)}
    area_um2 = mask.sum() * (PIXEL_NM / 1000) ** 2
    density = out["all"]["per_frame"] / area_um2
    out["poisson_basis"] = dict(area_um2=area_um2, blobs_per_um2_frame=density,
                                share_under_4=1 - np.exp(-density * np.pi * (4 * PIXEL_NM / 1000) ** 2))
    in_ids = blobs["id"][keep_in]
    out["share_K2plus"] = float(np.mean(blobs["K"][keep_in] >= 2))
    pos = np.isin(chosen["blob"], in_ids)
    ids, xy_c, N = chosen["blob"][pos], chosen["xy"][pos], chosen["N"][pos]
    bright = N > 500
    strict = 0
    order = np.argsort(ids[bright])
    ib, xb = ids[bright][order], xy_c[bright][order]
    cuts2 = np.r_[0, np.nonzero(np.diff(ib))[0] + 1, len(ib)]
    for a, e in zip(cuts2[:-1], cuts2[1:]):
        if e - a >= 2:
            d = np.hypot(*(xb[a:e, None] - xb[None, a:e]).transpose(2, 0, 1))
            np.fill_diagonal(d, 99)
            strict += bool(((d >= 1) & (d <= 4)).any())
    out["share_two_bright_1_4px"] = strict / max(len(in_ids), 1)
    return out


def star(points, bands):
    """N*: smallest grid N where FRC, count and bands agree with 15,000, there and at every larger N."""
    full = points[15000]
    good = {}
    for N in GRID_N:
        p, q = points[N], bands[N]
        ok_frc = abs(p["frc_nm"] / full["frc_nm"] - 1) <= 0.10
        ok_cnt = abs(p["count"] / full["count"] - 1) <= 0.10
        ok_band = (q["frc_q"][0] <= bands[15000]["frc_q"][1] and bands[15000]["frc_q"][0] <= q["frc_q"][1]
                   and q["count_q"][0] <= bands[15000]["count_q"][1] and bands[15000]["count_q"][0] <= q["count_q"][1])
        good[N] = dict(frc=bool(ok_frc), count=bool(ok_cnt), band=bool(ok_band), all=bool(ok_frc and ok_cnt and ok_band))
    result = 15000
    for N in reversed(GRID_N):
        if good[N]["all"]:
            result = N
        else:
            break
    return result, good


def p2b(workers, tag=""):
    store = bigrun.Store("E88")
    guard()
    suffix = f"_{tag}" if tag else ""
    drift = store.load_array(f"drift{suffix}_r7")
    report = {}
    masks = {}
    for rec in ("r7", "r32"):
        chosen, single, blobs = load_rec(rec, tag)
        masks[rec] = mask_share(rec)
        if rec == "r32":                        # same method; recording 7's numbers are not reused
            loc = b.good_rows(chosen, b.SIDE, PIXEL_NM)
            t, xy = loc["t"], loc["xy"]
            blocks = drift_blocks(t, xy)
            d = []
            for k in range(blocks.max() + 1):
                d.append(estimate(xy[blocks == k], blocks[blocks == k] * 0, xy[blocks != k])[0])
            d = np.array(d)
            store.save_array(f"drift{suffix}_r32", d)
        else:
            d = drift
        loc = b.good_rows(chosen, b.SIDE, PIXEL_NM)
        loc["xy"] = correct(loc["t"], loc["xy"], d)
        ev = merge_events(loc)
        FIELDS[rec] = Field(ev, masks[rec])
        report[rec] = dict(rows_filtered=len(loc["t"]), events=len(ev["t"]), events_in_mask=len(FIELDS[rec].t),
                           render_shape=FIELDS[rec].shape, blobs=len(blobs["t"]),
                           K_counts=np.bincount(blobs["K"], minlength=5).tolist(),
                           drift_range_nm=np.ptp(d * PIXEL_NM, axis=0).round(2).tolist(),
                           overlap=overlap(blobs, chosen, masks[rec]))
        print(rec, b.rounded(report[rec]), flush=True)
        store.save_json(f"p2b_decision{suffix}", b.rounded(report))
    jobs = [(rec, N, draw) for rec in FIELDS for N in GRID_N for draw in range(-1, DRAWS)]
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:          # fork: FIELDS is shared
        for i, r in enumerate(pool.map(curve_job, jobs, chunksize=8)):
            results.append(r)
            if i % 400 == 0:
                print(i, len(jobs), f"{time.time() - started:.0f} s", flush=True)
                guard()
    for rec in FIELDS:
        points, bands = {}, {}
        for N in GRID_N:
            mine = [r for r in results if r[0] == rec and r[1] == N]
            pt = [r for r in mine if r[2] < 0][0]
            dr = np.array([r[3:] for r in mine if r[2] >= 0], float)
            points[N] = dict(frc_nm=pt[3], count=pt[4], largest=pt[5])
            bands[N] = dict(frc_q=np.nanquantile(dr[:, 0], [0.025, 0.975]).tolist(),
                            count_q=np.quantile(dr[:, 1], [0.025, 0.975]).tolist(),
                            frc_median=float(np.nanmedian(dr[:, 0])), count_median=float(np.median(dr[:, 1])),
                            frc_nan=int(np.isnan(dr[:, 0]).sum()))
        n_star, good = star(points, bands)
        report[rec] |= dict(points=points, bands=bands, N_star=n_star, checks=good)
    report["N_both"] = max(report["r7"]["N_star"], report["r32"]["N_star"])
    report["seconds_curves"] = time.time() - started
    prim = {r: report[r]["overlap"]["all"]["primary"] for r in ("r7", "r32")}
    report["overlap_rare"] = bool(all(v < 0.05 for v in prim.values()))
    report["overlap_primary"] = prim
    store.save_json(f"p2b_decision{suffix}", b.rounded(report))
    print(b.rounded({k: v for k, v in report.items() if k not in ("r7", "r32")}), flush=True)
    for rec in ("r7", "r32"):
        print(rec, "N*", report[rec]["N_star"], flush=True)
        for N in GRID_N:
            p, q = report[rec]["points"][N], report[rec]["bands"][N]
            print(N, {k: round(v, 3) for k, v in p.items()}, np.round(q["frc_q"], 1), q["count_q"],
                  report[rec]["checks"][N], flush=True)


def p3():
    store = bigrun.Store("E88")
    guard()
    n = CHUNKS
    report, use = {}, {}
    for rec in b.TIFFS:
        rows = []
        for i in range(n):
            c = store.load_chunk(f"stats_{rec}", i)
            rows.append(b.photon_transfer(c["median"], c["diff_var"]))
        rows = np.array(rows)
        med = np.median(rows, axis=0)
        report[rec] = dict(slope_median=med[0], offset_median=med[1],
                           slope_rel_max=float(np.max(np.abs(rows[:, 0] / med[0] - 1))),
                           offset_dev_max=float(np.max(np.abs(rows[:, 1] - med[1]))),
                           slope_first_last=[rows[0, 0], rows[-1, 0]], offset_first_last=[rows[0, 1], rows[-1, 1]],
                           slopes=rows[:, 0].round(3).tolist(), offsets=rows[:, 1].round(2).tolist())
        use[rec] = dict(slope=float(med[0]), offset=float(med[1]))
    between = dict(slope_rel=abs(use["r32"]["slope"] / use["r7"]["slope"] - 1),
                   offset=abs(use["r32"]["offset"] - use["r7"]["offset"]))
    along_ok = all(report[r]["slope_rel_max"] <= 0.10 and report[r]["offset_dev_max"] <= 20 for r in b.TIFFS)
    ok = along_ok and between["slope_rel"] <= 0.10 and between["offset"] <= 20
    store.save_json("p3_calibration", b.rounded(dict(recordings=report, between=between, along_ok=along_ok, go=ok,
                                                     use=use, e58=dict(slope=13.86, offset=457.0))))
    print(ok, between, {r: {k: v for k, v in report[r].items() if k not in ("slopes", "offsets")} for r in report})


def p3_line():
    """Post-hoc, after P3 failed, and not a gate: block mean against block variance, one line across blocks.

    Each block's own fit rests on spatial scatter within a narrow range of pixel means (r about 0.5), and
    its slope and offset move together. Across blocks the x range is the background's fall along the movie."""
    store = bigrun.Store("E88")
    out = {}
    for rec in b.TIFFS:
        x, y = [], []
        for i in range(CHUNKS):
            c = store.load_chunk(f"stats_{rec}", i)
            keep = c["median"] < np.percentile(c["median"], 90)
            x.append(c["median"][keep].mean())
            y.append(c["diff_var"][keep].mean())
        x, y = np.array(x), np.array(y)
        (slope, icpt), cov = np.polyfit(x, y, 1, cov=True)
        fit = slope * x + icpt
        out[rec] = dict(slope=float(slope), offset=float(-icpt / slope), slope_se=float(np.sqrt(cov[0, 0])),
                        r=float(np.corrcoef(x, y)[0, 1]), resid_sd=float(np.std(y - fit)),
                        mean_range=[float(x.min()), float(x.max())])
    out["note"] = "Post-hoc sensitivity, not a gate. Labelled [estimated] in the plan."
    store.save_json("p3_line", b.rounded(out))
    print(b.rounded(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=("p3", "p3line", "classical", "p2", "p2b"))
    parser.add_argument("--rec", default="r7")
    parser.add_argument("--workers", type=int, default=13)
    parser.add_argument("--tag", default="", help="'line': the post-hoc calibration sensitivity run")
    args = parser.parse_args()
    if args.command == "p3":
        p3()
    elif args.command == "p3line":
        p3_line()
    elif args.command == "classical":
        run_classical(args.rec, args.workers, args.tag)
    elif args.command == "p2":
        p2(args.tag)
    else:
        p2b(args.workers, args.tag)


if __name__ == "__main__":
    sys.exit(main())
