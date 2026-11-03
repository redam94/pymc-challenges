"""Pack what E88's notebook draws into data/dannhauser2022_e88_summary.npz. Nothing is fitted here.

    taskset -c 14,15 .venv/bin/python tools/memguard.py --max-gb 8 -- .venv/bin/python tools/build_e88_pack.py

Needs the local tables of a FULL run (results/E88/chunks, tools/run_e88_full.py) and the raw TIFFs. A reader
needs neither: the notebook opens this file and the JSON in results/E88. Replaces the plan's stats-only
`pack` in build_e88_data.py, which now calls this. Choices below are rules, not picks by result:

- renders: good rows, classical drift of the same recording, per-block calibration, 0.25 px and 10 nm grids
- zoom window: the 3 um square with most classical events in recording 7's mask
- drift demo: the block with the largest gated shift in network A's per-block run on recording 7
- audit spots: from the 200 audited, for K = 1 and K = 2, the worst r_hat and the first spot at or under 1.01
- FRC random split: the same rng sequence as frc_split in run_e88_full.py, so the stored values come back
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_e88_data as b  # noqa: E402
import run_e88_audit as au  # noqa: E402
import run_e88_classical as c  # noqa: E402

from pymc_challenges import bigrun  # noqa: E402

RECS = ("r7", "r32")
NETS = ("a", "b")
SHOW_FRAMES = (500, 12000)
ZOOM_UM = 3.0
Z_QUANTILES = np.linspace(0.002, 0.998, 400)
ANIMATION_FRAMES = np.unique(np.round(np.geomspace(150, 15000, 36)).astype(int))
ANIMATION_PX = 40


def net_table(store, net, rec, tag=""):
    name = f"infer_{net}_{tag}_{rec}" if tag else f"infer_{net}_{rec}"
    all_ = store.gather(name, c.CHUNKS)
    return {k: all_[k] for k in ("t", "xy", "sd", "N", "p")}


def hist(xy, grid=0.25, side=b.SIDE):
    n = int(side / grid)
    return np.histogram2d(xy[:, 1], xy[:, 0], bins=n, range=[[-0.5, side - 0.5]] * 2)[0].astype(np.uint16)


def per_block(t):
    return np.bincount(t // b.BLOCK, minlength=c.CHUNKS).astype(np.int32)


def frc_curve(a, e, pix_nm=c.RENDER_NM):
    """c.frc_resolution's curve: the 5-ring smoothed FRC, its spatial period in nm, and the first ring read."""
    Fa, Fe = np.fft.rfft2(a), np.fft.rfft2(e)
    ny, nx = a.shape
    q = np.hypot(np.fft.rfftfreq(nx)[None, :], np.fft.fftfreq(ny)[:, None])
    width = 1.0 / min(ny, nx)
    ring = np.minimum((q / width).astype(int).ravel(), int(0.5 / width))
    num = np.bincount(ring, (Fa * np.conj(Fe)).real.ravel())
    d1, d2 = np.bincount(ring, (np.abs(Fa) ** 2).ravel()), np.bincount(ring, (np.abs(Fe) ** 2).ravel())
    frc = np.convolve(num / np.sqrt(np.maximum(d1 * d2, 1e-30)), np.ones(5) / 5, mode="same")
    start = int(1 / (2000.0 / pix_nm) / width) + 2
    rings = np.arange(start, len(frc) - 3)
    return pix_nm / (rings * width), frc[rings]


def split_arrays(field, labels):
    a = field.hist(field.xy[labels == 0], np.ones((labels == 0).sum())) * field.window
    e = field.hist(field.xy[labels == 1], np.ones((labels == 1).sum())) * field.window
    return a, e


def thin(period, frc, keep=140):
    take = np.unique(np.round(np.linspace(0, len(frc) - 1, keep)).astype(int))
    return np.c_[period[take], frc[take]].astype(np.float32)


def z_quantiles(loc, gap=3, radius=b.RADIUS_PX):
    _, _, d, s2, _ = b.linked(loc, gap, radius)
    return np.quantile((d / np.sqrt(s2)[:, None]).ravel(), Z_QUANTILES).astype(np.float32), len(d)


def add_frc(store, out):
    """Classical curves with the gated rng sequence; A and B with classical drift and the post-hoc sequence."""
    gated, posthoc = store.load_json("frc_split"), store.load_json("frc_split_cdrift")
    rng = np.random.default_rng(884)
    for rec in RECS:
        mask = c.mask_share(rec)
        chosen, _, _ = c.load_rec(rec)
        rows = {"chosen": b.good_rows(chosen, b.SIDE, c.PIXEL_NM)} | {
            net: b.good_rows(net_table(store, net, rec), b.SIDE, c.PIXEL_NM) for net in NETS}
        drift = {"chosen": store.load_array(f"drift_{rec}")} | {net: store.load_array(f"drift_{net}_{rec}") for net in NETS}
        for name, loc in rows.items():
            xy = c.correct(loc["t"], loc["xy"], drift[name])
            raw = c.Field(dict(t=loc["t"], xy=xy, sd=loc["sd"]), mask)
            labels = rng.integers(0, 2, len(raw.t))         # consumed for every method, kept for classical
            if name != "chosen":
                continue
            got = c.frc_resolution(*split_arrays(raw, labels), c.RENDER_NM)
            assert abs(got - gated[rec]["chosen"]["random_split_nm"]) < 1e-3, (got, rec)
            out[f"{rec}_frc_random_chosen"] = thin(*frc_curve(*split_arrays(raw, labels)))
            ev = c.Field(c.merge_events(dict(loc, xy=xy)), mask)
            got = c.frc_resolution(*split_arrays(ev, ev.tblock % 2), c.RENDER_NM)
            assert abs(got - gated[rec]["chosen"]["block_split_nm"]) < 1e-3, (got, rec)
            out[f"{rec}_frc_block_chosen"] = thin(*frc_curve(*split_arrays(ev, ev.tblock % 2)))
        for net in NETS:
            rng2 = np.random.default_rng(884)               # curves_cdrift restarts it for each network
            loc = b.good_rows(net_table(store, net, rec), b.SIDE, c.PIXEL_NM)
            xy = c.correct(loc["t"], loc["xy"], drift["chosen"])
            raw = c.Field(dict(t=loc["t"], xy=xy, sd=loc["sd"]), mask)
            labels = rng2.integers(0, 2, len(raw.t))
            got = c.frc_resolution(*split_arrays(raw, labels), c.RENDER_NM)
            assert abs(got - posthoc[rec][net]["random_split_nm"]) < 1e-3, (got, rec, net)
            out[f"{rec}_frc_random_{net}"] = thin(*frc_curve(*split_arrays(raw, labels)))
            ev = c.Field(c.merge_events(dict(loc, xy=xy)), mask)
            got = c.frc_resolution(*split_arrays(ev, ev.tblock % 2), c.RENDER_NM)
            assert abs(got - posthoc[rec][net]["block_split_nm"]) < 1e-3, (got, rec, net)
            out[f"{rec}_frc_block_{net}"] = thin(*frc_curve(*split_arrays(ev, ev.tblock % 2)))
        print("frc", rec, flush=True)


def add_recording(store, rec, out):
    stack = b.open_stack(b.RAW_DIR / b.TIFFS[rec][0])
    mask = c.mask_share(rec)
    out[f"{rec}_mask"] = mask.astype(np.uint8)
    frame_median = np.concatenate([store.load_chunk(f"stats_{rec}", i)["frame_median"] for i in range(c.CHUNKS)])
    out[f"{rec}_median_per_100"] = frame_median.reshape(-1, 100).mean(1).astype(np.float32)
    out[f"{rec}_frames"] = np.stack([np.asarray(stack["px"][f]) for f in SHOW_FRAMES]).astype(np.uint16)
    chosen, single, blobs = c.load_rec(rec)
    inside = c.in_mask(blobs["xy"], mask)
    out[f"{rec}_blobs_per_block"] = per_block(blobs["t"])
    out[f"{rec}_blobs_mask_per_block"] = per_block(blobs["t"][inside])
    loc = b.good_rows(chosen, b.SIDE, c.PIXEL_NM)
    drift = store.load_array(f"drift_{rec}")
    tables = {"chosen": loc}
    for net in NETS:
        tables[net] = b.good_rows(net_table(store, net, rec), b.SIDE, c.PIXEL_NM)
        out[f"{rec}_rows_per_block_{net}"] = per_block(net_table(store, net, rec)["t"])
        out[f"{rec}_rows_per_block_{net}_line"] = per_block(net_table(store, net, rec, "line")["t"])
    out[f"{rec}_rows_per_block_chosen"] = per_block(chosen["t"])
    out[f"{rec}_rows_per_block_chosen_line"] = per_block(c.load_rec(rec, "line")[0]["t"])
    for name, table in tables.items():
        corrected = c.correct(table["t"], table["xy"], drift)
        out[f"{rec}_render_{name}"] = hist(corrected)
    for name in ("chosen", "K1", "a", "b"):
        table = tables.get(name) or b.good_rows(single, b.SIDE, c.PIXEL_NM)
        keep = {k: v[c.in_mask(table["xy"], mask)] for k, v in table.items()}
        out[f"{rec}_z_{name}"], out[f"{rec}_z_pairs_{name}"] = z_quantiles(keep)
    for suffix in ("", "_line"):
        out[f"{rec}_drift{suffix}_chosen"] = store.load_array(f"drift{suffix}_{rec}").astype(np.float32)
        for net in NETS:
            out[f"{rec}_drift{suffix}_{net}"] = store.load_array(f"drift_{net}{suffix}_{rec}").astype(np.float32)
    print("recording", rec, flush=True)
    return tables, drift, mask, stack


def add_zoom(tables, drift, mask, stack, out):
    """The densest 3 um square in recording 7's mask, by classical events, and the animation's window around it."""
    corrected = c.correct(tables["chosen"]["t"], tables["chosen"]["xy"], drift)
    ev = c.merge_events(dict(tables["chosen"], xy=corrected))
    keep = c.in_mask(ev["xy"], mask)
    side = int(round(ZOOM_UM * 1000 / c.PIXEL_NM))
    density = np.histogram2d(ev["xy"][keep, 1], ev["xy"][keep, 0], bins=b.SIDE, range=[[-0.5, b.SIDE - 0.5]] * 2)[0]
    smooth = ndi.uniform_filter(density, side, mode="constant")
    margin = ANIMATION_PX // 2 + 1
    smooth[:margin], smooth[-margin:], smooth[:, :margin], smooth[:, -margin:] = 0, 0, 0, 0
    row, col = np.unravel_index(smooth.argmax(), smooth.shape)
    out["zoom_centre_xy"] = np.array([col, row], np.float32)
    half = ZOOM_UM * 500 / c.PIXEL_NM
    lo = np.array([col, row], float) - half
    n = int(ZOOM_UM * 1000 / c.RENDER_NM)
    for name, table in tables.items():
        xy = c.correct(table["t"], table["xy"], drift)
        hit = np.all((xy >= lo) & (xy < lo + 2 * half), axis=1)
        ij = ((xy[hit] - lo) * c.PIXEL_NM / c.RENDER_NM).astype(int)
        out[f"zoom_{name}"] = np.bincount(ij[:, 1] * n + ij[:, 0], minlength=n * n).reshape(n, n).astype(np.uint16)
    a = ANIMATION_PX // 2
    window = (row - a, row + a, col - a, col + a)
    out["animation_window"] = np.array(window, np.int32)
    out["animation_frames"] = ANIMATION_FRAMES.astype(np.int32)
    out["animation_raw"] = np.stack([np.asarray(stack["px"][f - 1][window[0]:window[1], window[2]:window[3]])
                                     for f in ANIMATION_FRAMES]).astype(np.uint16)
    inside = (ev["xy"][:, 0] >= window[2]) & (ev["xy"][:, 0] < window[3]) & (ev["xy"][:, 1] >= window[0]) & (ev["xy"][:, 1] < window[1])
    out["animation_xy"] = (ev["xy"][inside] - np.array([window[2], window[0]])).astype(np.float32)
    out["animation_t"] = ev["t"][inside].astype(np.uint16)
    print("zoom at", col, row, "events in window", int(inside.sum()), flush=True)


def add_drift_demo(store, tables, out):
    """The worst block of A's per-block run on recording 7: its render, and the rest of the movie's."""
    shifts = store.load_array("drift_a_r7")
    k = int(np.abs(shifts).max(1).argmax())
    out["drift_demo_block"] = np.array(k)
    for name in ("a", "chosen"):
        t, xy = tables[name]["t"], tables[name]["xy"]
        blocks = c.drift_blocks(t, xy)
        out[f"drift_demo_{name}_block"] = hist(xy[blocks == k])
        out[f"drift_demo_{name}_rest"] = hist(xy[blocks != k])
    print("drift demo block", k, shifts[k], flush=True)


def add_lineups(store, out):
    for rec in RECS:
        for net in NETS:
            chunk = store.load_chunk(f"lineup_{rec}_{net}", 0)
            out[f"lineup_{rec}_{net}"] = chunk["panels"].astype(np.float32)
            out[f"lineup_{rec}_{net}_real"] = chunk["real_position"]


def add_audit(store, out):
    n = 200
    rows = [store.load_chunk("audit_nuts", i) for i in range(n)]
    for name in ("K", "t", "divergences"):
        out[f"audit_{name}"] = np.array([int(r[name]) for r in rows], np.int32)
    for name in ("rhat", "ess", "nuts_sd_px", "laplace_sd_px"):
        out[f"audit_{name}"] = np.array([float(r[name]) for r in rows], np.float32)
    # the same draws as audit_nuts, to rebuild the four chosen spots' pixels
    cal = c.calibration("r7")
    _, _, blobs = c.load_rec("r7")
    inside = c.in_mask(blobs["xy"], c.mask_share("r7"))
    rng = np.random.default_rng(887)
    picks = []
    for K in (1, 2):
        pool = np.nonzero(inside & (blobs["K"] == K))[0]
        picks += [(K, i) for i in rng.choice(pool, n // 2, replace=False)]
    K, rhat = out["audit_K"], out["audit_rhat"]
    chosen_ids = []
    for k in (1, 2):
        which = np.nonzero(K == k)[0]
        chosen_ids += [int(which[rhat[which].argmax()]), int(which[rhat[which] <= 1.01][0])]
    spots = {key: [] for key in ("Z", "W", "cen", "logb", "id", "seed")}
    for idx in chosen_ids:
        spot_k, i = picks[idx]
        assert spot_k == K[idx] and int(blobs["t"][i]) == out["audit_t"][idx]
        Z, Wm, cen, _, logb = au.spot_inputs("r7", (int(blobs["t"][i]), blobs["xy"][i]), cal)
        spots["Z"].append(Z), spots["W"].append(Wm), spots["cen"].append(cen), spots["logb"].append(logb)
        spots["id"].append(idx), spots["seed"].append(int(i) % 2**31)
    out["spot_Z"] = np.array(spots["Z"], np.float32).reshape(4, b.PATCH, b.PATCH)
    out["spot_W"] = np.array(spots["W"], np.float32).reshape(4, b.PATCH, b.PATCH)
    out["spot_cen"] = np.array(spots["cen"], np.float64)
    out["spot_logb"] = np.array(spots["logb"], np.float64)
    out["spot_id"] = np.array(spots["id"], np.int32)
    out["spot_seed"] = np.array(spots["seed"], np.int64)
    print("spots", chosen_ids, rhat[chosen_ids], flush=True)


def add_simulated(store, out):
    raw = store.load_chunk("simbase_classical", 0)
    tables = {"chosen": b.good_rows({k: raw[f"chosen_{k}"] for k in ("t", "xy", "sd", "N")}, b.SIDE, c.PIXEL_NM)}
    for net in NETS:
        tables[net] = b.good_rows(store.load_chunk(f"simbase_{net}", 0), b.SIDE, c.PIXEL_NM)
    for name, table in tables.items():
        out[f"sim_z_{name}"], out[f"sim_z_pairs_{name}"] = z_quantiles(table)


def main():
    c.guard()
    store = bigrun.Store("E88")
    out = {"attribution": np.array(b.ATTRIBUTION), "block_frames": np.array(b.BLOCK), "show_frames": np.array(SHOW_FRAMES),
           "z_quantiles": Z_QUANTILES.astype(np.float32), "psf_band": store.load_array("psf_profile_boot").astype(np.float32),
           "zoom_um": np.array(ZOOM_UM)}
    kept = None
    for rec in RECS:
        got = add_recording(store, rec, out)
        kept = got if rec == "r7" else kept
        c.guard()
    tables, drift, mask, stack = kept
    add_zoom(tables, drift, mask, stack, out)
    add_drift_demo(store, tables, out)
    del tables, kept
    add_lineups(store, out)
    add_simulated(store, out)
    add_audit(store, out)
    add_frc(store, out)
    c.guard()
    bigrun.atomic_write(b.OUT, lambda f: np.savez_compressed(f, **out))
    sizes = sorted(((np.asarray(v).nbytes, k) for k, v in out.items()), reverse=True)[:12]
    print(f"wrote {b.OUT} ({b.OUT.stat().st_size / 1e6:.2f} MB), {len(out)} arrays. Largest raw: {sizes}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
