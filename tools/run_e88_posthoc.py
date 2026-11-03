"""E88 post-hoc diagnostics from the three audits of 2026-11-02. They change no gate.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTENSOR_FLAGS=base_compiledir=PATH \
        taskset -c 14,15 .venv/bin/python tools/memguard.py --max-gb 8 -- .venv/bin/python tools/run_e88_posthoc.py STAGE

Every number here is labelled post-hoc in the notebook and prints beside the gated value it qualifies.
Stages: note, nuts_cold, nuts_warm, stuck, sim, real, halo, psf, frames. `results/E88/posthoc_note.json` says
what each one does, and was written before any of them ran. Sub-jobs checkpoint as chunks, so a crash loses little.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import erf, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_e88_data as b  # noqa: E402
import run_e88_audit as au  # noqa: E402
import run_e88_classical as c  # noqa: E402
import run_e88_full as f  # noqa: E402

from pymc_challenges import bigrun  # noqa: E402

NM = c.PIXEL_NM
RAD = b.RADIUS_PX
RECS = ("r32", "r7")
PILE_BIN = 50                                   # rows in one 0.25 px bin, whole field, that make a pile
STATIC_BIN = 10                                 # the same, on the much shorter simulated movie
NOTE = dict(
    status="Post-hoc, from the audits of 2026-11-02. Written before these stages ran. No gate, line or stored verdict changes.",
    nuts="All 200 audit spots, same seeds, 4 chains, 1,000 + 1,000, in the non-centred form of the same posterior, so the start "
         "follows the data. Once with an empty compile cache and once warm. K = 2 is also read sorted on the wider axis.",
    stuck="The 87 stuck spots rerun in the stored centred form with per-chain logp, step size and an L-BFGS climb from each chain's last draw.",
    sim=f"Simulated baseline scored against the simulator's own emitters: a row is true within 2 px of an emitter in its frame. Static bins hold {STATIC_BIN} or more false rows.",
    real=f"Per-block rows in the mask, gap 3, 1.5 px. Common pairs: same frame, both ends within 0.5 px, in all four methods. "
         f"Floor fit: b.fit_floor, 40 bootstraps. Piles: bins of {PILE_BIN} or more whole-field rows, 0.25 px.",
    halo="Written ring and far field, K = 1 added, and faint rows taken from frames 200, 400 and 800 away as a control.",
    psf="The stored 20,000 spots, repeated at the same sites 100 and 300 frames away, and at random sites in the same frame.",
    frames="Count rule only, classical: block bootstrap of N*, 100 draws, 200-frame blocks, both calibrations; events thinned at random from "
           "all 15,000 frames; recording 7's count band at 10,500 and 15,000 in five fresh seed families.")


def once(store, name, index, fn):
    """A sub-job as a chunk of JSON: a resumed stage skips what it already has."""
    if index in store.pending(name, index + 1):
        row = json.loads(json.dumps(fn(), default=bigrun._json_value))
        store.save_chunk(name, index, dict(row=np.frombuffer(json.dumps(row).encode(), np.uint8)))
    return json.loads(store.load_chunk(name, index)["row"].tobytes())


def audit_picks():
    chosen, _, blobs = c.load_rec("r7")
    inside = c.in_mask(blobs["xy"], c.mask_share("r7"))
    rng = np.random.default_rng(887)
    picks = []
    for K in (1, 2):
        pool = np.nonzero(inside & (blobs["K"] == K))[0]
        picks += [(K, i) for i in rng.choice(pool, 100, replace=False)]
    return picks, chosen, blobs


def sort_on(axis, x, y, logN):
    order = np.argsort(axis, axis=-1)
    return tuple(np.take_along_axis(v, order, -1) for v in (x, y, logN))


def rhat_ess(chk):
    import arviz as az

    r = max(float(az.rhat(v[..., j])) for v in chk.values() for j in range(v.shape[-1]))
    return r, min(float(az.ess(v[..., j])) for v in chk.values() for j in range(v.shape[-1]))


def pooled_sd(x, y):
    return float(np.sqrt(0.5 * x.var(axis=(0, 1)).mean() + 0.5 * y.var(axis=(0, 1)).mean()))


def nuts_spot(store, compiled, cal, chosen, blobs, idx, K, i):
    import nutpie

    Z, Wm, cen, org, logb = au.spot_inputs("r7", (int(blobs["t"][i]), blobs["xy"][i]), cal)
    model = compiled[K].with_data(z=Z, w=Wm, cx=np.full(K, cen[0]), cy=np.full(K, cen[1]), lb=logb)
    trace = nutpie.sample(model, chains=4, tune=1000, draws=1000, cores=2, seed=int(i) % 2**31, progress_bar=False)
    p = trace.posterior
    x, y, lN = cen[0] + 2 * p["rx"].values, cen[1] + 2 * p["ry"].values, p["logN"].values
    extra = dict(logb=p["rb"].values[..., None], logs=p["logs"].values[..., None])
    wide = {}
    if K == 2:
        axis = x if np.abs(x[..., 0] - x[..., 1]).mean() >= np.abs(y[..., 0] - y[..., 1]).mean() else y
        xw, yw, lw = sort_on(axis, x, y, lN)
        wide = dict(rhat_wide=rhat_ess(dict(x=xw, y=yw, logN=lw, **extra))[0], sd_wide=pooled_sd(xw, yw))
        x, y, lN = sort_on(x, x, y, lN)
    r, ess = rhat_ess(dict(x=x, y=y, logN=lN, **extra))
    lp = trace.sample_stats["logp"].mean("draw").values
    table = chosen["blob"] == blobs["id"][i]
    laplace = chosen["sd"][table]
    stored = store.load_chunk("audit_nuts", idx)
    row = dict(idx=idx, K=K, rhat=r, ess=ess, divergences=int(trace.sample_stats["diverging"].sum()), sd=pooled_sd(x, y),
               laplace=float(np.mean(laplace)) if len(laplace) else None, gap=float(lp.max() - lp.min()),
               stored_rhat=float(stored["rhat"]), stored_sd=float(stored["nuts_sd_px"]), stored_div=int(stored["divergences"])) | wide
    if K == 1 and table.any():
        fit, photons = chosen["xy"][table] - org[[2, 1]], chosen["N"][table]
        row["fit_px"] = float(np.min(np.hypot(fit[:, 0] - x.mean(), fit[:, 1] - y.mean())))
        row["fit_N_ratio"] = float(photons[0] / np.exp(lN.mean()))
    return row


def nuts(store, tag):
    import nutpie

    picks, chosen, blobs = audit_picks()
    cal = c.calibration("r7")
    t0 = time.time()
    compiled = {K: nutpie.compile_pymc_model(au.spot_model_start(K)) for K in (1, 2)}
    compile_s = time.time() - t0
    rows = []
    for idx, (K, i) in enumerate(picks):
        rows.append(once(store, f"posthoc_nuts_{tag}", idx, lambda: nuts_spot(store, compiled, cal, chosen, blobs, idx, K, i)))
        if idx % 20 == 0:
            print(idx, rows[-1], f"{time.time() - t0:.0f} s", flush=True)
            c.guard()
    K = np.array([r["K"] for r in rows])
    rhat = np.array([r["rhat"] for r in rows])
    ratio = np.array([r["sd"] / r["laplace"] for r in rows])
    good = rhat <= 1.01
    wide = np.array([r.get("sd_wide", r["sd"]) / r["laplace"] for r in rows])
    good_wide = np.array([r.get("rhat_wide", r["rhat"]) for r in rows]) <= 1.01
    k1 = [r for r in rows if r["K"] == 1 and "fit_px" in r]
    med = lambda a: float(np.nanmedian(a))
    report = dict(status="post-hoc", tag=tag, compile_s=compile_s, spots=len(rows), median_sd_ratio=dict(
        all=med(ratio), K1=med(ratio[K == 1]), K2=med(ratio[K == 2]), converged=med(ratio[good]), K1_converged=med(ratio[(K == 1) & good]),
        K2_converged=med(ratio[(K == 2) & good]), K2_wider_axis=med(wide[K == 2]), K2_wider_axis_converged=med(wide[(K == 2) & good_wide])),
        over_1_01=dict(K1=int(((K == 1) & ~good).sum()), K2=int(((K == 2) & ~good).sum()), K2_wider_axis=int(((K == 2) & ~good_wide).sum())),
        max_rhat=dict(K1=float(rhat[K == 1].max()), K2=float(rhat[K == 2].max())), divergences=dict(
            all=sum(r["divergences"] for r in rows), K1=sum(r["divergences"] for r in rows if r["K"] == 1)),
        chain_gap_unconverged=dict(median=med([r["gap"] for r, g in zip(rows, good) if not g]), max=max(r["gap"] for r, g in zip(rows, good) if not g)),
        K1_fit=dict(of=len(k1), within_half_px=sum(r["fit_px"] < 0.5 for r in k1), median_px=med([r["fit_px"] for r in k1]),
                    median_N_ratio=med([r["fit_N_ratio"] for r in k1])))
    store.save_json(f"posthoc_nuts_{tag}", b.rounded(report))
    print(b.rounded(report), flush=True)


@f.stage("posthoc_nuts_cold", "cpu")
def nuts_cold(store):
    nuts(store, "cold")


@f.stage("posthoc_nuts_warm", "cpu")
def nuts_warm(store):
    nuts(store, "warm")


PIX = np.arange(11.0)


def neg_logp(th, K, Z, W, cen, lb):
    logb, logs, x, y, lN = th[0], th[1], th[2:2 + K], th[2 + K:2 + 2 * K], th[2 + 2 * K:]
    s = np.exp(logs)
    px = lambda u: 0.5 * (erf((PIX + 0.5 - u[..., None]) / (b.SQ2 * s)) - erf((PIX - 0.5 - u[..., None]) / (b.SQ2 * s)))
    img = np.sum(np.exp(lN)[:, None, None] * px(y)[:, :, None] * px(x)[:, None, :], axis=0)
    mu = (np.exp(logb) + img).ravel() / 2
    ll = np.sum(W * (Z * np.log(mu) - mu - gammaln(Z + 1)))
    prior = -0.5 * ((logb - lb) / 0.5) ** 2 - 0.5 * (logs / 0.2) ** 2 - 0.5 * np.sum(((x - cen[0]) / 2) ** 2) - 0.5 * np.sum(((y - cen[1]) / 2) ** 2)
    return -(ll + prior - 0.5 * np.sum((lN - np.log(1500.0)) ** 2))


def stuck_spot(compiled, cal, chosen, blobs, idx, K, i, stored):
    import nutpie

    Z, Wm, cen, org, logb = au.spot_inputs("r7", (int(blobs["t"][i]), blobs["xy"][i]), cal)
    model = compiled[K].with_data(z=Z, w=Wm, cx=np.full(K, cen[0]), cy=np.full(K, cen[1]), lb=logb)
    trace = nutpie.sample(model, chains=4, tune=1000, draws=1000, cores=2, seed=int(i) % 2**31, progress_bar=False)
    p, stats = trace.posterior, trace.sample_stats
    x, y, lN, lb_, ls_ = p["x"].values, p["y"].values, p["logN"].values, p["logb"].values, p["logs"].values
    lp = stats["logp"].mean("draw").values
    best = int(np.argmax(lp))
    xs, ys, ln = sort_on(x, x, y, lN) if K == 2 else (x, y, lN)
    climbed = []
    for chain in range(4):
        th = np.r_[lb_[chain, -1], ls_[chain, -1], x[chain, -1], y[chain, -1], lN[chain, -1]]
        climbed.append(-float(minimize(neg_logp, th, args=(K, Z, Wm, cen, logb), method="L-BFGS-B", options=dict(maxiter=500, maxfun=20000)).fun))
    table = chosen["blob"] == blobs["id"][i]
    fit, photons = chosen["xy"][table] - org[[2, 1]], chosen["N"][table]
    bx, by, bN = xs[best].mean(0), ys[best].mean(0), np.exp(ln[best].mean(0))
    dist = [float(np.min(np.hypot(fit[:, 0] - u, fit[:, 1] - v))) if len(fit) else None for u, v in zip(bx, by)]
    return dict(idx=idx, K=K, stored_rhat=float(stored["rhat"]), logp=lp.tolist(), step=stats["step_size"].mean("draw").values.tolist(),
                chain_sd=[pooled_sd(xs[j:j + 1], ys[j:j + 1]) for j in range(4)], best=best, laplace=float(stored["laplace_sd_px"]),
                climbed=climbed, best_N=bN.tolist(), fit_N=photons.tolist(), fit_px=dist)


@f.stage("posthoc_stuck", "cpu")
def stuck(store):
    import nutpie

    picks, chosen, blobs = audit_picks()
    cal = c.calibration("r7")
    compiled = {K: nutpie.compile_pymc_model(au.spot_model(K)) for K in (1, 2)}
    todo = [idx for idx in range(len(picks)) if float(store.load_chunk("audit_nuts", idx)["rhat"]) > 1.01]
    rows = []
    for count, idx in enumerate(todo):
        K, i = picks[idx]
        rows.append(once(store, "posthoc_stuck", count, lambda: stuck_spot(compiled, cal, chosen, blobs, idx, K, i, store.load_chunk("audit_nuts", idx))))
        if count % 10 == 0:
            print(count, idx, f"K {K}", flush=True)
            c.guard()
    K = np.array([r["K"] for r in rows])
    gap = np.array([max(r["logp"]) - min(r["logp"]) for r in rows])
    spread = np.array([max(r["climbed"]) - min(r["climbed"]) for r in rows])
    big = gap >= 20
    pack_ratio = {}
    all_K = np.array([int(store.load_chunk("audit_nuts", i)["K"]) for i in range(200)])
    sds = np.array([float(store.load_chunk("audit_nuts", i)["nuts_sd_px"]) for i in range(200)])
    laplace = np.array([float(store.load_chunk("audit_nuts", i)["laplace_sd_px"]) for i in range(200)])
    best_sd = sds.copy()
    for r in rows:
        best_sd[r["idx"]] = r["chain_sd"][r["best"]]
    pack_ratio = dict(all=float(np.nanmedian(best_sd / laplace)), K1=float(np.nanmedian((best_sd / laplace)[all_K == 1])), K2=float(np.nanmedian((best_sd / laplace)[all_K == 2])))
    k1 = [r for r in rows if r["K"] == 1 and r["fit_px"][0] is not None]
    report = dict(status="post-hoc, rerun of the stored form, not a reproduction", spots=len(rows), gap_20=dict(K1=int(big[K == 1].sum()), K2=int(big[K == 2].sum())),
                  of=dict(K1=int((K == 1).sum()), K2=int((K == 2).sum())), gap_20_flat_after_lbfgs=int((spread[big] < 5).sum()),
                  median_chain_step_max_over_min=float(np.median([max(r["step"]) / min(r["step"]) for r in rows])), best_chain_sd_ratio=pack_ratio,
                  K1_fit=dict(of=len(k1), on_best_chain=sum(r["fit_px"][0] < 0.5 for r in k1), median_N_ratio=float(np.median([r["fit_N"][0] / r["best_N"][0] for r in k1]))))
    store.save_json("posthoc_stuck", b.rounded(report))
    print(b.rounded(report), flush=True)


def true_rows(loc, by_frame, tol):
    t, xy = loc["t"], loc["xy"]
    hit = np.zeros(len(t), bool)
    for frame in np.unique(t):
        idx, sites = np.nonzero(t == frame)[0], by_frame.get(frame)
        if sites is not None:
            hit[idx] = cKDTree(sites).query(xy[idx])[0] <= tol
    return hit


def pile_map(xy, n=800):
    return np.histogram2d(xy[:, 1], xy[:, 0], bins=n, range=[[-0.5, 199.5]] * 2)[0]


def pile_of(xy, hh, n=800):
    ij = np.clip(((xy + 0.5) / 0.25).astype(int), 0, n - 1)
    return hh[ij[:, 1], ij[:, 0]]


@f.stage("posthoc_sim", "cpu")
def sim(store):
    truth = store.load_array("simbase_truth")
    tt = truth[:, 0].astype(int)
    by = {frame: truth[tt == frame, 1:3] for frame in np.unique(tt)}
    pack = np.load(b.OUT, allow_pickle=False)
    raw = store.load_chunk("simbase_classical", 0)
    tables = dict(chosen={k: raw[f"chosen_{k}"] for k in ("t", "xy", "sd", "N")}, K1={k: raw[f"single_{k}"] for k in ("t", "xy", "sd", "N")})
    for net in f.NETS:
        tables[net] = store.load_chunk(f"simbase_{net}", 0)
    report = dict(status="post-hoc", true_rows=len(truth))
    for name, table in tables.items():
        loc = b.good_rows(table, b.SIDE, NM)
        hit = true_rows(loc, by, 2.0)
        i, j, d, s2, _ = b.linked(loc, 3, RAD)
        r2 = np.sum(d**2, 1)
        both, neither = hit[i] & hit[j], ~hit[i] & ~hit[j]
        z_all = b.fit_z_scale(r2, s2, RAD)
        row = dict(rows=len(hit), rows_true=int(hit.sum()), precision=float(hit.mean()), pairs=len(i), pairs_both_true=int(both.sum()), pairs_neither=int(neither.sum()),
                   z_all=float(z_all[0]), mixture_true_share=float(z_all[1]), z_both_true=float(b.fit_z_scale(r2[both], s2[both], RAD)[0]))
        if neither.sum() > 50:
            row["z_neither"] = float(b.fit_z_scale(r2[neither], s2[neither], RAD)[0])
        false = ~hit
        hh = pile_map(loc["xy"][false])
        row["false_rows_in_static_bins"] = float((pile_of(loc["xy"][false], hh) >= STATIC_BIN).mean())
        if name in f.NETS:
            render = pack[f"r7_render_{name}"].astype(float)
            top = np.argsort(render.ravel())[-20:]
            rr, cc = np.unravel_index(top, render.shape)
            sites = cKDTree(np.c_[cc, rr] * 0.25 - 0.375)
            row["false_near_real_piles"] = float((sites.query(loc["xy"][false])[0] <= 1).mean())
            row["true_near_real_piles"] = float((sites.query(loc["xy"][hit])[0] <= 1).mean())
        report[name] = row
        print(name, b.rounded(row), flush=True)
    store.save_json("posthoc_sim", b.rounded(report))


def scatter_nm(d, w=None):
    """Per-axis scatter of one frame from the mixture with a constant variance, nm."""
    return float(b.fit_z_scale(np.sum(d**2, 1), np.full(len(d), 0.01), RAD, w)[0] * 0.1 * NM / np.sqrt(2))


def z_of(d, s2, w=None):
    return float(b.fit_z_scale(np.sum(d**2, 1), s2, RAD, w)[0])


def real_one(store, rec):
    mask = c.mask_share(rec)
    chosen, single, _ = c.load_rec(rec)
    full = dict(chosen=chosen, K1=single) | {net: f.net_table(store, net, rec) for net in f.NETS}
    good = {}
    for k, v in full.items():
        g = b.good_rows(v, b.SIDE, NM)
        good[k] = {name: col[c.in_mask(g["xy"], mask)] for name, col in g.items()}
    links = {k: b.linked(v, 3, RAD) for k, v in good.items()}
    out = dict(rows={k: len(v["t"]) for k, v in good.items()}, pairs={k: len(v[0]) for k, v in links.items()})
    keys = {k: np.c_[good[k]["t"][v[0]] * 100.0, good[k]["xy"][v[0]], good[k]["xy"][v[1]]] for k, v in links.items()}
    trees = {k: cKDTree(v) for k, v in keys.items()}
    base = keys["chosen"]
    keep, near = np.ones(len(base), bool), {}
    for k in ("K1", "a", "b"):
        dist, idx = trees[k].query(base, p=np.inf, distance_upper_bound=0.5)
        near[k] = idx
        keep &= np.isfinite(dist)
    which = {"chosen": np.nonzero(keep)[0]} | {k: near[k][keep] for k in ("K1", "a", "b")}
    out["common"] = dict(pairs=int(keep.sum()), z={}, scatter_nm={}, median_sd_nm={})
    for k, idx in which.items():
        d, s2 = links[k][2][idx], links[k][3][idx]
        out["common"]["z"][k], out["common"]["scatter_nm"][k] = z_of(d, s2), scatter_nm(d)
        out["common"]["median_sd_nm"][k] = float(np.median(np.sqrt(s2 / 2)) * NM)
    rng = np.random.default_rng(1)
    h = np.histogram2d(good["chosen"]["xy"][:, 1], good["chosen"]["xy"][:, 0], bins=200, range=[[-0.5, 199.5]] * 2)[0]
    yy, xx = np.mgrid[-2:3, -2:3]
    disc = (yy**2 + xx**2 <= 2.25).astype(float)
    dens = ndi.convolve(h, disc, mode="constant") / disc.sum()
    out["sd_tertiles"], out["density_tertiles"], out["floor"], out["by_gap_scatter_nm"] = {}, {}, {}, {}
    for k, (i, j, d, s2, tr) in links.items():
        sd = np.sqrt(s2 / 2) * NM
        mid = np.clip(np.round(0.5 * (good[k]["xy"][i] + good[k]["xy"][j])).astype(int), 0, 199)
        for label, value in (("sd_tertiles", sd), ("density_tertiles", dens[mid[:, 1], mid[:, 0]])):
            edges = np.quantile(value, [0, 1 / 3, 2 / 3, 1])
            out[label][k] = [z_of(d[(value >= lo) & (value <= hi)], s2[(value >= lo) & (value <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
        out["floor"][k] = b.fit_floor(d, s2, tr, RAD, rng, NM, boots=40)
        out["by_gap_scatter_nm"][k] = {gap: scatter_nm(b.linked(good[k], gap, RAD)[2]) for gap in (1, 3, 10)}
    out["piles"] = {}
    ref_i, ref_j = b.link_pairs(good["chosen"]["t"], good["chosen"]["xy"], 3, RAD)
    ref = f.pair_brightness(good["chosen"], ref_i, ref_j)
    for k in ("chosen", "K1", "a", "b"):
        hh = pile_map(np.asarray(full[k]["xy"]))
        i, j, d, s2, _ = links[k]
        pile = pile_of(good[k]["xy"], hh) >= PILE_BIN
        bad = pile[i] | pile[j]
        own = f.pair_brightness(good[k], i, j)
        w = np.ones(len(i)) if k == "chosen" else f.matched_weights(ref, own)[0]
        w2 = np.ones(int((~bad).sum())) if k == "chosen" else f.matched_weights(ref, own[~bad])[0]
        use, use2 = w > 0, w2 > 0
        out["piles"][k] = dict(touching=int(bad.sum()), of=len(i), z_unweighted=z_of(d, s2), z_unweighted_without=z_of(d[~bad], s2[~bad]),
                               z_matched=z_of(d[use], s2[use], w[use]), z_matched_without=z_of(d[~bad][use2], s2[~bad][use2], w2[use2]))
    gg = b.good_rows(chosen, b.SIDE, NM)
    out["persist_gap1_share_of_rows"] = len(b.link_pairs(gg["t"], gg["xy"], 1, RAD)[0]) / len(gg["t"])
    hh = pile_map(np.asarray(full["a"]["xy"]))
    y, x = np.array(np.unravel_index(hh.argmax(), hh.shape)) * 0.25 - 0.375
    y, x = int(round(y)), int(round(x))
    medians = np.stack([store.load_chunk(f"stats_{rec}", i)["median"] for i in range(c.CHUNKS)])
    window = medians[:, max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].max((1, 2))
    around = np.median(medians[:, max(y - 12, 0):y + 13, max(x - 12, 0):x + 13], axis=(1, 2))
    out["fullest_pile"] = dict(y=y, x=x, rows=int(hh.max()), peak_adu=window.round(0).tolist(), nearby_adu=around.round(0).tolist())
    return out


@f.stage("posthoc_real", "cpu")
def real(store):
    report = dict(status="post-hoc, unweighted unless named, one calibration (per block)")
    for index, rec in enumerate(RECS):
        report[rec] = once(store, "posthoc_real", index, lambda: real_one(store, rec))
        print(rec, b.rounded(report[rec]), flush=True)
    store.save_json("posthoc_real", b.rounded(report))


def halo_shift(loc, mask, shift):
    keep = (loc["xy"].min(1) >= b.EDGE_PX) & (loc["xy"].max(1) <= b.SIDE - 1 - b.EDGE_PX)
    t, xy, N = loc["t"][keep], loc["xy"][keep] * NM, loc["N"][keep]
    bright = (N >= f.HALO_BRIGHT) & c.in_mask(loc["xy"][keep], mask)
    faint = N < f.HALO_FAINT
    tb, xb, tf, xf = t[bright], xy[bright], (t[faint] + shift) % 15000, xy[faint]
    order = np.argsort(tf, kind="stable")
    tf, xf = tf[order], xf[order]
    start = np.searchsorted(tf, np.arange(15001))
    ring = far = 0
    for frame in np.unique(tb):
        q = xf[start[frame]:start[frame + 1]]
        if len(q):
            dist = np.hypot(*(xb[tb == frame][:, None] - q[None]).transpose(2, 0, 1)).ravel()
            ring += int(((dist >= f.RING[0]) & (dist <= f.RING[1])).sum())
            far += int(((dist >= f.FAR[0]) & (dist <= f.FAR[1])).sum())
    um2 = lambda lo, hi: np.pi * (hi**2 - lo**2) / 1e6
    return (ring / um2(*f.RING)) / (far / um2(*f.FAR)) if far else None


@f.stage("posthoc_halo", "cpu")
def halo(store):
    report = dict(status="post-hoc: K = 1 has one row per blob, so detector blindness inside a blob hides any halo")
    for rec in RECS:
        mask = c.mask_share(rec)
        chosen, single, _ = c.load_rec(rec)
        tables = dict(chosen=chosen, K1=single) | {net: f.net_table(store, net, rec) for net in f.NETS}
        report[rec] = {}
        for name, table in tables.items():
            shifted = [halo_shift(table, mask, s) for s in (200, 400, 800)]
            report[rec][name] = dict(ratio=f.halo_one(table, mask)["ratio"], shifted=float(np.mean(shifted)))
            print(rec, name, b.rounded(report[rec][name]), flush=True)
    store.save_json("posthoc_halo", b.rounded(report))


def wing_share(profile):
    g = f.normalise(profile)
    knots = np.arange(len(g)) * f.DR
    return float(2 * np.pi * np.sum((knots * g * f.DR)[knots >= 3])), g


@f.stage("posthoc_psf", "cpu")
def psf(store):
    cal = f.stored_cal(store, "r7", "")
    stack = b.open_stack(b.RAW_DIR / b.TIFFS["r7"][0])
    t, xy, N, pool = f.isolated_spots()
    r, e = f.radial_rows(t, xy, N, cal, stack)
    real_profile = f.stack_profile(r, e)
    centres = (np.arange(len(real_profile)) + 0.5) * f.DR
    use = centres < 9
    integral = lambda p: float(2 * np.pi * np.sum((centres * p * f.DR)[use]))
    beyond = lambda p: integral(np.where(centres >= 3, p, 0.0)) / integral(p)   # signed: a negative bin stays negative
    share, g = wing_share(real_profile)
    report = dict(status="post-hoc", spots=len(t), stored_share=store.load_json("psf_bg")["share_beyond_3px"], stack_share=share, stack_share_signed=beyond(real_profile), stack_integral=integral(real_profile),
                  stack_peak=float(g[0]), knots_px=(np.arange(len(g)) * f.DR).tolist(), stack=g.tolist(), raw=real_profile.tolist(), controls={})
    rng = np.random.default_rng(886)
    low, high = int(f.R_MAX), b.SIDE - 1 - int(f.R_MAX)
    for name in ("after_100", "after_300", "random_sites"):
        if name == "random_sites":
            tt, sites = t, rng.uniform(low, high, xy.shape)
        else:
            dt = int(name.split("_")[1])
            tt, sites = np.where(t + dt < 15000, t + dt, t - dt), xy
        r2, e2 = f.radial_rows(tt, sites, N, cal, stack)
        control = f.stack_profile(r2, e2)
        difference = real_profile - control
        g2 = f.normalise(np.maximum(difference, 0))
        report["controls"][name] = dict(control=control.tolist(), integral_control=integral(control), integral_difference=integral(difference),
                                        share_difference=beyond(difference), peak_difference=float(g2[0]), difference=difference.tolist(),
                                        pedestal_3_to_8px=float(np.mean(control[(centres >= 3) & (centres < 8)])), core_under_1px=float(np.mean(control[centres < 1])))
        print(name, {k: v for k, v in report["controls"][name].items() if k not in ("control", "difference")}, flush=True)
    sub = rng.choice(len(t), min(f.WIDTH_SPOTS, len(t)), replace=False)
    grid = np.exp(np.linspace(np.log(0.6), np.log(1.7), 45))
    fixed, free = [], []
    for w in grid:
        pred = f.lookup_np(r[sub] / w, g) / w**2
        keep = r[sub] < 5
        fixed.append(np.sum(np.where(keep, (e[sub] - pred) ** 2, 0.0), axis=1))
        amp = np.sum(np.where(keep, e[sub] * pred, 0.0), 1) / np.sum(np.where(keep, pred**2, 0.0), 1)
        free.append(np.sum(np.where(keep, (e[sub] - amp[:, None] * pred) ** 2, 0.0), axis=1))
    wf, wa = grid[np.argmin(fixed, 0)], grid[np.argmin(free, 0)]
    report["width"] = dict(stored_rule_median=float(np.median(wf)), stored_rule_log_sd=float(np.std(np.log(wf))), amplitude_free_median=float(np.median(wa)), amplitude_free_log_sd=float(np.std(np.log(wa))))
    store.save_json("posthoc_psf", b.rounded(report))
    print({k: v for k, v in b.rounded(report).items() if k not in ("stack", "raw", "controls", "knots_px")}, flush=True)


FIELD = None


def band_job(job):
    family, N, draw = job
    F = FIELD
    keep = F.t < N
    xy, sd, tb = F.xy[keep], F.sd[keep], F.tblock[keep]
    rng = np.random.default_rng([family, 7, N, draw])
    ids = np.unique(tb)
    counts = np.zeros(ids.max() + 1)
    for half in (0, 1):
        mine = ids[ids % 2 == half]
        counts[mine] = np.bincount(rng.integers(0, len(mine), len(mine)), minlength=len(mine))
    return N, F.clusters(xy + rng.normal(0, 1, xy.shape) * sd[:, None], counts[tb])[0]


def boot_one(store, tag, rec, draws=100):
    chosen, _, _ = c.load_rec(rec, tag)
    loc = b.good_rows(chosen, b.SIDE, NM)
    loc = {k: loc[k] for k in ("t", "xy", "sd", "N")}
    loc["xy"] = c.correct(loc["t"], loc["xy"], store.load_array(f"drift{'_line' if tag else ''}_{rec}"))
    F = c.Field(c.merge_events(loc), c.mask_share(rec))
    GRID = c.GRID_N
    base = [int(F.clusters(F.xy[F.t < N], np.ones((F.t < N).sum()))[0]) for N in GRID]
    rng = np.random.default_rng(int(rec[1:]) + 7)
    blocks, nb, stars = F.t // 200, 75, []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[blocks].astype(float)
        y = np.array([F.clusters(F.xy[(F.t < N) & (w > 0)], w[(F.t < N) & (w > 0)])[0] for N in GRID], float)
        ok = np.abs(y / y[-1] - 1) <= 0.10
        star = GRID[-1]
        for q in range(len(GRID) - 1, -1, -1):
            if not ok[q]:
                break
            star = GRID[q]
        stars.append(int(star))
    thinned = []
    for N in GRID:
        m = int((F.t < N).sum())
        pick = rng.choice(len(F.t), m, replace=False)
        thinned.append(int(F.clusters(F.xy[pick], np.ones(m))[0]))
    return dict(points=base, stars=stars, thinned=thinned, events=[int((F.t < N).sum()) for N in GRID])


@f.stage("posthoc_frames", "cpu")
def frames(store):
    global FIELD
    report = dict(status="post-hoc, count rule only, classical, not the gated three-part rule", grid=list(c.GRID_N), calibrations={})
    index = 0
    for tag in ("", "line"):
        rows = {}
        for rec in ("r7", "r32"):
            rows[rec] = once(store, "posthoc_frames", index, lambda: boot_one(store, tag, rec))
            index += 1
            print(tag or "per block", rec, rows[rec]["points"], rows[rec]["thinned"], flush=True)
            c.guard()
        both = np.maximum(rows["r7"]["stars"], rows["r32"]["stars"])
        values, counts = np.unique(both, return_counts=True)
        report["calibrations"][tag or "per_block"] = dict(rows, both_counts={int(v): int(n) for v, n in zip(values, counts)}, both_95=np.quantile(both, [0.025, 0.975]).tolist())
    loc = b.good_rows(c.load_rec("r7")[0], b.SIDE, NM)
    loc = {k: loc[k] for k in ("t", "xy", "sd", "N")}
    loc["xy"] = c.correct(loc["t"], loc["xy"], store.load_array("drift_r7"))
    FIELD = c.Field(c.merge_events(loc), c.mask_share("r7"))
    report["band_seeds"] = {}
    for family in (88, 1, 2, 3, 4, 5):
        jobs = [(family, N, draw) for N in (10500, 15000) for draw in range(200)]
        with ProcessPoolExecutor(2) as pool:
            done = list(pool.map(band_job, jobs, chunksize=8))
        q = {N: np.quantile([v for m, v in done if m == N], [0.025, 0.975]).tolist() for N in (10500, 15000)}
        report["band_seeds"][family] = dict(q10500=q[10500], q15000=q[15000], margin=q[10500][1] - q[15000][0])
        print(family, report["band_seeds"][family], flush=True)
    store.save_json("posthoc_frames", b.rounded(report))


STAGES = dict(nuts_cold=nuts_cold, nuts_warm=nuts_warm, stuck=stuck, sim=sim, real=real, halo=halo, psf=psf, frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("stage", choices=("note", *STAGES))
    which = parser.parse_args().stage
    if which == "note":
        return bigrun.Store("E88").save_json("posthoc_note", NOTE) and None
    STAGES[which]()


if __name__ == "__main__":
    sys.exit(main())
