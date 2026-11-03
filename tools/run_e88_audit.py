"""E88 pilot P7: NUTS audit of E58's spot model on spots from the recording-7 mask, nutpie on the CPU.

    OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTENSOR_FLAGS=base_compiledir=PATH \
        .venv/bin/python tools/run_e88_audit.py --spots 20

The model is E58's spot_model with the fitter's mask of neighbouring blobs, so it has the same posterior
as the Laplace fit it audits. Spots and z arrive through pm.Data, so each K compiles once. Two emitters
are exchangeable, so draws are sorted on x before r_hat: label switching is not a sampling failure.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_e88_data as b  # noqa: E402
import run_e88_classical as c  # noqa: E402

from pymc_challenges import bigrun  # noqa: E402

P = b.PATCH
PIX = np.arange(P)


def spot_model(K):
    import pymc as pm
    import pytensor.tensor as pt

    def pixel(u, s):
        return 0.5 * (pt.erf((PIX + 0.5 - u[..., None]) / (b.SQ2 * s)) - pt.erf((PIX - 0.5 - u[..., None]) / (b.SQ2 * s)))

    with pm.Model() as m:
        z = pm.Data("z", np.zeros(P * P))
        w = pm.Data("w", np.ones(P * P))
        cx, cy = pm.Data("cx", np.zeros(K)), pm.Data("cy", np.zeros(K))
        lb = pm.Data("lb", 5.0)
        logb = pm.Normal("logb", lb, 0.5)
        logs = pm.Normal("logs", 0.0, 0.2)
        x = pm.Normal("x", cx, 2.0, shape=K)
        y = pm.Normal("y", cy, 2.0, shape=K)
        logN = pm.Normal("logN", np.log(1500.0), 1.0, shape=K)
        s = pt.exp(logs)
        img = pt.sum(pt.exp(logN)[:, None, None] * pixel(y, s)[:, :, None] * pixel(x, s)[:, None, :], axis=0)
        mu = (pt.exp(logb) + img).ravel() / 2
        pm.Potential("ll", pt.sum(w * (z * pt.log(mu) - mu - pt.gammaln(z + 1))))
    return m


def spot_model_start(K):
    """The same posterior as `spot_model`, non-centred, so that the start follows the data.

    nutpie 0.16.11: `with_data` swaps the shared data of the logp, but `initial_point_func` keeps the
    compile-time placeholders, and `_make_model` ignores `init_mean` for a PyMC model. Every chain of
    `spot_model` therefore starts near x = y = 0, the corner of the patch. Here that start is the spot.
    """
    import pymc as pm
    import pytensor.tensor as pt

    def pixel(u, s):
        return 0.5 * (pt.erf((PIX + 0.5 - u[..., None]) / (b.SQ2 * s)) - pt.erf((PIX - 0.5 - u[..., None]) / (b.SQ2 * s)))

    with pm.Model() as m:
        z = pm.Data("z", np.zeros(P * P))
        w = pm.Data("w", np.ones(P * P))
        cx, cy = pm.Data("cx", np.zeros(K)), pm.Data("cy", np.zeros(K))
        lb = pm.Data("lb", 5.0)
        rb = pm.Normal("rb", 0.0, 1.0)
        logs = pm.Normal("logs", 0.0, 0.2)
        rx = pm.Normal("rx", 0.0, 1.0, shape=K)
        ry = pm.Normal("ry", 0.0, 1.0, shape=K)
        logN = pm.Normal("logN", np.log(1500.0), 1.0, shape=K)
        logb, x, y = lb + 0.5 * rb, cx + 2.0 * rx, cy + 2.0 * ry
        s = pt.exp(logs)
        img = pt.sum(pt.exp(logN)[:, None, None] * pixel(y, s)[:, :, None] * pixel(x, s)[:, None, :], axis=0)
        mu = (pt.exp(logb) + img).ravel() / 2
        pm.Potential("ll", pt.sum(w * (z * pt.log(mu) - mu - pt.gammaln(z + 1))))
    return m


def spot_inputs(rec, blob, cal):
    """Rebuild one blob's patch exactly as the classical job did."""
    t, xy = blob
    stack = b.open_stack(b.RAW_DIR / b.TIFFS[rec][0])
    start = t // c.SUB * c.SUB
    block = t // b.BLOCK
    offset, slope = cal["offset"][block], cal["slope"][block]
    z = np.maximum(np.round((np.asarray(stack["px"][start:start + c.SUB], float) - offset) / slope), 0.0)
    bg = c.rolling_background(stack, start, offset, slope)
    _, lab, n, _ = b.detect(z, bg)
    Z, Wm, org, cen = b.blob_patches(z, lab, n)
    abs_cen = cen + org[:, [2, 1]]
    hit = np.nonzero(org[:, 0] == t - start)[0]
    i = hit[np.hypot(*(abs_cen[hit] - xy).T).argmin()]
    return Z[i], Wm[i], cen[i], org[i], np.log(2 * np.median(bg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots", type=int, default=20)
    parser.add_argument("--init", action="store_true", help="diagnostic: start every chain at the Laplace mode")
    args = parser.parse_args()
    import arviz as az
    import nutpie

    state = c.guard()
    store = bigrun.Store("E88")
    cal = c.calibration("r7")
    chosen, _, blobs = c.load_rec("r7")
    mask = c.mask_share("r7")
    inside = c.in_mask(blobs["xy"], mask)
    rng = np.random.default_rng(887)
    picks = []
    per = args.spots // 2
    for K in (1, 2):
        pool = np.nonzero(inside & (blobs["K"] == K))[0]
        picks += [(K, i) for i in rng.choice(pool, per, replace=False)]
    t0 = time.time()
    compiled = {K: nutpie.compile_pymc_model(spot_model(K)) for K in (1, 2)}
    compile_s = time.time() - t0
    rows = []
    for K, i in picks:
        Z, Wm, cen, org, logb = spot_inputs("r7", (int(blobs["t"][i]), blobs["xy"][i]), cal)
        t0 = time.time()
        model = compiled[K].with_data(z=Z, w=Wm, cx=np.full(K, cen[0]), cy=np.full(K, cen[1]), lb=logb)
        init = {}
        if args.init:                            # the Laplace mode, in the model's unconstrained order
            row = chosen["blob"] == blobs["id"][i]
            xy, logN = chosen["xy"][row] - org[[2, 1]], np.log(chosen["N"][row])
            init = dict(init_mean=np.r_[logb, 0.0, xy[:, 0], xy[:, 1], logN])
        trace = nutpie.sample(model, chains=4, tune=1000, draws=1000, cores=4, seed=int(i) % 2**31, progress_bar=False,
                              **init)
        seconds = time.time() - t0
        post = trace.posterior
        x, y, logN = post["x"].values, post["y"].values, post["logN"].values
        if K == 2:                               # sort each draw on x
            order = np.argsort(x, axis=-1)
            x, y, logN = (np.take_along_axis(v, order, -1) for v in (x, y, logN))
        chk = dict(x=x, y=y, logN=logN, logb=post["logb"].values[..., None], logs=post["logs"].values[..., None])
        r = max(float(az.rhat(v[..., j])) for v in chk.values() for j in range(v.shape[-1]))
        ess = min(float(az.ess(v[..., j])) for v in chk.values() for j in range(v.shape[-1]))
        div = int(trace.sample_stats["diverging"].sum())
        table = chosen["blob"] == blobs["id"][i]
        laplace = chosen["sd"][table]
        sd_axis = float(np.sqrt(0.5 * x.var(axis=(0, 1)).mean() + 0.5 * y.var(axis=(0, 1)).mean()))
        row = dict(K=K, t=int(blobs["t"][i]), rhat=r, ess=ess, divergences=div, seconds=seconds,
                   nuts_sd_px=sd_axis, laplace_sd_px=float(np.mean(laplace)) if len(laplace) else None)
        rows.append(row)
        print(b.rounded(row), flush=True)
        c.guard()
    per_spot = float(np.mean([r["seconds"] for r in rows]))
    ratio = [r["nuts_sd_px"] / r["laplace_sd_px"] for r in rows if r["laplace_sd_px"]]
    report = dict(spots=rows, compile_s=compile_s, per_spot_s=per_spot, max_rhat=max(r["rhat"] for r in rows),
                  divergences=sum(r["divergences"] for r in rows), min_ess=min(r["ess"] for r in rows),
                  extrapolated_200_min_13cpu=(compile_s + 200 * per_spot / (13 // 4)) / 60,
                  median_sd_ratio=dict(all=float(np.median(ratio)),
                                       K1=float(np.median([r["nuts_sd_px"] / r["laplace_sd_px"] for r in rows if r["K"] == 1]))),
                  state=state)
    report["go"] = bool(report["max_rhat"] <= 1.01 and report["divergences"] == 0 and report["extrapolated_200_min_13cpu"] < 10)
    store.save_json("p7_audit_laplace_start" if args.init else "p7_audit", b.rounded(report))
    print(b.rounded({k: v for k, v in report.items() if k != "spots"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
