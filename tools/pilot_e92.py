"""E92 pilots P3, P4, P6, P7 (price), P9. Lines are in results/E92/gates_2026-11-03.json, written first.

    tools/e92_job.sh CAP THREADS -- .venv/bin/python tools/pilot_e92.py p4
    ... pilot_e92.py p3 --device gpu --prec 32 --n 20000 --op vg     # one configuration, fresh process
    ... pilot_e92.py p3scipy --prec 64 --n 20000
    ... pilot_e92.py p9path | p9eval --n 20000 | p6 | p7price --n 800

Each job stores its result with Store("E92"), so a crash loses only the job in flight. GPU jobs hold
bigrun.gpu_lock and set XLA_PYTHON_CLIENT_PREALLOCATE=false (the wrapper does).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e92_core as c  # noqa: E402

from pymc_challenges.bigrun import Store, gpu_lock, online_core_check  # noqa: E402

store = Store("E92")
MATMUL = os.environ.get("E92_MATMUL", "")
SUFFIX = f"_{MATMUL}" if MATMUL else ""
GATES = store.load_json("gates_2026-11-03")  # refuses if the gates were altered


def health():
    h = online_core_check()
    if h["offline_cpus"] != [10, 26] or h["machine_checks"] != 1:
        raise SystemExit(f"STOP: machine state {h}")
    return h


def jax_setup(device):
    os.environ["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    import jax

    jax.config.update("jax_enable_x64", True)
    if MATMUL:  # "highest" forces true float32 products; the default lets the GPU use TF32
        jax.config.update("jax_default_matmul_precision", MATMUL)
    return jax


def ladder_data(n, seed=0):
    d = c.load((2023, 2024))
    perm = np.random.default_rng(seed).permutation(len(d))[:n]
    d = d.iloc[np.sort(perm)]
    X = c.coords(d["lat"], d["lon"], d["t_days"])
    return X, d["anom"].to_numpy()


def timed(fn, *args, repeat):
    """Best wall of ``repeat`` calls after a first (compile) call; no previous result is held."""
    import jax

    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    finite = bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(out)))
    keep = [np.asarray(x).ravel()[:3].tolist() for x in jax.tree.leaves(out)]
    del out
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        runs.append(time.perf_counter() - t0)
        del out
    return {"first_s": first, "best_s": min(runs), "runs_s": runs, "finite": finite, "head": keep}


def p3(device, prec, n, op):
    health()
    jax = jax_setup(device)
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    from pymc_challenges.bigrun import vram_high_water

    dtype = jnp.float32 if prec == 32 else jnp.float64
    X, y = ladder_data(n)
    theta = jnp.asarray(c.THETA0)
    nu = 1.5
    repeat = 3 if n <= 5000 else 1
    dev = jax.devices()[0]
    Xd, yd = jax.device_put(jnp.asarray(X, dtype), dev), jax.device_put(jnp.asarray(y, dtype), dev)
    flops = {"chol": n**3 / 3, "kinv": 2 * n**3, "logp": n**3 / 3, "vg": n**3 / 3 + 2 * n**3}[op]
    ctx = gpu_lock() if device == "gpu" else _null()
    with ctx:
        if op in ("chol", "kinv"):
            K = jax.jit(lambda X: c.build_K(X, theta, nu))(Xd)
            if op == "chol":
                fn, args = jax.jit(jnp.linalg.cholesky), (K,)
            else:
                L = jax.jit(jnp.linalg.cholesky)(K)
                del K
                fn, args = jax.jit(lambda L: cho_solve((L, True), jnp.eye(L.shape[0], dtype=L.dtype))), (L,)
        elif op == "logp":
            fn, args = c.make_logp(nu), (theta, Xd, yd)
        else:
            fn, args = c.make_value_and_grad(nu), (theta, Xd, yd)
        jax.block_until_ready(args)
        before = vram_high_water(dev) if device == "gpu" else {}
        t = timed(fn, *args, repeat=repeat)
        peak = vram_high_water(dev) if device == "gpu" else {}
    matrix_gib = n * n * (4 if prec == 32 else 8) / 1024**3
    row = {"device": device, "prec": prec, "n": n, "op": op, **t, "flops": flops,
           "gflops": flops / t["best_s"] / 1e9, "matrix_gib": matrix_gib,
           "peak_gib": peak.get("peak_gb"), "inputs_gib_before": before.get("in_use_gb"),
           "bench_line_gib": 4.0 * matrix_gib + 0.047, "threads": os.environ.get("OMP_NUM_THREADS"),
           "health": health(), "labels": "measured; flops are nominal counts"}
    row["matmul_precision"] = MATMUL or "default"
    row["preallocate"] = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    store.save_json(f"ladder_{device}_f{prec}_{op}_{n}{SUFFIX}", row)
    print({k: row[k] for k in ("device", "prec", "n", "op", "first_s", "best_s", "gflops", "peak_gib",
                               "finite")}, flush=True)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def p3scipy(prec, n):
    health()
    import scipy.linalg.lapack as lapack

    jax = jax_setup("cpu")
    import jax.numpy as jnp

    X, _ = ladder_data(n)
    K64 = np.asarray(jax.jit(lambda X: c.build_K(X, jnp.asarray(c.THETA0), 1.5))(jnp.asarray(X)))
    K = np.asfortranarray(K64.astype(np.float32 if prec == 32 else np.float64))
    del K64
    potrf, potri = (lapack.spotrf, lapack.spotri) if prec == 32 else (lapack.dpotrf, lapack.dpotri)
    rows = {}
    reps = 3 if n <= 5000 else 2
    for op in ("chol", "kinv"):
        runs = []
        for _ in range(reps):
            A = K.copy(order="F")
            t0 = time.perf_counter()
            L, info = potrf(A, lower=1, overwrite_a=1, clean=0)
            dt = time.perf_counter() - t0
            if op == "kinv":
                t0 = time.perf_counter()
                Ki, info2 = potri(L, lower=1, overwrite_c=1)
                dt = time.perf_counter() - t0
                info = max(info, info2)
            runs.append(dt)
            del A
        flops = n**3 / 3 if op == "chol" else 2 * n**3 / 3
        rows[op] = {"best_s": min(runs), "runs_s": runs, "info": int(info), "flops": flops,
                    "gflops": flops / min(runs) / 1e9}
    row = {"device": "scipy_cpu", "prec": prec, "n": n, "ops": rows,
           "threads": os.environ.get("OPENBLAS_NUM_THREADS"), "health": health(),
           "labels": "measured; kinv is LAPACK potri (n^3 x 2/3 nominal flops) from the factor"}
    store.save_json(f"ladder_scipy_f{prec}_{n}", row)
    print(row, flush=True)


def p4():
    health()
    jax = jax_setup("cpu")
    import jax.numpy as jnp

    X, y = ladder_data(2000, seed=4)
    Xd, yd = jnp.asarray(X), jnp.asarray(y)
    thetas = [c.THETA0, *c.prior_draws(2, seed=4)]
    out = {}
    for nu in (0.5, 1.5):
        vg = c.make_value_and_grad(nu, block=512)
        lp = c.make_logp(nu)

        def full(theta):
            K = c.kernel_block(Xd, Xd, jnp.arange(2000), jnp.arange(2000), theta, nu)
            Lf = jnp.linalg.cholesky(K)
            a = jax.scipy.linalg.cho_solve((Lf, True), yd)
            return (-0.5 * yd @ a - jnp.sum(jnp.log(jnp.diag(Lf))) - 1000 * np.log(2 * np.pi)
                    + c.prior_logp(theta))

        ad = jax.jit(jax.grad(full))
        for k, th in enumerate(thetas):
            th = jnp.asarray(th)
            v, g = (np.asarray(z) for z in vg(th, Xd, yd))
            g_ad = np.asarray(ad(th))
            g_fd = np.zeros(10)
            for j in range(10):
                e = np.zeros(10)
                e[j] = 1e-5
                g_fd[j] = (float(lp(th + e, Xd, yd)) - float(lp(th - e, Xd, yd))) / 2e-5
            rel = {}
            for name, ref in (("autodiff", g_ad), ("finite_diff", g_fd)):
                scale = np.maximum(np.abs(ref), 1e-2 * np.abs(ref).max())
                rel[name] = (np.abs(g - ref) / scale).tolist()
            out[f"nu{nu}_theta{k}"] = {"theta": np.asarray(th).tolist(), "value": float(v),
                                       "value_full": float(full(th)), "hand": g.tolist(),
                                       "autodiff": g_ad.tolist(), "finite_diff": g_fd.tolist(), "rel": rel,
                                       "max_rel": max(max(r) for r in rel.values())}
            print(nu, k, out[f"nu{nu}_theta{k}"]["max_rel"], flush=True)
    worst = max(v["max_rel"] for v in out.values())
    store.save_json("pilot_p4", {"results": out, "max_rel": worst, "line": GATES["P4_gradient"]["go"],
                                 "pass": worst < 1e-3, "health": health(), "labels": "measured"})
    print("P4 max rel", worst, "pass", worst < 1e-3)


def map_fit(vg, X, y, theta0, maxiter=200, record=None):
    """L-BFGS on the negative log posterior; ``record`` collects (theta, value, gradient) per call."""
    import jax.numpy as jnp
    from scipy.optimize import minimize

    calls = []

    def f(th):
        v, g = vg(jnp.asarray(th), X, y)
        v, g = float(v), np.asarray(g, np.float64)
        calls.append((th.copy(), v, g))
        if not np.isfinite(v):
            return 1e30, np.zeros_like(th)
        return -v, -g

    t0 = time.perf_counter()
    res = minimize(f, np.asarray(theta0, float), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=3 * maxiter, ftol=1e-12, gtol=1e-4))
    return res, calls, time.perf_counter() - t0


def edges(theta):
    lo, hi = c.PRIOR_MU - 2.326 * c.PRIOR_SD, c.PRIOR_MU + 2.326 * c.PRIOR_SD
    return [c.NAMES[j] for j in range(10) if not lo[j] < theta[j] < hi[j]]


def p6():
    """Kernel and coordinates at SMALL, by whole-float held-out joint log score."""
    health()
    jax = jax_setup("gpu")
    import jax.numpy as jnp

    d = c.load((2024,), crop=c.SMALL_CROP)
    held = c.holdout(d["float_id"])
    y = d["anom"].to_numpy()
    draws = c.prior_draws(4000, seed=6)
    total_sd = np.sqrt(np.exp(2 * draws[:, 0]) + np.exp(2 * draws[:, 4]) + np.exp(2 * draws[:, 8]))
    prior_sd = np.quantile(total_sd, [0.025, 0.5, 0.975]).tolist()
    out = {"n_train": int((~held).sum()), "n_held": int(held.sum()),
           "held_floats": int(d.loc[held, "float_id"].nunique()), "prior_total_sd_q": prior_sd,
           "prior_ok": bool(prior_sd[0] > 0.1 and prior_sd[2] < 5), "options": {}}
    with gpu_lock():
        for nu in (0.5, 1.5):
            for proj in ("equirect", "mercator"):
                X = c.coords(d["lat"], d["lon"], d["t_days"], proj)
                Xt, Xs = jnp.asarray(X[~held]), jnp.asarray(X[held])
                yt, ys = jnp.asarray(y[~held]), y[held]
                vg = c.make_value_and_grad(nu)
                res, calls, secs = map_fit(vg, Xt, yt, c.THETA0)
                pred = c.make_predict(nu)
                scores = c.float_scores(pred, jnp.asarray(res.x), Xt, yt, Xs, ys,
                                        d.loc[held, "float_id"].to_numpy())
                key = f"nu{nu}_{proj}"
                out["options"][key] = {"theta": res.x.tolist(), "logpost": float(-res.fun),
                                       "nit": int(res.nit), "nfev": int(res.nfev), "message": str(res.message),
                                       "seconds": secs, "edges": edges(res.x), "float_scores": scores.tolist(),
                                       "score_sum": float(scores.sum())}
                print(key, round(-res.fun, 1), res.nit, round(secs, 1), "score", round(scores.sum(), 2),
                      "edges", edges(res.x), np.round(np.r_[np.exp(res.x[:9]), res.x[9]], 3).tolist(),
                      flush=True)
                jax.clear_caches()

    def compare(a, b):
        da = np.asarray(out["options"][a]["float_scores"]) - np.asarray(out["options"][b]["float_scores"])
        return {"sum_diff": float(da.sum()), "se": float(np.sqrt(len(da)) * da.std(ddof=1)),
                "z": float(da.sum() / (np.sqrt(len(da)) * da.std(ddof=1)))}

    nu_cmp = compare("nu1.5_equirect", "nu0.5_equirect")
    nu = 1.5 if nu_cmp["z"] > 2 else 0.5
    proj_cmp = compare(f"nu{nu}_mercator", f"nu{nu}_equirect")
    proj = "mercator" if proj_cmp["z"] > 2 else "equirect"
    out.update({"nu_1.5_minus_0.5": nu_cmp, "mercator_minus_equirect": proj_cmp, "chosen": {"nu": nu, "proj": proj},
                "line": GATES["P6_choices"]["go"], "health": health(), "labels": "measured"})
    store.save_json("small_choices", out)
    store.mark_done("small_choices", nu=nu, proj=proj)
    print("chosen", nu, proj, nu_cmp, proj_cmp)


def p9path():
    """A float64 MAP path at n = 5000 (the ladder subset), with the P6 kernel; 20 iterates kept."""
    health()
    jax = jax_setup("gpu")
    import jax.numpy as jnp

    ch = store.load_json("small_choices")["chosen"]
    d = c.load((2023, 2024))
    perm = np.sort(np.random.default_rng(0).permutation(len(d))[:5000])
    d = d.iloc[perm]
    X = jnp.asarray(c.coords(d["lat"], d["lon"], d["t_days"], ch["proj"]))
    y = jnp.asarray(d["anom"].to_numpy())
    with gpu_lock():
        res, calls, secs = map_fit(c.make_value_and_grad(ch["nu"]), X, y, c.THETA0)
    thetas = np.array([t for t, _, _ in calls])
    pick = np.unique(np.linspace(0, len(thetas) - 1, 20).round().astype(int))
    store.save_array("p9_iterates", thetas[pick])
    store.save_json("p9_path", {"nu": ch["nu"], "proj": ch["proj"], "n": 5000, "calls": len(calls),
                                "nit": int(res.nit), "seconds": secs, "theta_map": res.x.tolist(),
                                "logpost": float(-res.fun), "message": str(res.message), "picked": pick.tolist(),
                                "edges": edges(res.x), "labels": "measured"})
    print("path", len(calls), res.nit, round(secs, 1), np.round(np.r_[np.exp(res.x[:9]), res.x[9]], 3).tolist(),
          edges(res.x))


def p9eval(n, prec, device):
    """Log density and hand gradient at the stored iterates, at size n, in one precision and device."""
    health()
    jax = jax_setup(device)
    import jax.numpy as jnp

    path = store.load_json("p9_path")
    thetas = store.load_array("p9_iterates")
    d = c.load((2023, 2024))
    d = d.iloc[np.sort(np.random.default_rng(0).permutation(len(d))[:n])]
    X = c.coords(d["lat"], d["lon"], d["t_days"], path["proj"])
    dt = jnp.float32 if prec == 32 else jnp.float64
    Xd, yd = jnp.asarray(X, dt), jnp.asarray(d["anom"].to_numpy(), dt)
    vg = c.make_value_and_grad(path["nu"])
    vals, grads = [], []
    with gpu_lock() if device == "gpu" else _null():
        t0 = time.perf_counter()
        for th in thetas:
            v, g = vg(jnp.asarray(th), Xd, yd)
            vals.append(float(v))
            grads.append(np.asarray(g, np.float64))
        secs = time.perf_counter() - t0
    name = f"p9_vals_{n}_f{prec}_{device}{SUFFIX}"
    store.save_json(name, {"n": n, "prec": prec, "device": device, "matmul_precision": MATMUL or "default",
                           "values": vals, "grads": grads, "seconds": secs, "health": health(),
                           "labels": "measured"})
    print(name, round(secs, 1), "s", vals[:2], flush=True)


def p9cmp():
    """f32 GPU against the f64 reference (GPU where it fitted, else CPU), against the P9 lines."""
    out = {}
    for n in (5000, 10000, 20000):
        for suffix in ("", "_highest"):
            p32 = store.path / f"p9_vals_{n}_f32_gpu{suffix}.json"
            ref = next((store.path / f"p9_vals_{n}_f64_{dev}.json" for dev in ("gpu", "cpu")
                        if (store.path / f"p9_vals_{n}_f64_{dev}.json").exists()), None)
            if not p32.exists() or ref is None:
                continue
            a, b = store.load_json(p32.stem), store.load_json(ref.stem)
            v32, v64 = np.array(a["values"]), np.array(b["values"])
            g32, g64 = np.array(a["grads"]), np.array(b["grads"])
            cos = np.sum(g32 * g64, 1) / (np.linalg.norm(g32, axis=1) * np.linalg.norm(g64, axis=1))
            out[f"{n}{suffix}"] = {"reference": ref.stem, "abs_err": np.abs(v32 - v64),
                                   "max_abs_err": float(np.max(np.abs(v32 - v64))), "cosine": cos,
                                   "min_cosine": float(cos.min()),
                                   "grad_rel_err_max": float(np.max(np.linalg.norm(g32 - g64, axis=1)
                                                                    / np.linalg.norm(g64, axis=1))),
                                   "seconds_f32": a["seconds"], "seconds_f64": b["seconds"]}
            print(n, suffix, out[f"{n}{suffix}"]["max_abs_err"], out[f"{n}{suffix}"]["min_cosine"])
    top = out.get("20000")
    verdict = None if top is None else {
        "logp_err_20k_under_0.1": top["max_abs_err"] < 0.1,
        "cosine_over_0.9999_every_n": all(v["min_cosine"] > 0.9999 for k, v in out.items() if "_" not in k)}
    store.save_json("pilot_p9", {"results": out, "verdict": verdict, "line": GATES["P9_precision"]["go"],
                                 "labels": "measured"})
    print("P9 verdict", verdict)


def pymc_model(X, y, nu):
    import pymc as pm
    import pytensor.tensor as pt

    n = len(y)
    with pm.Model() as model:
        th = pm.Normal("theta", mu=c.PRIOR_MU, sigma=c.PRIOR_SD, shape=10)
        e = pt.exp(th[:9])
        Xc = pt.as_tensor_variable(X)
        dx = Xc[:, None, 0] - Xc[None, :, 0]
        dy = Xc[:, None, 1] - Xc[None, :, 1]
        dt = Xc[:, None, 2] - Xc[None, :, 2]
        r1 = ((dx - th[9] * dt) / e[1]) ** 2 + (dy / e[2]) ** 2 + (dt / e[3]) ** 2
        r2 = (dx / e[5]) ** 2 + (dy / e[6]) ** 2 + (dt / e[7]) ** 2

        def m(r2_):
            r = pt.sqrt(r2_ + 1e-12)  # differs from the JAX version only at r = 0, where it is O(1e-6)
            if nu == 0.5:
                return pt.exp(-r)
            s = np.sqrt(3.0) * r
            return (1 + s) * pt.exp(-s)

        t = X[:, 2] / 365.25
        H = c.BETA_SD[0] ** 2 + c.BETA_SD[1] ** 2 * np.outer(t, t)
        K = e[0] ** 2 * m(r1) + e[4] ** 2 * m(r2) + e[8] ** 2 * pt.eye(n) + H
        pm.MvNormal("y", mu=np.zeros(n), cov=K, observed=y)
    return model


def p7price(n):
    """nutpie-style logp+dlogp of the PyMC model on one thread, at size n."""
    health()
    ch = store.load_json("small_choices")["chosen"]
    d = c.load((2024,), crop=c.SMALL_CROP)
    d = d.iloc[np.sort(np.random.default_rng(7).permutation(len(d))[:n])]
    X = c.coords(d["lat"], d["lon"], d["t_days"], ch["proj"])
    model = pymc_model(X, d["anom"].to_numpy(), ch["nu"])
    t0 = time.perf_counter()
    f = model.logp_dlogp_function(ravel_inputs=True, mode="NUMBA")
    f.set_extra_values({})
    q0 = np.asarray(c.THETA0, float)
    f._pytensor_function(q0)
    compile_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    times = []
    for _ in range(50 if n <= 800 else 20):
        q = q0 + 0.05 * rng.normal(size=10)
        t0 = time.perf_counter()
        lp, g = f._pytensor_function(q)[:2]
        times.append(time.perf_counter() - t0)
    # the same point through the JAX hand model, to show the two are one model
    jax = jax_setup("cpu")
    import jax.numpy as jnp

    v_jax = float(c.make_logp(ch["nu"])(jnp.asarray(q0), jnp.asarray(X), jnp.asarray(d["anom"].to_numpy())))
    lp0 = float(f._pytensor_function(q0)[0])
    med = float(np.median(times))
    out = {"n": n, "nu": ch["nu"], "proj": ch["proj"], "compile_s": compile_s, "median_ms": med * 1e3,
           "min_ms": min(times) * 1e3, "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
           "logp_pymc": lp0, "logp_jax": v_jax, "chain_wall_min_at_21000": 21000 * med / 60,
           "health": health(), "labels": "measured; the chain wall is extrapolated from 21,000 gradients"}
    store.save_json(f"p7_price_{n}", out)
    print(out)


def laplace(vg, X, y, theta0, h=1e-4):
    """Mode by L-BFGS, Hessian by central differences of the hand gradient (symmetrised)."""
    import jax.numpy as jnp

    res, calls, secs = map_fit(vg, X, y, theta0, maxiter=300)
    H = np.zeros((10, 10))
    for j in range(10):
        e = np.zeros(10)
        e[j] = h
        gp = np.asarray(vg(jnp.asarray(res.x + e), X, y)[1])
        gm = np.asarray(vg(jnp.asarray(res.x - e), X, y)[1])
        H[j] = (gp - gm) / (2 * h)
    H = 0.5 * (H + H.T)
    cov = np.linalg.inv(-H)
    return res, cov, secs


def p7audit(n=400, draws=1000, tune=1000):
    """nutpie 4 chains against Laplace at the hand model's mode, SMALL crop subset of size n."""
    health()
    import arviz as az
    import pymc as pm

    ch = store.load_json("small_choices")["chosen"]
    d = c.load((2024,), crop=c.SMALL_CROP)
    d = d.iloc[np.sort(np.random.default_rng(7).permutation(len(d))[:n])]
    X = c.coords(d["lat"], d["lon"], d["t_days"], ch["proj"])
    y = d["anom"].to_numpy()
    jax = jax_setup("cpu")
    import jax.numpy as jnp

    res, cov, lap_s = laplace(c.make_value_and_grad(ch["nu"], block=400), jnp.asarray(X), jnp.asarray(y),
                              c.THETA0)
    lap_sd = np.sqrt(np.diag(cov)) if np.all(np.diag(cov) > 0) else np.full(10, np.nan)
    print("laplace", np.round(res.x, 3).tolist(), np.round(lap_sd, 3).tolist(), flush=True)
    model = pymc_model(X, y, ch["nu"])
    t0 = time.perf_counter()
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=4, nuts_sampler="nutpie", random_seed=92,
                          progressbar=False)
    wall = time.perf_counter() - t0
    post = idata.posterior["theta"].values.reshape(-1, 10)
    summ = az.summary(idata, var_names=["theta"])
    rhat = float(np.nanmax(summ["r_hat"].to_numpy()))
    mean, sd = post.mean(0), post.std(0)
    z = (mean - res.x) / sd
    ratio = lap_sd / sd
    div = int(idata.sample_stats["diverging"].values.sum())
    verdict = {"means_within_half_sd": bool(np.all(np.abs(z) < 0.5)),
               "sd_ratio_0.7_1.3": bool(np.all((ratio > 0.7) & (ratio < 1.3))), "rhat_1.01": rhat <= 1.01}
    store.save_array("p7_audit_draws_400", post[::4])
    store.save_json("p7_audit_400", {
        "n": n, "nu": ch["nu"], "proj": ch["proj"], "laplace_mode": res.x, "laplace_sd": lap_sd,
        "laplace_s": lap_s, "nuts_mean": mean, "nuts_sd": sd, "z_mode_vs_mean": z, "sd_ratio_laplace_over_nuts": ratio,
        "max_rhat": rhat, "min_ess_bulk": float(summ["ess_bulk"].min()), "divergences": div, "wall_s": wall,
        "tuning_steps": idata.posterior.attrs.get("tuning_steps"), "verdict": verdict,
        "pass": all(verdict.values()), "line": GATES["P7_audit_price"]["n400_audit"]["go"],
        "health": health(), "labels": "measured"})
    print("P7 n=400", verdict, "rhat", rhat, "div", div, "wall", round(wall), "\nz", np.round(z, 2).tolist(),
          "\nratio", np.round(ratio, 2).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("--device", default="gpu")
    p.add_argument("--prec", type=int, default=32)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--op", default="vg")
    a = p.parse_args()
    if a.job == "p3":
        p3(a.device, a.prec, a.n, a.op)
    elif a.job == "p3scipy":
        p3scipy(a.prec, a.n)
    elif a.job == "p4":
        p4()
    elif a.job == "p6":
        p6()
    elif a.job == "p9path":
        p9path()
    elif a.job == "p9eval":
        p9eval(a.n, a.prec, a.device)
    elif a.job == "p9cmp":
        p9cmp()
    elif a.job == "p7price":
        p7price(a.n)
    elif a.job == "p7audit":
        p7audit(a.n)
    else:
        raise SystemExit(f"unknown job {a.job}")


if __name__ == "__main__":
    main()
