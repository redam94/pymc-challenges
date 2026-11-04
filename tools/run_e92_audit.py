"""E92 FULL: the PyMC audit at n = 800 and 2,000 (lines in results/E92/gates_2026-11-03c.json).

    tools/e92_job.sh 4 4 -- .venv/bin/python tools/run_e92_audit.py laplace 2000     (cores 2-5)
    E92_BLAS_THREADS=1 E92_CORES=3 tools/e92_job.sh 4 1 -- .venv/bin/python tools/run_e92_audit.py chain 2000 1
    ... run_e92_audit.py summary 2000

Each chain is its own process and its own chunk (``pymc_audit_<n>``, index = chain), so a crash loses only
the chains in flight. ``summary`` joins the four chains, computes r_hat and ESS, and compares with Laplace.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e92_core as c  # noqa: E402
import pilot_e92 as p  # noqa: E402

store = p.store
GATES = store.load_json("gates_2026-11-03c")  # refuses if altered
TUNE, DRAWS, SEED0 = 1000, 1000, 920


def subset(n):
    ch = store.load_json("small_choices")["chosen"]
    d = c.load((2024,), crop=c.SMALL_CROP)
    d = d.iloc[np.sort(np.random.default_rng(7).permutation(len(d))[:n])]
    return ch, c.coords(d["lat"], d["lon"], d["t_days"], ch["proj"]), d["anom"].to_numpy()


def laplace(n):
    n = int(n)
    p.health()
    ch, X, y = subset(n)
    p.jax_setup("cpu")
    import jax.numpy as jnp

    t0 = time.perf_counter()
    res, cov, secs = p.laplace(c.make_value_and_grad(ch["nu"], block=min(n, 1000)), jnp.asarray(X),
                               jnp.asarray(y), c.THETA0)
    ev = np.linalg.eigvalsh(cov)
    store.save_json(f"pymc_audit_laplace_{n}", {
        "n": n, "mode": res.x, "cov": cov, "sd": np.sqrt(np.clip(np.diag(cov), 0, None)), "logpost": -res.fun,
        "nit": int(res.nit), "message": str(res.message), "cov_eigen_min": float(ev.min()),
        "edges": p.edges(res.x), "wall_s": time.perf_counter() - t0, "health": p.health(), "labels": "measured"})
    print("laplace", n, np.round(res.x, 3).tolist(), "sd", np.round(np.sqrt(np.diag(cov)), 3).tolist(), flush=True)
    store.mark_done(f"pymc_audit_laplace_{n}")


def chain(n, k):
    n, k = int(n), int(k)
    p.health()
    import pymc as pm

    ch, X, y = subset(n)
    model = p.pymc_model(X, y, ch["nu"])
    t0 = time.perf_counter()
    with model:
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=1, cores=1, nuts_sampler="nutpie",
                          random_seed=SEED0 + k, progressbar=False)
    wall = time.perf_counter() - t0
    post = idata.posterior["theta"].values.reshape(-1, 10)
    ss = idata.sample_stats
    steps = ss["n_steps"].values.ravel() if "n_steps" in ss else np.zeros(DRAWS)
    div = ss["diverging"].values.ravel()
    store.save_chunk(f"pymc_audit_{n}", k, {"theta": post, "n_steps": steps, "diverging": div.astype(np.int8),
                                            "wall_s": np.array(wall)})
    store.save_json(f"pymc_audit_{n}_chain{k}", {"n": n, "chain": k, "seed": SEED0 + k, "wall_s": wall,
                                                 "draw_steps": int(steps.sum()), "divergences": int(div.sum()),
                                                 "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                                                 "health": p.health(), "labels": "measured"})
    print("chain", n, k, "wall", round(wall), "div", int(div.sum()), "steps", int(steps.sum()), flush=True)


def summary(n):
    import arviz as az
    import xarray as xr

    n = int(n)
    if todo := store.pending(f"pymc_audit_{n}", 4):
        raise SystemExit(f"chains {todo} of n = {n} are not finished")
    ch = [store.load_chunk(f"pymc_audit_{n}", k) for k in range(4)]
    parts = {k: np.concatenate([np.atleast_1d(x[k]) for x in ch]) for k in ("theta", "n_steps", "diverging")}
    theta = parts["theta"].reshape(4, DRAWS, 10)
    ds = xr.Dataset({"theta": (("chain", "draw", "dim"), theta)})
    summ = az.summary(ds, var_names=["theta"])
    rhat = float(np.nanmax(summ["r_hat"].to_numpy()))
    post = theta.reshape(-1, 10)
    lap = store.load_json(f"pymc_audit_laplace_{n}")
    mode, lap_sd = np.asarray(lap["mode"]), np.asarray(lap["sd"])
    mean, sd = post.mean(0), post.std(0)
    z = (mean - mode) / sd
    ratio = lap_sd / sd
    verdict = {"means_within_half_sd": bool(np.all(np.abs(z) < 0.5)),
               "sd_ratio_0.7_1.3": bool(np.all((ratio > 0.7) & (ratio < 1.3))), "rhat_1.01": rhat <= 1.01}
    # The no-go action: importance sampling from NUTS (Gaussian at the NUTS moments as the proposal is
    # what Laplace would be if it were right; we report the sd the NUTS draws give, per coordinate).
    walls = [store.load_json(f"pymc_audit_{n}_chain{k}")["wall_s"] for k in range(4)]
    store.save_array(f"pymc_audit_draws_{n}", post[::4])
    store.save_json(f"pymc_audit_{n}", {
        "n": n, "laplace_mode": mode, "laplace_sd": lap_sd, "nuts_mean": mean, "nuts_sd": sd,
        "z_mode_vs_mean": z, "sd_ratio_laplace_over_nuts": ratio, "max_rhat": rhat,
        "rhat": summ["r_hat"].to_numpy(), "ess_bulk": summ["ess_bulk"].to_numpy(),
        "min_ess_bulk": float(summ["ess_bulk"].min()), "divergences": int(parts["diverging"].sum()),
        "chain_wall_s": walls, "draw_steps": int(parts["n_steps"].sum()), "verdict": verdict,
        "pass": all(verdict.values()), "failing_coordinates": [c.NAMES[j] for j in range(10)
                                                               if not (0.7 < ratio[j] < 1.3) or abs(z[j]) >= 0.5],
        "line": GATES["pymc_audit"]["go"], "labels": "measured"})
    store.mark_done(f"pymc_audit_{n}", passed=all(verdict.values()))
    print("audit", n, verdict, "rhat", round(rhat, 4), "\nz", np.round(z, 2).tolist(), "\nratio",
          np.round(ratio, 2).tolist(), flush=True)


if __name__ == "__main__":
    {"laplace": laplace, "chain": chain, "summary": summary}[sys.argv[1]](*sys.argv[2:])
