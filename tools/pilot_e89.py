"""E89's annual pilots P3-P7: the marginal GP (M1), the HSGP check (M2) and the value-of-information
loop, as the functions the notebook will reuse, with one subcommand per pilot.

    OPENBLAS_NUM_THREADS=1 taskset -c 1 .venv/bin/python tools/memguard.py --max-gb 3 -- \
        .venv/bin/python tools/pilot_e89.py p3-model --threads 1

Set the thread variables before starting Python. Results go to ``results/E89/`` as small JSON;
draws and county covariances go to ``~/.cache/pymc-challenges/e89/pilots``. The annual path never
executes JAX (``jax_executed`` says why that is the test), and a fit asserts it after saving. PyMC
is imported inside the functions that need it, so the VOI stage runs on NumPy alone, as
``plans/large/E89.md`` section 4 describes. Pilot results are in that plan.

Do not use more than one BLAS thread per chain when chains share cores: P3 measured a tail of 390 ms
against a median of 153 ms.
"""

import argparse
import json
import multiprocessing
import os
import re
import resource
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import logsumexp, ndtr

from build_e89_data import hsgp_variance, new_site_table
from pymc_challenges.bigrun import CACHE_DIR, Store

WORK = CACHE_DIR / "e89"
TABLES = WORK / "out"
RAW = WORK / "raw"
LINE = 9.05  # design values round to 0.1, so "above 9.0" means at least 9.05
YEARS = (2023, 2024, 2025)
METHODS = ("T640", "BAM", "other")  # FRM is the reference, so predictions are FRM-equivalent
SCALES = ("NEIGHBORHOOD", "URBAN SCALE", "REGIONAL SCALE", "MIDDLE SCALE", "MICROSCALE", "unknown")
HYPERS = ("eta", "ell", "tau")
PRIOR_ELL = (np.log(400.0), 0.6)
PRIOR_ETA = (np.log(0.2), 0.6)
PRIOR_TAU = (np.log(0.1), 0.6)
SEED = 8900
HALF_WIDTH_KM = (2262.0, 1419.0)  # recomputed from the county table in p5, and asserted close


def store():
    return Store("E89")


def cache():
    return Store("pilots", root=WORK)


def jax_executed():
    """Has JAX really run in this process? Never touch ``sys.modules["jax"]`` to find out.

    nutpie imports arviz, and ``arviz_base.io_numpyro`` calls ``lazy_loader.load("jax")``, so a stub
    sits in ``sys.modules`` on every fit. Any attribute access on it imports all of JAX, which also
    maps libcuda. ``"jax" in sys.modules`` is therefore always true here and proves nothing; a stub
    has no submodules, and real JAX has hundreds.
    """
    return any(name.startswith(("jax.", "jaxlib")) for name in sys.modules)


def runtime_check():
    """The annual path's promises: float64, JAX never executed, no CUDA library mapped."""
    maps = Path("/proc/self/maps").read_text()
    out = {"jax_in_sys_modules_literal": "jax" in sys.modules,
           "jax_module_type": type(sys.modules.get("jax")).__name__,
           "jax_executed": jax_executed(), "cuda_mapped": "libcuda" in maps,
           "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
           "blas_env": os.environ.get("OPENBLAS_NUM_THREADS"),
           "cores": sorted(os.sched_getaffinity(0))}
    if "pytensor" in sys.modules:
        out["floatX"] = sys.modules["pytensor"].config.floatX
    return out


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


# Tables and exclusions ---------------------------------------------------------------------------


def load_tables():
    """CONUS sites and counties, from the built tables."""
    sites = pd.read_csv(TABLES / "e89_sites.csv", dtype={"site": str, "fips": str})
    counties = pd.read_csv(TABLES / "e89_counties.csv", dtype={"fips": str})
    sites = sites[sites["conus"]].reset_index(drop=True)
    sites["scale"] = sites["scale"].fillna("unknown")
    assert set(sites["scale"]) <= set(SCALES), "a measurement scale the model does not know"
    return sites, counties


def excluded_sites(sites):
    """Window sites with no row in EPA's workbook and an exclusion covering the annual standard.

    The plan's rule was not recorded. This one: any 88101 monitor at the site carries an ``All`` or
    ``PM25 Annual 2024`` exclusion whose dates overlap 2023-2025.
    """
    from build_e89_data import design_values

    pattern = re.compile(r"(All|PM25 Annual 2024) \((\d{4}-\d\d-\d\d) - (\d{4}-\d\d-\d\d|Present)\)")
    with zipfile.ZipFile(RAW / "aqs_monitors.zip") as archive, \
            archive.open(archive.namelist()[0]) as member:
        monitors = pd.read_csv(member, dtype=str, keep_default_na=False,
                               usecols=["State Code", "County Code", "Site Number",
                                        "Parameter Code", "Exclusions"])
    monitors = monitors[monitors["Parameter Code"] == "88101"]
    ids = monitors["State Code"] + monitors["County Code"] + monitors["Site Number"]

    def covers(text):
        return any(start <= "2025-12-31" and (end == "Present" or end >= "2023-01-01")
                   for _, start, end in pattern.findall(text))

    flagged = set(ids[monitors["Exclusions"].map(covers)])
    workbook = design_values(RAW / "pm25_designvalues_2023_2025_final_06_12_26.xlsx")
    return sorted((set(sites["site"]) - set(workbook["site"])) & flagged)


def design(sites, density_scale=None):
    """What the model needs per site. Density is standardised on these sites unless a scale is given,
    which is how a fit's coefficients are carried to other sites.
    """
    density = sites["log_density"].to_numpy(float)
    centre, spread = density_scale or (density.mean(), density.std())
    shares = sites[[f"share_{m}" for m in METHODS]].to_numpy(float)
    has = sites[[f"has_{y}" for y in YEARS]].to_numpy(float)
    xy = sites[["x_km", "y_km"]].to_numpy(float)
    return {"site": sites["site"].to_numpy(str), "y": sites["log_mean"].to_numpy(float),
            "v": sites["var_log"].to_numpy(float), "density": (density - centre) / spread,
            "density_scale": (centre, spread), "methods": shares,
            "years": has / has.sum(axis=1, keepdims=True),
            "scale": sites["scale"].map(SCALES.index).to_numpy(int), "xy": xy,
            "distance": distance(xy, xy)}


def distance(a, b):
    return np.sqrt(((a[:, None] - b[None]) ** 2).sum(axis=-1))


def matern32(gap, ell):
    r = np.sqrt(3.0) * gap / ell
    return (1 + r) * np.exp(-r)


# Models ------------------------------------------------------------------------------------------


def mean_terms(d, pm):
    """Priors and the mean of the site model, shared by M1 and M2."""
    alpha = pm.Normal("alpha", 2.0, 0.5)
    beta_density = pm.Normal("beta_density", 0.0, 0.2)
    beta_method = pm.Normal("beta_method", 0.0, 0.2, dims="method")
    delta = pm.ZeroSumNormal("delta", sigma=0.1, dims="year")
    mu = (alpha + beta_density * d["density"] + pm.math.dot(d["methods"], beta_method)
          + pm.math.dot(d["years"], delta))
    eta = pm.LogNormal("eta", *PRIOR_ETA)
    ell = pm.LogNormal("ell", *PRIOR_ELL)
    tau = pm.LogNormal("tau", *PRIOR_TAU, dims="scale")
    return mu, eta, ell, tau


def m1(d):
    """Exact GP with the field integrated out, as section 4 writes it. 15 free parameters."""
    import pymc as pm
    import pytensor.tensor as pt

    coords = {"method": METHODS, "year": YEARS, "scale": SCALES, "site": d["site"]}
    with pm.Model(coords=coords) as model:
        mu, eta, ell, tau = mean_terms(d, pm)
        r = np.sqrt(3.0) * pt.constant(d["distance"]) / ell
        nugget = pt.diag(tau[d["scale"]] ** 2 + d["v"])
        pm.MvNormal("y", mu=mu, cov=eta ** 2 * (1 + r) * pt.exp(-r) + nugget, observed=d["y"],
                    dims="site")
    return model


def hsgp_modes(half, c, modes):
    """Square frequencies per axis and the box half-widths, for a product basis."""
    box = np.asarray(half, float) * c
    j1, j2 = np.meshgrid(*(np.arange(1, m + 1) for m in modes), indexing="ij")
    return j1.ravel(), j2.ravel(), box


def hsgp_basis(xy, j1, j2, box):
    """Dirichlet eigenfunctions on the box, at points relative to its centre."""
    return (np.sin(np.pi * j1 * (xy[:, [0]] + box[0]) / (2 * box[0])) / np.sqrt(box[0])
            * np.sin(np.pi * j2 * (xy[:, [1]] + box[1]) / (2 * box[1])) / np.sqrt(box[1]))


def matern32_density(omega2, ell):
    """Spectral density of the 2-D Matern 3/2 at squared frequency ``omega2``. Unit variance."""
    return 6 * np.pi * 3 ** 1.5 / ell ** 3 * (3 / ell ** 2 + omega2) ** -2.5


def m2(d, counties, centre, c, modes):
    """M1 plus a county 'has a monitor' term on a shared latent HSGP. Built to be timed in P5."""
    import pymc as pm
    import pytensor.tensor as pt

    j1, j2, box = hsgp_modes(HALF_WIDTH_KM, c, modes)
    omega2 = (np.pi * j1 / (2 * box[0])) ** 2 + (np.pi * j2 / (2 * box[1])) ** 2
    phi_s = hsgp_basis(d["xy"] - centre, j1, j2, box)
    phi_c = hsgp_basis(counties[["x_km", "y_km"]].to_numpy(float) - centre, j1, j2, box)
    density = (counties["log_density"].to_numpy(float) - d["density_scale"][0]) / d["density_scale"][1]
    coords = {"method": METHODS, "year": YEARS, "scale": SCALES, "mode": np.arange(j1.size)}
    with pm.Model(coords=coords) as model:
        mu, eta, ell, tau = mean_terms(d, pm)
        z = pm.Normal("z", 0.0, 1.0, dims="mode")
        weight = eta * pt.sqrt(matern32_density(omega2, ell)) * z
        pm.Normal("y", mu=mu + pt.dot(phi_s, weight), sigma=pt.sqrt(tau[d["scale"]] ** 2 + d["v"]),
                  observed=d["y"])
        rate = pm.Normal("rate", -1.0, 1.0)
        rate_density = pm.Normal("rate_density", 0.0, 1.0)
        share = pm.Normal("share", 0.0, 3.0)
        pm.Bernoulli("has_monitor", logit_p=rate + rate_density * density
                     + share * pt.dot(phi_c, weight),
                     observed=counties["monitored"].to_numpy(int))
    return model


def time_logp(model, evals, concurrent):
    """Time logp and gradient together, in the NUMBA graph that nutpie samples.

    ``concurrent`` lists how many copies run at once, each its own forked process, because the
    Python wrapper of a Numba function holds the GIL. P4's seconds per gradient check this against
    the sampler itself.
    """
    from pymc.blocking import DictToArrayBijection

    t0 = time.perf_counter()
    fn = model.logp_dlogp_function(ravel_inputs=True, mode="NUMBA")
    fn.set_extra_values({})
    point = model.initial_point()
    q = DictToArrayBijection.map({v.name: point[v.name] for v in fn._grad_vars}).data
    value, gradient = fn(q)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    out = {"compile_s": time.perf_counter() - t0, "n_dim": q.size, "logp_at_start": float(value)}

    def run(queue=None):
        times = []
        for _ in range(evals):
            t1 = time.perf_counter()
            fn(q)
            times.append(time.perf_counter() - t1)
        if queue is None:
            return times
        queue.put(times)

    for k in concurrent:
        if k == 1:
            ms = np.array(run()) * 1e3
        else:
            context = multiprocessing.get_context("fork")
            queue = context.Queue()
            workers = [context.Process(target=run, args=(queue,)) for _ in range(k)]
            for worker in workers:
                worker.start()
            ms = np.concatenate([queue.get() for _ in workers]) * 1e3
            for worker in workers:
                worker.join()
        out[f"{k}_at_once"] = {"median_ms": float(np.median(ms)), "min_ms": float(ms.min()),
                               "p90_ms": float(np.quantile(ms, 0.9)), "evals": ms.size}
    return out


# P3: rates ---------------------------------------------------------------------------------------


def rates():
    """The raw rates that replace the plan's 7 Gflop/s and 188 ms. Sizes small enough to be brief."""

    def best(fn, repeat=5):
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    rng = np.random.default_rng(SEED)
    out = {"blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
           "cores": sorted(os.sched_getaffinity(0))}
    for n in (419, 1000, 1030, 2000):
        a = rng.normal(size=(n, n))
        k = a @ a.T / n + np.eye(n)
        s = best(lambda: linalg.cholesky(k, lower=True, check_finite=False))
        out[f"cholesky_{n}"] = {"ms": s * 1e3, "gflop_s": n ** 3 / 3 / s / 1e9}
        if n == 1000:
            b = rng.normal(size=(n, 1))

            def composite():
                lower = linalg.cholesky(k, lower=True, check_finite=False)
                z = linalg.solve_triangular(lower, b, lower=True, check_finite=False)
                linalg.solve_triangular(lower, z, lower=True, trans="T", check_finite=False)
                return linalg.cho_solve((lower, True), np.eye(n), check_finite=False)

            out["composite_1000_ms"] = best(composite) * 1e3
            s = best(lambda: a @ a)
            out["matmul_1000"] = {"ms": s * 1e3, "gflop_s": 2 * n ** 3 / s / 1e9}
    values = rng.normal(size=4_000_000)
    out["ndtr_ns"] = best(lambda: ndtr(values)) / values.size * 1e9
    out["ndtr_ns_f32"] = best(lambda: ndtr(values.astype(np.float32))) / values.size * 1e9
    return out


def p3_model(args):
    sites, _ = load_tables()
    small = design(sites[sites["small_crop"]].reset_index(drop=True))
    full = design(sites)
    concurrent = [1] if args.threads == 1 else [1, 2]
    if args.alongside:  # one BLAS thread each, side by side, which is how chains share 2 cores
        concurrent = [2]
    out = {"blas_threads": args.threads, "alongside": args.alongside, "rates": rates(),
           "n_419": {"n": len(small["y"]), **time_logp(m1(small), 300, concurrent)},
           "n_1030": {"n": len(full["y"]), **time_logp(m1(full), 40, concurrent)},
           "runtime": runtime_check()}
    store().save_json(f"p3_timing_{args.threads}t{'_alongside' if args.alongside else ''}", out)
    print(json.dumps(out, indent=1, default=str))


# P4: fit -----------------------------------------------------------------------------------------


def prior_check():
    """Prior predictive on SMALL, in NumPy, with the ranges the priors are meant to allow."""
    sites, _ = load_tables()
    d = design(sites[sites["small_crop"]].reset_index(drop=True))
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(200):
        eta, ell = rng.lognormal(*PRIOR_ETA), rng.lognormal(*PRIOR_ELL)
        tau = rng.lognormal(*PRIOR_TAU, size=len(SCALES))
        mean = (rng.normal(2.0, 0.5) + rng.normal(0, 0.2) * d["density"]
                + d["methods"] @ rng.normal(0, 0.2, size=3))
        cov = eta ** 2 * matern32(d["distance"], ell) + np.diag(tau[d["scale"]] ** 2 + d["v"])
        rows.append(rng.multivariate_normal(mean, cov, method="cholesky"))
    y = np.array(rows)
    within = np.exp(y).mean(axis=1)
    out = {"draws": len(y), "site_sd_of_log_mean_prior": np.quantile(y.std(axis=1), [.05, .5, .95]),
           "site_sd_of_log_mean_observed": d["y"].std(),
           "crop_mean_ug_m3_prior": np.quantile(within, [.05, .5, .95]),
           "crop_mean_ug_m3_observed": np.exp(d["y"]).mean(),
           "share_of_sites_outside_1_40": float(((y < 0) | (y > np.log(40))).mean()),
           "prior_ell_km_p01_p10_p50_p90_p99": np.exp(
               PRIOR_ELL[0] + PRIOR_ELL[1] * np.array([-2.326, -1.282, 0, 1.282, 2.326]))}
    store().save_json("p4_prior_check", out)
    print(json.dumps(out, indent=1, default=lambda x: np.asarray(x).tolist()))


def excluded_list():
    sites, _ = load_tables()
    names = excluded_sites(sites)
    crop = sites[sites["site"].isin(names)]
    out = {"rule": excluded_sites.__doc__.split("\n\n")[1].strip(), "conus_sites": names,
           "count": len(names), "in_small_crop": crop.loc[crop["small_crop"], "site"].tolist()}
    store().save_json("p4_excluded_sites", out)
    print(out["count"], len(out["in_small_crop"]))


def fit_sites(drop_excluded):
    sites, counties = load_tables()
    crop = sites[sites["small_crop"]].reset_index(drop=True)
    if drop_excluded:
        names = store().load_json("p4_excluded_sites")["conus_sites"]
        crop = crop[~crop["site"].isin(names)].reset_index(drop=True)
    return crop, counties


def p4_fit(args):
    """One batch of chains. Chains share 2 cores, one BLAS thread each."""
    import nutpie

    crop, _ = fit_sites(args.tag == "drop")
    model = m1(design(crop))
    t0, cpu0 = time.perf_counter(), cpu_seconds()
    compiled = nutpie.compile_pymc_model(model)
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    trace = nutpie.sample(compiled, chains=args.chains, cores=args.chains, tune=args.tune,
                          draws=args.draws, seed=SEED + args.batch, progress_bar=False)
    sample_s, cpu_s = time.perf_counter() - t0, cpu_seconds() - cpu0
    stats = trace.sample_stats
    warm = trace.warmup_sample_stats
    arrays = {name: trace.posterior[name].to_numpy() for name in
              ("alpha", "beta_density", "beta_method", "delta", "eta", "ell", "tau")}
    arrays |= {"diverging": stats["diverging"].to_numpy(), "n_steps": stats["n_steps"].to_numpy()}
    cache().save_chunk(f"p4_{args.tag}", args.batch, arrays)
    runtime = runtime_check()
    out = {"tag": args.tag, "batch": args.batch, "sites": len(crop), "chains": args.chains,
           "tune": args.tune, "draws": args.draws, "adaptation": "nutpie default (diag)",
           "compile_s": compile_s, "sample_s": sample_s, "cpu_s": cpu_s,
           "gradients_warmup": int(warm["n_steps"].sum()),
           "gradients_draws": int(stats["n_steps"].sum()),
           "divergences": int(stats["diverging"].sum()),
           "divergences_warmup": int(warm["diverging"].sum()), "runtime": runtime}
    store().save_json(f"p4_{args.tag}_batch{args.batch}", out)
    print(json.dumps(out, indent=1, default=str))
    assert not runtime["jax_executed"], "the annual path executed JAX"
    assert runtime["floatX"] == "float64"
    assert not runtime["cuda_mapped"], "CUDA is mapped into the annual path"


def gather_fit(tag, batches):
    """Chains from the batches, stacked: each key is (chain, draw, ...)."""
    chunks = [cache().load_chunk(f"p4_{tag}", b) for b in batches]
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def p4_summary(args):
    """Convergence over all chains, the drop test, and the gates as written."""
    import arviz_stats.base as stats

    fit = gather_fit(args.tag, args.batches)

    def diag(values):
        values = values.reshape(*values.shape[:2], -1)
        rows = []
        for j in range(values.shape[2]):
            x = values[:, :, j]
            rows.append({"r_hat": float(stats.array_stats.rhat(x)),
                         "ess_bulk": float(stats.array_stats.ess(x, method="bulk")),
                         "ess_tail": float(stats.array_stats.ess(x, method="tail", prob=0.1)),
                         "median": float(np.median(x)), "p05": float(np.quantile(x, .05)),
                         "p95": float(np.quantile(x, .95))})
        return rows

    names = ("alpha", "beta_density", "beta_method", "delta", "eta", "ell", "tau")
    table = {name: diag(fit[name]) for name in names}
    every = [row for rows in table.values() for row in rows]
    hypers = [row for name in HYPERS for row in table[name]]
    meta = [store().load_json(f"p4_{args.tag}_batch{b}") for b in args.batches]
    ell = fit["ell"].ravel()
    low, high = np.exp(PRIOR_ELL[0] + PRIOR_ELL[1] * np.array([-2.326, 2.326]))
    out = {"tag": args.tag, "batches": meta, "parameters": table,
           "max_r_hat": max(r["r_hat"] for r in every),
           "min_ess_hyper": min(min(r["ess_bulk"], r["ess_tail"]) for r in hypers),
           "min_ess_all": min(min(r["ess_bulk"], r["ess_tail"]) for r in every),
           "divergences": sum(m["divergences"] for m in meta),
           "chains": int(fit["ell"].shape[0]),
           "wall_sum_s": sum(m["compile_s"] + m["sample_s"] for m in meta),
           "sample_wall_max_s": max(m["sample_s"] for m in meta),
           "sample_wall_sum_s": sum(m["sample_s"] for m in meta),
           "cpu_s": sum(m["cpu_s"] for m in meta),
           "ell_pinned": bool(np.quantile(ell, .05) < low or np.quantile(ell, .95) > high),
           "ell_prior_p01_p99": [low, high],
           "ell_p10_p50_p90": np.quantile(ell, [.1, .5, .9]).tolist()}
    store().save_json(f"p4_fit_{args.tag}", out)
    keep = ("max_r_hat", "min_ess_hyper", "min_ess_all", "divergences", "chains", "wall_sum_s",
            "sample_wall_max_s", "sample_wall_sum_s", "cpu_s", "ell_pinned", "ell_p10_p50_p90")
    print(json.dumps({k: out[k] for k in keep}, indent=1))
    for name in names:
        for row in table[name]:
            print(name, {k: round(v, 3) for k, v in row.items()})


# Conditionals ------------------------------------------------------------------------------------


def thin(fit, count):
    """``count`` draws evenly spaced through the chains, in chain order."""
    total = fit["ell"].shape[0] * fit["ell"].shape[1]
    take = np.linspace(0, total - 1, count).round().astype(int)
    return {k: v.reshape(total, *v.shape[2:])[take] for k, v in fit.items()
            if k not in ("diverging", "n_steps")}


def site_mean(d, beta):
    return (beta["alpha"] + beta["beta_density"] * d["density"] + d["methods"] @ beta["beta_method"]
            + d["years"] @ beta["delta"])


def county_mean_terms(table, d, beta):
    """Latent mean at county centres: FRM-equivalent, full window, so no method or year term."""
    density = (table["log_density"].to_numpy(float) - d["density_scale"][0]) / d["density_scale"][1]
    return beta["alpha"] + beta["beta_density"] * density


def conditional(d, points, table, draw):
    """Exact mean and covariance of the latent field at ``points`` for one posterior draw."""
    eta2, ell = draw["eta"] ** 2, draw["ell"]
    noise = draw["tau"][d["scale"]] ** 2 + d["v"]
    lower = linalg.cholesky(eta2 * matern32(d["distance"], ell) + np.diag(noise), lower=True,
                            check_finite=False)
    cross = eta2 * matern32(distance(points, d["xy"]), ell)
    half = linalg.solve_triangular(lower, cross.T, lower=True, check_finite=False)
    residual = linalg.solve_triangular(lower, d["y"] - site_mean(d, draw), lower=True,
                                       check_finite=False)
    mean = county_mean_terms(table, d, draw) + half.T @ residual
    return mean, eta2 * matern32(distance(points, points), ell) - half.T @ half


def conditionals(d, table, draws):
    """(mean, covariance) for every draw, as float32: (draws, counties) and (draws, c, c)."""
    points = table[["x_km", "y_km"]].to_numpy(float)
    count = len(draws["ell"])
    means = np.empty((count, len(points)), np.float32)
    covs = np.empty((count, len(points), len(points)), np.float32)
    for i in range(count):
        means[i], covs[i] = conditional(d, points, table, {k: v[i] for k, v in draws.items()})
    return means, covs


# Value of information ----------------------------------------------------------------------------


def reading_nodes(mean, variance, nodes):
    """Midpoints of ``nodes`` equal-probability bins of the mixture predictive of the reading."""
    sd = np.sqrt(variance)
    grid = np.linspace((mean - 6 * sd).min(), (mean + 6 * sd).max(), 256)
    cdf = ndtr((grid[:, None] - mean) / sd).mean(axis=1)
    return np.interp((np.arange(nodes) + 0.5) / nodes, cdf, grid)


def update_weights(y, mean, variance):
    """Each draw's posterior weight after reading ``y`` (axes: readings, draws)."""
    log = -0.5 * (y[:, None] - mean) ** 2 / variance - 0.5 * np.log(variance)
    return np.exp(log - logsumexp(log, axis=1, keepdims=True))


def probability_above(y, weights, mu, var, cross, mu_c, q, line):
    """P(above the line) per county after each reading: rank-one update in each draw, then mix."""
    gain = cross / q[:, None]
    shift = (y[:, None] - mu_c)[:, :, None]
    sd = np.sqrt(np.maximum(var - gain * cross, 1e-10))
    return (weights[:, :, None] * ndtr((mu + shift * gain - line) / sd)).sum(axis=1)


def wrong_calls(p, population):
    """Expected people on the wrong side of the line, calling each county at p = 0.5."""
    return (population * np.minimum(p, 1 - p)).sum(axis=-1)


def candidate_value(mu, var, cross, mu_c, var_c, noise, population, line, nodes=16):
    """Expected fall in wrong calls from one monitor (EVSI). Arrays are (draws, counties)."""
    q = var_c + noise
    y = reading_nodes(mu_c, q, nodes)
    p = probability_above(y, update_weights(y, mu_c, q), mu, var, cross, mu_c, q, line)
    before = ndtr((mu - line) / np.sqrt(var)).mean(axis=0)
    return wrong_calls(before, population) - wrong_calls(p, population).mean()


class Voi:
    """Greedy state: draws of county means and covariance, kept as float32.

    After a pick, each draw's covariance is downdated and its mean is left alone, because the
    reading is not known. The first round is an exact preposterior value. Later rounds value a
    monitor as if earlier readings came in at their predictive means, so ``check_batch`` measures
    what the picked set is worth jointly.
    """

    def __init__(self, means, covs, noise, population, xy, line=LINE, nodes=16):
        self.mu, self.s, self.nodes = means, covs, nodes
        self.noise, self.population = noise.astype(np.float32), population.astype(np.float32)
        self.line = np.float32(np.log(line))
        self.gap = np.float32(distance(xy, xy))
        self.var = np.einsum("ijj->ij", covs).copy()
        self.current()

    def current(self):
        self.p = ndtr((self.mu - self.line) / np.sqrt(np.maximum(self.var, 1e-10))).mean(axis=0)
        self.undecided = (self.p > 0.01) & (self.p < 0.99)

    def value(self, c, radius):
        local = np.flatnonzero((self.gap[c] < radius) & self.undecided)
        if local.size == 0:
            return 0.0, 0
        return float(candidate_value(self.mu[:, local], self.var[:, local], self.s[:, c][:, local],
                                     self.mu[:, c], self.s[:, c, c], self.noise,
                                     self.population[local], self.line, self.nodes)), local.size

    def score(self, candidates, radius, workers):
        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(lambda c: self.value(c, radius), candidates))
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

    def pick(self, c):
        cross = self.s[:, c].copy()
        q = cross[:, c] + self.noise
        for i in range(len(q)):
            self.s[i] -= np.outer(cross[i], cross[i] / q[i])
        self.var = self.var - cross ** 2 / q[:, None]
        self.current()


def greedy(voi, candidates, rounds, radius, top=None, workers=2):
    """Pick ``rounds`` sites. ``top`` rescoring only that many by last score is the lazy loop."""
    candidates = np.asarray(candidates)
    last = np.full(len(candidates), np.inf)
    free = np.ones(len(candidates), bool)
    picks, gains, seconds, scored, history, local = [], [], [], [], [], []
    for round_ in range(rounds):
        t0 = time.perf_counter()
        todo = np.flatnonzero(free)
        if top is not None and round_ > 0:
            todo = todo[np.argsort(-last[todo])[:top]]
        last[todo], counties = voi.score(candidates[todo], radius, workers)
        best = todo[np.argmax(last[todo])]
        picks.append(int(candidates[best]))
        gains.append(float(last[best]))
        scored.append(len(todo))
        local.append(float(counties.mean()))
        history.append(np.where(free, last, np.nan).copy())
        free[best] = False
        voi.pick(int(candidates[best]))
        seconds.append(time.perf_counter() - t0)
    return {"picks": picks, "gains": gains, "scored": scored, "seconds": seconds,
            "mean_local_counties": local, "history": np.array(history)}


def overlap(a, b):
    return len(set(a) & set(b))


def small_state(tag="main", draws=100, half="all"):
    """County draws for SMALL: from the cache, or computed and timed once.

    ``half`` keeps every second one of ``draws``, and only those are conditioned on, so that two
    halves of more draws than fit in memory can each run in their own process.
    """
    crop, counties = fit_sites(tag == "drop")
    table = counties[counties["small_crop"]].reset_index(drop=True)
    d = design(crop)
    fit = gather_fit(tag, (0, 1))
    chosen = thin(fit, draws)
    keep = {"all": slice(None), "odd": slice(1, None, 2), "even": slice(0, None, 2)}[half]
    chosen = {k: v[keep] for k, v in chosen.items()}
    noise = chosen["tau"][:, SCALES.index("NEIGHBORHOOD")] ** 2 + np.median(
        d["v"][crop["days"].to_numpy() >= 900])
    t0 = time.perf_counter()
    stem = f"{tag}_{draws}" + ("" if half == "all" else f"_{half}")
    path_m, path_s = cache().path / f"means_{stem}.npy", cache().path / f"covs_{stem}.npy"
    if path_m.exists() and path_s.exists():
        means, covs = np.load(path_m), np.load(path_s)
        seconds = None
    else:
        means, covs = conditionals(d, table, chosen)
        seconds = time.perf_counter() - t0
        np.save(path_m, means)
        np.save(path_s, covs)
    return {"table": table, "means": means, "covs": covs, "noise": noise, "chosen": chosen,
            "ell": float(np.median(fit["ell"])), "conditionals_s": seconds,
            "candidates": np.flatnonzero(~table["monitored"] & ~table["registered_only"])}


def make_voi(state, nodes=16):
    """No copy of the covariances, which are most of the memory: a greedy run consumes them."""
    table = state["table"]
    return Voi(state["means"], state["covs"], state["noise"], table["population"].to_numpy(float),
               table[["x_km", "y_km"]].to_numpy(float), nodes=nodes)


def brute_force(mu, cov, noise, population, c, sims, seed, line, control=False):
    """Simulate truth and reading, make the call, and count real wrong calls, with no shortcut.

    ``mu`` is (draws, counties), ``cov`` (draws, counties, counties), float64. Truth comes from a
    full Cholesky of each draw. Calls use the closed form's posterior, so if that were wrong the
    realised value would fall short of the value it promises. Returns the mean and se of the
    realised fall in wrong calls, paired on the same truth, and the closed form at 16 and 64 nodes.
    ``control`` also makes the calls with a reading believed 25x more precise than it is, which must
    fall short of the closed form, or the test cannot fail.
    """
    rng = np.random.default_rng(seed)
    count, width = mu.shape
    lower = [linalg.cholesky(s + 1e-7 * np.eye(width), lower=True) for s in cov]
    var = np.einsum("ijj->ij", cov)
    cross, var_c, mu_c = cov[:, c], cov[:, c, c], mu[:, c]
    q = var_c + noise
    before = ndtr((mu - line) / np.sqrt(var)).mean(axis=0) > 0.5
    gains, wrong = [], []
    for _ in range(sims // 500):
        which = rng.integers(count, size=500)
        truth = np.stack([mu[i] + lower[i] @ rng.normal(size=width) for i in which])
        y = truth[:, c] + np.sqrt(noise[which]) * rng.normal(size=500)
        p = probability_above(y, update_weights(y, mu_c, q), mu, var, cross, mu_c, q, line)
        above = truth > line
        gains.append(((before != above) * population).sum(axis=1)
                     - (((p > 0.5) != above) * population).sum(axis=1))
        if control:
            q_bad = var_c + noise / 25
            p = probability_above(y, update_weights(y, mu_c, q_bad), mu, var, cross, mu_c, q_bad, line)
            wrong.append(((before != above) * population).sum(axis=1)
                         - (((p > 0.5) != above) * population).sum(axis=1))
    gains = np.concatenate(gains)
    extra = {}
    if control:
        wrong = np.concatenate(wrong)
        extra = {"control_brute": float(wrong.mean()),
                 "control_se": float(wrong.std() / np.sqrt(wrong.size))}
    return {**extra, "draws": count, "counties": width, "sims": gains.size, "brute": float(gains.mean()),
            "se": float(gains.std() / np.sqrt(gains.size)),
            "closed_16": float(candidate_value(mu, var, cross, mu_c, var_c, noise, population, line)),
            "closed_64": float(candidate_value(mu, var, cross, mu_c, var_c, noise, population, line,
                                               nodes=64))}


def synthetic(count, width, offsets, rng):
    """A small field: ``count`` draws with Matern covariance and means offset across the line."""
    xy = rng.uniform(0, 600, size=(width, 2))
    means, covs = [], []
    line = np.log(LINE)
    for i in range(count):
        ell = rng.choice([150.0, 300.0]) if count > 1 else 200.0
        means.append(line + rng.normal(0, 0.02, width) + offsets[i % len(offsets)])
        covs.append(0.15 ** 2 * matern32(distance(xy, xy), ell))
    return np.array(means), np.array(covs), rng.uniform(1e4, 1e6, width)


def p6_unit(args):
    """Closed form against brute-force simulation at three configurations, one of them real."""
    rng = np.random.default_rng(SEED)
    results = {}
    line = np.log(LINE)

    mu, cov, pop = synthetic(1, 25, [0.0], rng)
    results["one_draw"] = brute_force(mu, cov, np.full(1, 0.05 ** 2), pop, 0, args.sims,
                                      args.seed + 1, line, args.control)

    mu, cov, pop = synthetic(40, 25, [-0.1, 0.1], rng)
    results["two_regimes"] = brute_force(mu, cov, np.full(40, 0.05 ** 2), pop, 0, args.sims,
                                         args.seed + 2, line, args.control)

    state = small_state()
    voi = make_voi(state)
    values, _ = voi.score(state["candidates"], 3 * state["ell"], args.workers)
    c = int(state["candidates"][np.argmax(values)])
    near = np.argsort(voi.gap[c])[:150]
    index = int(np.flatnonzero(near == c)[0])
    results["real_small_top_candidate"] = brute_force(
        state["means"][:, near].astype(float), state["covs"][np.ix_(range(100), near, near)].astype(
            float), state["noise"], state["table"]["population"].to_numpy(float)[near], index,
        args.sims, args.seed + 3, line, args.control)
    for row in results.values():
        row["z"] = (row["closed_16"] - row["brute"]) / row["se"]
        row["within_2_se"] = abs(row["z"]) <= 2
    results["runtime"] = runtime_check()
    store().save_json(f"p6_unit{args.name}", results)
    print(json.dumps(results, indent=1, default=str))


def p6_round(args):
    """One real round, scored on all candidates, on the threads given."""
    state = small_state()
    voi = make_voi(state)
    t0 = time.perf_counter()
    values, local = voi.score(state["candidates"], args.radius * state["ell"], args.workers)
    seconds = time.perf_counter() - t0
    out = {"workers": args.workers, "radius_ell": args.radius, "ell_km": state["ell"],
           "seconds": seconds, "candidates": len(values), "mean_local_counties": float(local.mean()),
           "share_of_counties_undecided": float(voi.undecided.mean()),
           "wrong_calls_now_people": float(wrong_calls(voi.p, voi.population)),
           "best_value_people": float(values.max()),
           "conditionals_s": state["conditionals_s"], "runtime": runtime_check()}
    store().save_json(f"p6_round_{args.workers}w_{args.radius:g}ell", out)
    print(json.dumps(out, indent=1, default=str))


def run_name(args):
    name = f"{args.tag}_{args.loop}_{args.radius:g}ell_{args.half}"
    return name if (args.draws, args.nodes) == (100, 16) else f"{name}_d{args.draws}_n{args.nodes}"


def p6_greedy(args):
    """One greedy run: full or lazy, at a radius, on all draws or one half."""
    state = small_state(args.tag, args.draws, args.half)
    voi = make_voi(state, args.nodes)
    state.pop("covs")
    t0 = time.perf_counter()
    run = greedy(voi, state["candidates"], args.rounds, args.radius * state["ell"],
                 top=50 if args.loop == "lazy" else None, workers=args.workers)
    wall = time.perf_counter() - t0
    history = run.pop("history")
    cache().save_array(f"p6_history_{run_name(args)}", history)
    table = state["table"]
    out = {**run, "loop": args.loop, "half": args.half, "radius_ell": args.radius, "tag": args.tag,
           "draws": len(state["means"]), "nodes": args.nodes,
           "conditionals_s": state["conditionals_s"],
           "p_above": voi.p.round(4).tolist() if args.keep_p else None,
           "ell_km": state["ell"], "wall_s": wall, "workers": args.workers,
           "names": (table["county"].astype(str) + ", " + table["state"].astype(str)).to_numpy()[
               run["picks"]].tolist(), "runtime": runtime_check()}
    store().save_json(f"p6_greedy_{run_name(args)}", out)
    print(json.dumps({k: out[k] for k in ("wall_s", "scored", "names", "gains", "seconds")},
                     indent=0))


def rises(history):
    """How often a site's gain rose after another pick: what the lazy loop assumes is rare."""
    a, b = history[:-1], history[1:]
    both = np.isfinite(a) & np.isfinite(b) & (a > 0)
    up = both & (b > a)
    ratio = b[both] / a[both]
    return {"pairs": int(both.sum()), "rose": int(up.sum()), "share": float(up.sum() / both.sum()),
            "rose_by_over_1pct": int((both & (b > 1.01 * a)).sum()),
            "rose_by_over_10pct": int((both & (b > 1.1 * a)).sum()),
            "ratio_p50_p99_max": np.quantile(ratio, [.5, .99, 1]).tolist()}


def check_batch(state, picks, sims, seed):
    """What a picked set is worth jointly, simulated, against the sum of its greedy gains.

    Added check, not a plan gate: rounds after the first use a mean-held-fixed approximation.
    Readings are drawn from the mixture; each draw is updated on all of them at once.
    """
    rng = np.random.default_rng(seed)
    mu, cov = state["means"].astype(float), state["covs"]
    noise, pop = state["noise"], state["table"]["population"].to_numpy(float)
    line = np.log(LINE)
    k = len(picks)
    count = len(mu)
    sets = []
    for i in range(count):
        a = cov[i][np.ix_(picks, picks)].astype(float) + noise[i] * np.eye(k)
        lower = linalg.cholesky(a, lower=True)
        solved = linalg.cho_solve((lower, True), cov[i][np.ix_(picks, range(cov.shape[1]))].astype(
            float))
        cross = cov[i][picks].astype(float)
        var = np.maximum(np.diag(cov[i]).astype(float) - np.einsum("ij,ij->j", cross, solved), 1e-10)
        sets.append((lower, solved, var))
    before = wrong_calls(ndtr((mu - line) / np.sqrt(np.einsum("ijj->ij", cov))).mean(axis=0), pop)
    after = []
    for _ in range(sims):
        i = rng.integers(count)
        y = mu[i, picks] + sets[i][0] @ rng.normal(size=k)
        log = np.empty(count)
        for j in range(count):
            r = y - mu[j, picks]
            z = linalg.solve_triangular(sets[j][0], r, lower=True, check_finite=False)
            log[j] = -0.5 * z @ z - np.log(np.diag(sets[j][0])).sum()
        w = np.exp(log - logsumexp(log))
        p = np.zeros(mu.shape[1])
        for j in np.flatnonzero(w > 1e-6):
            m = mu[j] + (y - mu[j, picks]) @ sets[j][1]
            p += w[j] * ndtr((m - line) / np.sqrt(sets[j][2]))
        after.append(wrong_calls(p, pop))
    after = np.array(after)
    return {"wrong_calls_before": float(before), "after_mean": float(after.mean()),
            "joint_value": float(before - after.mean()),
            "se": float(after.std() / np.sqrt(sims)), "sims": sims}


def p6_raised(args):
    """Split-half overlap at more draws or nodes, the no-go action of P6, and a nearness measure.

    ``within_100_km`` is an added check, not a plan gate: the share of one half's picks with a pick
    of the other half within 100 km, which is what a band of sites would claim.
    """
    gap = None
    rows = {}
    for label, suffix in (("A_100_per_half", "_d200_n16"), ("B_200_per_half", "_d400_n16"),
                          ("C_100_per_half_32_nodes", "_d200_n32")):
        odd = store().load_json(f"p6_greedy_main_lazy_3ell_odd{suffix}")
        even = store().load_json(f"p6_greedy_main_lazy_3ell_even{suffix}")
        if gap is None:
            table = load_tables()[1]
            table = table[table["small_crop"]].reset_index(drop=True)
            gap = distance(*[table[["x_km", "y_km"]].to_numpy(float)] * 2)
        near = (gap[np.ix_(odd["picks"], even["picks"])] < 100).any(axis=1).mean()
        rows[label] = {"overlap": overlap(odd["picks"], even["picks"]), "within_100_km": float(near),
                       "gain_sum": [sum(odd["gains"]), sum(even["gains"])],
                       "wall_s": [odd["wall_s"], even["wall_s"]],
                       "conditionals_s": [odd["conditionals_s"], even["conditionals_s"]],
                       "max_rss_gb": odd["runtime"]["max_rss_gb"]}
    main = store().load_json("p6_greedy_main_lazy_3ell_all_d200_n16")
    drop = store().load_json("p6_greedy_drop_lazy_3ell_all_d200_n16")
    p_main, p_drop = np.array(main["p_above"]), np.array(drop["p_above"])
    out = {"raises": rows, "main_vs_drop_200_draws": {
        "top20_overlap": overlap(main["picks"], drop["picks"]),
        "within_100_km": float((gap[np.ix_(main["picks"], drop["picks"])] < 100).any(axis=1).mean()),
        "note": "p_above is after the 20 picks, so both sides carry the mean-held-fixed step",
        "p_mean_abs_change": float(np.abs(p_main - p_drop).mean()),
        "p_p99_abs_change": float(np.quantile(np.abs(p_main - p_drop), .99)),
        "counties_changing_side_of_0.5": int(((p_main > .5) != (p_drop > .5)).sum())},
        "added_check": p6_raised.__doc__.split("\n\n")[1].strip()}
    store().save_json("p6_raised", out)
    print(json.dumps(out, indent=1))


def p6_summary(args):
    """The gates, from the stored runs, and the joint-value check."""
    def load(name):
        return store().load_json(f"p6_greedy_main_{name}")

    full3, lazy3, lazy2 = load("full_3ell_all"), load("lazy_3ell_all"), load("lazy_2ell_all")
    odd, even = load("full_3ell_odd"), load("full_3ell_even")
    odd_lazy, even_lazy = load("lazy_3ell_odd"), load("lazy_3ell_even")
    history = cache().load_array("p6_history_main_full_3ell_all")
    state = small_state()
    out = {"added_checks": ["joint value of picks", "size of gain rises"], "lazy_vs_full_top20": overlap(lazy3["picks"], full3["picks"]),
           "two_ell_vs_three_ell_lazy_top20": overlap(lazy2["picks"], lazy3["picks"]),
           "halves_top20_full": overlap(odd["picks"], even["picks"]),
           "halves_top20_lazy": overlap(odd_lazy["picks"], even_lazy["picks"]),
           "first_round_pick_same": full3["picks"][0] == lazy3["picks"][0],
           "gain_rises": rises(history),
           "lazy_gains_sum": sum(lazy3["gains"]), "full_gains_sum": sum(full3["gains"]),
           "wall_s": {"full": full3["wall_s"], "lazy": lazy3["wall_s"], "lazy_2ell": lazy2["wall_s"]},
           "first_round_s": {"full": full3["seconds"][0], "lazy_2ell": lazy2["seconds"][0]},
           "joint_value_of_full_top20": check_batch(state, full3["picks"], args.sims, 7),
           "joint_value_of_full_top5": check_batch(state, full3["picks"][:5], args.sims, 8),
           "greedy_gain_sum_top5": sum(full3["gains"][:5])}
    if (cache().path / "means_drop_100.npy").exists():
        out["dropped_sites_lazy_top20_vs_main"] = overlap(
            store().load_json("p6_greedy_drop_lazy_3ell_all")["picks"], lazy3["picks"])
    store().save_json("p6_summary", out)
    print(json.dumps(out, indent=1))


# P5: the HSGP approximation ----------------------------------------------------------------------


def hsgp_conditional(d, points, table, draw, c, modes, centre):
    """Posterior mean of the latent field under the HSGP, at fixed hyperparameters."""
    j1, j2, box = hsgp_modes(HALF_WIDTH_KM, c, modes)
    omega2 = (np.pi * j1 / (2 * box[0])) ** 2 + (np.pi * j2 / (2 * box[1])) ** 2
    prior = draw["eta"] ** 2 * matern32_density(omega2, draw["ell"])
    phi = hsgp_basis(d["xy"] - centre, j1, j2, box)
    noise = draw["tau"][d["scale"]] ** 2 + d["v"]
    precision = (phi.T / noise) @ phi + np.diag(1 / prior)
    weight = linalg.cho_solve(linalg.cho_factor(precision, check_finite=False),
                              phi.T @ ((d["y"] - site_mean(d, draw)) / noise), check_finite=False)
    return county_mean_terms(table, d, draw) + hsgp_basis(points - centre, j1, j2, box) @ weight


def p5_table(args):
    """Edge variance at fitted ell and the mean field against M1 exact, with no sampling."""
    sites, counties = load_tables()
    half = ((counties["x_km"].max() - counties["x_km"].min()) / 2,
            (counties["y_km"].max() - counties["y_km"].min()) / 2)
    assert np.allclose(half, HALF_WIDTH_KM, atol=1.0), half
    centre = np.array([(counties["x_km"].max() + counties["x_km"].min()) / 2,
                       (counties["y_km"].max() + counties["y_km"].min()) / 2])
    fit = gather_fit("main", (0, 1))
    ell10, ell50, ell90 = np.quantile(fit["ell"], [.1, .5, .9])
    median = {k: np.median(v.reshape(-1, *v.shape[2:]), axis=0) for k, v in fit.items()
              if k not in ("diverging", "n_steps")}
    d = design(sites, design(sites[sites["small_crop"]])["density_scale"])
    points = counties[["x_km", "y_km"]].to_numpy(float)
    rows = []
    for text in args.configs:
        c, modes = text.split(":")
        c, modes = float(c), tuple(int(m) for m in modes.split("x"))
        for name, ell in (("p10", ell10), ("p50", ell50), ("p90", ell90)):
            draw = {**median, "ell": ell}
            edge_ew = hsgp_variance(half, c, modes, ell, (half[0], 0.0))
            edge_ns = hsgp_variance(half, c, modes, ell, (0.0, half[1]))
            corner = hsgp_variance(half, c, modes, ell, tuple(half))
            exact = conditional_mean(d, points, counties, draw)
            approx = hsgp_conditional(d, points, counties, draw, c, modes, centre)
            gap = np.abs(exact - approx)
            rows.append({"c": c, "modes": list(modes), "ell_name": name, "ell_km": ell,
                         "centre_var": hsgp_variance(half, c, modes, ell, (0.0, 0.0)),
                         "edge_east_west": edge_ew, "edge_north_south": edge_ns, "corner": corner,
                         "gap_median": float(np.median(gap)), "gap_p95": float(np.quantile(gap, .95)),
                         "gap_p99": float(np.quantile(gap, .99)), "gap_max": float(gap.max())})
            print(rows[-1], flush=True)
    out = {"half_width_km": half, "fitted_ell_from": "SMALL fit P4, not a national fit",
           "ell_p10_p50_p90": [ell10, ell50, ell90], "rows": rows, "runtime": runtime_check()}
    store().save_json(args.name, out)


def conditional_mean(d, points, table, draw):
    """Exact posterior mean of the field: only what the mean needs, so no county-county matrix."""
    eta2, ell = draw["eta"] ** 2, draw["ell"]
    noise = draw["tau"][d["scale"]] ** 2 + d["v"]
    factor = linalg.cho_factor(eta2 * matern32(d["distance"], ell) + np.diag(noise),
                               check_finite=False)
    weight = linalg.cho_solve(factor, d["y"] - site_mean(d, draw), check_finite=False)
    cross = eta2 * matern32(distance(points, d["xy"]), ell)
    return county_mean_terms(table, d, draw) + cross @ weight


def p5_time(args):
    """M2's gradient, in the sampler's own function, at the design size."""
    sites, counties = load_tables()
    d = design(sites)
    centre = np.array([(counties["x_km"].max() + counties["x_km"].min()) / 2,
                       (counties["y_km"].max() + counties["y_km"].min()) / 2])
    concurrent = [1] if args.threads == 1 else [1, 2]
    out = {"c": args.c, "modes": args.modes, "points": len(d["y"]) + len(counties),
           "blas_threads": args.threads}
    out |= time_logp(m2(d, counties, centre, args.c, tuple(args.modes)), 200, concurrent)
    out["runtime"] = runtime_check()
    store().save_json(f"p5_gradient_{args.modes[0]}x{args.modes[1]}_{args.threads}t", out)
    print(json.dumps(out, indent=1, default=str))


# P7: backtest power ------------------------------------------------------------------------------


def p7(args):
    """The count gate from the one definition of a new site, and what that count can detect.

    The hold-out below is an added check, not a plan gate: hyperparameters and coefficients come
    from the SMALL fit and are applied nationally at their medians, so it is a look at the
    arithmetic of the backtest, not the backtest.
    """
    sites, _ = load_tables()
    years = pd.concat([pd.read_csv(RAW / f"e89_site_year_{y}.csv", dtype={"site": str})
                       for y in YEARS], ignore_index=True)
    table = new_site_table(sites, years)
    new = sites["first_year"] >= 2023
    old = sites["first_year"] < 2023
    scale = design(sites[sites["small_crop"]])["density_scale"]
    d_old = design(sites[old].reset_index(drop=True), scale)
    d_new = design(sites[new].reset_index(drop=True), scale)
    fit = gather_fit("main", (0, 1))
    draw = {k: np.median(v.reshape(-1, *v.shape[2:]), axis=0) for k, v in fit.items()
            if k not in ("diverging", "n_steps")}

    eta2, ell = draw["eta"] ** 2, draw["ell"]
    noise = draw["tau"][d_old["scale"]] ** 2 + d_old["v"]
    factor = linalg.cho_factor(eta2 * matern32(d_old["distance"], ell) + np.diag(noise))
    gap = distance(d_new["xy"], d_old["xy"])
    cross = eta2 * matern32(gap, ell)
    mean = site_mean(d_new, draw) + cross @ linalg.cho_solve(
        factor, d_old["y"] - site_mean(d_old, draw))
    var = eta2 - np.einsum("ij,ji->i", cross, linalg.cho_solve(factor, cross.T))
    total = var + draw["tau"][d_new["scale"]] ** 2 + d_new["v"]
    far = gap.min(axis=1) > 25
    error, z = d_new["y"] - mean, (d_new["y"] - mean) / np.sqrt(total)
    groups = {}
    for name, mask in (("all_new", np.ones(far.size, bool)), ("over_25_km", far)):
        n = int(mask.sum())
        groups[name] = {"sites": n, "bias_log": float(error[mask].mean()),
                        "bias_se_log": float(error[mask].std() / np.sqrt(n)),
                        "predictive_sd_log": float(np.sqrt(total[mask].mean())),
                        "coverage_90": float((np.abs(z[mask]) < 1.645).mean()),
                        "coverage_se_if_90": float(np.sqrt(.09 / n)),
                        "smallest_bias_detected_80pct_log": float(
                            2.8 * error[mask].std() / np.sqrt(n))}
    out = {"new_sites": table, "gate_over_25_km": table["over_25"],
           "gate_passes": table["over_25"] >= 30,
           "added_check_holdout_small_hyperparameters": groups}
    store().save_json("p7_backtest_power", out)
    print(json.dumps(out, indent=1, default=str))


def p2_extra(args):
    """The 90% test on the 8-10 band, recorded beside P2 and not folded into its gate.

    Added after P2's result was seen, on the owner's instruction.
    """
    p2 = store().load_json("p2_design_values")
    out = {"added_after_seeing_results": True, "threshold": 0.9, "rows": {}}
    for name in ("gate_no_concurred_events", "all_valid_sites"):
        share = p2[name]["agree_9.05_dv_8_10"]
        out["rows"][name] = {"sites": p2[name]["sites_dv_8_10"], "agree": share,
                             "passes": share >= 0.9}
    store().save_json("p2_band_extra", out)
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, *options):
        sub = commands.add_parser(name)
        sub.set_defaults(fn=fn)
        for flags, kwargs in options:
            sub.add_argument(*flags, **kwargs)

    workers = (["--workers"], dict(type=int, default=2))
    radius = (["--radius"], dict(type=float, default=3.0))
    add("p2-extra", p2_extra)
    add("p3-model", p3_model, (["--threads"], dict(type=int, required=True)),
        (["--alongside"], dict(action="store_true")))
    add("prior", lambda args: prior_check())
    add("excluded", lambda args: excluded_list())
    add("p4-fit", p4_fit, (["--tag"], dict(default="main")), (["--batch"], dict(type=int, default=0)),
        (["--chains"], dict(type=int, default=2)), (["--tune"], dict(type=int, default=400)),
        (["--draws"], dict(type=int, default=500)))
    add("p4-summary", p4_summary, (["--tag"], dict(default="main")),
        (["--batches"], dict(type=int, nargs="+", default=[0, 1])))
    add("p5-table", p5_table, (["--name"], dict(default="p5_table")),
        (["--configs"], dict(nargs="+", default=["1.5:40x25", "1.5:50x32", "2:50x32", "2:64x40"])))
    add("p5-time", p5_time, (["--threads"], dict(type=int, required=True)),
        (["--c"], dict(type=float, default=1.5)),
        (["--modes"], dict(type=int, nargs=2, default=[40, 25])))
    add("p6-unit", p6_unit, workers, (["--sims"], dict(type=int, default=20000)),
        (["--seed"], dict(type=int, default=0)), (["--control"], dict(action="store_true")),
        (["--name"], dict(default="")))
    add("p6-round", p6_round, workers, radius)
    add("p6-greedy", p6_greedy, workers, radius, (["--loop"], dict(default="full")),
        (["--tag"], dict(default="main")), (["--half"], dict(default="all")),
        (["--rounds"], dict(type=int, default=20)), (["--draws"], dict(type=int, default=100)),
        (["--nodes"], dict(type=int, default=16)), (["--keep-p"], dict(action="store_true")))
    add("p6-summary", p6_summary, (["--sims"], dict(type=int, default=300)))
    add("p6-raised", p6_raised)
    add("p7", p7)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
