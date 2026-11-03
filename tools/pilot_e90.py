"""E90 pilots P3 to P8. Gates are in results/E90/gates_2026-11-02.json, written first.

    python tools/pilot_e90.py p3cpu | baselines | ...
Run through a wrapper that checks the machine, pins cores and uses tools/memguard.py. Each job stores
its result with Store("E90"), so a crash loses only the job in flight.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e90_core as c  # noqa: E402
import build_e90_data as b  # noqa: E402

from pymc_challenges.bigrun import Store, gpu_lock, online_core_check, time_gradient, vram_high_water, precision_check  # noqa: E402

MAIN = "2022-01-13"
THREE = ("DE-BY", "DE-HB", "DE-HH")
store = Store("E90")


def health():
    h = online_core_check()
    if h["offline_cpus"] != [10, 26] or h["machine_checks"] != 1:
        raise SystemExit(f"STOP: machine state {h}")
    return h


def p3cpu():
    import pymc as pm
    import pytensor

    print("floatX", pytensor.config.floatX, "| compiledir", pytensor.config.compiledir, flush=True)
    triangle, days, population = c.load_wide()
    d = c.as_of_window(triangle, days, population, MAIN, states=THREE)
    print("cells", (d["y"] >= 0).sum(), "| kept7", d["kept7"].sum(1), "| tail", d["tail"][:, 0], flush=True)
    t0 = time.perf_counter()
    model = c.build_model(d)
    print(f"built in {time.perf_counter() - t0:.1f} s; {model.initial_point().__len__()} value vars, "
          f"{sum(np.size(v) for v in model.initial_point().values())} parameters", flush=True)

    t0 = time.perf_counter()
    f = model.logp_dlogp_function(ravel_inputs=True, mode="NUMBA")
    f.set_extra_values({})
    point = model.initial_point()
    q0 = np.concatenate([np.ravel(point[v.name]) for v in model.value_vars])
    f._pytensor_function(q0)
    compile_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    times, finite = [], True
    for _ in range(200):
        q = q0 + 0.05 * rng.normal(size=q0.size)
        t0 = time.perf_counter()
        logp, grad = f._pytensor_function(q)[:2]
        times.append(time.perf_counter() - t0)
        finite &= bool(np.isfinite(logp) and np.isfinite(grad).all())
    median_ms = float(np.median(times) * 1e3)
    print(f"numba dlogp: compile {compile_s:.1f} s, median {median_ms:.3f} ms, min {min(times) * 1e3:.3f} ms, "
          f"finite {finite}, logp {float(logp):.1f}", flush=True)

    with model:
        prior = pm.sample_prior_predictive(draws=2000, var_names=["mu7", "phi"], random_seed=1)
    mu7, phi = prior.prior["mu7"].values[0], prior.prior["phi"].values[0]
    totals = c.noisy_total(d, mu7[None], phi[None], seed=1)
    _, final, _ = c.truth(triangle, days, MAIN, population, states=THREE)
    low, high = np.quantile(totals, [0.025, 0.975], axis=0)
    covers = (low <= 0.1 * final) & (high >= 10 * final)
    expected = (mu7.sum(axis=-1).sum(axis=-1) * (1 + d["tail"][:, 0])).reshape(-1, len(THREE))
    exp_low, exp_high = np.quantile(expected, [0.025, 0.975], axis=0)
    print("supplementary, expected 7-day total before known cases:", exp_low, exp_high, flush=True)
    result = {"supplementary_expected_low": exp_low, "supplementary_expected_high": exp_high,
              "supplementary_covers": ((exp_low <= 0.1 * final) & (exp_high >= 10 * final)).tolist(),
              "gradient_ms_median": median_ms, "gradient_ms_min": min(times) * 1e3, "compile_s": compile_s,
              "parameters": int(q0.size), "cells": int((d["y"] >= 0).sum()), "finite": finite,
              "prior_low": low, "prior_high": high, "final": final, "covers": covers.tolist(),
              "prior_q999": np.quantile(totals, 0.999, axis=0), "health": health()}
    print({k: v for k, v in result.items()}, flush=True)
    store.save_json("p3_cpu", {**result, "go": {"gradient": median_ms <= 3.0, "prior": bool(covers.all())}})


def signals(triangle, days, population):
    """Section 4's baselines at every as-of date of the window, on the populations given."""
    span = np.arange(c.day_index(days, b.EVALUATION[0]), c.day_index(days, b.EVALUATION[1]) + 1)
    frozen = np.zeros((16, span.size))
    final = np.zeros_like(frozen)
    factor = np.zeros_like(frozen)
    stale = np.zeros_like(frozen)
    ladder = np.zeros_like(frozen)
    for i, state in enumerate(c.STATES):
        tri = np.where(triangle[i + 1, 0, :, :81] == b.MISSING, 0, triangle[i + 1, 0, :, :81]).astype(float)
        cum = tri.cumsum(axis=1)
        for j, a in enumerate(span):
            frozen[i, j] = sum(cum[r, a - r] for r in range(a - 6, a + 1))
            final[i, j] = cum[a - 6 : a + 1, 80].sum()
            stale[i, j] = sum(cum[r, a - r] for r in range(a - 13, a - 6))
            # factor: frozen x the median at-40 / frozen ratio over the last 28 dates complete at 40 days
            ratios = []
            for t in range(a - 40 - 27, a - 40 + 1):
                then = sum(cum[r, t - r] for r in range(t - 6, t + 1))
                later = cum[t - 6 : t + 1, 40].sum()
                if then > 0:
                    ratios.append(later / then)
            factor[i, j] = frozen[i, j] * np.median(ratios)
            # ladder: development factors from the last 42 rows, volume weighted, up to delay 40
            est = 0.0
            for r in range(a - 6, a + 1):
                value, age = cum[r, a - r], a - r
                for k in range(age, 40):
                    rows = [u for u in range(a - 41, a + 1) if u + k + 1 <= a]
                    top, bottom = cum[rows, k + 1].sum(), cum[rows, k].sum()
                    value *= top / bottom if bottom > 0 else 1.0
                est += value
            ladder[i, j] = est
    return span, frozen, final, factor, stale, ladder


def score_table(sig, pop):
    span, frozen, final, factor, stale, ladder = sig
    out = {}
    scale = 1e5 / pop[:, None]
    for name, x in (("frozen", frozen), ("revised_7d_stale", stale), ("factor", factor), ("ladder", ladder)):
        ok = (x > 0) & (final > 0)
        row = {"mae_log": float(np.abs(np.log(x[ok]) - np.log(final[ok])).mean()), "excluded": int((~ok).sum())}
        for tau in c.TAUS:
            up, truth_up = x * scale >= tau, final * scale >= tau
            row[f"tau{tau}"] = [int((up & ~truth_up).sum()), int((~up & truth_up).sum())]
        out[name] = row
    return out


def baselines():
    triangle, days, population = c.load_wide()
    t0 = time.perf_counter()
    sig = signals(triangle, days, population)
    print(f"signals in {time.perf_counter() - t0:.1f} s", flush=True)
    hub = b.read_population(b.CACHE / "hub" / "population_sizes.csv")
    table = {"destatis": score_table(sig, population[1:17, 0]), "hub_control": score_table(sig, hub[1:17, 0])}
    print(json.dumps(table, indent=0), flush=True)
    store.save_json("baselines_summary", table)
    store.save_array("baselines_signals", np.stack(sig[1:]))
    store.save_array("baselines_span", sig[0])


def set_precision(floatX):
    """PyTensor's floatX and JAX's x64 must agree; import pymc turns x64 on, so set it after."""
    import pymc  # noqa: F401
    import pytensor

    pytensor.config.floatX = floatX
    import jax

    jax.config.update("jax_enable_x64", floatX == "float64")
    return jax


def window(date, states=c.STATES):
    triangle, days, population = c.load_wide()
    return c.as_of_window(triangle, days, population, date, states=states), (triangle, days, population)


def gputime(floatX):
    jax = set_precision(floatX)
    import jax.numpy as jnp

    d, _ = window(MAIN)
    with gpu_lock():
        model = c.build_model(d)
        logdensity, _, q0, names = c.jax_functions(model)
        print("backend", jax.default_backend(), "dim", q0.size, q0.dtype, flush=True)
        out = {"floatX": floatX, "dim": int(q0.size)}
        if floatX == "float32":
            rng = np.random.default_rng(0)
            idx = jnp.asarray(rng.integers(0, 1_000_000, 10_000_000), jnp.int32)
            v = jnp.asarray(rng.normal(size=10_000_000), jnp.float32)
            scatter = jax.jit(lambda v: jnp.zeros(1_000_000, jnp.float32).at[idx].add(v))
            gather = jax.jit(lambda x: x[idx].sum())
            from pymc_challenges.bigrun import time_call

            out["scatter_10M"] = time_call(scatter, v)
            out["gather_10M"] = time_call(gather, v[:1_000_000])
            print(out, flush=True)
        out["gradient"] = time_gradient(logdensity, q0, chains=(1, 8, 32, 128), jitter=0.05)
        out["vram"] = vram_high_water()
        for row in out["gradient"]:
            print(row, flush=True)
        print(out["vram"], flush=True)
    store.save_json(f"p3_gpu_{floatX}", {**out, "health": health()})


def p4cpu(tag="base", tune="1000", draws="1000", target="0.8", low_rank="1", centred="0"):
    import pymc  # noqa: F401

    d, _ = window(MAIN)
    model = c.build_model(d, centred_drift=centred == "1")
    out, extra, _ = c.fit_cpu(d, model, tune=int(tune), draws=int(draws), target=float(target),
                              low_rank=low_rank == "1")
    store.save_chunk(f"p4_cpu_{tag}", 0, {k: np.asarray(v) for k, v in out.items()})
    store.save_json(f"p4_cpu_{tag}", {**extra, "health": health()})


def p4gpu(tag="c32", chains="32", warmup="1000", draws="1000", seed="1", centred="0", date=MAIN, target="0.651"):
    set_precision("float32")
    d, _ = window(date)
    with gpu_lock():
        model = c.build_model(d, centred_drift=centred == "1")
        out, extra, q = c.fit_gpu(d, model, chains=int(chains), warmup=int(warmup), draws=int(draws), seed=int(seed), target=float(target))
        vram = vram_high_water()
    print(json.dumps({k: v for k, v in extra.items()}, default=str), vram, flush=True)
    store.save_chunk(f"p4_gpu_{tag}", 0, {k: np.asarray(v) for k, v in out.items()})
    store.save_array(f"p4_gpu_{tag}_q_tail", q[:, -100:].astype(np.float32))
    store.save_json(f"p4_gpu_{tag}", {**extra, "vram": vram, "health": health()})


START = np.datetime64("2021-09-15")
GRID = [str(START + np.timedelta64(6 * k, "D")) for k in range(38)]
P5_DATES = ["2021-10-03", "2021-11-14", "2021-12-26", "2022-02-18", "2022-04-01"]


def chunk_id(date):
    return int((np.datetime64(date) - START) / np.timedelta64(1, "D"))


def gpufits(*dates):
    """One 16-state GPU fit per date, one at a time, the P4 settings. A finished date is skipped."""
    set_precision("float32")
    triangle, days, population = c.load_wide()
    for date in dates:
        if not store.pending("gpu_fits", chunk_id(date) + 1)[-1:] == [chunk_id(date)]:
            print("have", date, flush=True)
            continue
        health()
        d = c.as_of_window(triangle, days, population, date)
        with gpu_lock():
            model = c.build_model(d)
            out, extra, _ = c.fit_gpu(d, model, chains=32, warmup=1000, draws=1000, seed=chunk_id(date) + 11, target=0.85)
            vram = vram_high_water()
        store.save_chunk("gpu_fits", chunk_id(date), {k: np.asarray(v) for k, v in out.items()})
        store.save_json(f"gpu_fit_{date}", {**extra, "date": date, "vram": vram, "health": health()})


def poolfit(date, target="0.95"):
    """One 16-state nutpie fit in its own process: a member of the 3-at-a-time pool."""
    import pymc  # noqa: F401

    d, _ = window(date)
    t0 = time.time()
    out, extra, _ = c.fit_cpu(d, c.build_model(d), target=float(target))
    extra.update(started=t0, ended=time.time())
    store.save_chunk(f"pool_{date}", 0, {k: np.asarray(v) for k, v in out.items() if not k.startswith("u_")})
    store.save_json(f"pool_{date}", {**extra, "date": date, "health": health()})


def withdata():
    """Compile for another date, move to the main date with with_data, compare with a fresh compile."""
    import nutpie
    import pymc  # noqa: F401

    d_other, _ = window("2021-12-26")
    d, _ = window(MAIN)
    t0 = time.time()
    compiled = nutpie.compile_pymc_model(c.build_model(d_other)).with_data(**c.data_update(d))
    out, extra, _ = c.fit_cpu(d, compiled=compiled, seed=7)
    extra.update(started=t0, ended=time.time())
    store.save_chunk("pool_withdata", 0, {k: np.asarray(v) for k, v in out.items()})
    store.save_json("pool_withdata", {**extra, "health": health()})


def precision():
    jax = set_precision("float32")
    d, _ = window(MAIN)
    q = store.load_array("p4_gpu_c32t85_q_tail")
    draws = q.reshape(-1, q.shape[-1])[::4]
    with gpu_lock():
        model32 = c.build_model(d)
        f32, _, _, _ = c.jax_functions(model32)
        set_precision("float64")
        model64 = c.build_model(d)
        f64, _, _, _ = c.jax_functions(model64)
        set_precision("float32")
        result = precision_check(f32, f64, draws)
    print(result, flush=True)
    store.save_json("p3_precision", {**result, "line": 0.1, "health": health()})


def p4eval(ref="p4_cpu_base"):
    def latent(name):
        return store.load_chunk(name, 0)

    gpu, gpu8, cpu = latent("p4_gpu_c32t85"), latent("p4_gpu_c8t85"), latent(ref)
    z = lambda a, b_: (a["latent_mean"] - b_["latent_mean"]) / np.hypot(a["latent_mcse"], b_["latent_mcse"])
    z_chain, z_route = z(gpu8, gpu), z(gpu, cpu)
    dp = np.abs(gpu["p_cross"] - cpu["p_cross"])
    result = {"chain_z": z_chain, "chain_go": bool((np.abs(z_chain) <= 2).sum() >= 15 and (np.abs(z_chain) <= 3).all()),
              "route_z": z_route, "route_max_dp": float(dp.max()),
              "route_go": bool((np.abs(z_route) <= 3).all() and dp.max() <= 0.05),
              "latent_relative_gap": (gpu["latent_mean"] - cpu["latent_mean"]) / cpu["latent_mean"]}
    print(result, flush=True)
    store.save_json(f"p4_eval_{ref}", result)


def p6():
    triangle, days, population = c.load_wide()
    rows, gaps = [], []
    for g in P5_DATES:
        a = str(np.datetime64(g) + np.timedelta64(5, "D"))
        fit, full = store.load_chunk("gpu_fits", chunk_id(g)), store.load_chunk("gpu_fits", chunk_id(a))
        d = c.as_of_window(triangle, days, population, a)
        updated = c.daily_update(fit, d, 5)
        gap = np.abs(updated - full["p_cross"])
        gaps.append(gap)
        rows.append({"from": g, "to": a, "median": float(np.median(gap)), "max": float(gap.max())})
        print(rows[-1], flush=True)
    gaps = np.stack(gaps)
    result = {"rows": rows, "median_abs": float(np.median(gaps)), "max_abs": float(gaps.max()), "n": int(gaps.size),
              "share_over_0.10": float((gaps > 0.10).mean()),
              "go": bool(np.median(gaps) <= 0.03 and gaps.max() <= 0.10)}
    print(result, flush=True)
    store.save_json("p6", result)


def p8():
    triangle, days, population = c.load_wide()
    signals_ = store.load_array("baselines_signals")
    span = store.load_array("baselines_span")

    def score(dates):
        med, low, high, final_, factor_ = [], [], [], [], []
        for date in dates:
            fit = store.load_chunk("gpu_fits", chunk_id(date))
            _, final, pop = c.truth(triangle, days, date, population)
            j = int(np.flatnonzero(span == c.day_index(days, date))[0])
            factor_.append(signals_[2, :, j] * 1e5 / pop)
            final_.append(final * 1e5 / pop)
            med.append(fit["q"][3])
            low.append(fit["q"][1])
            high.append(fit["q"][5])
        med, low, high, final_, factor_ = map(np.concatenate, (med, low, high, final_, factor_))
        ok = (med > 0) & (final_ > 0) & (factor_ > 0)
        mae = float(np.abs(np.log(med[ok]) - np.log(final_[ok])).mean())
        mae_f = float(np.abs(np.log(factor_[ok]) - np.log(final_[ok])).mean())
        return {"n": int(med.size), "excluded": int((~ok).sum()), "mae_log": mae, "mae_log_factor": mae_f,
                "coverage_80": float(((low <= final_) & (final_ <= high)).mean())}

    gate = score(GRID[:8])
    gate["go"] = bool(gate["mae_log"] <= gate["mae_log_factor"] + 0.03 and 0.6 <= gate["coverage_80"] <= 0.95)
    diagnostic = score(P5_DATES)
    result = {"gate": gate, "diagnostic_p5_dates": diagnostic}
    print(result, flush=True)
    store.save_json("p8", result)


def p7(chains="32", warmup="1000", draws="1000", target="0.85"):
    set_precision("float32")
    triangle, days, population = c.load_wide()
    d = c.strata_window(triangle, days, population, MAIN)
    with gpu_lock():
        model = c.build_strata_model(d)
        print("strata parameters", sum(np.size(v) for v in model.initial_point().values()), "cells", d["y"][:, :, c.known_mask()].size, flush=True)
        out, extra, _ = c.fit_gpu(d, model, chains=int(chains), warmup=int(warmup), draws=int(draws), seed=71,
                                  target=float(target), outputs=("mu7", "phi"))
        vram = vram_high_water()
    print(json.dumps(extra, default=str), vram, flush=True)
    store.save_chunk("p7_gpu", 0, {k: np.asarray(v) for k, v in out.items() if not k.startswith("u_")})
    store.save_json("p7_gpu", {**extra, "vram": vram, "health": health()})


def strata_smoke():
    """CPU check that the strata window and model evaluate; no fit."""
    jax = set_precision("float32")
    triangle, days, population = c.load_wide()
    d = c.strata_window(triangle, days, population, MAIN)
    print("y", d["y"].shape, "cases", d["y"][d["y"] > 0].sum(), "scale", d["scale"].round(3), flush=True)
    model = c.build_strata_model(d)
    logdensity, forward, q0, names = c.jax_functions(model, ("mu7", "phi"))
    logp, grad = jax.value_and_grad(logdensity)(q0)
    mu7, phi = forward(q0)
    print(jax.default_backend(), q0.size, float(logp), bool(np.isfinite(np.asarray(grad)).all()), mu7.shape, phi.shape, flush=True)


def p6control():
    """Gap 0: updating a fit on its own date must return that fit's own P(cross)."""
    triangle, days, population = c.load_wide()
    gaps = []
    for g in P5_DATES:
        fit = store.load_chunk("gpu_fits", chunk_id(g))
        d = c.as_of_window(triangle, days, population, g)
        updated = c.daily_update(fit, d, 0)
        gap = np.abs(updated - fit["p_cross"])
        gaps.append(gap)
        q = np.unravel_index(gap.argmax(), gap.shape)
        print(g, "median", np.median(gap), "max", gap.max().round(3), "at tau", c.TAUS[q[0]], c.STATES[q[1]],
              "update", updated[q].round(3), "fit", fit["p_cross"][q].round(3), flush=True)
    gaps = np.stack(gaps)
    print("control median", np.median(gaps), "max", gaps.max(), "share>0.1", (gaps > 0.1).mean(), flush=True)
    a_gaps = store.load_json("p6")
    for g in P5_DATES:
        a = str(np.datetime64(g) + np.timedelta64(5, "D"))
        fit, full = store.load_chunk("gpu_fits", chunk_id(g)), store.load_chunk("gpu_fits", chunk_id(a))
        d = c.as_of_window(triangle, days, population, a)
        up = c.daily_update(fit, d, 5)
        gap = up - full["p_cross"]
        idx = np.argwhere(np.abs(gap) > 0.1)
        print(a, "n>0.1", len(idx), [(c.TAUS[i], c.STATES[j], float(up[i, j].round(2)), float(full["p_cross"][i, j].round(2))) for i, j in idx[:8]], flush=True)
        print("   frozen/kept7", d["kept7"].sum(1)[:6], flush=True)
    store.save_json("p6_control", {"median": float(np.median(gaps)), "max": float(gaps.max()), "share_over_0.10": float((gaps > 0.1).mean())})


def p5eval(scratch):
    """P5's gates, the ESS-per-second comparison on the same six dates, and the thermal record."""
    dates = [MAIN, *P5_DATES]
    p4_cpu = store.load_json(f"pool_{MAIN}")["wall_s"]
    p4_gpu = store.load_json("p4_gpu_c32t85")["wall_s"]
    rows, z_max, dp_max = [], 0.0, 0.0
    total_cpu = total_gpu = wall_gpu = sample_gpu = 0.0
    for date in dates:
        cpu = store.load_chunk(f"pool_{date}", 0)
        cj = store.load_json(f"pool_{date}")
        if date == MAIN:
            gpu, gj = store.load_chunk("p4_gpu_c32t85", 0), store.load_json("p4_gpu_c32t85")
        else:
            gpu, gj = store.load_chunk("gpu_fits", chunk_id(date)), store.load_json(f"gpu_fit_{date}")
        ess_c, ess_g = float(cpu["latent_ess"].min()), float(gpu["latent_ess"].min())
        z = np.abs((gpu["latent_mean"] - cpu["latent_mean"]) / np.hypot(gpu["latent_mcse"], cpu["latent_mcse"]))
        dp = np.abs(gpu["p_cross"] - cpu["p_cross"]).max()
        z_max, dp_max = max(z_max, float(z.max())), max(dp_max, float(dp))
        rows.append({"date": date, "cpu_wall_s": cj["wall_s"], "gpu_wall_s": gj["wall_s"], "cpu_ess": ess_c, "gpu_ess": ess_g,
                     "cpu_rhat": float(cpu["latent_rhat"].max()), "gpu_rhat": float(gpu["latent_rhat"].max()),
                     "cpu_div": cj["divergence_rate"], "gpu_div": gj["divergence_rate"], "z_max": float(z.max()), "dp_max": float(dp),
                     "single_fit_ratio": (ess_g / gj["wall_s"]) / (ess_c / cj["wall_s"])})
        total_cpu += ess_c
        total_gpu += ess_g
        wall_gpu += gj["wall_s"]
        sample_gpu += gj["sampling_s"]
        print(rows[-1], flush=True)
    logs = Path(scratch) / "logs"
    pool_wall = float((logs / "pool.end").read_text()) - float((logs / "pool.start").read_text())
    pool_rate, gpu_rate = total_cpu / pool_wall, total_gpu / wall_gpu
    fresh, moved = store.load_chunk("p4_cpu_base", 0), store.load_chunk("pool_withdata", 0)
    z_data = np.abs((fresh["latent_mean"] - moved["latent_mean"]) / np.hypot(fresh["latent_mcse"], moved["latent_mcse"]))
    t = [line.split(",") for line in (logs / "thermal.csv").read_text().splitlines()]
    t = [r for r in t if len(r) == 6 and r[1] and r[2]]
    result = {"rows": rows, "pool_wall_s": pool_wall, "pool_ess_per_s": pool_rate, "pool_ess_per_s_15_cores_extrapolated": pool_rate * 15 / 12,
              "gpu_ess_per_s": gpu_rate, "ratio": gpu_rate / pool_rate, "ratio_15_cores": gpu_rate / (pool_rate * 15 / 12),
              "gpu_sampling_only_ess_per_s": total_gpu / sample_gpu,
              "cpu_ratio_to_p4": [r["cpu_wall_s"] / p4_cpu for r in rows], "gpu_ratio_to_p4": [r["gpu_wall_s"] / p4_gpu for r in rows],
              "route_z_max": z_max, "route_dp_max": dp_max, "with_data_z_max": float(z_data.max()), "with_data_go": bool(z_data.max() <= 3),
              "thermal": {"max_tctl": max(float(r[1]) for r in t), "max_gpu": max(float(r[2]) for r in t), "samples": len(t),
                          "hw_slowdown": sum("Active" in r[3] and "Not" not in r[3] for r in t), "max_power_w": max(float(r[5]) for r in t)}}
    print({k: v for k, v in result.items() if k != "rows"}, flush=True)
    store.save_json("p5_eval", result)


def p7extra():
    """Not gated. float32 on the strata model at warmed-up draws, and what strata changed on the date."""
    set_precision("float32")
    triangle, days, population = c.load_wide()
    d = c.strata_window(triangle, days, population, MAIN)
    with gpu_lock():
        model = c.build_strata_model(d)
        _, _, q = c.fit_gpu(d, model, chains=32, warmup=400, draws=20, seed=72, target=0.85, outputs=("mu7", "phi"))
        f32, _, _, _ = c.jax_functions(model, ("mu7", "phi"))
        set_precision("float64")
        f64, _, _, _ = c.jax_functions(c.build_strata_model(d), ("mu7", "phi"))
        set_precision("float32")
        prec = precision_check(f32, f64, q.reshape(-1, q.shape[-1])[::2])
    strata, state = store.load_chunk("p7_gpu", 0), store.load_chunk("p4_gpu_c32t85", 0)
    gap = strata["p_cross"] - state["p_cross"]
    crossed = (strata["p_cross"] >= 0.5) != (state["p_cross"] >= 0.5)
    result = {"precision": prec, "dp_median": float(np.median(np.abs(gap))), "dp_max": float(np.abs(gap).max()),
              "decisions_changed_at_0.5": int(crossed.sum()),
              "latent_relative_gap": (strata["latent_mean"] - state["latent_mean"]) / state["latent_mean"]}
    print(result, flush=True)
    store.save_json("p7_extra", {**result, "health": health()})


def p6gap2():
    """Not a written gate: the same lines at gap 2, the longest gap on a 3-day grid."""
    triangle, days, population = c.load_wide()
    gaps = []
    for g in P5_DATES:
        a = str(np.datetime64(g) + np.timedelta64(2, "D"))
        fit, full = store.load_chunk("gpu_fits", chunk_id(g)), store.load_chunk("gpu_fits", chunk_id(a))
        gaps.append(np.abs(c.daily_update(fit, c.as_of_window(triangle, days, population, a), 2) - full["p_cross"]))
        print(g, a, "median", float(np.median(gaps[-1])), "max", float(gaps[-1].max()), flush=True)
    gaps = np.stack(gaps)
    result = {"median_abs": float(np.median(gaps)), "max_abs": float(gaps.max()), "share_over_0.10": float((gaps > 0.1).mean())}
    print(result, flush=True)
    store.save_json("p6_gap2", result)


def allscore():
    """Not gated: every 16-state GPU fit so far, against factor and frozen, with coverage."""
    triangle, days, population = c.load_wide()
    signals_, span = store.load_array("baselines_signals"), store.load_array("baselines_span")
    ids = [i for i in store._chunk_ids("gpu_fits") if i != chunk_id(MAIN)]  # FULL stores MAIN there too
    dates = [str(START + np.timedelta64(i, "D")) for i in ids] + [MAIN]
    rows = []
    for date in dates:
        fit = store.load_chunk("p4_gpu_c32t85" if date == MAIN else "gpu_fits", 0 if date == MAIN else chunk_id(date))
        _, final, pop = c.truth(triangle, days, date, population)
        j = int(np.flatnonzero(span == c.day_index(days, date))[0])
        inc = final * 1e5 / pop
        factor = signals_[2, :, j] * 1e5 / pop
        frozen = signals_[0, :, j] * 1e5 / pop
        q = fit["q"]
        rows.append({"date": date, "mae": np.abs(np.log(q[3]) - np.log(inc)).mean(), "factor": np.abs(np.log(factor) - np.log(inc)).mean(),
                     "frozen": np.abs(np.log(frozen) - np.log(inc)).mean(), "c50": ((q[2] <= inc) & (inc <= q[4])).mean(),
                     "c80": ((q[1] <= inc) & (inc <= q[5])).mean(), "c95": ((q[0] <= inc) & (inc <= q[6])).mean(),
                     "bias": (np.log(q[3]) - np.log(inc)).mean()})
        print({k: (round(v, 3) if not isinstance(v, str) else v) for k, v in rows[-1].items()}, flush=True)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "date"}
    print("all", len(rows), mean, flush=True)
    store.save_json("all_scores", {"rows": rows, "mean": mean})


JOBS = {"allscore": allscore, "p7extra": p7extra, "p6gap2": p6gap2, "p5eval": p5eval, "p6control": p6control, "strata_smoke": strata_smoke, "p6": p6, "p8": p8, "p7": p7, "p3cpu": p3cpu, "baselines": baselines, "gputime": gputime, "p4cpu": p4cpu, "p4gpu": p4gpu,
        "gpufits": gpufits, "poolfit": poolfit, "withdata": withdata, "precision": precision, "p4eval": p4eval}

if __name__ == "__main__":
    print("job", sys.argv[1:], "health", health(), flush=True)
    t0 = time.perf_counter()
    JOBS[sys.argv[1]](*sys.argv[2:])
    print(f"job done in {time.perf_counter() - t0:.1f} s; health {health()}", flush=True)
