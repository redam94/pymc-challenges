"""E90 core: as-of windows, the section 6 nowcast model, CPU and GPU fits, decision draws.

Imported by tools/pilot_e90.py and, later, tools/run_e90_full.py. See plans/large/E90.md, sections 3
and 6. Set OPENBLAS_NUM_THREADS=1 and PYTENSOR_FLAGS=base_compiledir=... before importing this.

Nothing after the as-of date moves a cell: counts come from `asof_nonnegative`, and only delay bins
fully known on the as-of date enter the likelihood. The window has a fixed shape (70 rows, 18 bins),
so the mask is the same on every date; only counts, weekdays and starting levels change.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_e90_data as b  # noqa: E402

T, K, B = 70, 7, 18
N_HAZARD = B - 1
W = 10
TAUS = (3, 6, 9)
FIRST_WIDE = "2021-04-06"  # the tail ratio on 2021-09-15 needs rows 80 days old, before the npz starts
BIN_EDGES = [(d, d) for d in range(15)] + [(15, 21), (22, 28), (29, 42)]
BIN_LAST = np.array([e[1] for e in BIN_EDGES])
WIDE = b.CACHE / "out" / "wide_triangle.npz"
STATES = b.LOCATIONS[1:]


def load_wide():
    """(triangle, days, population): the unprocessed file from its first date, cached outside git."""
    if not WIDE.exists():
        b.FIRST = FIRST_WIDE
        triangle, days = b.read_triangle(b.CACHE / "hub" / "COVID-19_hospitalizations.csv")
        with np.load(b.CACHE / "out" / "covid_hosp_vintages_de.npz") as npz:
            population = npz["population"]
        np.savez_compressed(WIDE, triangle=triangle, days=days.astype(str), population=population)
    with np.load(WIDE) as npz:
        return npz["triangle"], npz["days"].astype("datetime64[D]"), npz["population"]


def day_index(days, date):
    return int((np.datetime64(date) - days[0]) / np.timedelta64(1, "D"))


def known_mask():
    """(T, B): a bin enters only if all its delays are known; the same on every as-of date."""
    age = T - 1 - np.arange(T)
    return BIN_LAST[None, :] <= age[:, None]


def bin_counts(cells):
    """(T, 81) non-negative as-of counts, -1 where unknown -> (T, B), -1 where the bin is not whole."""
    out = np.full((T, B), -1, np.int64)
    known = known_mask()
    for k, (lo, hi) in enumerate(BIN_EDGES):
        out[known[:, k], k] = cells[known[:, k], lo : hi + 1].sum(axis=1)
    return out


def tail_params(block, asof, rows=28):
    """(mean ratio, gamma shape, gamma rate) of cases at delay 43-80 per case within 42.

    Uses the last `rows` rows complete at 80 days on `asof`. Negative tails floor at 0.
    """
    last = asof - 80
    cells = np.where(block == b.MISSING, 0, block)[last - rows + 1 : last + 1]
    inside, tail = max(float(cells[:, :43].sum()), 1.0), float(max(cells[:, 43:81].sum(), 0))
    return (tail + 0.5) / inside, tail + 0.5, inside


def as_of_window(triangle, days, population, date, states=STATES, age=0):
    """Everything one fit needs on one as-of date. `age` is an index into b.AGES."""
    asof = day_index(days, date)
    first = asof - T + 1
    rows = np.arange(first, asof + 1)
    y, kept7, tail = [], [], []
    for state in states:
        block = triangle[b.LOCATIONS.index(state), age]
        cells, _ = b.asof_nonnegative(block[first : asof + 1], T - 1)
        y.append(bin_counts(cells))
        kept7.append(np.where(cells < 0, 0, cells)[-K:].sum(axis=1))
        tail.append(tail_params(block, asof))
    y, kept7, tail = np.stack(y), np.stack(kept7).astype(float), np.array(tail)
    dow = (rows.astype("int64") + int(days[0].astype("datetime64[D]").astype("int64")) + 3) % 7
    rep = (dow[:, None] + np.arange(N_HAZARD)[None, :]) % 7  # reporting weekday; only daily bins use it
    start = y[:, :7, :].sum(axis=2).mean(axis=1)
    locs = [b.LOCATIONS.index(s) for s in states]
    return {"date": date, "asof": asof, "states": tuple(states), "y": y, "kept7": kept7, "tail": tail,
            "dow": dow, "rep": rep, "l0_mu": np.log(np.maximum(start, 1.0)),
            "population": population[locs, age].astype(float)}


def truth(triangle, days, date, population, states=STATES):
    """Frozen and final 7-day counts and the populations, (states,)."""
    t = day_index(days, date)
    locs = [b.LOCATIONS.index(s) for s in states]
    frozen = np.array([b.frozen_7d(triangle[l, 0], t) for l in locs], float)
    final = np.array([b.final_7d(triangle[l, 0], t) for l in locs], float)
    return frozen, final, population[locs, 0].astype(float)


def build_model(d, centred_drift=False, fixed_logit=None):
    """Section 6: hierarchical nowcast, centred likelihood, scales LogNormal as E86 advises.

    On a normal as-of window this is the model of the pilots. Two variants, for section 6's hindsight
    fits: a window from `finished_window` (any length, every bin known) adds the delay parameters to
    the outputs, and `fixed_logit` (S, rows, N_HAZARD) replaces the hazard block with fixed numbers.
    """
    import pymc as pm
    import pytensor.tensor as pt

    S, rows = len(d["states"]), d["y"].shape[1]
    weeks = (rows - 1) // 7 + 1
    finished = "known" in d
    known = d["known"] if finished else known_mask()
    ti, bi = np.nonzero(known)
    si = np.repeat(np.arange(S), ti.size)
    ti, bi = np.tile(ti, S), np.tile(bi, S)
    flat = np.ravel_multi_index((si, ti, bi), (S, rows, B))
    week = np.arange(rows) // 7
    basis = np.stack([np.cos(np.pi * k * (np.arange(N_HAZARD) + 0.5) / N_HAZARD) for k in range(3)], 1)
    basis = pm.floatX(basis / np.linalg.norm(basis, axis=0))
    daily = pm.floatX((np.arange(N_HAZARD) < 15).astype(float))
    coords = {"state": list(d["states"]), "weekday": list(range(7)), "basis": [0, 1, 2]}

    def walk(name, shape, sigma):
        """A random walk over the last axis starting at 0 at the first week."""
        if centred_drift:
            steps = pm.Normal(name, 0.0, sigma, shape=shape)
        else:
            steps = sigma * pm.Normal(name, 0.0, 1.0, shape=shape)
        return pt.concatenate([pt.zeros((*shape[:-1], 1)), pt.cumsum(steps, axis=-1)], axis=-1)

    with pm.Model(coords=coords) as model:
        y = pm.Data("y", d["y"][:, known].ravel().astype("int64"))
        dow = pm.Data("dow", d["dow"].astype("int64"))
        rep = pm.Data("rep", d["rep"].astype("int64"))
        l0_mu = pm.Data("l0_mu", pm.floatX(d["l0_mu"]))

        # level: integrated weekly growth, national walk plus partially pooled state deviations
        l0 = l0_mu + 0.5 * pm.Normal("l0_z", 0.0, 1.0, shape=S)
        g0 = pm.Normal("g0", 0.0, 0.02)
        s_nat = pm.LogNormal("s_nat", np.log(0.015), 0.5)
        s_dev = pm.LogNormal("s_dev", np.log(0.007), 0.5)
        s_dev0 = pm.LogNormal("s_dev0", np.log(0.01), 0.5)
        g_nat = g0 + walk("g_nat", (weeks - 1,), s_nat)
        g_start = s_dev0 * pm.Normal("g_dev0_z", 0.0, 1.0, shape=S)
        g_dev = g_start[:, None] + walk("g_dev", (S, weeks - 1), s_dev)
        daily_growth = (g_nat[None, :] + g_dev)[:, week]
        level = l0[:, None] + pt.concatenate([pt.zeros((S, 1)), pt.cumsum(daily_growth[:, 1:], axis=1)], axis=1)
        wd = pm.ZeroSumNormal("wd", sigma=0.5, dims=("state", "weekday"), n_zerosum_axes=1)
        log_lam = level + wd[:, dow]

        # hazard: delay baseline walk, low-rank state offsets, reporting weekday, weekly drift
        if fixed_logit is None:
            gamma0 = pm.Normal("gamma0", -1.3, 1.0)
            s_gamma = pm.LogNormal("s_gamma", np.log(0.3), 0.7)
            gamma = gamma0 + (s_gamma * pt.concatenate(
                [pt.zeros(1), pt.cumsum(pm.Normal("gamma_z", 0.0, 1.0, shape=N_HAZARD - 1))]))
            s_a = pm.LogNormal("s_a", np.log(0.3), 0.7, dims="basis")
            a = s_a[None, :] * pm.Normal("a_z", 0.0, 1.0, dims=("state", "basis"))
            gamma_s = pt.dot(a, basis.T)
            rho = pm.ZeroSumNormal("rho", sigma=1.0, dims=("state", "weekday"), n_zerosum_axes=1)
            s_beta = pm.LogNormal("s_beta", np.log(0.07), 0.7)
            s_beta_s = pm.LogNormal("s_beta_s", np.log(0.05), 0.7)
            beta = walk("beta", (weeks - 1,), s_beta)
            beta_s = walk("beta_s", (S, weeks - 1), s_beta_s)
            logit_h = (gamma[None, None, :] + gamma_s[:, None, :] + rho[:, rep] * daily[None, None, :]
                       + (beta[None, :] + beta_s)[:, week][:, :, None])
        else:
            logit_h = pt.as_tensor_variable(pm.floatX(fixed_logit))
        log_h, log_1mh = -pt.softplus(-logit_h), -pt.softplus(logit_h)
        log_p = (pt.concatenate([log_h, pt.zeros((S, rows, 1))], axis=2)
                 + pt.concatenate([pt.zeros((S, rows, 1)), pt.cumsum(log_1mh, axis=2)], axis=2))
        log_mu = log_lam[:, :, None] + log_p

        inv_sqrt_phi = pm.LogNormal("inv_sqrt_phi", np.log(0.25), 0.5, shape=S)
        phi = pm.Deterministic("phi", 1 / inv_sqrt_phi**2)
        pm.NegativeBinomial("n", mu=pt.exp(log_mu.ravel()[flat]), alpha=phi[si], observed=y)
        pm.Deterministic("mu7", pt.exp(log_mu[:, -K:, :]))
        if fixed_logit is not None:
            return model
        # what the daily update needs: hazard of the latest week, level and growth at the last row
        pm.Deterministic("base", gamma[None, :] + gamma_s + (beta[None, :] + beta_s)[:, week[-1]][:, None])
        pm.Deterministic("lvl_end", level[:, -1])
        pm.Deterministic("grw_end", daily_growth[:, -1])
        pm.Deterministic("rho_d", rho)
        pm.Deterministic("wd_d", wd)
        if finished:
            pm.Deterministic("gamma_d", gamma)
            pm.Deterministic("gamma_s_d", gamma_s)
            pm.Deterministic("beta_d", beta)
            pm.Deterministic("beta_s_d", beta_s)
            pm.Deterministic("frac7", pt.exp(log_p[:, :, :7]).sum(axis=2).mean(axis=1))
    return model


def finished_window(triangle, days, population, first_date, last_date):
    """Rows first_date..last_date as seen 80 days after the last row: every delay bin is known."""
    first, last = day_index(days, first_date), day_index(days, last_date)
    rows = last - first + 1
    y, tail = [], []
    for state in STATES:
        block = triangle[b.LOCATIONS.index(state), 0]
        cells, _ = b.asof_nonnegative(block[first : last + 81], rows + 79)
        binned = np.stack([cells[:rows, lo : hi + 1].sum(axis=1) for lo, hi in BIN_EDGES], axis=1)
        y.append(binned)
        tail.append(tail_params(block, last + 80))
    y = np.stack(y).astype(np.int64)
    all_rows = np.arange(first, last + 1)
    dow = (all_rows.astype("int64") + int(days[0].astype("datetime64[D]").astype("int64")) + 3) % 7
    rep = (dow[:, None] + np.arange(N_HAZARD)[None, :]) % 7
    start = y[:, :7, :].sum(axis=2).mean(axis=1)
    locs = [b.LOCATIONS.index(s) for s in STATES]
    return {"date": last_date, "first": first, "states": STATES, "y": y, "known": np.ones((rows, B), bool),
            "kept7": y[:, -K:, :].sum(axis=2).astype(float), "tail": np.array(tail), "dow": dow, "rep": rep,
            "l0_mu": np.log(np.maximum(start, 1.0)), "population": population[locs, 0].astype(float)}


def finished_logit(mean, d):
    """(S, rows, N_HAZARD) mean logit hazard of a finished fit; linear, so the mean of the parts."""
    weeks = np.arange(d["y"].shape[1]) // 7
    daily = (np.arange(N_HAZARD) < 15).astype(float)
    return (mean["gamma_d"][None, None, :] + mean["gamma_s_d"][:, None, :]
            + mean["rho_d"][:, d["rep"]] * daily[None, None, :]
            + (mean["beta_d"][None, :] + mean["beta_s_d"])[:, weeks][:, :, None])


def strata_window(triangle, days, population, date):
    """96 strata flattened to (S x A): the same window, age groups 1..6, the state's all-age tail."""
    whole = as_of_window(triangle, days, population, date)
    parts = [as_of_window(triangle, days, population, date, age=a) for a in range(1, 7)]
    S, A = len(STATES), 6
    y = np.stack([p["y"] for p in parts], axis=1)  # S, A, T, B
    kept7 = np.stack([p["kept7"] for p in parts], axis=1).reshape(S * A, K)
    tail = np.repeat(whole["tail"], A, axis=0)
    per_stratum = np.stack([p["l0_mu"] for p in parts], axis=1)
    known_all = whole["kept7"].sum(1)
    known_strata = kept7.reshape(S, A, K).sum((1, 2))
    return {"date": date, "asof": whole["asof"], "states": whole["states"], "y": y, "kept7": kept7, "tail": tail,
            "dow": whole["dow"], "rep": whole["rep"], "l0_mu": whole["l0_mu"], "l0_age": per_stratum,
            "population": np.ones(S * A), "population_all": whole["population"], "group": (S, A),
            "scale": known_all / np.maximum(known_strata, 1.0)}


def build_strata_model(d):
    """Variant 3: age shares the state's weekly path, with its own level, drift and delay offset."""
    import pymc as pm
    import pytensor.tensor as pt

    S, A = d["group"]
    known = known_mask()
    ti0, bi0 = np.nonzero(known)
    si = np.repeat(np.arange(S), A * ti0.size)
    ai = np.tile(np.repeat(np.arange(A), ti0.size), S)
    ti, bi = np.tile(ti0, S * A), np.tile(bi0, S * A)
    flat = np.ravel_multi_index((si, ai, ti, bi), (S, A, T, B))
    week = np.arange(T) // 7
    basis = np.stack([np.cos(np.pi * k * (np.arange(N_HAZARD) + 0.5) / N_HAZARD) for k in range(3)], 1)
    basis = pm.floatX(basis / np.linalg.norm(basis, axis=0))
    daily = pm.floatX((np.arange(N_HAZARD) < 15).astype(float))
    coords = {"state": list(d["states"]), "weekday": list(range(7)), "basis": [0, 1, 2]}

    def walk(name, shape, sigma):
        steps = sigma * pm.Normal(name, 0.0, 1.0, shape=shape)
        return pt.concatenate([pt.zeros((*shape[:-1], 1)), pt.cumsum(steps, axis=-1)], axis=-1)

    with pm.Model(coords=coords) as model:
        y = pm.Data("y", d["y"][:, :, known].ravel().astype("int64"))
        dow = pm.Data("dow", d["dow"].astype("int64"))
        rep = pm.Data("rep", d["rep"].astype("int64"))
        l0 = pm.floatX(d["l0_mu"])[:, None] + 0.5 * pm.Normal("l0_z", 0.0, 1.0, shape=(S, 1))
        l0_age = pm.floatX(d["l0_age"] - d["l0_mu"][:, None]) + 0.5 * pm.Normal("l0_age_z", 0.0, 1.0, shape=(S, A))
        g0 = pm.Normal("g0", 0.0, 0.02)
        s_nat = pm.LogNormal("s_nat", np.log(0.015), 0.5)
        s_dev = pm.LogNormal("s_dev", np.log(0.007), 0.5)
        s_dev0 = pm.LogNormal("s_dev0", np.log(0.01), 0.5)
        g_nat = g0 + walk("g_nat", (W - 1,), s_nat)
        g_dev = (s_dev0 * pm.Normal("g_dev0_z", 0.0, 1.0, shape=S))[:, None] + walk("g_dev", (S, W - 1), s_dev)
        s_age = pm.LogNormal("s_age", np.log(0.007), 0.5)
        g_age = walk("g_age", (A, W - 1), s_age)
        growth = (g_nat[None, None, :] + g_dev[:, None, :] + g_age[None, :, :])[:, :, week]
        level = (l0[:, :, None] + l0_age[:, :, None]
                 + pt.concatenate([pt.zeros((S, A, 1)), pt.cumsum(growth[:, :, 1:], axis=2)], axis=2))
        wd = pm.ZeroSumNormal("wd", sigma=0.5, dims=("state", "weekday"), n_zerosum_axes=1)
        log_lam = level + wd[:, dow][:, None, :]

        gamma0 = pm.Normal("gamma0", -1.3, 1.0)
        s_gamma = pm.LogNormal("s_gamma", np.log(0.3), 0.7)
        gamma = gamma0 + s_gamma * pt.concatenate(
            [pt.zeros(1), pt.cumsum(pm.Normal("gamma_z", 0.0, 1.0, shape=N_HAZARD - 1))])
        s_a = pm.LogNormal("s_a", np.log(0.3), 0.7, dims="basis")
        a = s_a[None, :] * pm.Normal("a_z", 0.0, 1.0, dims=("state", "basis"))
        gamma_s = pt.dot(a, basis.T)
        s_h = pm.LogNormal("s_h", np.log(0.2), 0.7)
        gamma_a = pt.dot(s_h * pm.Normal("h_z", 0.0, 1.0, shape=(A, 3)), basis.T)
        s_e = pm.LogNormal("s_e", np.log(0.1), 0.7)
        shift = s_e * pm.Normal("e_z", 0.0, 1.0, shape=(S, A))
        rho = pm.ZeroSumNormal("rho", sigma=1.0, dims=("state", "weekday"), n_zerosum_axes=1)
        s_beta = pm.LogNormal("s_beta", np.log(0.07), 0.7)
        s_beta_s = pm.LogNormal("s_beta_s", np.log(0.05), 0.7)
        beta = walk("beta", (W - 1,), s_beta)
        beta_s = walk("beta_s", (S, W - 1), s_beta_s)
        logit_h = (gamma[None, None, None, :] + gamma_s[:, None, None, :] + gamma_a[None, :, None, :]
                   + shift[:, :, None, None] + (rho[:, rep] * daily[None, None, :])[:, None, :, :]
                   + (beta[None, :] + beta_s)[:, week][:, None, :, None])
        log_h, log_1mh = -pt.softplus(-logit_h), -pt.softplus(logit_h)
        log_p = (pt.concatenate([log_h, pt.zeros((S, A, T, 1))], axis=3)
                 + pt.concatenate([pt.zeros((S, A, T, 1)), pt.cumsum(log_1mh, axis=3)], axis=3))
        log_mu = log_lam[:, :, :, None] + log_p
        inv_sqrt_phi = pm.LogNormal("inv_sqrt_phi", np.log(0.25), 0.5, shape=(S, A))
        phi = 1 / inv_sqrt_phi**2
        pm.NegativeBinomial("n", mu=pt.exp(log_mu.ravel()[flat]), alpha=phi[si, ai], observed=y)
        pm.Deterministic("mu7", pt.exp(log_mu[:, :, -K:, :]).reshape((S * A, K, B)))
        pm.Deterministic("phi", phi.ravel())
    return model


def daily_update(fit, d, gap, seed=0):
    """P(final >= tau) on as-of d["date"] from the fit `gap` days earlier, section 6's closed form.

    Delay parameters and growth come from the fit. Each row's total has a Gamma working prior with
    the fit's expected total as mean and shape phi_s; the row's reports so far are Poisson-thinned
    with the delay fraction, so the rest is a Gamma-Poisson draw. Uses nothing after the as-of date.
    """
    rng = np.random.default_rng(seed)
    phi = fit["u_phi"].astype(float)  # n, S
    base, rho, wd = fit["u_base"].astype(float), fit["u_rho_d"].astype(float), fit["u_wd_d"].astype(float)
    lvl, grw = fit["u_lvl_end"].astype(float), fit["u_grw_end"].astype(float)
    n, S = phi.shape
    total = np.zeros((n, S))
    tail = 1 + rng.gamma(d["tail"][:, 1], 1 / d["tail"][:, 2], size=(n, S))
    daily = (np.arange(N_HAZARD) < 15).astype(float)
    for i in range(K):
        age, offset = K - 1 - i, gap - (K - 1) + i
        dow = d["dow"][T - K + i]
        logit = base + rho[:, :, (dow + np.arange(N_HAZARD)) % 7] * daily
        hazard = 1 / (1 + np.exp(-logit))
        fraction = 1 - np.prod(1 - hazard[:, :, : age + 1], axis=2)
        expected = np.exp(lvl + offset * grw + wd[:, :, dow])
        known = d["kept7"][:, i]
        rate = phi / expected + fraction
        lam = rng.gamma(phi + known, 1 / rate)
        rest = rng.poisson(lam * (1 - fraction))
        total += (known + rest) * tail
    incidence = total / d["population"][None] * 1e5
    return np.stack([(incidence >= t).mean(0) for t in TAUS])


def data_update(d):
    """Keyword arguments for nutpie's `with_data`."""
    return {"y": d["y"][:, known_mask()].ravel().astype("int64"), "dow": d["dow"].astype("int64"),
            "rep": d["rep"].astype("int64"), "l0_mu": d["l0_mu"].astype(float)}


FORWARD = ("mu7", "phi", "base", "rho_d", "wd_d", "lvl_end", "grw_end")
FUTURE = BIN_LAST[None, :] > (K - 1 - np.arange(K))[:, None]  # (K, B): not known on the as-of date


def latent_total(d, mu7):
    """(..., S): the gate quantity. `mu7` is (..., S, K, B)."""
    future = (mu7 * FUTURE).sum(axis=-1)
    return ((d["kept7"] + future) * (1 + d["tail"][:, 0])[:, None]).sum(axis=-1)


def noisy_total(d, mu7, phi, seed=0):
    """(draws, S): one draw of the final 7-day count per posterior draw."""
    rng = np.random.default_rng(seed)
    mu7, phi = mu7.reshape(-1, *mu7.shape[-3:]).astype(float), phi.reshape(-1, phi.shape[-1]).astype(float)
    rate = rng.gamma(phi[:, :, None, None] * np.ones_like(mu7), mu7 / phi[:, :, None, None])
    future = (rng.poisson(rate) * FUTURE).sum(axis=-1)
    tail = 1 + rng.gamma(d["tail"][:, 1], 1 / d["tail"][:, 2], size=(mu7.shape[0], mu7.shape[1]))
    return ((d["kept7"][None] + future) * tail[:, :, None]).sum(axis=-1)


def rhat_ess(x):
    """(r_hat, bulk ESS, mcse of the mean, mean) per column of x (chains, draws, S)."""
    import arviz as az
    import xarray as xr

    ds = xr.Dataset({"x": (("chain", "draw", "s"), np.asarray(x, float))})
    rhat = az.rhat(ds, method="rank")["x"].values
    ess = az.ess(ds, method="bulk")["x"].values
    ess_mean = az.ess(ds, method="mean")["x"].values
    return rhat, ess, x.std(axis=(0, 1)) / np.sqrt(ess_mean), x.mean(axis=(0, 1))


def summarise(d, mu7, phi, wall, extra):
    """Diagnostics and decision quantities from draws (chains, draws, S, K, B) and (chains, draws, S)."""
    latent = latent_total(d, mu7)
    noisy = noisy_total(d, mu7, phi)
    population = d["population"]
    worst_stratum = None
    if "group" in d:  # strata: gates and decisions act on all-age state totals
        worst_stratum = float(rhat_ess(latent)[0].max())
        S, A = d["group"]
        latent = latent.reshape(*latent.shape[:-1], S, A).sum(-1) * d["scale"]
        noisy = noisy.reshape(noisy.shape[0], S, A).sum(-1) * d["scale"]
        population = d["population_all"]
    rhat, ess, mcse, mean = rhat_ess(latent)
    incidence = noisy / population[None] * 1e5
    extra = {**extra, "worst_stratum_rhat": worst_stratum}
    out = {"latent_rhat": rhat, "latent_ess": ess, "latent_mcse": mcse, "latent_mean": mean,
           "p_cross": np.stack([(incidence >= t).mean(0) for t in TAUS]),
           "q": np.quantile(incidence, [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], axis=0),
           "keep": incidence[:: max(1, incidence.shape[0] // 500)].astype(np.float32)}
    print(f"  {d['date']}: wall {wall:.1f} s, max r_hat {rhat.max():.3f}, min ESS {ess.min():.0f}, "
          f"ESS/s {ess.min() / wall:.2f}, {extra}", flush=True)
    return out


def fit_cpu(d, model=None, compiled=None, tune=1000, draws=1000, chains=4, seed=1, target=0.8, low_rank=True):
    """nutpie, float64: the fair baseline. Returns the summary and timing."""
    import nutpie

    t0 = time.perf_counter()
    if compiled is None:
        compiled = nutpie.compile_pymc_model(model)
    t1 = time.perf_counter()
    trace = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, cores=chains, seed=seed,
                          progress_bar=False, save_warmup=False, target_accept=target,
                          low_rank_modified_mass_matrix=low_rank, store_mass_matrix=False,
                          store_gradient=False, store_unconstrained=False)
    wall = time.perf_counter() - t0
    mu7, phi = trace.posterior["mu7"].values, trace.posterior["phi"].values
    stats = trace.sample_stats
    n_steps = float(stats["n_steps"].sum())
    div = int(stats["diverging"].sum())
    extra = {"divergences": div, "transitions": chains * draws, "divergence_rate": div / (chains * draws),
             "compile_s": t1 - t0, "wall_s": wall, "mean_n_steps": n_steps / (chains * draws),
             "chains": chains, "tune": tune, "draws": draws}
    return summarise(d, mu7, phi, wall, extra), extra, compiled


# GPU ---------------------------------------------------------------------------------------------


def jax_functions(model, names=None):
    """flat log density, flat q -> (mu7, phi), start point, for a model built in the wanted precision."""
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from pymc.sampling.jax import get_jaxified_graph, get_jaxified_logp

    logp = get_jaxified_logp(model)
    start = model.initial_point()
    value_names = [v.name for v in model.value_vars]
    q0, unravel = ravel_pytree([jnp.asarray(start[n]) for n in value_names])
    outs = model.replace_rvs_by_values([model[n] for n in (names or FORWARD)])
    graph = get_jaxified_graph(inputs=model.value_vars, outputs=outs)
    return (lambda q: logp(unravel(q))), (lambda q: graph(*unravel(q))), q0, value_names


def warm_chees(logdensity, chains, key, start, steps, step_size=0.01, target=0.651):
    import blackjax
    import jax
    import optax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    warmup = blackjax.chees_adaptation(logdensity, chains, mass_matrix_estimation="diagonal",
                                       target_acceptance_rate=target,
                                       adaptation_info_fn=get_filter_adapt_info_fn())
    (state, params), _ = warmup.run(key, start, step_size=step_size, optim=optax.adam(0.05), num_steps=steps)
    return jax.block_until_ready(state), params


def fit_gpu(d, model, chains=32, warmup=1000, draws=1000, thin=1, seed=1, batch=2000, stage1=0.3, target=0.651,
            outputs=FORWARD, keep_named=(), summary=True):
    """ChEES, two-stage warmup with straggler culling (E86). Returns summary, timing, positions.

    `keep_named` adds the full (chains, draws, ...) arrays of those outputs to the summary as n_<name>.
    `summary=False` skips the decision summary (the finished fit has no as-of date to summarise) and
    returns the largest mu7 of each chain instead.
    """
    import blackjax
    import jax
    import jax.numpy as jnp

    from pymc_challenges.bigrun import find_stragglers

    t0 = time.perf_counter()
    logdensity, forward, q0, _ = jax_functions(model, outputs)
    key = jax.random.key(seed)
    key, k0, k1, k2, k3 = jax.random.split(key, 5)
    start = q0 + 0.3 * jax.random.normal(k0, (chains, q0.size), q0.dtype)
    batch_logp = jax.jit(jax.vmap(logdensity))
    rng = np.random.default_rng(seed)

    def cull(state):
        logp = np.asarray(batch_logp(state.position))
        found = find_stragglers(logp, np.asarray(state.position)[:, _hyper_idx(model)])
        stuck = found["stuck"]
        if stuck.any() and (~stuck).any():
            source = np.arange(chains)
            source[stuck] = rng.choice(np.flatnonzero(~stuck), stuck.sum())
            state = jax.tree.map(lambda leaf: leaf[source], state)
        return state, {"stuck": int(stuck.sum()), "check_mode": int(found["check_mode"].sum()),
                       "worst_z": float(found["worst_z"]) if np.isfinite(found["worst_z"]) else 1e9}

    first = int(stage1 * warmup)
    t1 = time.perf_counter()
    state, params = warm_chees(logdensity, chains, k1, start, first, target=target)
    t2 = time.perf_counter()
    state, cull1 = cull(state)
    state, params = warm_chees(logdensity, chains, k2, state.position, warmup - first,
                               step_size=float(params["step_size"]), target=target)
    t3 = time.perf_counter()
    state, cull2 = cull(state)
    kernel = blackjax.dynamic_hmc(logdensity, **params)

    def one(s, ki):
        s, info = jax.vmap(kernel.step)(jax.random.split(ki, chains), s)
        return s, (info.acceptance_rate.mean(), info.is_divergent.sum(), info.num_integration_steps.mean())

    def keep(s, ki):
        s, (acc, div, steps) = jax.lax.scan(one, s, jax.random.split(ki, thin))
        return s, (s.position, acc.mean(), div.sum(), steps.sum())

    run = jax.jit(lambda s, ki, n: jax.lax.scan(keep, s, jax.random.split(ki, n)), static_argnums=2)
    kept, acc, div, steps, n_done = [], [], 0, 0.0, 0
    while n_done < draws:
        n = min(batch, draws - n_done)
        k3, ki = jax.random.split(k3)
        state, (q, a, dv, st) = run(state, ki, n)
        kept.append(np.asarray(jax.block_until_ready(q)))
        acc.append(np.asarray(a))
        div += int(np.asarray(dv).sum())
        steps += float(np.asarray(st).sum())
        n_done += n
    q = np.concatenate(kept).transpose(1, 0, 2)  # chains, draws, dim
    t4 = time.perf_counter()
    forward_batch = jax.jit(jax.vmap(forward))
    flat = q.reshape(-1, q.shape[-1])
    parts = []
    for i in range(0, len(flat), 2048):
        parts.append([np.asarray(x) for x in forward_batch(jnp.asarray(flat[i : i + 2048]))])
    named = {n: np.concatenate([p[j] for p in parts]) for j, n in enumerate(outputs)}
    mu7 = named["mu7"].reshape(chains, draws, *named["mu7"].shape[1:])
    phi = named["phi"].reshape(chains, draws, -1)
    step = max(1, len(flat) // 2000)
    update_draws = {f"u_{n}": named[n][::step] for n in outputs[1:]}
    update_draws.update({f"n_{n}": named[n].reshape(chains, draws, *named[n].shape[1:]) for n in keep_named})
    wall = time.perf_counter() - t0
    transitions = chains * draws * thin
    extra = {"divergences": div, "transitions": transitions, "divergence_rate": div / transitions,
             "wall_s": wall, "sampling_s": t4 - t3, "warmup_stage1_s": t2 - t1, "warmup_stage2_s": t3 - t2,
             "setup_s": t1 - t0, "target": target, "chains": chains, "warmup": warmup, "draws": draws, "thin": thin,
             "acceptance": float(np.mean(acc)), "step_size": float(params["step_size"]),
             "L_max": float(params["integration_steps_params"][0]), "gradients_per_chain": steps,
             "cull1": cull1, "cull2": cull2, "dtype": str(q0.dtype), "dim": int(q0.size)}
    if not summary:
        return {"mu7_max_by_chain": mu7.reshape(chains, -1).max(axis=1), **update_draws}, extra, q
    return {**summarise(d, mu7, phi, wall, extra), **update_draws}, extra, q


def _hyper_idx(model):
    """Positions of the scale parameters in the flat vector, for the straggler check."""
    index, out = 0, []
    start = model.initial_point()
    for v in model.value_vars:
        size = int(np.size(start[v.name]))
        if v.name.endswith("_log__"):
            out.extend(range(index, index + size))
        index += size
    return np.array(out)
