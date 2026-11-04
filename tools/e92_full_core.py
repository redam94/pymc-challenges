"""E92 FULL helpers: data sets, a checkpointed L-BFGS, exact prediction, Vecchia, design and scores.

Lines are in results/E92/gates_2026-11-03c.json. JAX is imported inside functions, after the caller set
JAX_PLATFORMS and x64 (pilot_e92.jax_setup).
"""

from __future__ import annotations

import functools
import time

import numpy as np
from scipy.spatial import cKDTree

import e92_core as c

DAY_2025 = 731.0  # 2025-01-01 in days from 2023-01-01
DAY_2026 = 1096.0
NU_T = 4.0  # Student-t nugget degrees of freedom, fixed in the gates
GYRE = dict(lat_min=20.0, lat_max=30.0, lon_min=-70.0, lon_max=-20.0)


def in_box(lat, lon, b):
    lat, lon = np.asarray(lat), np.asarray(lon)
    return (lat >= b["lat_min"]) & (lat < b["lat_max"]) & (lon >= b["lon_min"]) & (lon < b["lon_max"])


def region(lat, lon):
    """0 Gulf Stream box, 1 gyre, 2 rest."""
    r = np.full(np.shape(lat), 2)
    r[in_box(lat, lon, GYRE)] = 1
    r[in_box(lat, lon, c.GULF_STREAM)] = 0
    return r


def datasets(proj="mercator"):
    """Training, held-out (2023-2024) and 2025 frames, with Mercator coordinates in X."""
    d = c.load((2023, 2024, 2025))
    d["year"] = d["time"].str[:4].astype(int)
    d["held"] = c.holdout(d["float_id"])
    d[["x", "y", "t"]] = c.coords(d["lat"], d["lon"], d["t_days"], proj)
    fit = d[d["year"] <= 2024]
    train = fit[~fit["held"]].reset_index(drop=True)
    test = fit[fit["held"]].reset_index(drop=True)
    y25 = d[d["year"] == 2025].reset_index(drop=True)
    first = d.groupby("float_id")["year"].min()
    mincyc = y25.groupby("float_id")["cycle"].min()
    launched = set(first[first == 2025].index) & set(mincyc[mincyc <= 1].index)
    y25["launched_2025"] = y25["float_id"].isin(launched)
    return train, test, y25


def X_of(frame, extra=()):
    cols = [frame[["x", "y", "t"]].to_numpy(float)]
    cols += [np.asarray(e, float)[:, None] for e in extra]
    return np.concatenate(cols, axis=1)


def edges(theta):
    lo, hi = c.PRIOR_MU - 2.326 * c.PRIOR_SD, c.PRIOR_MU + 2.326 * c.PRIOR_SD
    return [c.NAMES[j] for j in range(10) if not lo[j] < theta[j] < hi[j]]


def natural(theta):
    return np.r_[np.exp(np.asarray(theta)[:9]), np.asarray(theta)[9]]


# Checkpointed L-BFGS on F = -log posterior ---------------------------------------------------------


def _direction(g, S, Y):
    q = g.copy()
    al = []
    for s, y in zip(S[::-1], Y[::-1]):
        r = 1.0 / (y @ s)
        a = r * (s @ q)
        q -= a * y
        al.append((r, a, s, y))
    if S:
        q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
    for r, a, s, y in al[::-1]:
        b = r * (y @ q)
        q += s * (a - b)
    return -q


def lbfgs(fun, x0, store, name, max_iter, gtol, max_eval, ftol=1e-8, ftol_iters=3, mem=10, log=print, health=None,
          init_SY=None):
    """Minimise fun (x -> F, gradient) with every accepted iteration checkpointed as a chunk of ``name``.

    A restart resumes from the newest intact chunk: x, F, gradient, the (s, y) memory and the counters,
    so a crash loses only the line search in flight. Returns the final state dict.
    """
    idx, st = store.resume(name) if store is not None else (None, None)
    if st is not None and int(st["finished"]):
        log(f"{name}: already finished")
        return st
    if st is None:
        x = np.asarray(x0, float)
        f, g = fun(x)
        S, Y, it, nev, small = [], [], 0, 1, 0
        if init_SY is not None:  # curvature pairs carried over from a cheaper phase (last 5, s.y > 0 only)
            pairs = [(s, y) for s, y in zip(*init_SY) if s @ y > 1e-10][-5:]
            S, Y = [s for s, _ in pairs], [y for _, y in pairs]
        hist = [np.r_[x, f, g]]
        idx = 0
    else:
        x, f, g = st["x"], float(st["f"]), st["g"]
        S, Y = list(st["S"]), list(st["Y"])
        it, nev, small = int(st["it"]), int(st["nev"]), int(st["small"])
        hist = list(st["hist"])
        log(f"{name}: resumed at iteration {it}, {nev} evaluations, F {f:.3f}")
    msg = ""
    t0 = time.perf_counter()
    while True:
        if np.max(np.abs(g)) < gtol:
            msg = f"max|g| {np.max(np.abs(g)):.3g} < {gtol}"
            break
        if it >= max_iter or nev >= max_eval:
            msg = f"limit: {it} iterations, {nev} evaluations"
            break
        if small >= ftol_iters:
            msg = f"relative change < {ftol} for {ftol_iters} iterations"
            break
        if health:
            health()
        d = _direction(g, S, Y)
        if g @ d >= 0:  # not a descent direction: drop the memory
            S, Y = [], []
            d = -g
        if not S:
            d = d * min(1.0, 0.2 / np.max(np.abs(d)))
        alpha, ok = 1.0, False
        for _ in range(20):
            xn = x + alpha * d
            fn, gn = fun(xn)
            nev += 1
            hist.append(np.r_[xn, fn, gn])
            if np.isfinite(fn) and fn <= f + 1e-4 * alpha * (g @ d):
                ok = True
                break
            alpha *= 0.3 if np.isfinite(fn) else 0.1
            if nev >= max_eval:
                break
        if not ok:
            msg = "line search failed"
            break
        s, yv = xn - x, gn - g
        if s @ yv > 1e-10:
            S.append(s)
            Y.append(yv)
            S, Y = S[-mem:], Y[-mem:]
        small = small + 1 if abs(fn - f) <= ftol * max(1.0, abs(f)) else 0
        x, f, g = xn, fn, gn
        it += 1
        idx += 1
        if store is not None:
            store.save_chunk(name, idx, _state(x, f, g, S, Y, it, nev, small, hist, 0, ""), keep=3)
        log(f"{name} it {it} nev {nev} F {f:.4f} max|g| {np.max(np.abs(g)):.3g} alpha {alpha:.3g} "
            f"{time.perf_counter() - t0:.0f}s")
    idx += 1
    st = _state(x, f, g, S, Y, it, nev, small, hist, 1, msg)
    if store is not None:
        store.save_chunk(name, idx, st, keep=3)
    log(f"{name}: finished, {msg}")
    return st


def _state(x, f, g, S, Y, it, nev, small, hist, finished, msg):
    return {"x": np.asarray(x), "f": np.array(f), "g": np.asarray(g),
            "S": np.asarray(S).reshape(-1, len(x)), "Y": np.asarray(Y).reshape(-1, len(x)),
            "it": np.array(it), "nev": np.array(nev), "small": np.array(small), "hist": np.asarray(hist),
            "finished": np.array(finished), "msg": np.frombuffer(msg.encode().ljust(80)[:80], np.uint8)}


def msg_of(st):
    return bytes(np.asarray(st["msg"], np.uint8)).decode().strip()


def neg(vg, X, y, dtype):
    """F(x) = -log posterior and its gradient, as float64 NumPy, from a JAX value-and-gradient."""
    import jax.numpy as jnp

    Xd, yd = jnp.asarray(X, dtype), jnp.asarray(y, dtype)

    def f(th):
        v, g = vg(jnp.asarray(th), Xd, yd)
        v, g = float(v), np.asarray(g, np.float64)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return np.inf, np.zeros_like(g)
        return -v, -g

    return f


# Exact prediction -----------------------------------------------------------------------------------


@functools.lru_cache
def make_factor(nu, block=1024):
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    # Three programs, with K donated to the Cholesky: one program holding K, its symmetrised copy and L
    # asked XLA for 17 GiB at n = 23,939 in float64 (measured), over the 16.56 GiB pool; this peaks at 12.9.
    bk = jax.jit(lambda th, X: c.build_K(X, th, nu, block))
    ch = jax.jit(lambda K: jax.lax.linalg.cholesky(K, symmetrize_input=False), donate_argnums=0)
    sv = jax.jit(lambda L, y: cho_solve((L, True), y))

    def fac(theta, X, y):
        L = ch(bk(theta, X))
        return L, sv(L, y)

    return fac


@functools.lru_cache
def make_cross(nu):
    """V = L^-1 K(X, Xs) and mean K(Xs, X) a, latent variance diag K(Xs, Xs) - colsum V^2, for a block."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    def cross(theta, L, a, X, Xs):
        n, m = X.shape[0], Xs.shape[0]
        Ks = c.kernel_block(Xs, X, jnp.arange(m) + n, jnp.arange(n), theta, nu)
        V = solve_triangular(L, Ks.T, lower=True)
        kss = c.kernel_block(Xs, Xs, jnp.arange(m) + n, jnp.arange(m) + n, theta, nu, sigma_noise=False)
        return Ks @ a, V, kss

    return jax.jit(cross)


def predict(nu, theta, L, a, X, Xs, groups=None, block=2048):
    """Latent mean and variance at Xs, and per-group latent covariance blocks if ``groups`` is given."""
    import jax.numpy as jnp

    cross = make_cross(nu)
    th = jnp.asarray(theta)
    mean, var, covs = np.zeros(len(Xs)), np.zeros(len(Xs)), {}
    if groups is None:
        for s in range(0, len(Xs), block):
            mu, V, kss = cross(th, L, a, X, jnp.asarray(Xs[s:s + block], X.dtype))
            mean[s:s + block] = np.asarray(mu, np.float64)
            var[s:s + block] = np.asarray(jnp.diag(kss) - jnp.sum(V * V, 0), np.float64)
        return mean, var, covs
    for gid in np.unique(groups):
        m = np.flatnonzero(groups == gid)
        mu, V, kss = cross(th, L, a, X, jnp.asarray(Xs[m], X.dtype))
        cov = np.asarray(kss - V.T @ V, np.float64)
        mean[m], var[m], covs[gid] = np.asarray(mu, np.float64), np.diag(cov), cov
    return mean, var, covs


def joint_scores(mean, covs, noise_var, y, groups):
    """Joint Gaussian log density of each group's observations: latent covariance plus nugget."""
    out = {}
    for gid, cov in covs.items():
        m = np.flatnonzero(groups == gid)
        nv = noise_var[m] if np.ndim(noise_var) else np.full(len(m), noise_var)
        ch = np.linalg.cholesky(cov + np.diag(nv))
        z = np.linalg.solve(ch, y[m] - mean[m])
        out[gid] = float(-0.5 * z @ z - np.log(np.diag(ch)).sum() - 0.5 * len(m) * np.log(2 * np.pi))
    return out


@functools.lru_cache
def make_resid(nu, block=1024):
    """a = K^-1 y and diag(K^-1), float64, for the scale-mixture weight update."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    def resid(theta, X, y):
        K = c.build_K(X, theta, nu, block)
        L = jnp.linalg.cholesky(K)
        a = cho_solve((L, True), y)
        Li = solve_triangular(L, jnp.eye(X.shape[0], dtype=X.dtype), lower=True)
        return a, jnp.sum(Li * Li, 0)

    return jax.jit(resid)


def t_weights(theta, a, kinv_diag, mult, nu_t=NU_T):
    """New 1/w from the exact posterior of each noise term e_i given y (current multipliers ``mult``)."""
    sn2 = np.exp(2 * theta[8])
    s2 = sn2 * mult
    e_mean = s2 * a
    e_var = np.maximum(s2 - s2 * s2 * kinv_diag, 1e-12)
    w = (nu_t + 1) / (nu_t + (e_mean**2 + e_var) / sn2)
    return 1.0 / w


# Amplitude map from B's windows ------------------------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * c.EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def log_amp(lat, lon, centres, log_sd, norm=0.0, bw=500.0):
    """500 km Gaussian-weighted mean of the windows' log signal sd, less ``norm``."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    out = np.empty(len(lat))
    for s in range(0, len(lat), 4096):
        d = haversine_km(lat[s:s + 4096, None], lon[s:s + 4096, None], centres[None, :, 0], centres[None, :, 1])
        w = np.exp(-0.5 * (d / bw) ** 2) + 1e-300
        out[s:s + 4096] = (w * log_sd[None]).sum(1) / w.sum(1)
    return out - norm


# Vecchia ---------------------------------------------------------------------------------------------


def scaled(X, theta):
    e = np.exp(np.asarray(theta)[:9])
    return np.c_[X[:, 0] / e[1], X[:, 1] / e[2], X[:, 2] / e[3]]


def maxmin_order(Z, seed=0):
    """Exact maxmin ordering (start nearest the centroid)."""
    n = len(Z)
    first = int(np.argmin(((Z - Z.mean(0)) ** 2).sum(1)))
    order = np.empty(n, int)
    order[0] = first
    dmin = ((Z - Z[first]) ** 2).sum(1)
    dmin[first] = -1
    for k in range(1, n):
        j = int(np.argmax(dmin))
        order[k] = j
        dmin = np.minimum(dmin, ((Z - Z[j]) ** 2).sum(1))  # chosen points stay at -1
        dmin[j] = -1
    return order


def neighbours(Zo, m, brute=1500, ratio=1.05):
    """For points already in maxmin order: up to m nearest earlier points (-1 padded).

    Exact for the first ``brute`` points; after that, from a tree of the points before each geometric
    block, so a few same-block earlier points are skipped: the conditioning sets stay earlier points, so
    the Vecchia density is still a valid density.
    """
    n = len(Zo)
    nb = -np.ones((n, m), int)
    for i in range(1, min(brute, n)):
        d = ((Zo[:i] - Zo[i]) ** 2).sum(1)
        k = min(m, i)
        nb[i, :k] = np.argsort(d)[:k]
    s = brute
    while s < n:
        e = min(n, max(s + 1, int(s * ratio)))
        tree = cKDTree(Zo[:s])
        _, j = tree.query(Zo[s:e], k=m)
        nb[s:e] = j
        s = e
    return nb


@functools.lru_cache
def make_vecchia(nu):
    import jax
    import jax.numpy as jnp

    def one(theta, xi, i, yi, Xn, nb, yn, valid):
        Knn = c.kernel_block(Xn, Xn, nb, nb, theta, nu)
        eye = jnp.eye(Xn.shape[0], dtype=Xn.dtype)
        vv = valid[:, None] & valid[None, :]
        Knn = jnp.where(vv, Knn, eye)
        kin = jnp.where(valid, c.kernel_block(xi[None], Xn, i[None], nb, theta, nu)[0], 0.0)
        kii = c.kernel_block(xi[None], xi[None], i[None], i[None], theta, nu)[0, 0]
        Lc = jnp.linalg.cholesky(Knn)
        b = jax.scipy.linalg.solve_triangular(Lc, kin, lower=True)
        r = jax.scipy.linalg.solve_triangular(Lc, jnp.where(valid, yn, 0.0), lower=True)
        mu, var = b @ r, kii - b @ b
        return -0.5 * (yi - mu) ** 2 / var - 0.5 * jnp.log(var) - 0.5 * np.log(2 * np.pi)

    def logp(theta, X, y, nb):
        valid = nb >= 0
        nbc = jnp.where(valid, nb, 0)
        ll = jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
            theta, X, jnp.arange(X.shape[0]), y, X[nbc], nbc, y[nbc], valid)
        return jnp.sum(ll)

    lp = jax.jit(logp)

    def vg(theta, X, y, nb):
        return jax.value_and_grad(lambda th: logp(th, X, y, nb) + c.prior_logp(th))(theta)

    return lp, jax.jit(vg)


# Scores ----------------------------------------------------------------------------------------------


def crps_gauss(y, mu, sd):
    from scipy.stats import norm

    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_mc(y, draws):
    """CRPS from draws (profiles x draws): E|X - y| - E|X - X'| / 2, the second by the sorted-sample form."""
    s = np.sort(draws, axis=1)
    k = s.shape[1]
    term1 = np.abs(s - y[:, None]).mean(1)
    w = (2 * np.arange(1, k + 1) - k - 1) / (k * k)
    return term1 - (s * w).sum(1)


def t_predictive_draws(mu, var, sn, k=4000, seed=92, nu_t=NU_T):
    rng = np.random.default_rng(seed)
    return (mu[:, None] + np.sqrt(var)[:, None] * rng.standard_normal((len(mu), k))
            + sn * rng.standard_t(nu_t, size=(len(mu), k)))
