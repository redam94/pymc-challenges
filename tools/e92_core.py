"""E92 core: data, coordinates, and the exact GP's log density with a hand-coded blocked gradient.

Kernel (plans/large/E92.md, "The model"):

    k = s1^2 M(r1) + s2^2 M(r2) + sn^2 delta + H B H^T
    r1^2 = ((dx - c dt)/lx1)^2 + (dy/ly1)^2 + (dt/lt1)^2,   r2 the same with lx2, ly2, lt2 and no c

x, y in km, t in days from 2023-01-01, c in km/day (positive: features move east). H = [1, t in years]
with B = diag(1 C, 1 C/yr)^2 is the marginalised intercept and trend. theta holds 9 log values and c:
[log s1, log lx1, log ly1, log lt1, log s2, log lx2, log ly2, log lt2, log sn, c].

Memory rules: K is built in row blocks with in-place updates (never X[:, None] - X[None]); the gradient
is 0.5 tr((a a^T - K^-1) dK) summed over row blocks, with dK of one block by autodiff and never held
whole. The log determinant is summed in float64 whatever the factor's precision.

JAX is imported by the functions that need it, after the caller has set JAX_PLATFORMS and x64.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

NAMES = ["log_s1", "log_lx1", "log_ly1", "log_lt1", "log_s2", "log_lx2", "log_ly2", "log_lt2",
         "log_sn", "c"]
# Prior: LogNormal(median, sd) for the 9 scales, Normal(0, 5 km/day) for c
PRIOR_MEDIAN = np.array([1.0, 300.0, 200.0, 15.0, 0.5, 1500.0, 1000.0, 200.0, 0.3, 0.0])
PRIOR_SD = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 5.0])
PRIOR_MU = np.r_[np.log(PRIOR_MEDIAN[:9]), 0.0]
THETA0 = PRIOR_MU.copy()  # the plan's illustrative scales: 300 km, 200 km, 15 d; 1,500 km, 1,000 km, 200 d
EARTH_KM = 6371.0
LAT0 = 37.5
BETA_SD = np.array([1.0, 1.0])  # intercept (C) and trend (C a year)
T0 = pd.Timestamp("2023-01-01", tz="UTC")
SMALL_CROP = dict(lat_min=25.0, lat_max=45.0, lon_min=-75.0, lon_max=-40.0)
GULF_STREAM = dict(lat_min=30.0, lat_max=45.0, lon_min=-75.0, lon_max=-45.0)


def load(years=(2023, 2024), crop=None) -> pd.DataFrame:
    from pymc_challenges import data

    d = data.load("argo_natl_t100")
    d = d[d["time"].str[:4].astype(int).isin(years)].copy()
    if crop:
        d = d[(d["lat"] >= crop["lat_min"]) & (d["lat"] < crop["lat_max"])
              & (d["lon"] >= crop["lon_min"]) & (d["lon"] < crop["lon_max"])]
    d["t_days"] = (pd.to_datetime(d["time"], utc=True) - T0).dt.total_seconds().to_numpy() / 86400
    return d.reset_index(drop=True)


def holdout(float_ids, share=0.10) -> np.ndarray:
    """True for held-out floats: a fixed hash of the float id, so the split never depends on order."""
    h = np.array([int(hashlib.sha256(f"e92:{f}".encode()).hexdigest()[:8], 16) / 2**32 for f in float_ids])
    return h < share


def coords(lat, lon, t_days, proj="equirect") -> np.ndarray:
    """(n, 3) of x km, y km, t days. Equirectangular about 37.5N, or Mercator (equatorial km)."""
    lam, phi = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
    if proj == "equirect":
        x, y = EARTH_KM * np.cos(np.radians(LAT0)) * lam, EARTH_KM * phi
    elif proj == "mercator":
        x, y = EARTH_KM * lam, EARTH_KM * np.log(np.tan(np.pi / 4 + phi / 2))
    else:
        raise ValueError(proj)
    return np.c_[x, y, np.asarray(t_days, float)]


def prior_logp(theta):
    import jax.numpy as jnp

    z = (theta - PRIOR_MU) / PRIOR_SD
    return -0.5 * jnp.sum(z * z) - jnp.sum(jnp.log(PRIOR_SD)) - 5.0 * np.log(2 * np.pi)


def prior_draws(n, seed=0) -> np.ndarray:
    return PRIOR_MU + PRIOR_SD * np.random.default_rng(seed).normal(size=(n, len(NAMES)))


def _matern(r2, nu):
    import jax.numpy as jnp

    pos = r2 > 0
    r = jnp.where(pos, jnp.sqrt(jnp.where(pos, r2, 1.0)), 0.0)
    if nu == 0.5:
        return jnp.exp(-r)
    if nu == 1.5:
        s = float(np.sqrt(3.0)) * r
        return (1 + s) * jnp.exp(-s)
    if nu == "inf":  # ExpQuad: only for the float32 failure demonstration
        return jnp.exp(-0.5 * r2)
    raise ValueError(nu)


def kernel_block(Xa, Xb, ia, ib, theta, nu, sigma_noise=True):
    """k(Xa, Xb) with the nugget where global indices ia == ib, plus the intercept-and-trend term.

    Columns 0-2 of X are x km, y km, t days. Optional column 3 is a log amplitude D (FULL's A', K = D R D
    + noise) and optional column 4 a nugget variance multiplier (1/w, the Student-t scale mixture).
    With three columns this is the pilots' kernel, unchanged.
    """
    import jax.numpy as jnp

    dt_ = Xa.dtype
    e = jnp.exp(theta[:9]).astype(dt_)
    c = theta[9].astype(dt_)
    dx = Xa[:, None, 0] - Xb[None, :, 0]
    dy = Xa[:, None, 1] - Xb[None, :, 1]
    dt = Xa[:, None, 2] - Xb[None, :, 2]
    r1 = ((dx - c * dt) / e[1]) ** 2 + (dy / e[2]) ** 2 + (dt / e[3]) ** 2
    r2 = (dx / e[5]) ** 2 + (dy / e[6]) ** 2 + (dt / e[7]) ** 2
    k = e[0] ** 2 * _matern(r1, nu) + e[4] ** 2 * _matern(r2, nu)
    if Xa.shape[1] >= 4:
        k = k * jnp.exp(Xa[:, None, 3] + Xb[None, :, 3])
    if sigma_noise:
        nug = e[8] ** 2 * (Xa[:, 4][:, None] if Xa.shape[1] >= 5 else 1.0)
        k = k + jnp.where(ia[:, None] == ib[None, :], nug, 0.0).astype(dt_)
    ta, tb = Xa[:, 2] / 365.25, Xb[:, 2] / 365.25
    k = k + (float(BETA_SD[0] ** 2) + float(BETA_SD[1] ** 2) * ta[:, None] * tb[None, :]).astype(dt_)
    return k


def build_K(X, theta, nu, block=1024):
    """Dense K in X's dtype, filled in row blocks in place. The last block overlaps; it rewrites the
    same values."""
    import jax
    import jax.numpy as jnp

    n = X.shape[0]
    block = min(block, n)
    nb = -(-n // block)
    idx = jnp.arange(n)

    def body(b, K):
        s = jnp.minimum(b * block, n - block)
        Xa = jax.lax.dynamic_slice_in_dim(X, s, block)
        ia = jax.lax.dynamic_slice_in_dim(idx, s, block)
        return jax.lax.dynamic_update_slice_in_dim(K, kernel_block(Xa, X, ia, idx, theta, nu), s, 0)

    return jax.lax.fori_loop(0, nb, body, jnp.zeros((n, n), X.dtype))


def _factor(X, y, theta, nu, block):
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    K = build_K(X, theta, nu, block)
    L = jnp.linalg.cholesky(K)
    a = cho_solve((L, True), y)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).astype(jnp.float64)))
    quad = jnp.dot(y.astype(jnp.float64), a.astype(jnp.float64))
    loglik = -0.5 * quad - 0.5 * logdet - 0.5 * X.shape[0] * np.log(2 * np.pi)
    return L, a, loglik


def make_logp(nu, block=1024, with_prior=True):
    """theta, X, y -> log density (float64 scalar). Precision is X's dtype."""
    import jax

    def logp(theta, X, y):
        _, _, ll = _factor(X, y, theta, nu, block)
        return ll + (prior_logp(theta) if with_prior else 0.0)

    return jax.jit(logp)


def make_value_and_grad(nu, block=1024, with_prior=True, kinv="full"):
    """theta, X, y -> (log density, gradient), both float64; the gradient by the blocked trace."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    def vg(theta, X, y):
        n = X.shape[0]
        b_ = min(block, n)
        nb = -(-n // b_)
        L, a, ll = _factor(X, y, theta, nu, b_)
        Kinv = cho_solve((L, True), jnp.eye(n, dtype=X.dtype))
        idx = jnp.arange(n)

        def body(b, acc):
            s = jnp.minimum(b * b_, n - b_)
            Xa = jax.lax.dynamic_slice_in_dim(X, s, b_)
            ia = jax.lax.dynamic_slice_in_dim(idx, s, b_)
            aa = jax.lax.dynamic_slice_in_dim(a, s, b_)
            Wr = aa[:, None] * a[None, :] - jax.lax.dynamic_slice_in_dim(Kinv, s, b_, 0)
            Wr = Wr * (ia >= b * b_)[:, None].astype(X.dtype)  # the overlapping last block counts once

            def f(th):
                return jnp.sum(Wr * kernel_block(Xa, X, ia, idx, th, nu))

            return acc + jax.grad(f)(theta.astype(X.dtype)).astype(jnp.float64)

        g = 0.5 * jax.lax.fori_loop(0, nb, body, jnp.zeros(theta.shape, jnp.float64))
        if with_prior:
            lp, gp = jax.value_and_grad(prior_logp)(theta)
            return ll + lp, g + gp
        return ll, g

    return jax.jit(vg)


def make_predict(nu, block=1024):
    """Predictive mean and covariance of y at Xs given (X, y): for held-out floats' joint scores."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    def pred(theta, X, y, Xs):
        n, m = X.shape[0], Xs.shape[0]
        L, a, _ = _factor(X, y, theta, nu, block)
        idx, js = jnp.arange(n), jnp.arange(m) + n
        Ks = kernel_block(Xs, X, js, idx, theta, nu)  # (m, n); m is small
        Kss = kernel_block(Xs, Xs, js, js, theta, nu)
        mean = Ks @ a
        V = solve_triangular(L, Ks.T, lower=True)
        return mean, Kss - V.T @ V

    return jax.jit(pred)


def float_scores(pred, theta, X, y, Xs, ys, groups) -> np.ndarray:
    """Joint Gaussian predictive log density of each held-out group (float) given the training data."""
    out = []
    for g in np.unique(groups):
        m = groups == g
        mean, cov = (np.asarray(v, np.float64) for v in pred(theta, X, y, Xs[m]))
        r = ys[m] - mean
        c = np.linalg.cholesky(cov + 1e-9 * np.eye(len(r)))
        z = np.linalg.solve(c, r)
        out.append(-0.5 * z @ z - np.sum(np.log(np.diag(c))) - 0.5 * len(r) * np.log(2 * np.pi))
    return np.array(out)
