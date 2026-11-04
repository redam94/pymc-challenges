"""E92 FULL on the GPU: stages in plans/large/E92.md, lines in results/E92/gates_2026-11-03c.json.

    tools/e92_job.sh 16 8 -- .venv/bin/python tools/run_e92_full.py next        the next stage, or "none"
    tools/e92_job.sh 16 8 -- .venv/bin/python tools/run_e92_full.py STAGE [LIMIT]

tools/e92_full_driver.sh runs them in order, a fresh process each (or each LIMIT chunks), under the
memory guard, with the machine checked before every process and every chunk. Every long loop is
checkpointed by Store, so a crash loses at most one chunk (one L-BFGS iteration, one gradient, one
window, one design draw). GPU processes take bigrun.gpu_lock and run with a preallocated pool
(XLA_PYTHON_CLIENT_PREALLOCATE=true: the growing allocator ran out of memory at n = 20k in f64, P3).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e92_core as c  # noqa: E402
import e92_full_core as fc  # noqa: E402
import pilot_e92 as p  # noqa: E402

from pymc_challenges.bigrun import gpu_lock, sha256_file, vram_high_water  # noqa: E402

store = p.store
GATES = store.load_json("gates_2026-11-03c")  # refuses if altered
NU = 0.5  # P6
H_STEP = 1e-4
N_DRAWS = 11  # draw 0 = the mode, then 10
N_REP = 4
N_SITES, N_PICK, N_CYC = 200, 20, 36
FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gpu():
    jax = p.jax_setup("gpu")
    return jax


def cleanup():
    import jax

    jax.clear_caches()
    gc.collect()


def peak():
    try:
        return vram_high_water()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# Data and models ---------------------------------------------------------------------------------------

_DATA = {}


def data():
    if not _DATA:
        tr, te, y25 = fc.datasets("mercator")
        _DATA.update(train=tr, test=te, y25=y25)
    return _DATA["train"], _DATA["test"], _DATA["y25"]


def ocean_fn():
    import build_e92_data as b

    from pymc_challenges import data as reg

    with np.load(reg.path("natural_earth_50m_land_natl"), allow_pickle=False) as z:
        rings = [z[k] for k in z.files]
    with np.load(reg.path("argo_natl_t100_climatology"), allow_pickle=False) as z:
        coef, clat, clon = z["coef"], z["cell_lat"], z["cell_lon"]

    def ocean(lat, lon):
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        i = np.clip(np.round(lat - clat[0]).astype(int), 0, len(clat) - 1)
        j = np.clip(np.round(lon - clon[0]).astype(int), 0, len(clon) - 1)
        ok = np.isfinite(coef[i, j, 0]) & (lat > 20.0) & (lat < 55.0) & (lon > -80.0) & (lon < 0.0)
        if ok.any():
            ok[ok] = b.open_atlantic(lat[ok], lon[ok], rings)
        return ok

    return ocean


def mode_of(name):
    st = store.resume(name)[1]
    if st is None or not int(st["finished"]):
        raise SystemExit(f"{name} has not finished")
    return np.asarray(st["x"]), st


def is_t(model):
    return model.endswith("t")


def amp_info():
    return store.load_json("amp_map") if (store.path / "amp_map.json").exists() else None


def logD(lat, lon):
    a = amp_info()
    return fc.log_amp(lat, lon, np.asarray(a["centres"]), np.asarray(a["log_sd"]), a["norm"])


def model_spec(model):
    """theta, training X (with the model's extra columns), y, and a builder of X for new points."""
    tr, _, _ = data()
    y = tr["anom"].to_numpy()
    name = {"A": "fit_A_f64", "At": "fit_At_final", "Afix": "fit_Afix_f64", "Afixt": "fit_Afixt_final"}[model]
    theta, _ = mode_of(name)
    amp = model.startswith("Afix")
    cols = []
    if amp or is_t(model):
        cols.append(logD(tr["lat"], tr["lon"]) if amp else np.zeros(len(tr)))
    if is_t(model):
        cols.append(store.load_array(f"{name}_mult"))
    X = fc.X_of(tr, cols)

    def new(frame_or_lat, lon=None, xyz=None):
        if xyz is None:
            f = frame_or_lat
            xyz, lat, lon_ = f[["x", "y", "t"]].to_numpy(float), f["lat"].to_numpy(), f["lon"].to_numpy()
        else:
            lat, lon_ = frame_or_lat, lon
        extra = []
        if amp or is_t(model):
            extra.append(logD(lat, lon_) if amp else np.zeros(len(xyz)))
        if is_t(model):
            extra.append(np.full(len(xyz), fc.NU_T / (fc.NU_T - 2)))
        return np.concatenate([xyz, *[e[:, None] for e in extra]], axis=1) if extra else xyz

    noise_new = np.exp(2 * theta[8]) * (fc.NU_T / (fc.NU_T - 2) if is_t(model) else 1.0)
    return theta, X, y, new, noise_new


def primary():
    """A-t replaces A only if A sat at an edge and A-t did not (the gates' edge action)."""
    if store.done("fit_At_final"):
        info = next(s for s in store.stages() if s["stage"] == "fit_At_final")
        if not info.get("edges"):
            return "At"
    return "A"


# Stages -------------------------------------------------------------------------------------------------


def data_fingerprint():
    from pymc_challenges import data as reg
    from pymc_challenges.bigrun import run_level

    env = {**os.environ, "PYMC_CHALLENGES_RUN": "full", "PYMC_CHALLENGES_CONFIRM_FULL": "yes"}
    level = run_level(needs=dict(ram_gb=12, vram_gb=14, disk_gb=2), env=env)
    path = reg.path("argo_natl_t100")
    digest = sha256_file(path)
    pilot = store.load_json("pilot_data")
    expect = next((v for k, v in _walk(pilot) if "sha256" in k.lower() and isinstance(v, str)
                   and v.startswith("bc11b8ff")), None)
    tr, te, y25 = data()
    out = {"table": str(path.name), "sha256": digest, "matches_pilot": digest == expect, "pull_date": "2026-11-02",
           "n_train": len(tr), "n_test": len(te), "test_floats": int(te["float_id"].nunique()),
           "train_floats": int(tr["float_id"].nunique()), "n_2025": len(y25),
           "n_2025_heldout": int(y25["held"].sum()), "launched_2025_floats": int(y25.loc[y25["launched_2025"],
                                                                                         "float_id"].nunique()),
           "launched_2025_profiles_nonheld": int((y25["launched_2025"] & ~y25["held"]).sum()),
           "run_level": level, "health": p.health(), "labels": "measured"}
    if not out["matches_pilot"]:
        raise SystemExit(f"STOP: table SHA-256 {digest} is not the pilot's")
    store.save_json("data_fingerprint", out)
    store.mark_done("data_fingerprint", n_train=len(tr))
    log(out)


def _walk(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}.{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def p8():
    gpu()
    import jax.numpy as jnp

    t0 = time.perf_counter()
    small = store.load_json("small_choices")
    theta = np.asarray(small["options"]["nu0.5_mercator"]["theta"])
    tr, _, _ = data()
    d = tr[(tr["year"] == 2024) & fc.in_box(tr["lat"], tr["lon"], c.SMALL_CROP)]
    d = d.iloc[np.sort(np.random.default_rng(8).permutation(len(d))[:3000])]
    X, y = fc.X_of(d), d["anom"].to_numpy()
    order = fc.maxmin_order(fc.scaled(X, theta))
    Xo, yo = X[order], y[order]
    out = {"n": len(y), "theta": theta, "rows": {}}
    with gpu_lock():
        exact = float(c.make_logp(NU, with_prior=False)(jnp.asarray(theta), jnp.asarray(Xo), jnp.asarray(yo)))
        out["exact"] = exact
        for m in (15, 30, 60):
            nb = fc.neighbours(fc.scaled(Xo, theta), m, brute=3000)
            lp, _ = fc.make_vecchia(NU)
            v = float(lp(jnp.asarray(theta), jnp.asarray(Xo), jnp.asarray(yo), jnp.asarray(nb)))
            out["rows"][m] = {"vecchia": v, "gap": exact - v, "gap_per_obs": (exact - v) / len(y)}
            log("P8 m", m, "exact", round(exact, 2), "vecchia", round(v, 2), "per obs", (exact - v) / len(y))
        out["vram"] = peak()
    g30 = abs(out["rows"][30]["gap_per_obs"])
    out.update({"pass_m30": g30 < 0.005, "m_for_C": 30 if g30 < 0.005 else 60,
                "line": GATES["P8_vecchia"]["go"], "wall_s": time.perf_counter() - t0, "health": p.health(),
                "labels": "measured"})
    store.save_json("p8_vecchia", out)
    store.mark_done("p8", passed=out["pass_m30"], m_for_C=out["m_for_C"])


# Design machinery ---------------------------------------------------------------------------------------


def tracks(train):
    """Real 36-step tracks: (start lat, lon), steps (km east, km north) and day gaps, from 2023-2024."""
    starts, steps, gaps = [], [], []
    for _, f in train.sort_values(["float_id", "t_days"]).groupby("float_id"):
        cyc, t, la, lo = (f[k].to_numpy() for k in ("cycle", "t_days", "lat", "lon"))
        ok = (np.diff(cyc) == 1) & (np.diff(t) > 5) & (np.diff(t) < 15)
        run = 0
        for i in range(len(ok)):
            run = run + 1 if ok[i] else 0
            if run >= N_CYC - 1 and (run - (N_CYC - 1)) % 12 == 0:
                s = i + 1 - (N_CYC - 1)
                seg = slice(s, i + 2)
                lat_s, lon_s = la[seg], lo[seg]
                mid = np.radians(0.5 * (lat_s[1:] + lat_s[:-1]))
                dx = np.diff(lon_s) * 111.19 * np.cos(mid)
                dy = np.diff(lat_s) * 111.19
                starts.append((lat_s[0], lon_s[0]))
                steps.append(np.c_[dx, dy])
                gaps.append(np.diff(t[seg]))
    return np.array(starts), np.array(steps), np.array(gaps)


def candidate_sites(ocean, n, box=None, seed=92):
    from scipy.stats import qmc

    b = box or dict(lat_min=20.0, lat_max=55.0, lon_min=-80.0, lon_max=0.0)
    h = qmc.Halton(d=2, seed=seed).random(20 * n)
    lat = b["lat_min"] + 0.5 + h[:, 0] * (b["lat_max"] - b["lat_min"] - 1.0)
    lon = b["lon_min"] + 0.5 + h[:, 1] * (b["lon_max"] - b["lon_min"] - 1.0)
    ok = ocean(lat, lon)
    return np.c_[lat[ok], lon[ok]][:n]


def drift(sites, starts, steps, gaps, ocean, rep, seed=92, fixed=False, box=None):
    """(sites, 36) lat, lon, t and a validity mask for one drift replicate, common to every rule."""
    rng = np.random.default_rng(seed + 1000 * rep)
    S = len(sites)
    lat, lon, t = np.zeros((S, N_CYC)), np.zeros((S, N_CYC)), np.zeros((S, N_CYC))
    valid = np.zeros((S, N_CYC), bool)
    d0 = rng.uniform(0, 60, S)
    picks = []
    for s in range(S):
        dist = fc.haversine_km(sites[s, 0], sites[s, 1], starts[:, 0], starts[:, 1])
        near = np.flatnonzero(dist < 600)
        if len(near) == 0:
            near = np.argsort(dist)[:20]
        k = rng.choice(near)
        picks.append(k)
        la, lo = [sites[s, 0]], [sites[s, 1]]
        for st in steps[k]:
            if fixed:
                la.append(la[-1]), lo.append(lo[-1])
                continue
            nla = la[-1] + st[1] / 111.19
            lo.append(lo[-1] + st[0] / (111.19 * np.cos(np.radians(0.5 * (la[-1] + nla)))))
            la.append(nla)
        tt = fc.DAY_2025 + d0[s] + np.r_[0, np.cumsum(gaps[k])]
        la, lo = np.array(la), np.array(lo)
        ok = (tt < fc.DAY_2026) & ocean(la, lo)
        if box is not None:
            ok &= fc.in_box(la, lo, box)
        first_bad = np.flatnonzero(~ok)
        keep = N_CYC if len(first_bad) == 0 else first_bad[0]
        lat[s], lon[s], t[s] = la, lo, tt
        valid[s, :keep] = True
    return lat, lon, t, valid, np.array(picks)


def targets(ocean, box=None, step=2.0):
    b = box or dict(lat_min=20.0, lat_max=55.0, lon_min=-80.0, lon_max=0.0)
    la = np.arange(b["lat_min"] + step / 2, b["lat_max"], step)
    lo = np.arange(b["lon_min"] + step / 2, b["lon_max"], step)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    LA, LO = LA.ravel(), LO.ravel()
    ok = ocean(LA, LO)
    LA, LO = LA[ok], LO[ok]
    days = fc.DAY_2025 + np.array([45.0, 134.0, 226.0, 318.0])
    return np.repeat(LA, 4), np.repeat(LO, 4), np.tile(days, len(LA)), np.c_[LA, LO]


def posterior_rest(nu, theta, X, Xnet, noise_net, Xrest, dtype):
    """Posterior covariance (f64 NumPy on the device as JAX) of the rest given training and the network."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    th = jnp.asarray(theta)
    Xd = jnp.asarray(X, dtype)
    K = c.build_K(Xd, th, nu)
    L = jnp.linalg.cholesky(K)
    del K
    Q = jnp.asarray(np.concatenate([Xnet, Xrest]), dtype)
    n, q = Xd.shape[0], Q.shape[0]
    rows = []
    for s in range(0, q, 4096):
        Ks = c.kernel_block(Q[s:s + 4096], Xd, jnp.arange(min(4096, q - s)) + n + s, jnp.arange(n), th, nu)
        rows.append(Ks)
    V = solve_triangular(L, jnp.concatenate(rows).T, lower=True)
    del rows, L
    Sig = jax.jit(lambda Q: c.build_K(Q, th, nu, block=1024))(Q)
    Sig = Sig - V.T @ V
    del V
    # build_K adds the nugget on the diagonal: remove it so Sig is the latent covariance
    sn2 = jnp.exp(2 * th[8]).astype(dtype)
    diag_nug = sn2 * (Q[:, 4] if Q.shape[1] >= 5 else jnp.ones(Q.shape[0], dtype))
    di = jnp.arange(Q.shape[0])
    Sig = Sig.at[di, di].add(-diag_nug)
    m = len(Xnet)
    if m == 0:
        return Sig.astype(jnp.float64), jnp.diag(Sig).astype(jnp.float64)
    dm = jnp.arange(m)
    Snn = Sig[:m, :m].at[dm, dm].add(jnp.asarray(noise_net, dtype))
    Lnn = jnp.linalg.cholesky(Snn)
    W = solve_triangular(Lnn, Sig[:m, m:], lower=True)
    P = (Sig[m:, m:] - W.T @ W).astype(jnp.float64)
    prior_var = jnp.diag(Sig[m:, m:]).astype(jnp.float64)
    return P, prior_var


@__import__("functools").lru_cache
def make_greedy():
    import jax
    import jax.numpy as jnp

    def site_scores(P, T, idx, valid, noise, w):
        Pi = jnp.where(valid, idx, 0)
        B = P[T][:, Pi]  # (nT, S, 36)
        B = jnp.where(valid[None], B, 0.0)
        C = P[Pi[:, :, None], Pi[:, None, :]]
        vv = valid[:, :, None] & valid[:, None, :]
        eye = jnp.eye(idx.shape[1])
        C = jnp.where(vv, C, eye[None]) + eye[None] * jnp.where(valid, noise, 0.0)[:, :, None]
        Lc = jnp.linalg.cholesky(C)
        Bt = jnp.transpose(B, (1, 2, 0))  # (S, 36, nT)
        Z = jax.scipy.linalg.solve_triangular(Lc, Bt, lower=True)
        return jnp.einsum("sjt,t->s", Z * Z, w)

    def update(P, j, idx, valid, noise):
        Pi = jnp.where(valid[j], idx[j], 0)
        Bc = jnp.where(valid[j][None], P[:, Pi], 0.0)
        C = P[Pi[:, None], Pi[None, :]]
        vv = valid[j][:, None] & valid[j][None, :]
        eye = jnp.eye(idx.shape[1])
        C = jnp.where(vv, C, eye) + eye * jnp.where(valid[j], noise[j], 0.0)
        Lc = jnp.linalg.cholesky(C)
        Z = jax.scipy.linalg.solve_triangular(Lc, Bc.T, lower=True)
        return P - Z.T @ Z

    def set_removal(P, T, flat_idx, flat_noise, w):
        B = P[T][:, flat_idx]
        C = P[flat_idx[:, None], flat_idx[None, :]] + jnp.diag(flat_noise)
        Lc = jnp.linalg.cholesky(C)
        Z = jax.scipy.linalg.solve_triangular(Lc, B.T, lower=True)
        return jnp.sum(jnp.sum(Z * Z, 0) * w)

    return jax.jit(site_scores), jax.jit(update), jax.jit(set_removal)


def run_design(P, nT, idx, valid, noise, weights, picks_n, gap_picks, random_sets, keep_maps=False):
    """Greedy picks under each weight vector, plus the removal of the gap and random sets (uniform weight)."""
    import jax.numpy as jnp

    scores_fn, update_fn, set_fn = make_greedy()
    T = jnp.arange(nT)
    idx_j, valid_j, noise_j = jnp.asarray(idx), jnp.asarray(valid), jnp.asarray(noise)
    out = {}
    for wname, w in weights.items():
        w = jnp.asarray(w / w.sum())
        Pk = P
        picks, trace, maps = [], [float(jnp.sum(jnp.diag(P)[:nT] * w))], [np.asarray(jnp.diag(P)[:nT], np.float32)]
        for _ in range(picks_n):
            sc = np.array(scores_fn(Pk, T, idx_j, valid_j, noise_j, w))
            sc[picks] = -np.inf
            sc[~np.asarray(valid).any(1)] = -np.inf
            j = int(np.argmax(sc))
            picks.append(j)
            Pk = update_fn(Pk, j, idx_j, valid_j, noise_j)
            trace.append(float(jnp.sum(jnp.diag(Pk)[:nT] * w)))
            if keep_maps:
                maps.append(np.asarray(jnp.diag(Pk)[:nT], np.float32))
        out[wname] = {"picks": np.array(picks), "trace": np.array(trace),
                      "removed": trace[0] - trace[-1], **({"maps": np.array(maps)} if keep_maps else {})}
        del Pk
    w = jnp.ones(nT) / nT

    def removal(sites):
        flat = np.concatenate([idx[s][valid[s]] for s in sites])
        fn = np.concatenate([noise[s][valid[s]] for s in sites])
        return float(set_fn(P, T, jnp.asarray(flat), jnp.asarray(fn), w))

    out["gap_removed"] = removal(gap_picks)
    out["random_removed"] = np.array([removal(s) for s in random_sets])
    out["before"] = float(jnp.mean(jnp.diag(P)[:nT]))
    return out


def gap_fill(sites, net_latlon, n, allowed=None):
    dmin = np.full(len(sites), np.inf)
    for s in range(0, len(net_latlon), 2000):
        d = fc.haversine_km(sites[:, None, 0], sites[:, None, 1], net_latlon[None, s:s + 2000, 0],
                            net_latlon[None, s:s + 2000, 1])
        dmin = np.minimum(dmin, d.min(1))
    if allowed is not None:  # the operable-water rerun: excluded sites can never be the farthest
        dmin = np.where(allowed, dmin, -1.0)
    picks = []
    for _ in range(n):
        j = int(np.argmax(np.where(np.isin(np.arange(len(sites)), picks), -2, dmin)))
        picks.append(j)
        dmin = np.minimum(dmin, fc.haversine_km(sites[j, 0], sites[j, 1], sites[:, 0], sites[:, 1]))
    return np.array(picks)


def design_inputs(box=None, n_sites=N_SITES, seed=92, year_train=None):
    """Sites, targets, tracks and the network (2025 non-held positions), for FULL or a crop."""
    tr, _, y25 = data()
    ocean = ocean_fn()
    net = y25[~y25["held"]]
    if box is not None:
        net = net[fc.in_box(net["lat"], net["lon"], box)]
    sites = candidate_sites(ocean, n_sites, box=box, seed=seed)
    starts, steps, gaps = tracks(tr)
    tla, tlo, tt, cells = targets(ocean, box=box)
    return dict(ocean=ocean, net=net, sites=sites, starts=starts, steps=steps, gaps=gaps, tla=tla, tlo=tlo, tt=tt,
                cells=cells)


def xyz(lat, lon, t):
    return c.coords(lat, lon, t, "mercator")


def p10():
    """Design sensitivity at SMALL: best 5 against random 5 and gap 5, P6 MAP, common drift."""
    gpu()
    import jax.numpy as jnp

    t0 = time.perf_counter()
    small = store.load_json("small_choices")
    theta = np.asarray(small["options"]["nu0.5_mercator"]["theta"])
    crop = c.SMALL_CROP
    tr, _, _ = data()
    d = tr[(tr["year"] == 2024) & fc.in_box(tr["lat"], tr["lon"], crop)]
    di = design_inputs(box=crop, n_sites=40, seed=10)
    X = fc.X_of(d)
    Xnet = fc.X_of(di["net"])
    Xt = xyz(di["tla"], di["tlo"], di["tt"])
    sn2 = np.exp(2 * theta[8])
    rows = []
    rs = np.random.default_rng(10)
    random_sets = [rs.choice(len(di["sites"]), 5, replace=False) for _ in range(50)]
    gap = gap_fill(di["sites"], di["net"][["lat", "lon"]].to_numpy(), 5)
    with gpu_lock():
        for rep in range(N_REP):
            la, lo, tt, valid, _ = drift(di["sites"], di["starts"], di["steps"], di["gaps"], di["ocean"], rep, seed=10,
                                         box=crop)
            Xc = xyz(la.ravel(), lo.ravel(), tt.ravel())
            nT = len(Xt)
            P, _ = posterior_rest(NU, theta, X, Xnet, np.full(len(Xnet), sn2), np.concatenate([Xt, Xc]), jnp.float64)
            idx = nT + np.arange(len(Xc)).reshape(valid.shape)
            res = run_design(P, nT, idx, valid, np.full(valid.shape, sn2), {"uniform": np.ones(nT)}, 5, gap, random_sets)
            rows.append({"rep": rep, "best": res["uniform"]["removed"], "picks": res["uniform"]["picks"],
                         "gap": res["gap_removed"], "random_mean": float(res["random_removed"].mean()),
                         "before": res["before"], "valid_cycles": int(valid.sum())})
            log("P10 rep", rep, rows[-1])
            del P
            cleanup()
        vram = peak()
    best = np.mean([r["best"] for r in rows])
    rnd = np.mean([r["random_mean"] for r in rows])
    gp = np.mean([r["gap"] for r in rows])
    out = {"rows": rows, "best_over_random": best / rnd, "best_over_gap": best / gp, "pass": best / rnd >= 1.25,
           "n_train": len(d), "n_network": len(Xnet), "n_targets": len(Xt), "n_sites": len(di["sites"]),
           "line": GATES["P10_design_small"]["go"], "vram": vram, "wall_s": time.perf_counter() - t0,
           "health": p.health(), "labels": "measured"}
    store.save_json("p10_design_small", out)
    store.mark_done("p10", passed=out["pass"], best_over_random=out["best_over_random"])
    log("P10", out["best_over_random"], out["best_over_gap"])


# Fitting ------------------------------------------------------------------------------------------------


def fit_phase(name, X, y, x0, prec, **kw):
    gpu()
    import jax.numpy as jnp

    t0 = time.perf_counter()
    dtype = jnp.float32 if prec == 32 else jnp.float64
    with gpu_lock():
        fun = fc.neg(c.make_value_and_grad(NU), X, y, dtype)
        st = fc.lbfgs(fun, x0, store, name, log=log, health=p.health, **kw)
        vram = peak()
    theta = np.asarray(st["x"])
    info = {"theta": theta, "natural": fc.natural(theta), "F": float(st["f"]), "max_abs_g": float(np.max(np.abs(st["g"]))),
            "iterations": int(st["it"]), "evaluations": int(st["nev"]), "message": fc.msg_of(st),
            "edges": fc.edges(theta), "prec": prec, "n": len(y), "wall_s_this_process": time.perf_counter() - t0,
            "vram": vram, "health": p.health(), "labels": "measured"}
    store.save_json(name, info)
    store.mark_done(name, edges=info["edges"], F=info["F"], evaluations=info["evaluations"])
    log(name, "done", info["message"], "edges", info["edges"], np.round(info["natural"], 3).tolist())
    return theta


F32 = dict(max_iter=60, gtol=1.0, max_eval=150)
F64 = dict(max_iter=25, gtol=0.02, max_eval=25, ftol=1e-12)


def fit_A_f32():
    tr, _, _ = data()
    x0 = np.asarray(store.load_json("small_choices")["options"]["nu0.5_mercator"]["theta"])
    fit_phase("fit_A_f32", fc.X_of(tr), tr["anom"].to_numpy(), x0, 32, **F32)


def fit_A_f64():
    tr, _, _ = data()
    x0, st = mode_of("fit_A_f32")
    fit_phase("fit_A_f64", fc.X_of(tr), tr["anom"].to_numpy(), x0, 64, init_SY=(st["S"], st["Y"]), **F64)


def fit_t(model, base_cols, x0, out_name, rounds=8):
    """Student-t nugget by scale-mixture weights: weights from f32 exact residual posteriors, theta refit."""
    gpu()
    import jax.numpy as jnp

    tr, _, _ = data()
    y = tr["anom"].to_numpy()
    t0 = time.perf_counter()
    done_r, st = store.resume(f"{out_name}_rounds")
    r0 = 0 if done_r is None else int(st["round"]) + 1
    mult = np.ones(len(y)) if done_r is None else st["mult"]
    theta = np.asarray(x0) if done_r is None else st["theta"]
    with gpu_lock():
        resid = fc.make_resid(NU)
        for r in range(r0, rounds):
            p.health()
            X = fc.X_of(tr, [*base_cols, mult])
            a, kd = (np.asarray(v, np.float64) for v in resid(jnp.asarray(theta), jnp.asarray(X, jnp.float32),
                                                               jnp.asarray(y, jnp.float32)))
            mult = fc.t_weights(theta, a, kd, mult)
            X = fc.X_of(tr, [*base_cols, mult])
            fun = fc.neg(c.make_value_and_grad(NU), X, y, jnp.float32)
            st = fc.lbfgs(fun, theta, None, f"{out_name}_r{r}", max_iter=20, gtol=1.0, max_eval=50, log=log,
                          health=p.health)
            theta = np.asarray(st["x"])
            store.save_chunk(f"{out_name}_rounds", r, {"round": np.array(r), "theta": theta, "mult": mult,
                                                        "F": np.array(st["f"])}, keep=2)
            log(out_name, "round", r, "F", float(st["f"]), "share mult>2", float((mult > 2).mean()),
                np.round(fc.natural(theta), 3).tolist())
            cleanup()
        vram = peak()
    store.save_array(f"{out_name}_mult", mult)
    store.save_json(f"{out_name}_rounds", {"theta": theta, "rounds": rounds, "wall_s": time.perf_counter() - t0,
                                           "vram": vram, "mult_quantiles": np.quantile(mult, [0.5, 0.9, 0.99, 0.999]),
                                           "labels": "measured"})
    store.mark_done(f"{out_name}_rounds")


def fit_t_final(model, base_cols, out_name):
    """f64 polish at the final weights (the mult array saved as ``<out_name>_final_mult`` for model_spec)."""
    tr, _, _ = data()
    mult = store.load_array(f"{out_name}_mult")
    store.save_array(f"{out_name}_final_mult", mult)
    theta0 = np.asarray(store.load_json(f"{out_name}_rounds")["theta"])
    fit_phase(f"{out_name}_final", fc.X_of(tr, [*base_cols, mult]), tr["anom"].to_numpy(), theta0, 64,
              max_iter=12, gtol=0.02, max_eval=12, ftol=1e-12)


def fit_At():
    tr, _, _ = data()
    x0, _ = mode_of("fit_A_f64")
    fit_t("At", [np.zeros(len(tr))], x0, "fit_At")


def fit_At_final():
    tr, _, _ = data()
    fit_t_final("At", [np.zeros(len(tr))], "fit_At")


def hessian(limit=99):
    gpu()
    import jax.numpy as jnp

    model = primary()
    theta, X, y, _, _ = model_spec(model)
    todo = store.pending("hessian", 20)[: int(limit)]
    with gpu_lock():
        vg = c.make_value_and_grad(NU)
        Xd, yd = jnp.asarray(X), jnp.asarray(y)
        for k in todo:
            p.health()
            j, sign = divmod(k, 2)
            e = np.zeros(10)
            e[j] = H_STEP * (1 if sign == 0 else -1)
            t0 = time.perf_counter()
            v, g = vg(jnp.asarray(theta + e), Xd, yd)
            store.save_chunk("hessian", k, {"theta": theta + e, "value": np.array([float(v)]),
                                            "grad": np.asarray(g, np.float64)})
            log("hessian", k, round(time.perf_counter() - t0, 1), "s")
        vram = peak()
    if not store.pending("hessian", 20):
        G = store.gather("hessian", 20)["grad"].reshape(10, 2, 10)
        H = (G[:, 0] - G[:, 1]) / (2 * H_STEP)
        H = 0.5 * (H + H.T)
        ev = np.linalg.eigvalsh(H)
        clipped = False
        if ev.max() >= 0:  # not negative definite: the gates' no-go (clip) is recorded
            w, U = np.linalg.eigh(-H)
            w = np.maximum(w, 1e-6 * w.max())
            H = -(U * w) @ U.T
            clipped = True
        cov = np.linalg.inv(-H)
        store.save_json("hessian", {"model": model, "theta": theta, "H": H, "eigen": ev, "negative_definite": bool(ev.max() < 0),
                                    "clipped": clipped, "cov": cov, "sd": np.sqrt(np.diag(cov)), "step": H_STEP,
                                    "vram": vram, "labels": "measured"})
        store.mark_done("hessian", model=model, negative_definite=bool(ev.max() < 0))
        log("hessian sd", np.round(np.sqrt(np.diag(cov)), 4).tolist())


def sobol(limit=99):
    gpu()
    import jax.numpy as jnp
    from scipy.stats import norm, qmc

    hs = store.load_json("hessian")
    model = hs["model"]
    theta, X, y, _, _ = model_spec(model)
    cov = np.asarray(hs["cov"])
    Lc = np.linalg.cholesky(cov)
    u = qmc.Sobol(d=10, scramble=True, seed=92).random(64)
    z = norm.ppf(u)
    pts = theta + z @ Lc.T
    logq = -0.5 * (z * z).sum(1)  # up to a constant shared by every point
    todo = store.pending("sobol", 64)[: int(limit)]
    with gpu_lock():
        lp = c.make_logp(NU)
        Xd, yd = jnp.asarray(X), jnp.asarray(y)
        for k in todo:
            p.health()
            v = float(lp(jnp.asarray(pts[k]), Xd, yd))
            store.save_chunk("sobol", k, {"theta": pts[k][None], "logp": np.array([v]), "logq": np.array([logq[k]])})
            log("sobol", k, v)
    if not store.pending("sobol", 64):
        from arviz_stats.base import array_stats

        g = store.gather("sobol", 64, axis=0)
        th = g["theta"].reshape(64, 10)
        lw_raw = np.asarray(g["logp"]).ravel() - np.asarray(g["logq"]).ravel()
        lw_raw = np.where(np.isfinite(lw_raw), lw_raw, -np.inf)
        lw, k = array_stats.psislw(lw_raw - np.max(lw_raw), r_eff=1.0)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        ok = float(k) < 0.7
        rng = np.random.default_rng(92)
        if ok:
            u0 = (rng.uniform() + np.arange(10)) / 10
            draws = th[np.searchsorted(np.cumsum(w), u0)]
        else:
            draws = th[:10]
        store.save_json("sobol", {"model": model, "theta": th, "log_ratio": lw_raw, "weights": w, "pareto_k": float(k),
                                  "pass": ok, "ess": float(1 / np.sum(w**2)), "design_draws": np.vstack([theta, draws]),
                                  "weighted_mean": w @ th, "weighted_sd": np.sqrt(w @ (th - w @ th) ** 2),
                                  "labels": "measured"})
        store.mark_done("sobol", pareto_k=float(k), passed=ok)
        log("sobol k", float(k), "ess", 1 / np.sum(w**2))


def heldout(model):
    """Per-profile latent mean and variance at the held-out floats, and per-float joint log scores."""
    gpu()
    import jax.numpy as jnp

    t0 = time.perf_counter()
    theta, X, y, new, noise_new = model_spec(model)
    _, te, _ = data()
    Xs = new(te)
    groups = te["float_id"].to_numpy()
    with gpu_lock():
        fac = fc.make_factor(NU)
        L, a = fac(jnp.asarray(theta), jnp.asarray(X), jnp.asarray(y))
        mean, var, covs = fc.predict(NU, theta, L, a, jnp.asarray(X), Xs, groups=groups)
        vram = peak()
    yte = te["anom"].to_numpy()
    joint = fc.joint_scores(mean, covs, noise_new, yte, groups) if not is_t(model) else {}
    out = {"model": model, "theta": theta, "mean": mean, "latent_var": var, "noise_var": noise_new,
           "sn": float(np.exp(theta[8])), "student_t": is_t(model), "nu_t": fc.NU_T if is_t(model) else None,
           "float_id": groups, "joint_log_score": joint, "wall_s": time.perf_counter() - t0, "vram": vram,
           "labels": "measured"}
    store.save_json(f"heldout_{model}", out)
    z = (yte - mean) / np.sqrt(var + noise_new)
    store.mark_done(f"heldout_{model}", share_abs_z_over_3=float(np.mean(np.abs(z) > 3)))
    log("heldout", model, "rmse", float(np.sqrt(np.mean((yte - mean) ** 2))), "|z|>3", float(np.mean(np.abs(z) > 3)))


def maps():
    gpu()
    import jax.numpy as jnp

    ocean = ocean_fn()
    la, lo = np.arange(20.5, 55, 1.0), np.arange(-79.5, 0, 1.0)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    ok = ocean(LA.ravel(), LO.ravel())
    cl, co = LA.ravel()[ok], LO.ravel()[ok]
    days = 365 + 15.0 + 30.4 * np.arange(12)  # mid-month 2024
    out = {"cell_lat": cl, "cell_lon": co, "days": days}
    models = [primary()] + (["Afix"] if store.done("fit_Afix_f64") else [])
    with gpu_lock():
        for model in models:
            theta, X, y, new, _ = model_spec(model)
            fac = fc.make_factor(NU)
            Xd = jnp.asarray(X, jnp.float32)
            L, a = fac(jnp.asarray(theta), Xd, jnp.asarray(y, jnp.float32))
            Xs = new(np.tile(cl, 12), np.tile(co, 12), xyz=xyz(np.tile(cl, 12), np.tile(co, 12), np.repeat(days, len(cl))))
            mean, var, _ = fc.predict(NU, theta, L, a, Xd, Xs)
            store.save_array(f"map_mean_{model}", mean.reshape(12, -1).astype(np.float32))
            store.save_array(f"map_sd_{model}", np.sqrt(np.maximum(var, 0)).reshape(12, -1).astype(np.float32))
            del L, a
            cleanup()
        vram = peak()
    store.save_json("maps", {**out, "models": models, "vram": vram, "labels": "measured; latent anomaly, f32"})
    store.mark_done("maps", models=models)


def design_chunks():
    """(chunk id, model, draw, replicate, fixed) for every design chunk, in a fixed order."""
    rows = []
    prim = primary()
    for dr in range(N_DRAWS):
        for rep in range(N_REP):
            rows.append((len(rows), prim, dr, rep, False))
    rows.append((len(rows), prim, 0, 0, True))
    for model in ["Afix", "Afixt", "A"]:
        if model == prim:
            continue
        for rep in range(N_REP):
            rows.append((len(rows), model, 0, rep, False))
    return rows


def design(limit=99, which="primary"):
    gpu()
    import jax.numpy as jnp

    di = design_inputs()
    sites, nT = di["sites"], len(di["tla"])
    gs = fc.in_box(di["tla"], di["tlo"], c.GULF_STREAM).astype(float)
    rs = np.random.default_rng(92)
    random_sets = [rs.choice(len(sites), N_PICK, replace=False) for _ in range(50)]
    gap = gap_fill(sites, di["net"][["lat", "lon"]].to_numpy(), N_PICK)
    rows = design_chunks()
    prim = primary()
    avail = {"Afix": store.done("heldout_Afix"), "Afixt": store.done("heldout_Afixt"), "A": store.done("fit_A_f64")}
    pend = set(store.pending("design", len(rows)))
    todo = [r for r in rows if r[0] in pend and ((r[1] == prim) if which == "primary" else (r[1] != prim and avail[r[1]]))]
    draws = np.asarray(store.load_json("sobol")["design_draws"]) if which == "primary" else None
    with gpu_lock():
        for cid, model, dr, rep, fixed in todo[: int(limit)]:
            p.health()
            t0 = time.perf_counter()
            theta, X, y, new, noise_new = model_spec(model)
            if dr > 0:
                theta = draws[dr]
                noise_new = np.exp(2 * theta[8]) * (fc.NU_T / (fc.NU_T - 2) if is_t(model) else 1.0)
            la, lo, tt, valid, tk = drift(sites, di["starts"], di["steps"], di["gaps"], di["ocean"], rep, fixed=fixed)
            net = di["net"]
            Xnet = new(net)
            Xt = new(di["tla"], di["tlo"], xyz=xyz(di["tla"], di["tlo"], di["tt"]))
            Xc = new(la.ravel(), lo.ravel(), xyz=xyz(la.ravel(), lo.ravel(), tt.ravel()))
            P, prior_var = fc_posterior(theta, X, Xnet, noise_new, np.concatenate([Xt, Xc]))
            idx = nT + np.arange(len(Xc)).reshape(valid.shape)
            weights = {"uniform": np.ones(nT)}
            if dr == 0 and not fixed:
                weights["gulf"] = gs.copy()
            res = run_design(P, nT, idx, valid, np.full(valid.shape, noise_new), weights, N_PICK, gap, random_sets,
                             keep_maps=(dr == 0 and rep == 0 and not fixed))
            arrays = {"picks": res["uniform"]["picks"], "trace": res["uniform"]["trace"],
                      "gap_removed": np.array(res["gap_removed"]), "random_removed": res["random_removed"],
                      "before": np.array(res["before"]), "prior_mean_var": np.array(float(prior_var[:nT].mean())),
                      "valid": valid, "tracks": tk, "gap_picks": gap, "meta": np.array([cid, dr, rep, int(fixed)]),
                      "theta": theta}
            if "gulf" in res:
                arrays["gulf_picks"], arrays["gulf_trace"] = res["gulf"]["picks"], res["gulf"]["trace"]
            if "maps" in res["uniform"]:
                arrays["var_maps"] = res["uniform"]["maps"]
            store.save_chunk("design", cid, arrays)
            store.save_json(f"design_{cid:03d}", {"model": model, "draw": dr, "rep": rep, "fixed": fixed,
                                                   "greedy": res["uniform"]["removed"], "gap": res["gap_removed"],
                                                   "random_mean": float(res["random_removed"].mean()),
                                                   "before": res["before"], "seconds": time.perf_counter() - t0,
                                                   "vram": peak(), "labels": "measured"})
            log("design", cid, model, dr, rep, fixed, "greedy", round(res["uniform"]["removed"], 4), "gap",
                round(res["gap_removed"], 4), "random", round(float(res["random_removed"].mean()), 4),
                round(time.perf_counter() - t0, 1), "s")
            del P
            cleanup()
    left = [r for r in rows if r[0] in set(store.pending("design", len(rows)))]
    if which == "primary" and not [r for r in left if r[1] == prim]:
        store.save_json("design_inputs", {"sites": sites, "target_cells": di["cells"], "target_days": [45, 134, 226, 318],
                                          "gap_picks": gap, "random_sets": np.array(random_sets), "rows": rows,
                                          "n_network": len(di["net"]), "n_targets": nT, "n_tracks": len(di["starts"])})
        store.save_array("design_tracks_steps", di["steps"].astype(np.float32))
        store.save_array("design_tracks_starts", di["starts"])
        store.mark_done("design")
    if which == "alt" and not [r for r in left if r[1] != prim and avail.get(r[1])]:
        store.mark_done("design_alt", models=[m for m in avail if avail[m] and m != prim])


# The operable-water rerun -------------------------------------------------------------------------------
# A declared change made after the result was seen (2026-11-04). The first design only excluded land, so
# gap-filling chose shelf and sound sites (Long Island Sound, the Grand Banks, Browns Bank, Little Bahama
# Bank, the Labrador Shelf) where a float cannot profile to 2,000 m. No bathymetry is in the repository, so
# the proxy is Argo's own record: a point is operable if at least OPERABLE_MIN profiles of 2015-2022 (the
# climatology's base years, already cached) lie within OPERABLE_KM. The radius and count were chosen after
# looking at the counts at the shelf sites above (at 150 km the Grand Banks still pass, with 22-38). The
# mask applies to launch sites and to drift (a track ends where it leaves operable water); targets, sites,
# tracks and seeds are the first design's, and the first design's chunks are left untouched.
OPERABLE_KM, OPERABLE_MIN = 75.0, 10


@__import__("functools").lru_cache
def _base_tree():
    import build_e92_data as b
    import pandas as pd
    from scipy.spatial import cKDTree

    base = pd.read_csv(b.BASE, usecols=["lat", "lon"])
    return cKDTree(_unit(base["lat"].to_numpy(), base["lon"].to_numpy())), len(base)


def _unit(lat, lon):
    p, l_ = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(p) * np.cos(l_), np.cos(p) * np.sin(l_), np.sin(p)]


def base_counts(lat, lon, km=OPERABLE_KM):
    tree, _ = _base_tree()
    chord = 2 * np.sin(km / (2 * c.EARTH_KM))
    return np.asarray(tree.query_ball_point(_unit(np.atleast_1d(lat), np.atleast_1d(lon)), chord,
                                            return_length=True))


def operable_fn(ocean=None):
    ocean = ocean or ocean_fn()

    def operable(lat, lon):
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        ok = ocean(lat, lon)
        if ok.any():
            ok[ok] = base_counts(lat[ok], lon[ok]) >= OPERABLE_MIN
        return ok

    return operable


def design_masked(limit=99):
    """The design rerun on operable water: chunks ``design_masked``, same rows, draws and seeds as ``design``."""
    gpu()

    di = design_inputs()
    operable = operable_fn(di["ocean"])
    sites, nT = di["sites"], len(di["tla"])
    allowed = operable(sites[:, 0], sites[:, 1])
    rs = np.random.default_rng(92)
    random_sets = [np.flatnonzero(allowed)[rs.choice(int(allowed.sum()), N_PICK, replace=False)] for _ in range(50)]
    gap = gap_fill(sites, di["net"][["lat", "lon"]].to_numpy(), N_PICK, allowed=allowed)
    rows = design_chunks()
    prim = primary()
    draws = np.asarray(store.load_json("sobol")["design_draws"])
    todo = [r for r in rows if r[0] in set(store.pending("design_masked", len(rows)))]
    with gpu_lock():
        for cid, model, dr, rep, fixed in todo[: int(limit)]:
            p.health()
            t0 = time.perf_counter()
            theta, X, y, new, noise_new = model_spec(model)
            if dr > 0:
                theta = draws[dr]
                noise_new = np.exp(2 * theta[8]) * (fc.NU_T / (fc.NU_T - 2) if is_t(model) else 1.0)
            la, lo, tt, valid, tk = drift(sites, di["starts"], di["steps"], di["gaps"], operable, rep, fixed=fixed)
            assert not valid[~allowed].any()
            Xnet = new(di["net"])
            Xt = new(di["tla"], di["tlo"], xyz=xyz(di["tla"], di["tlo"], di["tt"]))
            Xc = new(la.ravel(), lo.ravel(), xyz=xyz(la.ravel(), lo.ravel(), tt.ravel()))
            P, prior_var = fc_posterior(theta, X, Xnet, noise_new, np.concatenate([Xt, Xc]))
            idx = nT + np.arange(len(Xc)).reshape(valid.shape)
            weights = {"uniform": np.ones(nT)}
            if dr == 0 and not fixed and model == prim:
                weights["gulf"] = fc.in_box(di["tla"], di["tlo"], c.GULF_STREAM).astype(float)
            res = run_design(P, nT, idx, valid, np.full(valid.shape, noise_new), weights, N_PICK, gap, random_sets,
                             keep_maps=(dr == 0 and rep == 0 and not fixed and model == prim))
            arrays = {"picks": res["uniform"]["picks"], "trace": res["uniform"]["trace"],
                      "gap_removed": np.array(res["gap_removed"]), "random_removed": res["random_removed"],
                      "before": np.array(res["before"]), "prior_mean_var": np.array(float(prior_var[:nT].mean())),
                      "valid": valid, "tracks": tk, "gap_picks": gap, "meta": np.array([cid, dr, rep, int(fixed)]),
                      "theta": theta}
            if "gulf" in res:
                arrays["gulf_picks"], arrays["gulf_trace"] = res["gulf"]["picks"], res["gulf"]["trace"]
            if "maps" in res["uniform"]:
                arrays["var_maps"] = res["uniform"]["maps"]
            store.save_chunk("design_masked", cid, arrays)
            store.save_json(f"design_masked_{cid:03d}", {"model": model, "draw": dr, "rep": rep, "fixed": fixed,
                                                          "greedy": res["uniform"]["removed"], "gap": res["gap_removed"],
                                                          "random_mean": float(res["random_removed"].mean()),
                                                          "before": res["before"], "seconds": time.perf_counter() - t0,
                                                          "vram": peak(), "labels": "measured"})
            log("design_masked", cid, model, dr, rep, fixed, "greedy", round(res["uniform"]["removed"], 4), "gap",
                round(res["gap_removed"], 4), "random", round(float(res["random_removed"].mean()), 4),
                round(time.perf_counter() - t0, 1), "s")
            del P
            cleanup()
    if not store.pending("design_masked", len(rows)):
        store.save_json("design_masked_inputs", {
            "sites": sites, "allowed": allowed, "site_base_counts": base_counts(sites[:, 0], sites[:, 1]),
            "gap_picks": gap, "random_sets": np.array(random_sets), "rows": rows, "operable_km": OPERABLE_KM,
            "operable_min": OPERABLE_MIN, "n_base_profiles": _base_tree()[1],
            "why": "declared change after the first design was seen: sites and drift limited to operable water"})
        store.mark_done("design_masked", excluded_sites=int((~allowed).sum()))


def fc_posterior(theta, X, Xnet, noise_new, Xrest):
    import jax.numpy as jnp

    return posterior_rest(NU, theta, X, Xnet, np.full(len(Xnet), noise_new), Xrest, jnp.float32)


def design_alt(limit=99):
    design(limit, which="alt")


# Racer B: local windows -------------------------------------------------------------------------------


def windows():
    tr, _, _ = data()
    out = []
    for la in np.arange(22.0, 55.0, 4.0):
        for lo in np.arange(-76.0, 0.0, 8.0):
            d = fc.haversine_km(la, lo, tr["lat"].to_numpy(), tr["lon"].to_numpy())
            inside = np.flatnonzero(d < 1000)
            if len(inside) > 4000:
                inside = inside[np.argsort(d[inside])[:4000]]
            if len(inside) >= 800:
                out.append((la, lo, np.sort(inside)))
    return out


def fit_B(limit=99):
    gpu()
    import jax.numpy as jnp

    tr, _, _ = data()
    X, y = fc.X_of(tr), tr["anom"].to_numpy()
    wins = windows()
    x0, _ = mode_of("fit_A_f64")
    todo = store.pending("fit_B", len(wins))[: int(limit)]
    with gpu_lock():
        for k in todo:
            p.health()
            la, lo, idx = wins[k]
            t0 = time.perf_counter()
            fun = fc.neg(c.make_value_and_grad(NU), X[idx], y[idx], jnp.float64)
            st = fc.lbfgs(fun, x0, None, f"B{k}", max_iter=80, gtol=0.05, max_eval=150, log=lambda *a: None)
            th = np.asarray(st["x"])
            store.save_chunk("fit_B", k, {"centre": np.array([la, lo]), "theta": th, "n": np.array(len(idx)),
                                          "F": np.array(st["f"]), "nev": np.array(st["nev"]),
                                          "seconds": np.array(time.perf_counter() - t0)})
            log("B window", k, (la, lo), len(idx), "nev", int(st["nev"]), fc.msg_of(st), "edges", fc.edges(th),
                round(time.perf_counter() - t0, 1), "s")
            cleanup()
        vram = peak()
    if not store.pending("fit_B", len(wins)):
        g = [store.load_chunk("fit_B", k) for k in range(len(wins))]
        th = np.array([x["theta"] for x in g])
        centres = np.array([x["centre"] for x in g])
        log_sd = 0.5 * np.log(np.exp(2 * th[:, 0]) + np.exp(2 * th[:, 4]))
        raw = fc.log_amp(tr["lat"].to_numpy(), tr["lon"].to_numpy(), centres, log_sd)
        store.save_json("amp_map", {"centres": centres, "log_sd": log_sd, "norm": float(raw.mean()), "bandwidth_km": 500})
        store.save_json("fit_B", {"centres": centres, "theta": th, "n": [int(x["n"]) for x in g],
                                  "evaluations": [int(x["nev"]) for x in g], "seconds": [float(x["seconds"]) for x in g],
                                  "edges": [fc.edges(t) for t in th], "vram": vram, "labels": "measured"})
        store.mark_done("fit_B", windows=len(wins))


def heldout_B():
    gpu()
    import jax.numpy as jnp

    t0 = time.perf_counter()
    tr, te, _ = data()
    X, y = fc.X_of(tr), tr["anom"].to_numpy()
    wins = windows()
    B = store.load_json("fit_B")
    th, centres = np.asarray(B["theta"]), np.asarray(B["centres"])
    groups = te["float_id"].to_numpy()
    mean, var, noise, joint, which = (np.zeros(len(te)), np.zeros(len(te)), np.zeros(len(te)), {}, {})
    Xs_all, yte = fc.X_of(te), te["anom"].to_numpy()
    with gpu_lock():
        for gid in np.unique(groups):
            m = np.flatnonzero(groups == gid)
            d = fc.haversine_km(te["lat"].to_numpy()[m].mean(), te["lon"].to_numpy()[m].mean(), centres[:, 0], centres[:, 1])
            k = int(np.argmin(d))
            idx = wins[k][2]
            fac = fc.make_factor(NU)
            L, a = fac(jnp.asarray(th[k]), jnp.asarray(X[idx]), jnp.asarray(y[idx]))
            mu, v, covs = fc.predict(NU, th[k], L, a, jnp.asarray(X[idx]), Xs_all[m], groups=np.zeros(len(m)))
            mean[m], var[m], noise[m] = mu, v, np.exp(2 * th[k][8])
            joint[gid] = fc.joint_scores(mu, covs, np.exp(2 * th[k][8]), yte[m], np.zeros(len(m)))[0]
            which[gid] = k
        vram = peak()
    out = {"model": "B", "mean": mean, "latent_var": var, "noise_var": noise, "float_id": groups,
           "joint_log_score": joint, "window": which, "student_t": False, "wall_s": time.perf_counter() - t0,
           "vram": vram, "labels": "measured"}
    store.save_json("heldout_B", out)
    store.mark_done("heldout_B")
    log("heldout B rmse", float(np.sqrt(np.mean((yte - mean) ** 2))))


# A': amplitude map ------------------------------------------------------------------------------------


def fit_Afix_f32():
    tr, _, _ = data()
    x0, _ = mode_of("fit_A_f64")
    fit_phase("fit_Afix_f32", fc.X_of(tr, [logD(tr["lat"], tr["lon"])]), tr["anom"].to_numpy(), x0, 32, **F32)


def fit_Afix_f64():
    tr, _, _ = data()
    x0, st = mode_of("fit_Afix_f32")
    fit_phase("fit_Afix_f64", fc.X_of(tr, [logD(tr["lat"], tr["lon"])]), tr["anom"].to_numpy(), x0, 64,
              init_SY=(st["S"], st["Y"]), **F64)


def fit_Afixt():
    tr, _, _ = data()
    x0, _ = mode_of("fit_Afix_f64")
    fit_t("Afixt", [logD(tr["lat"], tr["lon"])], x0, "fit_Afixt")


def fit_Afixt_final():
    tr, _, _ = data()
    fit_t_final("Afixt", [logD(tr["lat"], tr["lon"])], "fit_Afixt")


# Racer C: Vecchia ------------------------------------------------------------------------------------


def fit_C():
    gpu()
    import jax.numpy as jnp

    t0 = time.perf_counter()
    tr, te, _ = data()
    X, y = fc.X_of(tr), tr["anom"].to_numpy()
    m = int(store.load_json("p8_vecchia")["m_for_C"])
    theta, _ = mode_of("fit_A_f64")
    info = {"m": m, "rounds": []}
    with gpu_lock():
        for rnd in range(2):
            p.health()
            t1 = time.perf_counter()
            order = fc.maxmin_order(fc.scaled(X, theta))
            nb = fc.neighbours(fc.scaled(X[order], theta), m)
            Xo, yo = jnp.asarray(X[order]), jnp.asarray(y[order])
            _, vg = fc.make_vecchia(NU)
            nbj = jnp.asarray(nb)

            def fun(th):
                v, g = vg(jnp.asarray(th), Xo, yo, nbj)
                v, g = float(v), np.asarray(g, np.float64)
                return (-v, -g) if np.isfinite(v) else (np.inf, np.zeros(10))

            st = fc.lbfgs(fun, theta, store, f"fit_C_r{rnd}", max_iter=100, gtol=0.02, max_eval=200, log=log,
                          health=p.health)
            theta = np.asarray(st["x"])
            info["rounds"].append({"theta": theta, "F": float(st["f"]), "evaluations": int(st["nev"]),
                                   "message": fc.msg_of(st), "seconds": time.perf_counter() - t1, "edges": fc.edges(theta)})
            log("C round", rnd, info["rounds"][-1])
        # held-out: each float conditioned on the union of its profiles' m nearest training points
        Z = fc.scaled(X, theta)
        tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(Z)
        groups = te["float_id"].to_numpy()
        Xs, yte = fc.X_of(te), te["anom"].to_numpy()
        mean, var, joint = np.zeros(len(te)), np.zeros(len(te)), {}
        for gid in np.unique(groups):
            mm = np.flatnonzero(groups == gid)
            _, j = tree.query(fc.scaled(Xs[mm], theta), k=m)
            U = np.unique(j.ravel())
            fac = fc.make_factor(NU)
            L, a = fac(jnp.asarray(theta), jnp.asarray(X[U]), jnp.asarray(y[U]))
            mu, v, covs = fc.predict(NU, theta, L, a, jnp.asarray(X[U]), Xs[mm], groups=np.zeros(len(mm)))
            mean[mm], var[mm] = mu, v
            joint[gid] = fc.joint_scores(mu, covs, np.exp(2 * theta[8]), yte[mm], np.zeros(len(mm)))[0]
        vram = peak()
    store.save_json("fit_C", {**info, "theta": theta, "natural": fc.natural(theta), "wall_s": time.perf_counter() - t0,
                              "vram": vram, "labels": "measured"})
    store.save_json("heldout_C", {"model": "C", "theta": theta, "mean": mean, "latent_var": var,
                                  "noise_var": float(np.exp(2 * theta[8])), "float_id": groups, "joint_log_score": joint,
                                  "student_t": False, "labels": "measured"})
    store.mark_done("fit_C", m=m, edges=fc.edges(theta))
    store.mark_done("heldout_C")
    log("heldout C rmse", float(np.sqrt(np.mean((yte - mean) ** 2))))


# Failure runs ---------------------------------------------------------------------------------------


FAIL_ROWS = [(n, ratio, prec) for n in (5000, 10000, 20000) for ratio in (1e3, 1e4, 1e5) for prec in (64, 32)]


def failure_runs(limit=4):
    gpu()
    import jax
    import jax.numpy as jnp

    tr, _, _ = data()
    todo = store.pending("failure_runs", len(FAIL_ROWS) + 1)[: int(limit)]
    with gpu_lock():
        for k in todo:
            p.health()
            t0 = time.perf_counter()
            if k == len(FAIL_ROWS):  # Matern 1/2 at the primary mode, full n, f32 against the f64 value
                model = primary()
                theta, X, y, _, _ = model_spec(model)
                v32 = float(c.make_logp(NU)(jnp.asarray(theta), jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32)))
                v64 = float(c.make_logp(NU)(jnp.asarray(theta), jnp.asarray(X), jnp.asarray(y)))
                store.save_chunk("failure_runs", k, {"v32": np.array(v32), "v64": np.array(v64), "n": np.array(len(y))})
                log("failure matern full", v32, v64, v32 - v64)
                continue
            n, ratio, prec = FAIL_ROWS[k]
            d = tr.iloc[np.sort(np.random.default_rng(0).permutation(len(tr))[:n])]
            dt = jnp.float32 if prec == 32 else jnp.float64
            X = jnp.asarray(fc.X_of(d), dt)
            yv = jnp.asarray(d["anom"].to_numpy(), dt)
            theta = np.array([-20.0, 0, 0, 0, 0.0, np.log(1500), np.log(1000), np.log(200),
                              0.5 * np.log(1.0 / ratio), 0.0])

            def run(X, yv):
                K = c.build_K(X, jnp.asarray(theta), "inf")
                L = jnp.linalg.cholesky(K)
                logdet = 2 * jnp.sum(jnp.log(jnp.diag(L).astype(jnp.float64)))
                a = jax.scipy.linalg.cho_solve((L, True), yv)
                quad = jnp.dot(yv.astype(jnp.float64), a.astype(jnp.float64))
                v = jnp.ones(X.shape[0], X.dtype) / np.sqrt(X.shape[0])
                for _ in range(30):
                    v = K @ v
                    lam = jnp.linalg.norm(v)
                    v = v / lam
                return logdet, quad, lam.astype(jnp.float64), jnp.all(jnp.isfinite(L))

            logdet, quad, lam, finite = (np.asarray(z) for z in jax.jit(run)(X, yv))
            store.save_chunk("failure_runs", k, {"n": np.array(n), "ratio": np.array(ratio), "prec": np.array(prec),
                                                 "logdet": np.array(float(logdet)), "quad": np.array(float(quad)),
                                                 "lambda_max": np.array(float(lam)), "finite": np.array(bool(finite)),
                                                 "seconds": np.array(time.perf_counter() - t0)})
            log("failure", n, ratio, prec, float(logdet), float(quad), bool(finite), round(time.perf_counter() - t0, 1))
            cleanup()
    if not store.pending("failure_runs", len(FAIL_ROWS) + 1):
        rows = []
        for k, (n, ratio, prec) in enumerate(FAIL_ROWS):
            if prec == 32:
                a, b = store.load_chunk("failure_runs", k), store.load_chunk("failure_runs", k - 1)
                ld = float(a["logdet"] - b["logdet"]) if bool(a["finite"]) else None
                obj = (-0.5 * float(a["quad"] - b["quad"]) - 0.5 * ld) if ld is not None else None
                rows.append({"n": n, "ratio": ratio, "f32_finite": bool(a["finite"]), "logdet_err": ld,
                             "objective_err": obj, "cond_est": float(b["lambda_max"]) * ratio,
                             "silently_wrong": bool(a["finite"]) and ld is not None and abs(ld) > 1})
        mf = store.load_chunk("failure_runs", len(FAIL_ROWS))
        store.save_json("failure_runs", {"expquad": rows, "matern_full": {"v32": float(mf["v32"]), "v64": float(mf["v64"]),
                                                                          "err": float(mf["v32"] - mf["v64"]), "n": int(mf["n"])},
                                         "eps32": float(np.finfo(np.float32).eps),
                                         "labels": "measured; cond_est = lambda_max x ratio (lambda_min is about the nugget): estimated"})
        store.mark_done("failure_runs")


# Natural experiment -------------------------------------------------------------------------------


def natural_experiment():
    gpu()
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    t0 = time.perf_counter()
    model = primary()
    theta, X, y, new, noise_new = model_spec(model)
    _, _, y25 = data()
    obs, hel = y25[~y25["held"]], y25[y25["held"]]
    ocean = ocean_fn()
    la, lo = np.arange(21.0, 55, 2.0), np.arange(-79.0, 0, 2.0)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    ok = ocean(LA.ravel(), LO.ravel())
    cl, co = LA.ravel()[ok], LO.ravel()[ok]
    days = fc.DAY_2025 + 15.0 + 30.4 * np.arange(12)
    Xo, Xh = new(obs), new(hel)
    Xm = new(np.tile(cl, 12), np.tile(co, 12), xyz=xyz(np.tile(cl, 12), np.tile(co, 12), np.repeat(days, len(cl))))
    with gpu_lock():
        th = jnp.asarray(theta)
        fac = fc.make_factor(NU)
        Xd = jnp.asarray(X, jnp.float32)
        L, a = fac(th, Xd, jnp.asarray(y, jnp.float32))
        Q = np.concatenate([Xo, Xh, Xm])
        mean_q, _, _ = fc.predict(NU, theta, L, a, Xd, Q)
        del L, a
        cleanup()
        P, _ = posterior_rest(NU, theta, X, np.zeros((0, Q.shape[1])), np.zeros(0), Q, jnp.float32)
        no, nh = len(Xo), len(Xh)
        yo = obs["anom"].to_numpy()
        res = {}
        for name, keep in (("with", np.ones(no, bool)), ("without", ~obs["launched_2025"].to_numpy())):
            ki = np.flatnonzero(keep)
            rest = np.r_[no + np.arange(nh), (no + nh) + np.arange(len(Xm))]
            Soo = P[ki][:, ki] + jnp.eye(len(ki)) * noise_new
            Lo = jnp.linalg.cholesky(Soo)
            W = solve_triangular(Lo, P[ki][:, rest], lower=True)
            r = solve_triangular(Lo, jnp.asarray(yo[ki] - mean_q[ki]), lower=True)
            mu = mean_q[rest] + np.asarray(W.T @ r)
            var = np.asarray(jnp.diag(P[rest][:, rest]) - jnp.sum(W * W, 0))
            res[name] = {"mean_h": mu[:nh], "var_h": var[:nh], "mean_m": mu[nh:], "var_m": var[nh:], "n_used": len(ki)}
            del W, Lo, Soo
        vram = peak()
    yh = hel["anom"].to_numpy()
    gid = hel["float_id"].to_numpy()
    store.save_json("natural_experiment", {
        "model": model, "noise_var": noise_new, "float_id": gid, "y": yh, "lat": hel["lat"].to_numpy(), "lon": hel["lon"].to_numpy(),
        **{f"{k}_{n}": v for n, r in res.items() for k, v in r.items() if k.endswith("_h") or k == "n_used"},
        "launched_floats": int(obs.loc[obs["launched_2025"], "float_id"].nunique()),
        "launched_profiles": int(obs["launched_2025"].sum()), "wall_s": time.perf_counter() - t0, "vram": vram,
        "labels": "measured"})
    store.save_array("map2025_mean", res["with"]["mean_m"].reshape(12, -1).astype(np.float32))
    store.save_array("map2025_sd", np.sqrt(np.maximum(res["with"]["var_m"], 0)).reshape(12, -1).astype(np.float32))
    store.save_json("map2025", {"cell_lat": cl, "cell_lon": co, "days": days, "model": model})
    pred = np.mean(res["without"]["var_h"] - res["with"]["var_h"])
    real = np.mean((yh - res["without"]["mean_h"]) ** 2 - (yh - res["with"]["mean_h"]) ** 2)
    store.mark_done("natural_experiment", predicted=float(pred), realised=float(real))
    log("natural experiment predicted", pred, "realised", real)


# Order --------------------------------------------------------------------------------------------


def stage_info(name):
    return next((s for s in store.stages() if s["stage"] == name), None)


def needed(stage):
    if stage in ("fit_At", "fit_At_final", "heldout_At"):
        info = stage_info("fit_A_f64")
        return bool(info and info.get("edges"))
    if stage in ("fit_Afixt", "fit_Afixt_final", "heldout_Afixt"):
        info = stage_info("heldout_Afix")
        return bool(info and info.get("share_abs_z_over_3", 0) > 0.01)
    return True


ORDER = ["data_fingerprint", "p8", "p10", "fit_A_f32", "fit_A_f64", "heldout_A", "fit_At", "fit_At_final",
         "heldout_At", "hessian", "sobol", "design", "fit_B", "heldout_B", "fit_Afix_f32", "fit_Afix_f64",
         "heldout_Afix", "fit_Afixt", "fit_Afixt_final", "heldout_Afixt", "maps", "design_alt", "fit_C",
         "failure_runs", "natural_experiment"]
DONE_NAME = {"fit_At": "fit_At_rounds", "fit_Afixt": "fit_Afixt_rounds", "p8": "p8", "p10": "p10"}


def following():
    for s in ORDER:
        if not needed(s):
            continue
        if not store.done(DONE_NAME.get(s, s)):
            if s == "heldout_At" and not store.done("fit_At_final"):
                continue
            print(s)
            return
    print("none")


JOBS = {"next": following, "data_fingerprint": data_fingerprint, "p8": p8, "p10": p10, "fit_A_f32": fit_A_f32,
        "fit_A_f64": fit_A_f64, "heldout_A": lambda: heldout("A"), "fit_At": fit_At, "fit_At_final": fit_At_final,
        "heldout_At": lambda: heldout("At"), "hessian": hessian, "sobol": sobol, "design": design, "fit_B": fit_B,
        "heldout_B": heldout_B, "fit_Afix_f32": fit_Afix_f32, "fit_Afix_f64": fit_Afix_f64,
        "heldout_Afix": lambda: heldout("Afix"), "fit_Afixt": fit_Afixt, "fit_Afixt_final": fit_Afixt_final,
        "heldout_Afixt": lambda: heldout("Afixt"), "maps": maps, "design_alt": design_alt, "fit_C": fit_C,
        "failure_runs": failure_runs, "natural_experiment": natural_experiment,
        "design_masked": design_masked}  # not in ORDER: a declared rerun after FULL, started by hand

if __name__ == "__main__":
    job = sys.argv[1]
    if job == "next":
        following()
        sys.exit(0)
    log("job", sys.argv[1:], "gates", sha256_file(store.path / "gates_2026-11-03c.json")[:16], "health", p.health())
    t0 = time.perf_counter()
    JOBS[job](*sys.argv[2:])
    store.save_json(f"wall_{job}_{int(time.time())}", {"job": job, "args": sys.argv[2:], "wall_s": time.perf_counter() - t0,
                                                      "started": time.strftime(FORMAT, time.gmtime(time.time() - (time.perf_counter() - t0))),
                                                      "vram": peak() if "jax.numpy" in sys.modules else None})
    log(f"job done in {time.perf_counter() - t0:.1f} s; health {p.health()}")
