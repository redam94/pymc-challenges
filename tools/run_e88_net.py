"""E88 network pilots P4, P5, P6 and P8': a DECODE-style U-Net in equinox, simulator inside the JIT.

    .venv/bin/python tools/run_e88_net.py p4 --device gpu|cpu
    .venv/bin/python tools/run_e88_net.py train --name p5 --steps 5000 [--base 32]
    .venv/bin/python tools/run_e88_net.py validate --name p5 [--base 32]
    .venv/bin/python tools/run_e88_net.py p6 --name p5

Written fresh from Speiser et al. 2021 (Nat Methods 18:1082); no DECODE code is copied. Gates are in
results/E88/pilot_gates.md. Plain JAX: this script imports neither PyMC nor blackjax. Training saves
weights, Adam state and the loss log through bigrun every --ckpt steps; keys are fold_in(base, step), so a
resumed run continues the same sequence.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import erf, logsumexp
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

jax.config.update("jax_enable_x64", False)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_e88_data as b  # noqa: E402

from pymc_challenges import bigrun  # noqa: E402

CROP, MARGIN, MAX_EMITTERS, BATCH = 64, 2, 80, 16
PIXEL_UM = 0.127
AREA_UM2 = ((CROP + 2 * MARGIN) * PIXEL_UM) ** 2
DENSITY = (0.02, 0.8)                           # per um^2 per frame, log-uniform
BG = (60.0, 250.0)                              # photons per pixel, flat per crop
LOGN = (np.log(1500.0), 0.6)                    # E58's brightness
LOGS = (np.log(1.05), 0.2)                      # PSF sd, px
READ_SD_COUNTS = 5.0 / 13.86
SCALE = 100.0                                   # effective counts -> network input
THRESHOLD = 0.5
EXPECT_OFFLINE, EXPECT_MC = [10, 26], 1


def guard():
    state = bigrun.online_core_check()
    if state["offline_cpus"] != EXPECT_OFFLINE or state["machine_checks"] != EXPECT_MC:
        raise SystemExit(f"SAFETY STOP: {state}, expected offline {EXPECT_OFFLINE}, checks {EXPECT_MC}")
    return state


# Simulator -----------------------------------------------------------------------------------------


def wilson_hilferty(key, mu):
    """Fixed-cost draw of effective counts for mean ``mu`` photons in the pixel.

    EMCCD output scaled by 2g has mean and variance mu / 2, so it is treated as Gamma(mu / 2, 1), which
    the cube-root transform of a normal draw approximates without a rejection loop."""
    k1, k2 = jax.random.split(key)
    k = jnp.maximum(mu / 2, 0.5)
    c = 1 - 1 / (9 * k) + jax.random.normal(k1, mu.shape) / (3 * jnp.sqrt(k))
    x = k * jnp.maximum(c, 0.0) ** 3 + READ_SD_COUNTS * jax.random.normal(k2, mu.shape)
    return jnp.maximum(jnp.round(x), 0.0)


def pixel_integral(u, s, n=CROP):
    edges = jnp.arange(n)
    lo = (edges - 0.5 - u[..., None]) / (jnp.sqrt(2.0) * s[..., None])
    hi = (edges + 0.5 - u[..., None]) / (jnp.sqrt(2.0) * s[..., None])
    return 0.5 * (erf(hi) - erf(lo))


def simulate_crop(key):
    """One 3-frame crop in effective counts, and the middle frame's truth (padded to MAX_EMITTERS)."""
    ks = jax.random.split(key, 8)
    density = jnp.exp(jax.random.uniform(ks[0], minval=jnp.log(DENSITY[0]), maxval=jnp.log(DENSITY[1])))
    q = jnp.clip(density * AREA_UM2 / MAX_EMITTERS, 0.0, 1.0)
    on = jax.random.bernoulli(ks[1], q, (3, MAX_EMITTERS))
    xy = jax.random.uniform(ks[2], (3, MAX_EMITTERS, 2), minval=-0.5 - MARGIN, maxval=CROP - 0.5 + MARGIN)
    N = jnp.exp(LOGN[0] + LOGN[1] * jax.random.normal(ks[3], (3, MAX_EMITTERS)))
    s = jnp.exp(LOGS[0] + LOGS[1] * jax.random.normal(ks[4], ())) * jnp.ones((3, MAX_EMITTERS))
    bg = jax.random.uniform(ks[5], minval=BG[0], maxval=BG[1])
    Ex, Ey = pixel_integral(xy[..., 0], s), pixel_integral(xy[..., 1], s)
    mu = bg + jnp.einsum("fm,fmy,fmx->fyx", on * N, Ey, Ex)
    z = wilson_hilferty(ks[6], mu)
    inside = (xy[1] >= -0.5).all(-1) & (xy[1] <= CROP - 0.5).all(-1) & on[1]
    return z, dict(xy=xy[1], logN=jnp.log(N[1] / 1500.0), mask=inside, bg=bg / 2 / SCALE)


simulate_batch = jax.vmap(simulate_crop)


# Network -------------------------------------------------------------------------------------------


class Block(eqx.Module):
    a: eqx.nn.Conv2d
    c: eqx.nn.Conv2d

    def __init__(self, nin, nout, key):
        k1, k2 = jax.random.split(key)
        self.a = eqx.nn.Conv2d(nin, nout, 3, padding=1, key=k1)
        self.c = eqx.nn.Conv2d(nout, nout, 3, padding=1, key=k2)

    def __call__(self, x):
        return jax.nn.elu(self.c(jax.nn.elu(self.a(x))))


class UNet(eqx.Module):
    e1: Block
    e2: Block
    e3: Block
    d2: Block
    d1: Block
    head: eqx.nn.Conv2d

    def __init__(self, base, key):
        ks = jax.random.split(key, 6)
        self.e1 = Block(3, base, ks[0])
        self.e2 = Block(base, 2 * base, ks[1])
        self.e3 = Block(2 * base, 4 * base, ks[2])
        self.d2 = Block(6 * base, 2 * base, ks[3])
        self.d1 = Block(3 * base, base, ks[4])
        self.head = eqx.nn.Conv2d(base, 8, 1, key=ks[5])

    def __call__(self, x):
        def down(u):
            return eqx.nn.AvgPool2d(2, 2)(u)

        def up(u):
            return jax.image.resize(u, (u.shape[0], 2 * u.shape[1], 2 * u.shape[2]), "nearest")

        a = self.e1(x)
        c = self.e2(down(a))
        e = self.e3(down(c))
        u = self.d2(jnp.concatenate([up(e), c]))
        return self.head(self.d1(jnp.concatenate([up(u), a])))


def new_model(base, key):
    model = UNet(base, key)
    return eqx.tree_at(lambda m: m.head.bias, model, model.head.bias.at[0].set(-4.0))


def unpack(out):
    p = jnp.clip(jax.nn.sigmoid(out[0]), 1e-6, 1 - 1e-6)
    return dict(p=p, dx=1.5 * jnp.tanh(out[1]), dy=1.5 * jnp.tanh(out[2]),
                sx=jnp.exp(jnp.clip(out[3], -6.0, 1.0)), sy=jnp.exp(jnp.clip(out[4], -6.0, 1.0)),
                logN=out[5], sN=jnp.exp(jnp.clip(out[6], -6.0, 2.0)), bg=out[7])


def crop_loss(model, z, truth):
    o = unpack(model(z / SCALE))
    yy, xx = jnp.mgrid[:CROP, :CROP].astype(jnp.float32)
    total = o["p"].sum()
    var = (o["p"] * (1 - o["p"])).sum() + 1e-3
    count = truth["mask"].sum()
    l_count = (count - total) ** 2 / (2 * var) + 0.5 * jnp.log(2 * jnp.pi * var)
    logw = (jnp.log(o["p"]) - jnp.log(total)).ravel()
    mx, my = (xx + o["dx"]).ravel(), (yy + o["dy"]).ravel()
    sx, sy, sN = o["sx"].ravel(), o["sy"].ravel(), o["sN"].ravel()
    mN = o["logN"].ravel()

    def one(t_xy, t_logN):
        lp = (-0.5 * ((t_xy[0] - mx) / sx) ** 2 - jnp.log(sx) - 0.5 * ((t_xy[1] - my) / sy) ** 2 - jnp.log(sy)
              - 0.5 * ((t_logN - mN) / sN) ** 2 - jnp.log(sN))
        return logsumexp(logw + lp)

    l_loc = -jnp.sum(truth["mask"] * jax.vmap(one)(truth["xy"], truth["logN"]))
    l_bg = jnp.mean((o["bg"] - truth["bg"]) ** 2)
    return l_count + l_loc + 100.0 * l_bg


def make_opt(steps):
    schedule = optax.warmup_cosine_decay_schedule(0.0, 3e-4, 200, max(steps, 400), 3e-5)
    return optax.chain(optax.clip_by_global_norm(30.0), optax.adam(schedule))


def make_step(opt):
    @eqx.filter_jit
    def step(params, static, state, key):
        z, truth = simulate_batch(jax.random.split(key, BATCH))

        def loss(p):
            m = eqx.combine(p, static)
            return jnp.mean(jax.vmap(crop_loss, in_axes=(None, 0, 0))(m, z, truth))

        value, grads = jax.value_and_grad(loss)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, value

    return step


def setup(base, steps, seed=88):
    key = jax.random.key(seed)
    params, static = eqx.partition(new_model(base, jax.random.fold_in(key, 1)), eqx.is_inexact_array)
    opt = make_opt(steps)
    return key, params, static, opt, opt.init(params), make_step(opt)


def n_weights(params):
    return sum(leaf.size for leaf in jax.tree.leaves(params))


def to_arrays(tree):
    return {f"a{i:04d}": np.asarray(leaf) for i, leaf in enumerate(jax.tree.leaves(tree))}


def from_arrays(tree, arrays):
    leaves, treedef = jax.tree.flatten(tree)
    return jax.tree.unflatten(treedef, [jnp.asarray(arrays[f"a{i:04d}"], leaf.dtype) for i, leaf in enumerate(leaves)])


def train(name, steps, base, ckpt, log_every=50):
    store = bigrun.Store("E88")
    state_seen = guard()
    with bigrun.gpu_lock():
        key, params, static, opt, opt_state, step = setup(base, steps)
        device = str(jax.devices()[0])
        tree = (params, opt_state)
        start, log, log_steps = 0, [], []
        at, saved = store.resume(f"ckpt_{name}")
        if at is not None:
            tree = from_arrays(tree, saved)
            start = int(saved["step"])
            log, log_steps = saved["log"].tolist(), saved["log_steps"].tolist()
            print(f"resumed from step {start}", flush=True)
        params, opt_state = tree
        print(f"{name}: base {base}, {n_weights(params)} weights, {device}, steps {start} to {steps}, {state_seen}", flush=True)
        t0 = time.time()
        for s in range(start, steps):
            params, opt_state, value = step(params, static, opt_state, jax.random.fold_in(key, s))
            if (s + 1) % log_every == 0:
                v = float(value)
                if not np.isfinite(v):
                    raise SystemExit(f"NaN loss at step {s + 1}")
                log.append(v)
                log_steps.append(s + 1)
                print(f"step {s + 1} loss {v:.3f} {(s + 1 - start) / (time.time() - t0):.1f} steps/s", flush=True)
            if (s + 1) % ckpt == 0 or s + 1 == steps:
                jax.block_until_ready((params, opt_state))
                store.save_chunk(f"ckpt_{name}", s + 1,
                                 to_arrays((params, opt_state)) | dict(step=np.array(s + 1), log=np.array(log),
                                                                       log_steps=np.array(log_steps)), keep=3)
                guard()
        seconds = time.time() - t0
    store.save_json(f"train_{name}", dict(steps=steps, start=start, base=base, seconds=seconds, weights=n_weights(params),
                                          device=device, last_loss=log[-1] if log else None))
    print(f"{name}: done {steps - start} steps in {seconds:.0f} s", flush=True)


def load_model(name, base, steps):
    store = bigrun.Store("E88")
    _, params, static, _, opt_state, _ = setup(base, steps)
    at, saved = store.resume(f"ckpt_{name}")
    if at is None:
        raise SystemExit(f"no checkpoint for {name}")
    params, _ = from_arrays((params, opt_state), saved)
    return params, static, at


@eqx.filter_jit
def forward(params, static, x):
    model = eqx.combine(params, static)
    return jax.vmap(lambda u: jnp.stack(list(unpack(model(u)).values())))(x / SCALE)


def predict(params, static, stack, batch=8):
    """(T, 3, H, W) counts -> (T, 8, H, W) maps as NumPy."""
    return np.concatenate([np.asarray(forward(params, static, jnp.asarray(stack[i:i + batch], jnp.float32)))
                           for i in range(0, len(stack), batch)])


def detections(maps, threshold=THRESHOLD):
    """Local maxima of p with 3 x 3 summed p over threshold; position is the p-weighted mean over 3 x 3."""
    p = maps[:, 0]
    summed = ndi.uniform_filter(p, size=(1, 3, 3)) * 9
    peak = (p == ndi.maximum_filter(p, size=(1, 3, 3))) & (summed >= threshold)
    t, r, c = np.nonzero(peak)
    T, H, W = p.shape
    xs, ys, sds = [], [], []
    for tt, rr, cc in zip(t, r, c):
        a, e, f, g = max(rr - 1, 0), min(rr + 2, H), max(cc - 1, 0), min(cc + 2, W)
        yy, xx = np.mgrid[a:e, f:g]
        w = p[tt, a:e, f:g]
        w = w / w.sum()
        xs.append(np.sum(w * (xx + maps[tt, 1, a:e, f:g])))
        ys.append(np.sum(w * (yy + maps[tt, 2, a:e, f:g])))
        sds.append(np.hypot(maps[tt, 3, rr, cc], maps[tt, 4, rr, cc]) / np.sqrt(2))
    return t, np.c_[xs, ys], np.array(sds)


def exact_counts(mu, rng, gain=6.93, offset=457.0, scale=13.86, read_sd=5.0):
    n = rng.poisson(mu)
    charge = np.where(n > 0, rng.gamma(np.maximum(n, 1), gain), 0.0)
    adu = np.round(charge + offset + rng.normal(0.0, read_sd, n.shape))
    return np.maximum(np.round((adu - offset) / scale), 0.0)


def movie(rng, T, density, N, bg=110.0, sig=1.05, H=200, lognormal=False):
    """T + 2 independent frames with exact EMCCD noise; truth for frames 1..T."""
    area = (H * PIXEL_UM) ** 2
    frames, truth = [], []
    for f in range(T + 2):
        n = rng.poisson(density * area)
        xy = rng.uniform(-0.5, H - 0.5, (n, 2))
        brightness = rng.lognormal(np.log(1500.0), 0.6, n) if lognormal else np.full(n, N)
        s = np.full(n, sig)
        Ex, Ey = b.pixel_gauss(xy[:, 0], s, H)[0], b.pixel_gauss(xy[:, 1], s, H)[0]
        frames.append(exact_counts(bg + np.einsum("m,my,mx->yx", brightness, Ey, Ex), rng))
        truth.append(np.c_[np.full(n, f), xy])
    return np.array(frames, np.float32), np.vstack(truth)


def triples(z):
    return np.stack([z[:-2], z[1:-1], z[2:]], axis=1)


def score(t_est, xy_est, truth, T, tol):
    tp, fp, fn, err = 0, 0, 0, []
    for f in range(T):
        a = truth[truth[:, 0] == f + 1, 1:3]
        e = xy_est[t_est == f]
        if not len(a) or not len(e):
            fn += len(a)
            fp += len(e)
            continue
        D = np.hypot(a[:, None, 0] - e[None, :, 0], a[:, None, 1] - e[None, :, 1])
        r, c = linear_sum_assignment(D)
        ok = D[r, c] <= tol
        tp += int(ok.sum())
        fn += len(a) - int(ok.sum())
        fp += len(e) - int(ok.sum())
        err.append(e[c[ok]] - a[r[ok]])
    err = np.vstack(err) if err else np.zeros((0, 2))
    return dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), rmse_px_axis=float(np.sqrt(np.mean(err**2))) if len(err) else None,
                bias_px=err.mean(0).tolist() if len(err) else None, matched=tp)


def crlb_axis(N, bg, sig=1.05, P=11):
    th = np.array([[np.log(bg), np.log(sig), P / 2 - 0.5, P / 2 - 0.5, np.log(N)]])
    best = []
    for shift in np.linspace(-0.4, 0.4, 5):
        t = th.copy()
        t[0, 2] += shift
        t[0, 3] += 0.3 * shift
        mu, J = b.model_and_jacobian(t, 1, P)
        I = b.info(J / 2, 2 / mu)              # effective counts: mean mu / 2
        best.append(np.linalg.inv(I)[0, 2, 2])
    return float(np.sqrt(np.mean(best)))


def validate(name, base, steps):
    store = bigrun.Store("E88")
    guard()
    with bigrun.gpu_lock():
        params, static, at = load_model(name, base, steps)
        rng = np.random.default_rng(880)
        out = dict(checkpoint_step=at, crlb_px=crlb_axis(2000.0, 110.0))
        out["crlb_nm"] = out["crlb_px"] * 127
        rows = {"sparse_2000": dict(density=0.089, N=2000.0), "sparse_e58_brightness": dict(density=0.089, N=0, lognormal=True),
                "dense_2000": dict(density=0.70, N=2000.0), "dense_e58_brightness": dict(density=0.70, N=0, lognormal=True)}
        for label, cfg in rows.items():
            z, truth = movie(rng, 100, **cfg)
            maps = predict(params, static, triples(z))
            if not np.isfinite(maps).all():
                raise SystemExit(f"NaN in outputs, {label}")
            t, xy, sd = detections(maps)
            row = {f"tol{tol}": score(t, xy, truth, 100, tol) for tol in (1.0, 2.0)}
            mid = truth[(truth[:, 0] >= 1) & (truth[:, 0] <= 100)]
            row |= dict(true=len(mid), found=len(t), sum_p=float(maps[:, 0].sum()), median_sd_px=float(np.median(sd)) if len(sd) else None)
            if row["tol2.0"]["rmse_px_axis"]:
                row["rmse_over_crlb"] = row["tol2.0"]["rmse_px_axis"] / out["crlb_px"]
            out[label] = row
            print(label, b.rounded(row), flush=True)
        s = out["sparse_2000"]
        out["go"] = bool(s["tol2.0"]["recall"] >= 0.90 and s["rmse_over_crlb"] <= 1.3)
    store.save_json(f"p5_{name}", b.rounded(out))
    print("verdict", out["go"], out["crlb_nm"], flush=True)


def p4(device, steps):
    store = bigrun.Store("E88")
    guard()
    report = dict(device=device, steps=steps, cpus=sorted(os.sched_getaffinity(0)))

    def run():
        key, params, static, opt, state, step = setup(32, 40000)
        report["weights"] = n_weights(params)
        t0 = time.perf_counter()
        for s in range(3):
            params, state, value = step(params, static, state, jax.random.fold_in(key, s))
        jax.block_until_ready((params, state, value))
        report["compile_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for s in range(3, 3 + steps):
            params, state, value = step(params, static, state, jax.random.fold_in(key, s))
        jax.block_until_ready((params, state, value))
        seconds = time.perf_counter() - t0
        report.update(seconds=seconds, steps_per_s=steps / seconds, finite=bool(np.isfinite(float(value))),
                      loss=float(value), weight_sum=float(sum(jnp.sum(x) for x in jax.tree.leaves(params))),
                      jax_device=str(jax.devices()[0]))

    if device == "gpu":
        with bigrun.gpu_lock():
            run()
            report["vram"] = bigrun.vram_high_water()
            wh = wh_check()
    else:
        run()
        wh = None
    store.save_json(f"p4_{device}", b.rounded(report | dict(wilson_hilferty=wh)))
    print(b.rounded(report), wh, flush=True)


def wh_check(n=2_000_000):
    """Total variation, on the grid of effective counts, of the JAX draw against the exact chain."""
    rng = np.random.default_rng(884)
    out = {}
    for mu in (20.0, 88.0, 136.0, 2000.0):
        exact = exact_counts(np.full(n, mu), rng, gain=6.93, offset=457.0, scale=13.86)
        ours = np.asarray(wilson_hilferty(jax.random.key(int(mu)), jnp.full((n,), mu, jnp.float32)))
        exact2 = exact_counts(np.full(n, mu), rng, gain=6.93, offset=457.0, scale=13.86)
        hi = int(max(exact.max(), ours.max())) + 2
        pe, po, p2 = (np.bincount(a.astype(int), minlength=hi) / n for a in (exact, ours, exact2))
        out[str(int(mu))] = dict(tv=0.5 * float(np.abs(pe - po).sum()), tv_floor=0.5 * float(np.abs(pe - p2).sum()))
    return out


def p6(name, base, steps):
    store = bigrun.Store("E88")
    guard()
    cal = store.load_json("p3_line")["r7"]     # one scale for the network; P3's per-block values are not physical
    stack = b.open_stack(b.RAW_DIR / b.TIFFS["r7"][0])
    centres = np.r_[np.arange(1, 168), np.arange(7000, 7167), np.arange(14800, 14966)]
    t0 = time.perf_counter()
    idx = np.stack([centres - 1, centres, centres + 1], 1)
    z = np.maximum(np.round((np.asarray(stack["px"][idx.ravel()], np.float32) - cal["offset"]) / cal["slope"]), 0.0)
    z = z.reshape(len(centres), 3, b.SIDE, b.SIDE)
    read_s = time.perf_counter() - t0
    with bigrun.gpu_lock():
        params, static, at = load_model(name, base, steps)
        report = dict(frames=len(centres), read_s=read_s, checkpoint_step=at)
        for batch in (8, 32):
            forward(params, static, jnp.zeros((batch, 3, b.SIDE, b.SIDE), jnp.float32)).block_until_ready()
            t0 = time.perf_counter()
            total = jnp.zeros(())
            outs = []
            for i in range(0, len(z), batch):
                x = z[i:i + batch]
                if len(x) < batch:
                    continue
                y = forward(params, static, jnp.asarray(x, jnp.float32))
                total = total + jnp.sum(y)
                outs.append(y)
            total.block_until_ready()
            seconds = time.perf_counter() - t0
            used = sum(len(o) for o in outs)
            report[f"batch{batch}"] = dict(frames_per_s=used / seconds, seconds=seconds, used=used, checksum=float(total))
            maps = np.concatenate([np.asarray(o) for o in outs])
        report["vram"] = bigrun.vram_high_water()
    p = maps[:, 0]
    yy, xx = np.mgrid[:b.SIDE, :b.SIDE]
    trim = (yy >= 8) & (yy < b.SIDE - 8) & (xx >= 8) & (xx < b.SIDE - 8)
    ids = (yy % 8) * 8 + xx % 8
    grid = np.bincount(ids[trim], p.mean(0)[trim], 64) / np.bincount(ids[trim], minlength=64)
    t, xy, sd = detections(maps)
    report |= dict(grid_max_over_min=float(grid.max() / grid.min()), sum_p_per_frame=float(p.sum() / len(p)),
                   found_per_frame=len(t) / len(p), finite=bool(np.isfinite(maps).all()),
                   grid_mod2=float(np.bincount(((yy % 2) * 2 + xx % 2)[trim], p.mean(0)[trim], 4).max()
                                   / np.bincount(((yy % 2) * 2 + xx % 2)[trim], p.mean(0)[trim], 4).min()))
    store.save_json(f"p6_{name}", b.rounded(report))
    print(b.rounded(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=("p4", "train", "validate", "p6"))
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--timing-steps", type=int, default=500)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--ckpt", type=int, default=1000)
    parser.add_argument("--name", default="p5")
    args = parser.parse_args()
    if args.command == "p4":
        p4(args.device, args.timing_steps)
    elif args.command == "train":
        train(args.name, args.steps, args.base, args.ckpt)
    elif args.command == "validate":
        validate(args.name, args.base, args.steps)
    else:
        p6(args.name, args.base, args.steps)


if __name__ == "__main__":
    sys.exit(main())
