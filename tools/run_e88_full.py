"""E88 FULL: networks A and B, inference on both recordings, the section 7 checks and the decision.

    taskset -c 14,15 .venv/bin/python tools/memguard.py --max-gb 16 -- .venv/bin/python tools/run_e88_full.py STAGE

Stages, in order: psf_bg train_a validate_a coverage_a infer_a train_b validate_b coverage_b infer_b
simbase infer_sim audit_nuts paired halo replicate curves frc_split ood532 summary. Each one skips if it is
marked done and resumes from its chunks otherwise, so running the list again resumes the job.

The gates and every definition are in results/E88/full_gates.md, written before any of this ran. The stored
classical tables are reused, never rerun. Plain JAX in the network stages; PyMC and nutpie only in audit_nuts,
which runs with JAX_PLATFORMS=cpu. Every stage appends its times, device and cores to the file named by
E88_BUSY_LOG, and every held GPU lock too, so that another job's timings can be checked against them.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import contextlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_e88_data as b  # noqa: E402
import run_e88_classical as c  # noqa: E402
import run_e88_net as n  # noqa: E402

from pymc_challenges import bigrun  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SMOKE = os.environ.get("E88_SMOKE") == "1"
ROOT = os.environ.get("E88_STORE_ROOT") or None
BUSY = os.environ.get("E88_BUSY_LOG")
SCRATCH = Path(os.environ.get("E88_SCRATCH", "/tmp"))
CORES = "14,15"
WORKERS = 2
BASE = 32
STEPS = dict(a=4, b=2) if SMOKE else dict(a=40000, b=12000)
CKPT = 2 if SMOKE else 1000
SEEDS = dict(a=88, b=89)
CHUNK_FRAMES = 6 if SMOKE else b.BLOCK          # frames read per chunk (smoke only reads the head of a chunk)
BATCH = 2 if SMOKE else 10
BOOTS = 8 if SMOKE else 400
SPOTS = 4 if SMOKE else 200
PSF_SPOTS, WIDTH_SPOTS = (400, 100) if SMOKE else (20000, 2000)
DRAWS = 2 if SMOKE else c.DRAWS
NETS = ("a", "b")
CALS = ("", "line")                             # "" is P3's written action, per block; "line" the sensitivity
RECS = ("r7", "r32")
GRID_PHOTONS, GRID_DENSITY = ((2000.0,), (0.089,)) if SMOKE else ((500.0, 1000.0, 2000.0, 4000.0), (0.02, 0.089, 0.3, 0.7, 1.5))
COVER_T = 4 if SMOKE else 40
REP_WINDOWS = (1000, 7000, 13000)
REP_T = 100
LINEUP = 9
STAY = 0.5
R_MAX, DR = 12.0, 0.25
HALO_BRIGHT, HALO_FAINT = 2000.0, 1000.0
HALO_EDGES = np.arange(0.0, 1550.0, 50.0)
RING, FAR = (301.0, 462.0), (1000.0, 1500.0)

if ROOT:                                        # smoke test: another store, every module, no GPU lock
    _Store = bigrun.Store
    bigrun.Store = lambda name, root=None: _Store(name, root=ROOT)
if SMOKE:
    bigrun.gpu_lock = contextlib.nullcontext


def get_store():
    return bigrun.Store("E88", root=ROOT)


def busy(job, start, end, device, note=""):
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(start))} {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(end))} "
            f"device={device} cores={CORES} job={job} {note}".rstrip())
    print("busy:", line, flush=True)
    if BUSY:
        with open(BUSY, "a") as f:
            f.write(line + "\n")


@contextlib.contextmanager
def held(job):
    """gpu_lock, with the wait and the held span in busy.log."""
    asked = time.time()
    with bigrun.gpu_lock():
        got = time.time()
        print(f"{job}: GPU lock after {got - asked:.0f} s", flush=True)
        try:
            yield got - asked
        finally:
            busy(f"{job}:gpu_lock_held", got, time.time(), "gpu", f"waited_s={got - asked:.0f}")


def stage(name, device):
    """Skip a finished stage; guard and time the rest."""

    def wrap(fn):
        def run(*args, **kwargs):
            store = get_store()
            if store.done(name):
                print(f"{name}: already done", flush=True)
                return None
            state = n.guard()
            started = time.time()
            print(f"{name}: start, {state}, cpus {sorted(os.sched_getaffinity(0))}", flush=True)
            try:
                out = fn(store, *args, **kwargs)
            finally:
                busy(name, started, time.time(), device)
            n.guard()
            store.mark_done(name, seconds=time.time() - started)
            print(f"{name}: done in {time.time() - started:.0f} s", flush=True)
            return out

        return run

    return wrap


# PSF and background, recording 7 only ---------------------------------------------------------------


def frames_z(stack, idx, cal):
    """Effective counts for frame numbers ``idx``, each with its own block's calibration."""
    idx = np.asarray(idx)
    block = np.minimum(idx // b.BLOCK, c.CHUNKS - 1)
    off = np.asarray(cal["offset"], float)[block].reshape(-1, 1, 1)
    slope = np.asarray(cal["slope"], float)[block].reshape(-1, 1, 1)
    return np.maximum(np.round((np.asarray(stack["px"][idx], np.float32) - off) / slope), 0.0).astype(np.float32)


def isolated_spots():
    chosen, _, blobs = c.load_rec("r7")
    keep = (chosen["K"] == 1) & (chosen["N"] >= 2000) & (chosen["sd"] * c.PIXEL_NM < 30)
    keep &= (chosen["xy"].min(1) >= R_MAX) & (chosen["xy"].max(1) <= b.SIDE - 1 - R_MAX)
    order = np.argsort(blobs["t"], kind="stable")
    t, xy = blobs["t"][order], blobs["xy"][order]
    cuts = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]
    ids = blobs["id"][order]
    isolated = []
    for a, e in zip(cuts[:-1], cuts[1:]):
        d = np.hypot(*(xy[a:e, None] - xy[a:e][None]).transpose(2, 0, 1))
        np.fill_diagonal(d, 99)
        isolated.append(ids[a:e][d.min(1) >= R_MAX])
    keep &= np.isin(chosen["blob"], np.concatenate(isolated))
    rows = np.nonzero(keep)[0]
    rng = np.random.default_rng(885)
    rows = rng.choice(rows, min(PSF_SPOTS, len(rows)), replace=False)
    return chosen["t"][rows], chosen["xy"][rows], chosen["N"][rows], int(keep.sum())


def radial_rows(t, xy, N, cal, stack):
    """Per spot: distances, excess in photons / N, inside 12 px, with a local background."""
    side = 2 * int(R_MAX) + 1
    grid = np.arange(side)
    rr, ee, order = [], [], np.argsort(t)
    for f, chunk in zip(*np.unique(t[order], return_index=True)):
        frame = frames_z(stack, [f], cal)[0]
        for i in order[chunk:]:
            if t[i] != f:
                break
            r0, c0 = int(round(xy[i, 1])) - int(R_MAX), int(round(xy[i, 0])) - int(R_MAX)
            patch = frame[r0:r0 + side, c0:c0 + side]
            dy, dx = grid[:, None] + r0 - xy[i, 1], grid[None, :] + c0 - xy[i, 0]
            r = np.hypot(dx, dy)
            ring = (r >= 9) & (r <= 12)
            rr.append(r.ravel())
            ee.append(((patch - np.median(patch[ring])) * 2 / N[i]).ravel())
    return np.array(rr), np.array(ee)


def stack_profile(r, e):
    bins = np.minimum((r / DR).astype(int), int(R_MAX / DR))
    total = np.bincount(bins.ravel(), e.ravel(), int(R_MAX / DR) + 1)
    count = np.bincount(bins.ravel(), minlength=int(R_MAX / DR) + 1)
    return total / np.maximum(count, 1)


def normalise(profile):
    """Profile at bin centres -> lookup at r = k DR, zero past 9 px, summing to 1 over integer pixels."""
    centres = (np.arange(len(profile)) + 0.5) * DR
    knots = np.arange(int(R_MAX / DR) + 1) * DR
    f = np.interp(knots, centres, profile)
    f[knots >= 9.0] = 0.0
    f = np.maximum(f, 0.0)
    g = np.arange(-15, 16)
    area = lookup_np(np.hypot(*np.meshgrid(g, g)), f).sum()
    return f / area


def lookup_np(r, f):
    u = np.clip(r / DR, 0, len(f) - 1.001)
    i = u.astype(int)
    w = u - i
    return f[i] * (1 - w) + f[i + 1] * w


@stage("psf_bg", "cpu")
def psf_bg(store):
    cal = stored_cal(store, "r7", "")
    stack = b.open_stack(b.RAW_DIR / b.TIFFS["r7"][0])
    t, xy, N, pool = isolated_spots()
    r, e = radial_rows(t, xy, N, cal, stack)
    print(f"{len(t)} spots of {pool}", flush=True)
    f = normalise(stack_profile(r, e))
    rng = np.random.default_rng(885)
    boots = []
    for _ in range(8 if SMOKE else 200):
        pick = rng.integers(0, len(t), len(t))
        boots.append(normalise(stack_profile(r[pick], e[pick])))
    boots = np.array(boots)
    # defocus: per-spot width scale, least squares of the stacked profile on the inner 5 px
    sub = rng.choice(len(t), min(WIDTH_SPOTS, len(t)), replace=False)
    grid = np.exp(np.linspace(np.log(0.6), np.log(1.7), 45))
    sse = []
    for w in grid:
        pred = lookup_np(r[sub] / w, f) / w**2
        sse.append(np.sum(np.where(r[sub] < 5, (e[sub] - pred) ** 2, 0.0), axis=1))
    width = grid[np.argmin(np.array(sse), axis=0)]
    log_w_sd = float(np.std(np.log(width)))
    knots = np.arange(len(f)) * DR
    gauss = np.exp(-knots**2 / (2 * 1.05**2)) / (2 * np.pi * 1.05**2)
    wing = float(2 * np.pi * np.sum((knots * f * DR)[knots >= 3]))
    wing_g = float(2 * np.pi * np.sum((knots * gauss * DR)[knots >= 3]))
    maps = np.stack([2 * np.maximum((store.load_chunk("stats_r7", i)["median"] - cal["offset"][i]) / cal["slope"][i], 0.0)
                     for i in range(c.CHUNKS)]).astype(np.float32)
    store.save_array("psf_profile", f.astype(np.float32))
    store.save_array("psf_profile_boot", np.quantile(boots, [0.025, 0.975], axis=0).astype(np.float32))
    store.save_array("bg_maps", maps)
    report = dict(spots=len(t), pool=pool, log_w_sd=log_w_sd, width_median=float(np.median(width)),
                  width_q=np.quantile(width, [0.05, 0.95]).tolist(), share_beyond_3px=wing, share_beyond_3px_gauss_105=wing_g,
                  peak=float(f[0]), peak_gauss_105=float(gauss[0]), bg_photons_range=[float(np.median(maps[-1])), float(np.median(maps[0]))],
                  dr_px=DR, recording="r7 only", profile=f.round(6).tolist(), knots_px=knots.tolist())
    store.save_json("psf_bg", b.rounded(report))
    print(b.rounded({k: v for k, v in report.items() if k not in ("profile", "knots_px")}), flush=True)


def stored_cal(store, rec, tag):
    if tag == "line":
        line = store.load_json("p3_line")[rec]
        return dict(offset=[line["offset"]] * c.CHUNKS, slope=[line["slope"]] * c.CHUNKS)
    rows = store.load_json("p3_calibration")["recordings"][rec]
    return dict(offset=rows["offsets"], slope=rows["slopes"])


# Network B's simulator -------------------------------------------------------------------------------


def world_b(store):
    return dict(prof=jnp.asarray(store.load_array("psf_profile"), jnp.float32), bg=jnp.asarray(store.load_array("bg_maps"), jnp.float32),
                log_w_sd=jnp.asarray(store.load_json("psf_bg")["log_w_sd"], jnp.float32))


def profile_image(prof, xy, w, side=n.CROP):
    g = jnp.arange(side, dtype=jnp.float32)
    r = jnp.sqrt((g[None, None, :] - xy[:, 0, None, None]) ** 2 + (g[None, :, None] - xy[:, 1, None, None]) ** 2) / w[:, None, None]
    u = jnp.clip(r / DR, 0.0, prof.shape[0] - 1.001)
    i = u.astype(jnp.int32)
    frac = u - i
    return (prof[i] * (1 - frac) + prof[i + 1] * frac) / w[:, None, None] ** 2


def simulate_crop_b(key, world):
    M = n.MAX_EMITTERS
    ks = jax.random.split(key, 12)
    density = jnp.exp(jax.random.uniform(ks[0], minval=jnp.log(n.DENSITY[0]), maxval=jnp.log(n.DENSITY[1])))
    q = jnp.clip(density * n.AREA_UM2 / M, 0.0, 1.0)
    stay = jax.random.uniform(ks[1], minval=0.3, maxval=0.7)
    off_on = q * (1 - stay) / (1 - q)
    s0 = jax.random.bernoulli(ks[2], q, (M,))
    s1 = jax.random.bernoulli(ks[3], jnp.where(s0, stay, off_on))
    s2 = jax.random.bernoulli(ks[4], jnp.where(s1, stay, off_on))
    on = jnp.stack([s0, s1, s2]).astype(jnp.float32)
    xy = jax.random.uniform(ks[5], (M, 2), minval=-0.5 - n.MARGIN, maxval=n.CROP - 0.5 + n.MARGIN)
    N = jnp.exp(n.LOGN[0] + n.LOGN[1] * jax.random.normal(ks[6], (M,)))
    w = jnp.exp(world["log_w_sd"] * jax.random.normal(ks[7], (M,)))
    block = jax.random.randint(ks[8], (), 0, world["bg"].shape[0])
    corner = jax.random.randint(ks[9], (2,), 0, b.SIDE - n.CROP + 1)
    bg = jax.lax.dynamic_slice(world["bg"][block], (corner[0], corner[1]), (n.CROP, n.CROP))
    flips = jax.random.bernoulli(ks[10], 0.5, (2,))
    bg = jnp.where(flips[0], bg[::-1], bg)
    bg = jnp.where(flips[1], bg[:, ::-1], bg)
    mu = bg + jnp.einsum("fm,m,myx->fyx", on, N, profile_image(world["prof"], xy, w))
    z = n.wilson_hilferty(ks[11], mu)
    inside = (xy >= -0.5).all(-1) & (xy <= n.CROP - 0.5).all(-1) & (on[1] > 0)
    return z, dict(xy=xy, logN=jnp.log(N / 1500.0), mask=inside, bg=bg / 2 / n.SCALE)


def make_step_b(opt):
    @eqx.filter_jit
    def step(params, static, state, key, world):
        z, truth = jax.vmap(simulate_crop_b, in_axes=(0, None))(jax.random.split(key, n.BATCH), world)

        def loss(p):
            m = eqx.combine(p, static)
            return jnp.mean(jax.vmap(n.crop_loss, in_axes=(None, 0, 0))(m, z, truth))

        value, grads = jax.value_and_grad(loss)(params)
        updates, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, updates), state, value

    return step


def load_net(store, net):
    return n.load_model(net, BASE, STEPS[net])


def train_net(store, net):
    steps = STEPS[net]
    device_name = None
    with held(f"train_{net}") as waited:
        key, params, static, opt, opt_state, step = n.setup(BASE, steps, seed=SEEDS[net])
        extra = ()
        if net == "b":
            params, _, at = load_net(store, "a")
            opt_state = opt.init(params)
            step = make_step_b(opt)
            extra = (world_b(store),)
            print(f"warm start from A at step {at}", flush=True)
        device_name = str(jax.devices()[0])
        start, log, log_steps = 0, [], []
        at, saved = store.resume(f"ckpt_{net}")
        if at is not None:
            params, opt_state = n.from_arrays((params, opt_state), saved)
            start = int(saved["step"])
            log, log_steps = saved["log"].tolist(), saved["log_steps"].tolist()
            print(f"resumed from step {start}", flush=True)
        t0 = time.time()
        for s in range(start, steps):
            params, opt_state, value = step(params, static, opt_state, jax.random.fold_in(key, s), *extra)
            if (s + 1) % 50 == 0 or SMOKE:
                v = float(value)
                if not np.isfinite(v):
                    raise SystemExit(f"NaN loss at step {s + 1}")
                log.append(v)
                log_steps.append(s + 1)
                if (s + 1) % 1000 == 0 or SMOKE:
                    print(f"step {s + 1} loss {v:.3f} {(s + 1 - start) / (time.time() - t0):.1f} steps/s", flush=True)
            if (s + 1) % CKPT == 0 or s + 1 == steps:
                jax.block_until_ready((params, opt_state))
                store.save_chunk(f"ckpt_{net}", s + 1, n.to_arrays((params, opt_state))
                                 | dict(step=np.array(s + 1), log=np.array(log), log_steps=np.array(log_steps)), keep=3)
                n.guard()
        seconds = time.time() - t0
        vram = bigrun.vram_high_water()
    store.save_json(f"train_{net}", b.rounded(dict(steps=steps, start=start, base=BASE, seconds=seconds, steps_per_s=(steps - start) / max(seconds, 1e-9),
                                                   weights=n.n_weights(params), device=device_name, last_loss=log[-1] if log else None,
                                                   lock_wait_s=waited, vram=vram, xla_flags=os.environ.get("XLA_FLAGS", ""),
                                                   log_steps=log_steps[::20], log=log[::20])))
    print(f"train {net}: {steps - start} steps in {seconds:.0f} s, {vram}", flush=True)


train_a = stage("train_a", "gpu")(lambda store: train_net(store, "a"))
train_b = stage("train_b", "gpu")(lambda store: train_net(store, "b"))


def validate(store, net):
    n.validate(net, BASE, STEPS[net])


validate_a = stage("validate_a", "gpu")(lambda store: validate(store, "a"))
validate_b = stage("validate_b", "gpu")(lambda store: validate(store, "b"))


# Coverage on simulation ------------------------------------------------------------------------------


def movie_empirical(rng, T, density, N, f, log_w_sd, bg_map, stay=STAY, H=b.SIDE):
    """T + 2 frames: persistent blinks, empirical PSF, a real background map, exact noise. Truth for all frames."""
    area = (H * n.PIXEL_UM) ** 2
    rate = density * area * (1 - stay)
    events = []
    for t in range(-20, T + 2):
        k = rng.poisson(rate)
        xy = rng.uniform(-0.5, H - 0.5, (k, 2))
        w = np.exp(log_w_sd * rng.normal(size=k))
        length = rng.geometric(1 - stay, k)
        events.extend((t, xy[i, 0], xy[i, 1], w[i], length[i]) for i in range(k))
    ev = np.array(events).reshape(-1, 5)
    g = np.arange(H)
    frames, truth = [], []
    for t in range(T + 2):
        on = ev[(ev[:, 0] <= t) & (t < ev[:, 0] + ev[:, 4])]
        mu = bg_map.astype(float).copy()
        for x, y, w in on[:, 1:4]:
            r0, r1, c0, c1 = max(int(y) - 12, 0), min(int(y) + 13, H), max(int(x) - 12, 0), min(int(x) + 13, H)
            r = np.hypot(g[r0:r1, None] - y, g[None, c0:c1] - x)
            mu[r0:r1, c0:c1] += N * lookup_np(r / w, f) / w**2
        frames.append(n.exact_counts(mu, rng))
        truth.append(np.c_[np.full(len(on), t), on[:, 1:3]])
    return np.array(frames, np.float32), np.vstack(truth)


def detect_maps(maps, threshold=n.THRESHOLD):
    """n.detections, vectorised, with the other maps at the peak kept."""
    p = maps[:, 0]
    summed = ndi.uniform_filter(p, size=(1, 3, 3)) * 9
    peak = (p == ndi.maximum_filter(p, size=(1, 3, 3))) & (summed >= threshold)
    t, r, cc = np.nonzero(peak)
    T, H, W = p.shape
    sw, sx, sy = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, ce = r + dr, cc + dc
            ok = (rr >= 0) & (rr < H) & (ce >= 0) & (ce < W)
            weight, dx, dy = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
            weight[ok] = p[t[ok], rr[ok], ce[ok]]
            dx[ok] = maps[t[ok], 1, rr[ok], ce[ok]]
            dy[ok] = maps[t[ok], 2, rr[ok], ce[ok]]
            sw += weight
            sx += weight * (ce + dx)
            sy += weight * (rr + dy)
    s_x, s_y = maps[t, 3, r, cc], maps[t, 4, r, cc]
    return dict(t=t, xy=np.c_[sx / sw, sy / sw], sd=np.hypot(s_x, s_y) / np.sqrt(2), N=1500.0 * np.exp(maps[t, 5, r, cc]),
                p=summed[t, r, cc], sx=s_x, sy=s_y, sN=maps[t, 6, r, cc])


def cell_score(table, truth, T, tol=2.0):
    tp, fp, fn, err, sds = 0, 0, 0, [], []
    for f in range(T):
        a = truth[truth[:, 0] == f + 1, 1:3]
        row = table["t"] == f
        e = table["xy"][row]
        if not len(a) or not len(e):
            fn, fp = fn + len(a), fp + len(e)
            continue
        D = np.hypot(a[:, None, 0] - e[None, :, 0], a[:, None, 1] - e[None, :, 1])
        r, q = linear_sum_assignment(D)
        ok = D[r, q] <= tol
        tp += int(ok.sum())
        fn += len(a) - int(ok.sum())
        fp += len(e) - int(ok.sum())
        err.append(e[q[ok]] - a[r[ok]])
        sds.append(np.c_[table["sx"][row][q[ok]], table["sy"][row][q[ok]]])
    out = dict(true=tp + fn, found=tp + fp, recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1))
    if err and tp:
        err, sds = np.vstack(err), np.vstack(sds)
        u = err / sds
        out |= dict(rmse_px=float(np.sqrt(np.mean(err**2))), coverage95=float(np.mean(np.abs(u) < 1.96)), z_mad=float(1.4826 * np.median(np.abs(u))))
    return out


def coverage(store, net):
    params, static, at = load_net(store, net)
    rng = np.random.default_rng(881)
    f = store.load_array("psf_profile").astype(float)
    log_w_sd = store.load_json("psf_bg")["log_w_sd"]
    bg_map = store.load_array("bg_maps")[10].astype(float)
    cells = []
    for world in ("gauss", "empirical"):
        for N in GRID_PHOTONS:
            for density in GRID_DENSITY:
                if world == "gauss":
                    z, truth = n.movie(rng, COVER_T, density, N)
                else:
                    z, truth = movie_empirical(rng, COVER_T, density, N, f, log_w_sd, bg_map)
                cells.append((world, N, density, z, truth))
        print(world, "simulated", flush=True)
    results, sums = [], []
    with held(f"coverage_{net}"):
        for cell in cells:
            maps = n.predict(params, static, n.triples(cell[3]), batch=BATCH)
            if not np.isfinite(maps).all():
                raise SystemExit(f"NaN in outputs at {cell[:3]}")
            results.append(detect_maps(maps))
            sums.append(float(maps[:, 0].sum()))
    rows = []
    for (world, N, density, z, truth), table, s in zip(cells, results, sums):
        row = dict(world=world, photons=N, density=density, in_prior=bool(n.DENSITY[0] <= density <= n.DENSITY[1]), crlb_px=n.crlb_axis(N, 110.0))
        row |= cell_score(table, truth, COVER_T)
        row["sum_p_over_true"] = s / max(row["true"], 1)
        if "rmse_px" in row:
            row["rmse_over_crlb"] = row["rmse_px"] / row["crlb_px"]
        rows.append(row)
        print(b.rounded(row), flush=True)
    store.save_json(f"coverage_{net}", b.rounded(dict(checkpoint_step=at, background=110.0, frames=COVER_T, rows=rows)))


coverage_a = stage("coverage_a", "gpu")(lambda store: coverage(store, "a"))
coverage_b = stage("coverage_b", "gpu")(lambda store: coverage(store, "b"))


# Inference on real frames ----------------------------------------------------------------------------


def infer_name(net, tag, rec):
    return f"infer_{net}{'_' + tag if tag else ''}_{rec}"


def infer(store, net):
    params, static, at = load_net(store, net)
    report = {}
    for tag in CALS:
        for rec in RECS:
            name = infer_name(net, tag, rec)
            todo = store.pending(name, c.CHUNKS)
            cal = stored_cal(store, rec, tag)
            stack = b.open_stack(b.RAW_DIR / b.TIFFS[rec][0])
            gpu_s, frames = 0.0, 0
            with held(name):
                for block in todo:
                    ids = block * b.BLOCK + np.arange(CHUNK_FRAMES)
                    z = frames_z(stack, np.clip(np.arange(ids[0] - 1, ids[-1] + 2), 0, b.N_FRAMES - 1), cal)
                    t0 = time.perf_counter()
                    maps = n.predict(params, static, n.triples(z), batch=BATCH)
                    gpu_s += time.perf_counter() - t0
                    frames += len(ids)
                    if not np.isfinite(maps).all():
                        raise SystemExit(f"NaN in outputs at {name} {block}")
                    table = detect_maps(maps)
                    table["t"] = table["t"] + ids[0]
                    table |= dict(sum_p=maps[:, 0].sum((1, 2)), pmap=maps[:, 0].mean(0).astype(np.float32), bg_mean=maps[:, 7].mean((1, 2)))
                    store.save_chunk(name, block, table)
                    if block % 10 == 0:
                        print(name, block, len(table["t"]), flush=True)
                        n.guard()
                vram = bigrun.vram_high_water()
            report[name] = dict(frames=frames, predict_seconds=gpu_s, frames_per_s=frames / max(gpu_s, 1e-9), vram=vram)
            print(b.rounded(report[name]), flush=True)
            store.save_json(f"infer_{net}", b.rounded(report))


infer_a = stage("infer_a", "gpu")(lambda store: infer(store, "a"))
infer_b = stage("infer_b", "gpu")(lambda store: infer(store, "b"))


def net_table(store, net, rec, tag=""):
    all_ = store.gather(infer_name(net, tag, rec), c.CHUNKS)
    return {k: all_[k] for k in ("t", "xy", "sd", "N", "p")}


def grid_ratio(pmap):
    yy, xx = np.mgrid[:b.SIDE, :b.SIDE]
    trim = (yy >= 8) & (yy < b.SIDE - 8) & (xx >= 8) & (xx < b.SIDE - 8)
    ids = (yy % 8) * 8 + xx % 8
    grid = np.bincount(ids[trim], pmap[trim], 64) / np.bincount(ids[trim], minlength=64)
    return float(grid.max() / grid.min())


# Simulated baseline for the paired test --------------------------------------------------------------


def linked_movie(rng, bg_counts, sites, events, follow, gain, offset, scale, read_sd=5.0, T=2000, sig=1.05):
    """b.simulate_linked_movie's model, rendered on patches: its full-frame arrays per row need 14 GB here."""
    H = bg_counts.shape[0]
    mu = np.repeat(2 * bg_counts[None], T, axis=0)
    pick = rng.integers(0, len(sites), events)
    start, length = rng.integers(0, T, events), rng.geometric(1 - follow, events)
    truth = []
    half = 8
    for e in range(events):
        x, y, N = sites[pick[e]]
        for f in range(start[e], min(start[e] + length[e], T)):
            truth.append((f, x, y, e, N))
        r0, c0 = int(np.clip(round(y) - half, 0, H - 2 * half - 1)), int(np.clip(round(x) - half, 0, H - 2 * half - 1))
        s = np.array([sig])
        Ey = b.pixel_gauss(np.array([y - r0]), s, 2 * half + 1)[0][0]
        Ex = b.pixel_gauss(np.array([x - c0]), s, 2 * half + 1)[0][0]
        last = min(start[e] + length[e], T)
        mu[start[e]:last, r0:r0 + 2 * half + 1, c0:c0 + 2 * half + 1] += N * np.outer(Ey, Ex)
    counts = rng.poisson(mu)
    charge = np.where(counts > 0, rng.gamma(np.maximum(counts, 1), gain), 0.0)
    adu = np.round(charge + offset + rng.normal(0.0, read_sd, counts.shape))
    return np.maximum(np.round((adu - offset) / scale), 0.0).astype(np.float32), np.array(truth)


def localise_batched(z):
    """b.localise in batches of c.FIT_BATCH, as the classical job does, with one background for the movie."""
    bg = np.median(z, axis=0)
    logb = np.log(2 * np.median(bg))
    _, lab, count, _ = b.detect(z, bg)
    Z, Wm, org, cen = b.blob_patches(z, lab, count)
    tables = {"chosen": [], "single": []}
    Ks = []
    for i in range(0, count, c.FIT_BATCH):
        s_ = slice(i, i + c.FIT_BATCH)
        res = b.count_emitters(Z[s_], b.PATCH, b.Prior(logb=(logb, 0.5), centre=cen[s_]), 4, Wm[s_])
        K = np.array([r[1] for r in res]).T.argmax(1)
        Ks.append(K)
        tables["chosen"].append(b.localisation_table(res, org[s_], K, 4))
        tables["single"].append(b.localisation_table(res, org[s_], np.ones(len(K), int), 1))
    out = {f"{name}_{key}": np.concatenate([p[key] for p in parts]) for name, parts in tables.items() for key in parts[0]}
    return count, np.bincount(np.concatenate(Ks), minlength=5), out


@stage("simbase", "cpu")
def simbase(store):
    rng = np.random.default_rng(886)
    cal = stored_cal(store, "r7", "")
    bg = np.maximum((store.load_chunk("stats_r7", 0)["median"] - cal["offset"][0]) / cal["slope"][0], 0.0).astype(float)
    chosen, _, _ = c.load_rec("r7")
    loc = b.good_rows(chosen, b.SIDE, c.PIXEL_NM)
    sites = np.c_[loc["xy"], loc["N"]]
    slope, offset = cal["slope"][0], cal["offset"][0]
    events = 40 if SMOKE else 20000
    z, truth = linked_movie(rng, bg, sites, events, 0.55, slope / 2, offset, slope, T=40 if SMOKE else 2000)
    np.save(SCRATCH / "simbase_z.npy", z.astype(np.uint16))
    store.save_array("simbase_truth", truth)
    print("simulated", z.shape, len(truth), flush=True)
    blobs, K, table = localise_batched(z)
    store.save_chunk("simbase_classical", 0, table)
    store.save_json("simbase", dict(frames=len(z), events=events, stay=0.55, blobs=int(blobs), K_counts=K.tolist(), rows=len(truth)))
    print("classical read the simulated movie:", blobs, K, flush=True)


@stage("infer_sim", "gpu")
def infer_sim(store):
    z = np.load(SCRATCH / "simbase_z.npy", mmap_mode="r")
    for net in NETS:
        params, static, at = load_net(store, net)
        parts = []
        with held(f"infer_sim_{net}"):
            for start in range(0, len(z) - 2, 400):          # the whole movie's maps would be 26 GB
                maps = n.predict(params, static, n.triples(np.asarray(z[start:start + 402], np.float32)), batch=BATCH)
                table = detect_maps(maps)
                table["t"] = table["t"] + start + 1
                parts.append(table)
        store.save_chunk(f"simbase_{net}", 0, {k: np.concatenate([p[k] for p in parts]) for k in ("t", "xy", "sd", "N", "p")})


# NUTS audit ------------------------------------------------------------------------------------------


@stage("audit_nuts", "cpu")
def audit_nuts(store):
    import arviz as az
    import nutpie

    import run_e88_audit as au

    cal = c.calibration("r7")
    chosen, _, blobs = c.load_rec("r7")
    mask = c.mask_share("r7")
    inside = c.in_mask(blobs["xy"], mask)
    rng = np.random.default_rng(887)
    picks = []
    for K in (1, 2):
        pool = np.nonzero(inside & (blobs["K"] == K))[0]
        picks += [(K, i) for i in rng.choice(pool, SPOTS // 2, replace=False)]
    t0 = time.time()
    compiled = {K: nutpie.compile_pymc_model(au.spot_model(K)) for K in (1, 2)}
    compile_s = time.time() - t0
    print(f"compiled in {compile_s:.0f} s", flush=True)
    todo = store.pending("audit_nuts", len(picks))
    for count, idx in enumerate(todo):
        K, i = picks[idx]
        Z, Wm, cen, org, logb = au.spot_inputs("r7", (int(blobs["t"][i]), blobs["xy"][i]), cal)
        t0 = time.time()
        model = compiled[K].with_data(z=Z, w=Wm, cx=np.full(K, cen[0]), cy=np.full(K, cen[1]), lb=logb)
        trace = nutpie.sample(model, chains=4, tune=1000, draws=1000, cores=WORKERS, seed=int(i) % 2**31, progress_bar=False)
        seconds = time.time() - t0
        post = trace.posterior
        x, y, logN = post["x"].values, post["y"].values, post["logN"].values
        if K == 2:
            order = np.argsort(x, axis=-1)
            x, y, logN = (np.take_along_axis(v, order, -1) for v in (x, y, logN))
        chk = dict(x=x, y=y, logN=logN, logb=post["logb"].values[..., None], logs=post["logs"].values[..., None])
        r = max(float(az.rhat(v[..., j])) for v in chk.values() for j in range(v.shape[-1]))
        ess = min(float(az.ess(v[..., j])) for v in chk.values() for j in range(v.shape[-1]))
        table = chosen["blob"] == blobs["id"][i]
        laplace = chosen["sd"][table]
        store.save_chunk("audit_nuts", idx, dict(
            K=np.array(K), t=np.array(int(blobs["t"][i])), rhat=np.array(r), ess=np.array(ess),
            divergences=np.array(int(trace.sample_stats["diverging"].sum())), seconds=np.array(seconds),
            nuts_sd_px=np.array(float(np.sqrt(0.5 * x.var(axis=(0, 1)).mean() + 0.5 * y.var(axis=(0, 1)).mean()))),
            laplace_sd_px=np.array(float(np.mean(laplace)) if len(laplace) else np.nan),
            mean_xy=np.c_[x.mean((0, 1)) + org[2], y.mean((0, 1)) + org[1]], sd_xy=np.c_[x.std((0, 1)), y.std((0, 1))]))
        if count % 10 == 0:
            print(f"spot {idx} of {len(picks)}: K {K}, r_hat {r:.3f}, {seconds:.1f} s", flush=True)
            n.guard()
    rows = [store.load_chunk("audit_nuts", i) for i in range(len(picks))]
    K = np.array([int(r["K"]) for r in rows])
    rhat = np.array([float(r["rhat"]) for r in rows])
    ratio = np.array([float(r["nuts_sd_px"]) / float(r["laplace_sd_px"]) for r in rows])
    good = rhat <= 1.01
    net = {}
    for name in NETS:
        table = net_table(store, name, "r7")
        dist, sd_ratio = [], []
        for r in rows:
            row = table["t"] == int(r["t"])
            for xy, sd in zip(r["mean_xy"], r["sd_xy"]):
                if row.any():
                    d = np.hypot(*(table["xy"][row] - xy).T)
                    j = d.argmin()
                    if d[j] <= 1.5:
                        dist.append(d[j] * c.PIXEL_NM)
                        sd_ratio.append(float(np.sqrt(0.5 * np.sum(sd**2))) / table["sd"][row][j])
        net[name] = dict(matched=len(dist), of=int(sum(len(r["mean_xy"]) for r in rows)),
                         median_distance_nm=float(np.median(dist)) if dist else None,
                         median_nuts_over_net_sd=float(np.median(sd_ratio)) if sd_ratio else None)
    med = lambda a: float(np.nanmedian(a)) if len(a) else None
    per_spot = float(np.mean([float(r["seconds"]) for r in rows]))
    report = dict(spots=len(rows), K_counts=[int((K == 1).sum()), int((K == 2).sum())], compile_s=compile_s, per_spot_s=per_spot,
                  max_rhat=float(rhat.max()), over_1_01=dict(K1=int(((K == 1) & ~good).sum()), K2=int(((K == 2) & ~good).sum())),
                  divergences=int(sum(int(r["divergences"]) for r in rows)), min_ess=float(min(float(r["ess"]) for r in rows)),
                  median_sd_ratio=dict(all=med(ratio), K1=med(ratio[K == 1]), K2=med(ratio[K == 2]), converged=med(ratio[good]),
                                       K1_converged=med(ratio[(K == 1) & good]), K2_converged=med(ratio[(K == 2) & good])),
                  check4_within_0_8_1_25=bool(0.8 <= med(ratio) <= 1.25), nets=net, cores=WORKERS)
    store.save_json("audit_nuts_report", b.rounded(report))
    print(b.rounded(report), flush=True)


# Paired-frame test -----------------------------------------------------------------------------------


TABLES = {}


def pair_brightness(loc, i, j):
    return np.sqrt(loc["N"][i] * loc["N"][j])


def matched_weights(ref_bright, own_bright, bins=8):
    edges = np.quantile(ref_bright, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_h = np.histogram(ref_bright, edges)[0] / len(ref_bright)
    own_h = np.histogram(own_bright, edges)[0] / max(len(own_bright), 1)
    which = np.clip(np.searchsorted(edges, own_bright, side="right") - 1, 0, bins - 1)
    ratio = np.where(own_h > 0, ref_h / np.maximum(own_h, 1e-30), 0.0)
    return ratio[which], float(ref_h[own_h == 0].sum())


def weighted_z(d, s2, track, radius, weight, rng, boots):
    r2 = np.sum(d**2, axis=1)
    z, w_true, x = b.fit_z_scale(r2, s2, radius, weight)
    ids, inv = np.unique(track, return_inverse=True)
    boot = []
    for _ in range(boots):
        draw = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv].astype(float)
        boot.append(b.fit_z_scale(r2, s2, radius, weight * draw, start=x)[0])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(z_sd=z, z_sd_lo=lo, z_sd_hi=hi, halfwidth95=(hi - lo) / 2, boot_se=float(np.std(boot)), true_share=w_true)


def paired_job(job):
    """One (recording, calibration tag, method) at gap and radius, matched to each reference."""
    key, method, refs, gap, radius, boots, seed = job
    loc = TABLES[key][method]
    rng = np.random.default_rng(seed)
    i, j, d, s2, tr = b.linked(loc, gap, radius)
    out = dict(key=key, method=method, gap=gap, radius=radius, pairs=len(i), median_sd_nm=float(np.median(loc["sd"])) * c.PIXEL_NM, rows=len(loc["t"]))
    if len(i) < 30:
        return out | dict(note="under 30 pairs")
    out["unweighted"] = b.z_summary(d, s2, tr, radius, rng, boots)
    for ref in refs:
        rl = TABLES[key][ref]
        ri, rj = b.link_pairs(rl["t"], rl["xy"], gap, radius)
        if ref == method:
            weight, dropped = np.ones(len(i)), 0.0
        else:
            weight, dropped = matched_weights(pair_brightness(rl, ri, rj), pair_brightness(loc, i, j))
        keep = weight > 0
        out[f"matched_{ref}"] = weighted_z(d[keep], s2[keep], tr[keep], radius, weight[keep], rng, boots) | dict(dropped_share_of_reference=dropped)
    return out


def verdict(z, ref):
    rel = "no worse than classical" if z <= ref else ("worse" if z > 1.10 * ref else "between")
    ab = "pass" if 1.0 <= z <= 1.15 else ("failure" if z > 1.25 else ("undefined" if z > 1.15 else "below 1.0"))
    return dict(absolute=ab, relative=rel)


@stage("paired", "cpu")
def paired(store):
    jobs = []
    for rec in RECS:
        mask = c.mask_share(rec)
        for tag in CALS:
            chosen, single, _ = c.load_rec(rec, tag)
            methods = dict(chosen=chosen, K1=single) | {net: net_table(store, net, rec, tag) for net in NETS}
            key = f"{rec}{'_' + tag if tag else ''}"
            TABLES[key] = {}
            for name, table in methods.items():
                loc = b.good_rows(table, b.SIDE, c.PIXEL_NM)
                TABLES[key][name] = {k: v[c.in_mask(loc["xy"], mask)] for k, v in loc.items()}
                jobs.append((key, name, ("chosen", "K1"), 3, b.RADIUS_PX, BOOTS, 880))
                if not tag:
                    jobs.append((key, name, ("chosen", "K1"), 1, b.RADIUS_PX, BOOTS // 4, 881))
                    jobs.extend((key, name, ("chosen",), 3, radius, BOOTS // 4, 882) for radius in b.RADII_PX if radius != b.RADIUS_PX)
    sim = {"chosen": {}, "K1": {}}
    raw = store.load_chunk("simbase_classical", 0)
    sim["chosen"] = b.good_rows({k: raw[f"chosen_{k}"] for k in ("t", "xy", "sd", "N")}, b.SIDE, c.PIXEL_NM)
    sim["K1"] = b.good_rows({k: raw[f"single_{k}"] for k in ("t", "xy", "sd", "N")}, b.SIDE, c.PIXEL_NM)
    for net in NETS:
        sim[net] = b.good_rows(store.load_chunk(f"simbase_{net}", 0), b.SIDE, c.PIXEL_NM)
    TABLES["simulated"] = sim
    jobs.extend(("simulated", name, ("chosen",), gap, b.RADIUS_PX, BOOTS // 4, 883) for name in sim for gap in (1, 3))
    done = {}
    todo = store.pending("paired", len(jobs))
    print(f"{len(todo)} of {len(jobs)} paired jobs", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for idx, row in zip(todo, pool.map(paired_job, [jobs[i] for i in todo])):
            store.save_chunk("paired", idx, dict(row=np.frombuffer(json.dumps(row, default=bigrun._json_value).encode(), np.uint8)))
            print(idx, row["key"], row["method"], row["gap"], row["radius"], row["pairs"], f"{time.time() - started:.0f} s", flush=True)
            n.guard()
    rows = [json.loads(store.load_chunk("paired", i)["row"].tobytes()) for i in range(len(jobs))]
    for row in rows:
        done.setdefault(row["key"], {}).setdefault(row["method"], {})[f"gap{row['gap']}_R{row['radius']}"] = row
    report = dict(rows=done, verdicts={})
    for key in done:
        if key == "simulated":
            continue
        for net in NETS:
            row = done[key][net].get("gap3_R1.5", {})
            ref = done[key]["chosen"].get("gap3_R1.5", {})
            if "matched_chosen" in row and "matched_chosen" in ref:
                z, zc = row["matched_chosen"]["z_sd"], ref["matched_chosen"]["z_sd"]
                report["verdicts"][f"{key}_{net}"] = dict(z=z, classical_z=zc, halfwidth95=row["matched_chosen"]["halfwidth95"],
                                                          one_se=row["matched_chosen"]["boot_se"], k1_z=done[key]["K1"].get("gap3_R1.5", {}).get("matched_K1", {}).get("z_sd")) | verdict(z, zc)
    store.save_json("paired_report", b.rounded(report))
    print(b.rounded(report["verdicts"]), flush=True)


# Companion halo --------------------------------------------------------------------------------------


def halo_one(loc, mask):
    keep = (loc["xy"].min(1) >= b.EDGE_PX) & (loc["xy"].max(1) <= b.SIDE - 1 - b.EDGE_PX)
    t, xy, N = loc["t"][keep], loc["xy"][keep] * c.PIXEL_NM, loc["N"][keep]
    bright = (N >= HALO_BRIGHT) & c.in_mask(loc["xy"][keep], mask)
    faint = N < HALO_FAINT
    order = np.argsort(t, kind="stable")
    t, xy, bright, faint = t[order], xy[order], bright[order], faint[order]
    cuts = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]
    counts = np.zeros(len(HALO_EDGES) - 1)
    ring, far, n_bright = 0, 0, int(bright.sum())
    for a, e in zip(cuts[:-1], cuts[1:]):
        p, q = xy[a:e][bright[a:e]], xy[a:e][faint[a:e]]
        if len(p) and len(q):
            d = np.hypot(*(p[:, None] - q[None]).transpose(2, 0, 1)).ravel()
            d = d[d > 0]
            counts += np.histogram(d, HALO_EDGES)[0]
            ring += int(((d >= RING[0]) & (d <= RING[1])).sum())
            far += int(((d >= FAR[0]) & (d <= FAR[1])).sum())
    um2 = lambda lo, hi: np.pi * (hi**2 - lo**2) / 1e6
    area = um2(HALO_EDGES[:-1], HALO_EDGES[1:])
    d_ring = ring / max(n_bright, 1) / um2(*RING)
    d_far = far / max(n_bright, 1) / um2(*FAR)
    return dict(bright=n_bright, faint=int(faint.sum()), ring_pairs=ring, far_pairs=far, density_ring=d_ring, density_far=d_far,
                ratio=d_ring / d_far if d_far > 0 else None, density=(counts / max(n_bright, 1) / area).tolist(), edges_nm=HALO_EDGES.tolist())


@stage("halo", "cpu")
def halo(store):
    report = {}
    for rec in RECS:
        mask = c.mask_share(rec)
        chosen, _, _ = c.load_rec(rec)
        rows = dict(chosen=halo_one(chosen, mask))
        for net in NETS:
            rows[net] = halo_one(net_table(store, net, rec), mask)
        for name, row in rows.items():
            row["failure"] = bool(row["ratio"] is not None and row["ratio"] > 2)
            print(rec, name, b.rounded({k: v for k, v in row.items() if k not in ("density", "edges_nm")}), flush=True)
        report[rec] = rows
    store.save_json("halo_report", b.rounded(report))


# Population replicate --------------------------------------------------------------------------------


def blob_stats(z, bg, rng, limit=2000):
    """Area and peak of detector blobs, and the mean profile about their peaks, for frames ``z``."""
    S, lab, count, thr = b.detect(z, bg)
    if count == 0:
        return dict(area=np.zeros(0), peak=np.zeros(0), profile=np.zeros(9), blobs=0)
    excess = z - bg
    objects = ndi.find_objects(lab)
    area, peak, centres = [], [], []
    for i, sl in enumerate(objects):
        here = lab[sl] == i + 1
        e = excess[sl] * here
        area.append(int(here.sum()))
        peak.append(float(e.max()))
        k = np.unravel_index(e.argmax(), e.shape)
        centres.append((sl[0].start + k[0], sl[1].start + k[1], sl[2].start + k[2]))
    area, peak, centres = np.array(area), np.array(peak), np.array(centres)
    pick = rng.choice(count, min(limit, count), replace=False)
    total, seen = np.zeros(9), np.zeros(9)
    H = z.shape[1]
    grid = np.arange(-8, 9)
    r = np.hypot(grid[:, None], grid[None, :]).round().astype(int)
    for t, y, x in centres[pick]:
        if 8 <= y < H - 8 and 8 <= x < H - 8:
            patch = excess[t, y - 8:y + 9, x - 8:x + 9] / max(excess[t, y, x], 1.0)
            use = r <= 8
            total += np.bincount(r[use], patch[use], 9)
            seen += np.bincount(r[use], minlength=9)
    return dict(area=area, peak=peak, profile=total / np.maximum(seen, 1), blobs=int(count))


def ks(a, e):
    if not len(a) or not len(e):
        return None
    grid = np.sort(np.r_[a, e])
    return float(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a) - np.searchsorted(np.sort(e), grid, side="right") / len(e)).max())


def render_frame(rng, rate, sites, photons, bg, kind, f, log_w_sd, slope, offset):
    count = rng.poisson(rate)
    xy = sites[rng.integers(0, len(sites), count)]
    N = photons[rng.integers(0, len(photons), count)]
    H = bg.shape[0]
    g = np.arange(H)
    mu = 2 * bg.astype(float)
    w = np.exp(log_w_sd * rng.normal(size=count))
    for (x, y), Ni, wi in zip(xy, N, w):
        r0, r1, c0, c1 = max(int(y) - 12, 0), min(int(y) + 13, H), max(int(x) - 12, 0), min(int(x) + 13, H)
        if r0 >= r1 or c0 >= c1:
            continue
        r = np.hypot(g[r0:r1, None] - y, g[None, c0:c1] - x)
        if kind == "b":
            mu[r0:r1, c0:c1] += Ni * lookup_np(r / wi, f) / wi**2
        else:
            mu[r0:r1, c0:c1] += Ni * np.exp(-(r**2) / (2 * 1.05**2)) / (2 * np.pi * 1.05**2)
    return n.exact_counts(mu, rng, gain=slope / 2, offset=offset, scale=slope)


@stage("replicate", "cpu")
def replicate(store):
    rng = np.random.default_rng(889)
    f = store.load_array("psf_profile").astype(float)
    log_w_sd = store.load_json("psf_bg")["log_w_sd"]
    report = {}
    for rec in RECS:
        stack = b.open_stack(b.RAW_DIR / b.TIFFS[rec][0])
        cal = stored_cal(store, rec, "")
        mask = c.mask_share(rec)
        report[rec] = {}
        tables = {net: net_table(store, net, rec) for net in NETS}
        for net, table in tables.items():
            real, sim, panels = {"area": [], "peak": [], "profile": []}, {"area": [], "peak": [], "profile": []}, []
            T = 8 if SMOKE else REP_T
            for start in REP_WINDOWS:
                idx = start + np.arange(T)
                z = frames_z(stack, idx, cal)
                block = start // b.BLOCK
                bg = c.rolling_background(stack, start, cal["offset"][block], cal["slope"][block])
                slope, offset = float(cal["slope"][block]), float(cal["offset"][block])
                near = np.abs(table["t"] - start - T / 2) <= T / 2
                far = np.abs(table["t"] - start - T / 2) >= 500
                sites, photons = table["xy"][far], table["N"][far]
                rate = near.sum() / T
                rep = np.array([render_frame(rng, rate, sites, photons, bg, net, f, log_w_sd if net == "b" else 0.0, slope, offset) for _ in range(T)], np.float32)
                a, e = blob_stats(z, bg, rng), blob_stats(rep, bg, rng)
                for key in real:
                    real[key].append(a[key])
                    sim[key].append(e[key])
                if start == REP_WINDOWS[0]:
                    ys, xs = np.nonzero(mask)
                    y0, x0 = int(np.clip(np.median(ys) - 20, 0, b.SIDE - 40)), int(np.clip(np.median(xs) - 20, 0, b.SIDE - 40))
                    where = int(rng.integers(0, LINEUP))
                    panels = [rep[(k * 7) % T, y0:y0 + 40, x0:x0 + 40] for k in range(LINEUP - 1)]
                    panels.insert(where, z[T // 2, y0:y0 + 40, x0:x0 + 40])
            real_p, sim_p = np.mean(real["profile"], axis=0), np.mean(sim["profile"], axis=0)
            area_r, area_s = np.concatenate(real["area"]), np.concatenate(sim["area"])
            peak_r, peak_s = np.concatenate(real["peak"]), np.concatenate(sim["peak"])
            row = dict(blobs_real=len(area_r), blobs_replicate=len(area_s), ks_area=ks(area_r, area_s), ks_peak=ks(peak_r, peak_s),
                       median_area=[float(np.median(area_r)), float(np.median(area_s))], median_peak=[float(np.median(peak_r)), float(np.median(peak_s))],
                       profile_real=real_p.tolist(), profile_replicate=sim_p.tolist(),
                       profile_ratio_3_7px=(sim_p[3:8] / np.where(real_p[3:8] != 0, real_p[3:8], np.nan)).tolist(), lineup_real_position=where)
            report[rec][net] = row
            store.save_chunk(f"lineup_{rec}_{net}", 0, dict(panels=np.array(panels, np.float32), real_position=np.array(where)))
            print(rec, net, b.rounded({k: v for k, v in row.items() if "profile_r" not in k}), flush=True)
            store.save_json("replicate_report", b.rounded(report))
            n.guard()


# The frames decision ---------------------------------------------------------------------------------


def field_of(store, net, rec, tag, mask, drift=None):
    loc = b.good_rows(net_table(store, net, rec, tag), b.SIDE, c.PIXEL_NM)
    t, xy = loc["t"], loc["xy"]
    blocks = c.drift_blocks(t, xy)
    d = drift
    if d is None:
        d = np.array([c.estimate(xy[blocks == k], blocks[blocks == k] * 0, xy[blocks != k])[0] for k in range(blocks.max() + 1)])
    loc["xy"] = c.correct(t, xy, d)
    ev = c.merge_events(loc)
    return loc, ev, d


def curves_one(store, net, tag, classical_drift=False):
    suffix = f"{net}{'_' + tag if tag else ''}" + ("_cdrift" if classical_drift else "")
    report = {}
    grid = c.GRID_N
    for rec in RECS:
        mask = c.mask_share(rec)
        given = store.load_array(f"drift_{tag + '_' if tag else ''}{rec}") if classical_drift else None
        loc, ev, d = field_of(store, net, rec, tag, mask, given)
        if not classical_drift:
            store.save_array(f"drift_{suffix}_{rec}", d)
        c.FIELDS[rec] = c.Field(ev, mask)
        name = f"curve_{suffix}_{rec}"
        todo = store.pending(name, len(grid))
        print(name, f"{len(todo)} of {len(grid)} to do", len(loc["t"]), len(ev["t"]), flush=True)
        started = time.time()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for i in todo:
                jobs = [(rec, grid[i], draw) for draw in range(-1, DRAWS)]
                rows = list(pool.map(c.curve_job, jobs, chunksize=4))
                store.save_chunk(name, i, dict(rows=np.array([r[2:] for r in rows], float)))
                print(name, grid[i], f"{time.time() - started:.0f} s", flush=True)
                n.guard()
        points, bands = {}, {}
        for i, N in enumerate(grid):
            rows = store.load_chunk(name, i)["rows"]
            pt, dr = rows[0], rows[1:, 1:]
            points[N] = dict(frc_nm=float(pt[1]), count=float(pt[2]), largest=float(pt[3]))
            bands[N] = dict(frc_q=np.nanquantile(dr[:, 0], [0.025, 0.975]).tolist(), count_q=np.quantile(dr[:, 1], [0.025, 0.975]).tolist(),
                            frc_median=float(np.nanmedian(dr[:, 0])), count_median=float(np.median(dr[:, 1])), frc_nan=int(np.isnan(dr[:, 0]).sum()))
        n_star, good = c.star(points, bands)
        report[rec] = dict(rows_filtered=len(loc["t"]), events=len(ev["t"]), events_in_mask=len(c.FIELDS[rec].t), median_sd_nm=float(np.median(loc["sd"])) * c.PIXEL_NM,
                           share_over_12000_adu_rule=float(np.mean(loc["N"] >= 1730)), drift_range_nm=np.ptp(d * c.PIXEL_NM, axis=0).round(2).tolist(),
                           points=points, bands=bands, N_star=n_star, checks=good)
        print(suffix, rec, "N*", n_star, {N: round(p["count"]) for N, p in points.items()}, flush=True)
    report["N_both"] = max(report[r]["N_star"] for r in RECS)
    store.save_json(f"decision_{suffix}", b.rounded(report))


@stage("curves", "cpu")
def curves(store):
    for net in NETS:
        for tag in CALS:
            if not store.done(f"curve_{net}{'_' + tag if tag else ''}"):
                curves_one(store, net, tag)
                store.mark_done(f"curve_{net}{'_' + tag if tag else ''}")


@stage("curves_cdrift", "cpu")
def curves_cdrift(store):
    """Post-hoc, not a gate: full_posthoc_drift.json. The gated runs' drift failed in 1-4 blocks of 15."""
    for net in NETS:
        for tag in CALS:
            done = f"curve_{net}{'_' + tag if tag else ''}_cdrift"
            if not store.done(done):
                curves_one(store, net, tag, classical_drift=True)
                store.mark_done(done)
    report = {}
    for rec in RECS:
        mask = c.mask_share(rec)
        report[rec] = {}
        for net in NETS:
            loc = b.good_rows(net_table(store, net, rec), b.SIDE, c.PIXEL_NM)
            loc["xy"] = c.correct(loc["t"], loc["xy"], store.load_array(f"drift_{rec}"))
            rng = np.random.default_rng(884)
            raw = c.Field(dict(t=loc["t"], xy=loc["xy"], sd=loc["sd"]), mask)
            random_nm = split_frc(raw, rng.integers(0, 2, len(raw.t)))
            ev = c.Field(c.merge_events(loc), mask)
            block_nm = split_frc(ev, ev.tblock % 2)
            report[rec][net] = dict(random_split_nm=random_nm, block_split_nm=block_nm, gap_nm=block_nm - random_nm)
    store.save_json("frc_split_cdrift", b.rounded(report))
    print(b.rounded(report), flush=True)
    decisions = {f"{net}{'_' + tag if tag else ''}": store.load_json(f"decision_{net}{'_' + tag if tag else ''}_cdrift") for net in NETS for tag in CALS}
    out = {k: dict(N_both=v["N_both"], r7=v["r7"]["N_star"], r32=v["r32"]["N_star"]) for k, v in decisions.items()}
    store.save_json("decision_posthoc_cdrift", dict(note="post-hoc, classical drift, not a gate", decision=out))
    store.write_manifest(git_hash=bigrun.git_hash(), versions=bigrun.library_versions("jax", "equinox", "optax", "numpy", "scipy", "pymc", "nutpie"),
                         data_fingerprint={k: v[1] for k, v in b.TIFFS.items()}, hardware="5950X (cores 14,15 only, 10 and 26 offline), RTX 3090 Ti",
                         started=os.environ.get("E88_STARTED", ""), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    problems = store.verify()
    store.save_json("verify", dict(problems=problems, at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print("decision", out, "verify", len(problems), problems[:5], flush=True)


# FRC: random against block split ---------------------------------------------------------------------


def split_frc(field, labels):
    a = field.hist(field.xy[labels == 0], np.ones((labels == 0).sum())) * field.window
    e = field.hist(field.xy[labels == 1], np.ones((labels == 1).sum())) * field.window
    return c.frc_resolution(a, e, c.RENDER_NM)


@stage("frc_split", "cpu")
def frc_split(store):
    rng = np.random.default_rng(884)
    report = {}
    for rec in RECS:
        mask = c.mask_share(rec)
        chosen, _, _ = c.load_rec(rec)
        report[rec] = {}
        methods = {"chosen": b.good_rows(chosen, b.SIDE, c.PIXEL_NM)} | {net: b.good_rows(net_table(store, net, rec), b.SIDE, c.PIXEL_NM) for net in NETS}
        drift = {"chosen": store.load_array(f"drift_{rec}")} | {net: store.load_array(f"drift_{net}_{rec}") for net in NETS}
        for name, loc in methods.items():
            loc["xy"] = c.correct(loc["t"], loc["xy"], drift[name])
            raw = c.Field(dict(t=loc["t"], xy=loc["xy"], sd=loc["sd"]), mask)
            random_nm = split_frc(raw, rng.integers(0, 2, len(raw.t)))
            ev = c.Field(c.merge_events(loc), mask)
            block_nm = split_frc(ev, ev.tblock % 2)
            report[rec][name] = dict(random_split_nm=random_nm, block_split_nm=block_nm, gap_nm=block_nm - random_nm)
            print(rec, name, b.rounded(report[rec][name]), flush=True)
    store.save_json("frc_split", b.rounded(report))


# Out of domain ---------------------------------------------------------------------------------------


@stage("ood532", "gpu")
def ood532(store):
    name = "dSTORM532_Nc82_7.tif"
    url = f"{b.BASE}/{name}/content"
    out = dict(file=name, frames=[3000, 3001, 3002, 3003], certificate="never tested here")
    try:
        raw = np.array([np.frombuffer(b._get(url, b.frame_offset(k), b.frame_offset(k) + b.PIXELS), "<u2").reshape(b.SIDE, b.SIDE)
                        for k in out["frames"]], np.float32)
    except OSError as err:
        out["error"] = f"fetch failed: {err}"
        store.save_json("ood532", out)
        print(out, flush=True)
        return
    cal = stored_cal(store, "r7", "line")
    z = np.maximum(np.round((raw - cal["offset"][0]) / cal["slope"][0]), 0.0)
    out["median_counts"] = float(np.median(z))
    for net in NETS:
        params, static, _ = load_net(store, net)
        with held(f"ood532_{net}"):
            maps = n.predict(params, static, n.triples(z), batch=2)
        table = detect_maps(maps)
        out[net] = dict(found_per_frame=len(table["t"]) / len(maps), sum_p_per_frame=float(maps[:, 0].sum() / len(maps)), median_sd_nm=float(np.median(table["sd"]) * c.PIXEL_NM) if len(table["t"]) else None)
    store.save_json("ood532", b.rounded(out))
    print(b.rounded(out), flush=True)


# Certificate and summary -----------------------------------------------------------------------------


def outcome(paired_v, halos):
    def fails(net):
        v = paired_v[f"r32_{net}"]
        h = halos["r32"][net]
        return dict(paired_z=v["z"] > 1.25, relative_worse=v["relative"] == "worse", halo=bool(h["ratio"] is not None and h["ratio"] > 2))

    fa, fb = fails("a"), fails("b")
    if not any(fa.values()):
        return "C", fa, fb
    return ("A" if not any(fb[k] for k, v in fa.items() if v) else "B"), fa, fb


@stage("summary", "cpu")
def summary(store):
    pilots = dict(P3="NO-GO", P2="pass", P2b_overlap="rare (2.2% / 1.65%), Q1 stop fired, overridden by the owner", P4="pass (203 steps/s, 44x over 13 CPUs, a pilot rate)",
                  P5="NO-GO (1.38 x CRLB at 5,000 steps)", P6_grid="NO-GO (1.78, line 1.25)", P7="NO-GO (r_hat 2.57, 2,074 divergences)", P8p="first attempt NO-GO; second pass with deterministic ops")
    paired_r = store.load_json("paired_report")
    halos = store.load_json("halo_report")
    decision = {}
    for net in NETS:
        for tag in CALS:
            suffix = f"{net}{'_' + tag if tag else ''}"
            d = store.load_json(f"decision_{suffix}")
            decision[suffix] = dict(N_both=d["N_both"], r7=d["r7"]["N_star"], r32=d["r32"]["N_star"])
    p2b = store.load_json("p2b_decision")
    line = store.load_json("p2b_decision_line")
    decision["classical"] = dict(N_both=p2b["N_both"], r7=p2b["r7"]["N_star"], r32=p2b["r32"]["N_star"])
    decision["classical_line"] = dict(N_both=line["N_both"], r7=line["r7"]["N_star"], r32=line["r32"]["N_star"])
    letter, fa, fb = outcome(paired_r["verdicts"], halos)
    grid = {}
    for net in NETS:
        for rec in RECS:
            chunks = [store.load_chunk(infer_name(net, "", rec), i)["pmap"] for i in range(c.CHUNKS)]
            grid[f"{net}_{rec}"] = grid_ratio(np.mean(chunks, axis=0))
    audit = store.load_json("audit_nuts_report")
    certificate = dict(
        instrument="Network B (and A) DECODE-style U-Net, base 32, tested only as listed",
        trained_on=dict(a="E58's forward model, Gaussian PSF, flat background, independent frames", b="empirical radial PSF, defocus, real block-median maps and persistence, recording 7 only"),
        ranges=dict(density_per_um2=list(n.DENSITY), photons_median=1500, background_photons_a=list(n.BG)),
        simulation=dict(a=store.load_json("coverage_a")["rows"], b=store.load_json("coverage_b")["rows"]),
        real=dict(paired=paired_r["verdicts"], halo={r: {m: halos[r][m]["ratio"] for m in halos[r]} for r in RECS}, nuts_median_sd_ratio=audit["median_sd_ratio"]),
        never_tested=["532 channel", "any preparation but these two recordings", "density above 1.5 or below 0.02 per um^2 on simulation", "ground truth on real frames"],
        p6_grid_ratio_all_frames=grid, attribution=b.ATTRIBUTION)
    store.save_json("certificate", b.rounded(certificate))
    report = dict(pilots_on_record=pilots, p5_at_final=dict(a=store.load_json("p5_a").get("sparse_2000"), b=store.load_json("p5_b").get("sparse_2000")),
                  decision=decision, outcome=letter, outcome_D=decision.get("b", {}).get("N_both") == decision["classical"]["N_both"], fails_a=fa, fails_b=fb,
                  grid=grid, train=dict(a=store.load_json("train_a"), b=store.load_json("train_b")), infer=dict(a=store.load_json("infer_a"), b=store.load_json("infer_b")),
                  owner_ruling="Its oke if the network isn't definitive tried and failed is still useful information", p4_cpu_ratio_pilot=44, attribution=b.ATTRIBUTION)
    store.save_json("summary", b.rounded(report))
    finish = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    store.write_manifest(git_hash=bigrun.git_hash(), versions=bigrun.library_versions("jax", "equinox", "optax", "numpy", "scipy", "pymc", "nutpie"),
                         data_fingerprint={k: v[1] for k, v in b.TIFFS.items()}, hardware="5950X (cores 14,15 only, 10 and 26 offline), RTX 3090 Ti",
                         started=os.environ.get("E88_STARTED", ""), finished=finish)
    problems = store.verify()
    print("verify:", problems[:10], len(problems), flush=True)
    store.save_json("verify", dict(problems=problems, at=finish))
    print(b.rounded({k: v for k, v in report.items() if k not in ("train", "infer")}), flush=True)


STAGES = dict(psf_bg=psf_bg, train_a=train_a, validate_a=validate_a, coverage_a=coverage_a, infer_a=infer_a, train_b=train_b, validate_b=validate_b,
              coverage_b=coverage_b, infer_b=infer_b, simbase=simbase, infer_sim=infer_sim, audit_nuts=audit_nuts, paired=paired, halo=halo,
              replicate=replicate, curves=curves, frc_split=frc_split, ood532=ood532, summary=summary, curves_cdrift=curves_cdrift)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("stage", choices=tuple(STAGES))
    STAGES[parser.parse_args().stage]()


if __name__ == "__main__":
    sys.exit(main())
