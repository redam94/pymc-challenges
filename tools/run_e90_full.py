"""E90 FULL: stage fits, hindsight variant and 96 strata, on the GPU. See plans/large/E90.md, section 14.

    python tools/run_e90_full.py prep                  once: move the pilots' fits into their stages
    python tools/run_e90_full.py next                  print the next stage, or "none"
    python tools/run_e90_full.py states|hindsight|strata N    up to N fits, then exit

Lines, dates, seeds and the restart rule are in results/E90/gates_full_2026-11-02.json, written first.
Each fit is one chunk, so a crash loses the fit in flight. A wrapper starts a fresh process every few
fits, under tools/memguard.py, and checks the machine; so does every fit here. Scoring is in
tools/score_e90.py. Every fit gets a seed fixed by its date: nothing here looks at a result to pick one.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e90_core as c  # noqa: E402
import pilot_e90 as p  # noqa: E402

from pymc_challenges.bigrun import gpu_lock, sha256_file, vram_high_water  # noqa: E402

store = p.store
GATES = "gates_full_2026-11-02"
HARD = ("2021-12-03", "2021-12-17", "2021-12-29", "2022-01-13", "2022-01-25", "2022-02-04", "2022-03-02", "2022-04-23")
FINISHED_ROWS = ("2021-09-25", "2022-04-23")  # first hard date less 69 days, last hard date
N_DATES = 227
ALL_DATES = [str(p.START + np.timedelta64(k, "D")) for k in range(N_DATES)]
LINE = {"r_hat": 1.01, "ess": 400, "divergence": 0.001}
FAILED_OFFSET = {"gpu_fits": 1000, "hindsight": 2000, "strata": 3000}
SEED_OFFSET = {"gpu_fits": 0, "hindsight": 2000, "strata": 4000}
DRAWS = dict(chains=32, warmup=1000, draws=1000, target=0.85)


def machine():
    """Refuse to go on if the machine has changed or /home is short of the 30 GB floor."""
    h = p.health()
    free = shutil.disk_usage(store.path).free / 1e9
    if free < 30:
        raise SystemExit(f"STOP: /home has {free:.1f} GB free")
    return h


def gate(out, extra):
    rhat, ess, div = float(out["latent_rhat"].max()), float(out["latent_ess"].min()), float(extra["divergence_rate"])
    return {"r_hat": rhat, "ess": ess, "divergence": div,
            "pass": rhat <= LINE["r_hat"] and ess >= LINE["ess"] and div <= LINE["divergence"]}


def run_with_restart(stage, date, cid, name, fit):
    """The first seed, then one restart on a failed line; both attempts are kept. `fit(seed)` is one fit."""
    base = cid + 11 + SEED_OFFSET[stage]
    failed = f"failed_{stage}_{date}"
    start = 1 if (store.path / f"{failed}.json").exists() else 0
    for attempt in range(start, 2):
        machine()
        seed = base + 1000 * attempt
        out, extra, vram = fit(seed)
        verdict = gate(out, extra)
        record = {**extra, "date": date, "seed": seed, "attempt": attempt, "gate": verdict, "vram": vram,
                  "health": p.health(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        arrays = {k: np.asarray(v) for k, v in out.items() if not k.startswith("n_")}
        if verdict["pass"] or attempt == 1:
            store.save_chunk(stage, cid, arrays)
            store.save_json(name, record)
            print(f"{stage} {date} attempt {attempt}: pass {verdict['pass']}, {verdict}", flush=True)
            return
        store.save_chunk("failed_runs", cid + FAILED_OFFSET[stage], arrays)
        store.save_json(failed, record)
        print(f"{stage} {date} failed a line, restarting: {verdict}", flush=True)


def cleanup():
    import gc

    import jax

    jax.clear_caches()
    gc.collect()


def prep():
    """The pilots' fits that are FULL fits. The restart of 2021-12-26 was run in P5, seed 99."""
    cid = p.chunk_id("2021-12-26")
    done = store.load_json("gpu_fit_2021-12-26")
    if "gate" not in done:
        store.save_chunk("failed_runs", cid + FAILED_OFFSET["gpu_fits"], store.load_chunk("gpu_fits", cid))
        store.save_json("failed_gpu_fits_2021-12-26", {**done, "seed": cid + 11, "attempt": 0})
        chunk, extra = store.load_chunk("p4_gpu_r1226", 0), store.load_json("p4_gpu_r1226")
        store.save_chunk("gpu_fits", cid, chunk)
        store.save_json("gpu_fit_2021-12-26", {**extra, "date": "2021-12-26", "seed": 99, "attempt": 1,
                                               "gate": gate(chunk, extra), "source": "p4_gpu_r1226"})
    main = p.chunk_id(p.MAIN)
    if main in store.pending("gpu_fits", N_DATES):
        chunk, extra = store.load_chunk("p4_gpu_c32t85", 0), store.load_json("p4_gpu_c32t85")
        store.save_chunk("gpu_fits", main, chunk)
        store.save_json(f"gpu_fit_{p.MAIN}", {**extra, "date": p.MAIN, "seed": 1, "attempt": 0,
                                              "gate": gate(chunk, extra), "source": "p4_gpu_c32t85"})
    if main in store.pending("strata", N_DATES):
        chunk, extra = store.load_chunk("p7_gpu", 0), store.load_json("p7_gpu")
        store.save_chunk("strata", main, chunk)
        store.save_json(f"strata_{p.MAIN}", {**extra, "date": p.MAIN, "seed": 71, "attempt": 0,
                                             "gate": gate(chunk, extra), "source": "p7_gpu"})
    left = set(store.pending("gpu_fits", N_DATES))
    for date in ALL_DATES:  # the 21 other pilot fits: same settings, stored without a verdict
        name, cid = f"gpu_fit_{date}", p.chunk_id(date)
        if cid not in left:
            extra = store.load_json(name)
            if "gate" not in extra:
                store.save_json(name, {**extra, "seed": cid + 11, "attempt": 0,
                                       "gate": gate(store.load_chunk("gpu_fits", cid), extra)})
    print("prep done; pending", len(todo("states")), flush=True)


def todo(stage):
    def left(name):
        missing = set(store.pending(name, N_DATES))
        return [d for d in (ALL_DATES if name == "gpu_fits" else HARD) if p.chunk_id(d) in missing]

    if stage == "states":
        return left("gpu_fits")
    if stage == "hindsight":
        return (["finished"] if store.pending("finished_fit", 1) else []) + left("hindsight")
    return left("strata")


def states(n):
    p.set_precision("float32")
    triangle, days, population = c.load_wide()
    for date in todo("states")[: int(n)]:
        cid = p.chunk_id(date)
        d = c.as_of_window(triangle, days, population, date)

        def fit(seed):
            with gpu_lock():
                model = c.build_model(d)
                out, extra, _ = c.fit_gpu(d, model, seed=seed, **DRAWS)
                return out, extra, vram_high_water()

        run_with_restart("gpu_fits", date, cid, f"gpu_fit_{date}", fit)
        cleanup()


def finished_fit(triangle, days, population):
    """Section 6, variant 2: delay parameters from the finished triangle over all eight windows."""
    d = c.finished_window(triangle, days, population, *FINISHED_ROWS)
    names = ("mu7", "phi", "gamma_d", "gamma_s_d", "rho_d", "beta_d", "beta_s_d", "frac7")
    for attempt, seed in enumerate((9001, 9002)):
        machine()
        with gpu_lock():
            model = c.build_model(d)
            out, extra, _ = c.fit_gpu(d, model, seed=seed, outputs=names, keep_named=names[2:], summary=False, **DRAWS)
            vram = vram_high_water()
        rhat, ess, _, _ = c.rhat_ess(out["n_frac7"])
        verdict = {"r_hat": float(rhat.max()), "ess": float(ess.min()), "divergence": extra["divergence_rate"]}
        verdict["pass"] = (verdict["r_hat"] <= LINE["r_hat"] and verdict["ess"] >= LINE["ess"]
                           and verdict["divergence"] <= LINE["divergence"])
        print("finished fit", verdict, extra["wall_s"], "largest mu7 by chain", out["mu7_max_by_chain"].round(0).tolist(),
              "frac7 by chain, state 0", out["n_frac7"][:, :, 0].mean(axis=1).round(3).tolist(), flush=True)
        if verdict["pass"] or attempt == 1:
            arrays = {n: out[f"n_{n}"].mean(axis=(0, 1)) for n in names[2:]}
            store.save_chunk("finished_fit", 0, arrays)
            store.save_json("finished_fit", {**extra, "seed": seed, "attempt": attempt, "gate": verdict,
                                             "vram": vram, "health": p.health(), "rows": FINISHED_ROWS})
            return
        store.save_json("failed_finished_fit", {**extra, "seed": seed, "gate": verdict})


def hindsight_window(triangle, days, population, date, mean, long):
    """The as-of window with the finished hazard and the tail of rows known only later."""
    d = c.as_of_window(triangle, days, population, date)
    tails = [c.tail_params(triangle[c.b.LOCATIONS.index(s), 0], d["asof"] + 80) for s in c.STATES]
    d = {**d, "tail": np.array(tails)}
    first = d["asof"] - c.T + 1 - long["first"]
    return d, c.finished_logit(mean, long)[:, first : first + c.T]


def hindsight(n):
    p.set_precision("float32")
    triangle, days, population = c.load_wide()
    left = int(n)
    for date in todo("hindsight"):
        if left == 0:
            return
        if date == "finished":
            finished_fit(triangle, days, population)
            cleanup()
            left -= 1
            continue
        long = c.finished_window(triangle, days, population, *FINISHED_ROWS)
        d, logit = hindsight_window(triangle, days, population, date, store.load_chunk("finished_fit", 0), long)

        def fit(seed):
            with gpu_lock():
                model = c.build_model(d, fixed_logit=logit)
                out, extra, _ = c.fit_gpu(d, model, seed=seed, outputs=("mu7", "phi"), **DRAWS)
                return out, extra, vram_high_water()

        run_with_restart("hindsight", date, p.chunk_id(date), f"hindsight_{date}", fit)
        cleanup()
        left -= 1


def strata(n):
    p.set_precision("float32")
    triangle, days, population = c.load_wide()
    for date in todo("strata")[: int(n)]:
        d = c.strata_window(triangle, days, population, date)

        def fit(seed):
            with gpu_lock():
                model = c.build_strata_model(d)
                out, extra, _ = c.fit_gpu(d, model, seed=seed, outputs=("mu7", "phi"), **DRAWS)
                return out, extra, vram_high_water()

        run_with_restart("strata", date, p.chunk_id(date), f"strata_{date}", fit)
        cleanup()


def following():
    for stage in ("states", "hindsight", "strata"):
        if left := todo(stage):
            print(stage, len(left), left[0])
            return
    print("none 0 -")


JOBS = {"prep": prep, "next": following, "states": states, "hindsight": hindsight, "strata": strata}

if __name__ == "__main__":
    if sys.argv[1] != "next":
        print("job", sys.argv[1:], "gates", sha256_file(store.path / f"{GATES}.json")[:16], "health", machine(), flush=True)
    t0 = time.perf_counter()
    JOBS[sys.argv[1]](*sys.argv[2:])
    if sys.argv[1] != "next":
        print(f"job done in {time.perf_counter() - t0:.1f} s; health {machine()}", flush=True)
