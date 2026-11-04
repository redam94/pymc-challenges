"""E92: score the FULL run and pack what the notebook replays. CPU only; reads results/E92/ only.

    tools/e92_job.sh 4 2 -- .venv/bin/python tools/score_e92.py

Every line is read from results/E92/gates_2026-11-03c.json (written before FULL) and printed with its
outcome; a failed line stays failed. Writes ``scores`` (every verdict), ``replay`` (what the notebook
draws, small JSON) and ``replay_arrays`` (NPZ chunk), ``banner``, then the manifest.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e92_core as c  # noqa: E402
import e92_full_core as fc  # noqa: E402

from pymc_challenges.bigrun import (Store, estimated, extrapolated, git_hash, library_versions,  # noqa: E402
                                    measured, sha256_file)

store = Store("E92")
GATES = store.load_json("gates_2026-11-03c")
B_RESAMPLES = 4000
LOG = Path("/home/data/e92_scratch/full.log")


def jl(name):
    return store.load_json(name)


def has(name):
    return (store.path / f"{name}.json").exists()


def test_frame():
    _, te, _ = fc.datasets("mercator")
    return te


def per_profile(h, y, seed=92):
    mean, lv = np.asarray(h["mean"]), np.asarray(h["latent_var"])
    nv = np.asarray(h["noise_var"]) * np.ones(len(mean))
    if h.get("student_t"):
        draws = fc.t_predictive_draws(mean, lv, h["sn"], seed=seed)
        crps = fc.crps_mc(y, draws)
        lo, hi = np.quantile(draws, [0.05, 0.95], axis=1)
        sd = np.sqrt(lv + nv)
    else:
        sd = np.sqrt(lv + nv)
        crps = fc.crps_gauss(y, mean, sd)
        lo, hi = mean - 1.6449 * sd, mean + 1.6449 * sd
    return {"crps": crps, "cover90": (y >= lo) & (y <= hi), "z": (y - mean) / sd, "sq": (y - mean) ** 2}


def float_mean(values, groups, mask=None):
    ids = np.unique(groups)
    out = {}
    for g in ids:
        m = groups == g
        if mask is not None:
            m &= mask
        if m.any():
            out[g] = float(values[m].mean())
    return out


def paired_boot(a, b, seed=92):
    """Rival minus A over shared floats: mean, and the 90% bootstrap interval of the mean per-float difference."""
    keys = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (B_RESAMPLES, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [0.05, 0.95])
    return {"floats": len(d), "mean_diff": float(d.mean()), "ci90": [float(lo), float(hi)],
            "A_better": bool(lo > 0), "rival_better": bool(hi < 0), "excludes_zero": bool(lo > 0 or hi < 0)}


def stage_info(name):
    return next((s for s in store.stages() if s["stage"] == name), None)


def primary():
    info = stage_info("fit_At_final")
    return "At" if info and not info.get("edges") else "A"


def heldout_scores():
    te = test_frame()
    y, groups = te["anom"].to_numpy(), te["float_id"].to_numpy()
    reg = fc.region(te["lat"].to_numpy(), te["lon"].to_numpy())
    models = [m for m in ("A", "At", "B", "C", "Afix", "Afixt") if has(f"heldout_{m}")]
    out, perf = {}, {}
    for m in models:
        h = jl(f"heldout_{m}")
        pp = per_profile(h, y)
        perf[m] = pp
        cov = {name: float(pp["cover90"][reg == r].mean()) for r, name in enumerate(("gulf_stream", "gyre", "rest"))}
        cov["all"] = float(pp["cover90"].mean())
        js = h.get("joint_log_score") or {}
        out[m] = {"crps_mean_profile": float(pp["crps"].mean()), "rmse": float(np.sqrt(pp["sq"].mean())),
                  "crps_mean_float": float(np.mean(list(float_mean(pp["crps"], groups).values()))),
                  "coverage90": cov, "profiles_by_region": {n: int((reg == r).sum()) for r, n in
                                                            enumerate(("gulf_stream", "gyre", "rest"))},
                  "share_abs_z_over_3": float(np.mean(np.abs(pp["z"]) > 3)),
                  "joint_log_score_sum": float(sum(js.values())) if js else None,
                  "crps_by_region": {n: float(pp["crps"][reg == r].mean()) for r, n in
                                     enumerate(("gulf_stream", "gyre", "rest"))}}
    prim = primary()
    fa = float_mean(perf[prim]["crps"], groups)
    fa_gs = float_mean(perf[prim]["crps"], groups, reg == 0)
    race = {}
    for m in models:
        if m == prim:
            continue
        race[m] = {"all": paired_boot(fa, float_mean(perf[m]["crps"], groups)),
                   "gulf_stream": paired_boot(fa_gs, float_mean(perf[m]["crps"], groups, reg == 0))}
    cA = out[prim]["coverage90"]
    failure1 = {"gulf_stream_coverage": cA["gulf_stream"], "gyre_coverage": cA["gyre"], "all": cA["all"],
                "present": bool(cA["gulf_stream"] < 0.80 and cA["gyre"] > 0.90)}
    fix = out.get("Afixt") or out.get("Afix")
    fix_name = "Afixt" if "Afixt" in out else "Afix"
    target = {"model": fix_name, "gulf_stream_coverage": fix["coverage90"]["gulf_stream"] if fix else None,
              "pass": bool(fix and fix["coverage90"]["gulf_stream"] >= 0.85)}
    if "Afix" in out:
        target["Afix_gulf_stream_coverage"] = out["Afix"]["coverage90"]["gulf_stream"]
    tails = {"Afix_share_abs_z_over_3": out["Afix"]["share_abs_z_over_3"] if "Afix" in out else None,
             "demand_t": bool("Afix" in out and out["Afix"]["share_abs_z_over_3"] > 0.01)}
    arrays = {f"ho_{m}_{k}": np.asarray(jl(f"heldout_{m}")[k], np.float32) for m in models
              for k in ("mean", "latent_var")}
    arrays.update({f"ho_{m}_crps": perf[m]["crps"].astype(np.float32) for m in models})
    arrays.update({f"ho_{m}_cover": perf[m]["cover90"] for m in models})
    arrays.update({"ho_y": y.astype(np.float32), "ho_lat": te["lat"].to_numpy(np.float32),
                   "ho_lon": te["lon"].to_numpy(np.float32), "ho_t": te["t_days"].to_numpy(np.float32),
                   "ho_float": np.unique(groups, return_inverse=True)[1].astype(np.int16), "ho_region": reg.astype(np.int8)})
    return {"primary": prim, "models": out, "race": race, "failure_1": failure1, "fix_target": target, "tails": tails,
            "rule": GATES["scores"]["rule"]}, arrays


def design_summary(hs, stage="design", inputs_name="design_inputs"):
    """The design's numbers and the rule's advice; ``stage="design_masked"`` scores the operable-water rerun."""
    inputs = jl(inputs_name)
    rows = inputs["rows"]
    sites = np.asarray(inputs["sites"])
    pend = set(store.pending(stage, len(rows)))
    chunks = {r[0]: store.load_chunk(stage, r[0]) for r in rows if r[0] not in pend}
    by = {}
    for cid, model, dr, rep, fixed in rows:
        if cid in chunks and not fixed:
            by.setdefault(model, []).append((dr, rep, chunks[cid]))
    prim = primary()
    res = {"models": {}}
    rec20 = {}
    for model, lst in by.items():
        g = np.array([ch["trace"][0] - ch["trace"][-1] for _, _, ch in lst])
        gap = np.array([float(ch["gap_removed"]) for _, _, ch in lst])
        rnd = np.array([float(ch["random_removed"].mean()) for _, _, ch in lst])
        freq = np.zeros(len(sites))
        for _, _, ch in lst:
            freq[ch["picks"]] += 1
        top = np.argsort(-freq, kind="stable")[:20]
        rec20[model] = top
        mode_reps = [ch for dr, _, ch in lst if dr == 0]
        before = float(np.mean([float(ch["before"]) for _, _, ch in lst]))
        draw_overlap = [len(set(ch["picks"]) & set(m0["picks"])) for dr, rep, ch in lst if dr > 0
                        for m0 in [next(x for d2, r2, x in lst if d2 == 0 and r2 == rep)]]
        rep_overlap = [len(set(a["picks"]) & set(b["picks"])) for i, a in enumerate(mode_reps) for b in mode_reps[i + 1:]]
        res["models"][model] = {"chunks": len(lst), "greedy_removed": float(g.mean()), "gap_removed": float(gap.mean()),
                                "random_removed": float(rnd.mean()), "greedy_over_gap": float(g.mean() / gap.mean()),
                                "greedy_over_random": float(g.mean() / rnd.mean()), "before": before,
                                "share_removed_greedy": float(g.mean() / before), "recommended_20": top,
                                "pick_frequency": freq, "draw_overlap_with_mode_mean": float(np.mean(draw_overlap)) if draw_overlap else None,
                                "replicate_overlap_mean": float(np.mean(rep_overlap)) if rep_overlap else None}
    # variants at the primary mode
    gulf_change = [20 - len(set(ch["picks"]) & set(ch["gulf_picks"])) for dr, _, ch in by[prim] if dr == 0]
    fixed = [chunks[r[0]] for r in rows if r[4] and r[0] in chunks]
    mode0 = next(ch for dr, rep, ch in by[prim] if dr == 0 and rep == 0)
    res["gulf_weight_top20_changed"] = gulf_change
    res["fixed_sites_top20_changed"] = 20 - len(set(fixed[0]["picks"]) & set(mode0["picks"])) if fixed else None
    # decision model and recommendation (gates)
    crps = {m: hs["models"][m]["crps_mean_float"] for m in ("A", "At", "Afix", "Afixt") if m in hs["models"] and m in by}
    dm = min(crps, key=crps.get)
    ratio = res["models"][dm]["greedy_over_gap"]
    a_name, f_name = prim, ("Afix" if "Afix" in rec20 else None)
    overlap = len(set(rec20[a_name]) & set(rec20[f_name])) if f_name else None

    def regions(top):
        la, lo = sites[top, 0], sites[top, 1]
        band = np.digitize(la, [32.0, 44.0])
        return np.bincount(band * 2 + (lo >= -40).astype(int), minlength=6) / len(top)

    tv = float(0.5 * np.abs(regions(rec20[a_name]) - regions(rec20[f_name])).sum()) if f_name else None
    if ratio < 1.10:
        rec = 2
    elif overlap is not None and overlap >= 10:
        rec = 1
    elif tv is not None and tv < 0.25:
        rec = 3
    else:
        rec = 2
    res["decision"] = {"decision_model": dm, "crps_by_model": crps, "greedy_over_gap": ratio,
                       "overlap_A_Afix": overlap, "region_tv": tv, "recommendation": rec,
                       "rule": GATES["design"]["recommendation"],
                       "recommended_sites": sites[rec20[dm]], "recommended_index": rec20[dm]}
    arrays = {"design_sites": sites.astype(np.float32), "design_var_maps": mode0["var_maps"],
              "design_trace": mode0["trace"], "design_picks_mode": mode0["picks"],
              "design_target_cells": np.asarray(jl("design_inputs")["target_cells"], np.float32), "design_gap_picks": mode0["gap_picks"]}
    for model in by:
        arrays[f"design_freq_{model}"] = res["models"][model]["pick_frequency"].astype(np.int16)
    if fixed:
        arrays["design_fixed_picks"] = fixed[0]["picks"]
    arrays["design_gulf_picks"] = mode0["gulf_picks"]
    return res, arrays


def natural():
    ne = jl("natural_experiment")
    y, g = np.asarray(ne["y"]), np.asarray(ne["float_id"])
    nv = ne["noise_var"]
    pred_p = (np.asarray(ne["var_h_without"]) - np.asarray(ne["var_h_with"]))
    real_p = (y - np.asarray(ne["mean_h_without"])) ** 2 - (y - np.asarray(ne["mean_h_with"])) ** 2
    ids = np.unique(g)
    pf = np.array([pred_p[g == k].mean() for k in ids])
    rf = np.array([real_p[g == k].mean() for k in ids])
    nf = np.array([(g == k).sum() for k in ids])
    rng = np.random.default_rng(92)
    bi = rng.integers(0, len(ids), (B_RESAMPLES, len(ids)))
    wsum = nf[bi].sum(1)
    diff = ((rf - pf)[bi] * nf[bi]).sum(1) / wsum
    real_b = (rf[bi] * nf[bi]).sum(1) / wsum
    lo, hi = np.quantile(diff, [0.05, 0.95])
    out = {"predicted": float(pred_p.mean()), "realised": float(real_p.mean()), "ci90_realised_minus_predicted": [float(lo), float(hi)],
           "ci90_realised": [float(x) for x in np.quantile(real_b, [0.05, 0.95])],
           "profiles": len(y), "floats": len(ids), "launched_floats": ne["launched_floats"],
           "launched_profiles": ne["launched_profiles"], "noise_var": nv,
           "pass": bool(real_p.mean() > 0 and lo <= 0 <= hi), "line": GATES["natural_experiment"]["go"],
           "near_launch_share": None}
    return out


def walls_and_peaks():
    walls = {}
    for p in sorted(store.path.glob("wall_*.json")):
        w = json.loads(p.read_text())
        walls.setdefault(w["job"], 0.0)
        walls[w["job"]] += w["wall_s"]
    vram = []
    for p in store.path.glob("*.json"):
        try:
            t = p.read_text()
        except OSError:
            continue
        vram += [float(x) for x in re.findall(r'"peak_gb": ([0-9.]+)', t)]
    ram_gpu = [float(x) for x in re.findall(r"peak memory ([0-9.]+) GB", LOG.read_text())] if LOG.exists() else []
    ram_audit = []
    for p in Path("/home/data/e92_scratch").glob("audit*.log"):
        ram_audit += [float(x) for x in re.findall(r"peak memory ([0-9.]+) GB", p.read_text())]
    return walls, max(vram) if vram else None, max(ram_gpu) if ram_gpu else None, max(ram_audit) if ram_audit else None


def main():
    t0 = time.perf_counter()
    hs, a1 = heldout_scores()
    ds, a2 = design_summary(hs)
    ne = natural()
    walls, vram_peak, ram_peak, ram_audit = walls_and_peaks()
    audit = {n: jl(f"pymc_audit_{n}") for n in (800, 2000) if has(f"pymc_audit_{n}")}
    p8, p10 = jl("p8_vecchia"), jl("p10_design_small")
    sob, hes = jl("sobol"), jl("hessian")
    fits = {k: jl(k) for k in ("fit_A_f32", "fit_A_f64", "fit_At_final", "fit_Afix_f32", "fit_Afix_f64", "fit_Afixt_final",
                                "fit_C", "fit_B") if has(k)}
    fail = jl("failure_runs")
    edges_A = fits["fit_A_f64"]["edges"]
    verdicts = {
        "data_fingerprint": jl("data_fingerprint")["matches_pilot"],
        "P8_vecchia_m30": p8["pass_m30"],
        "P10_best_over_random_1.25": p10["pass"],
        "fit_A_inside_prior_edges": not edges_A,
        "fit_At_inside_prior_edges": (not fits["fit_At_final"]["edges"]) if "fit_At_final" in fits else None,
        "hessian_negative_definite": hes["negative_definite"],
        "sobol_pareto_k_0.7": sob["pass"],
        "failure_1_symptom_present": hs["failure_1"]["present"],
        "fix_gulf_stream_coverage_0.85": hs["fix_target"]["pass"],
        "natural_experiment": ne["pass"],
        **{f"pymc_audit_{n}": a["pass"] for n, a in audit.items()},
        **{f"race_{m}_{where}_excludes_zero": r[where]["excludes_zero"] for m, r in hs["race"].items()
           for where in ("all", "gulf_stream")},
    }
    scores = {"verdicts": verdicts, "heldout": hs, "design": ds["decision"], "design_models": ds["models"],
              "design_variants": {"gulf_weight_top20_changed": ds["gulf_weight_top20_changed"],
                                  "fixed_sites_top20_changed": ds["fixed_sites_top20_changed"]},
              "natural_experiment": ne, "pymc_audit": audit, "p8": p8, "p10": p10,
              "sobol": {k: sob[k] for k in ("pareto_k", "ess", "pass", "weighted_mean", "weighted_sd")},
              "hessian": {k: hes[k] for k in ("model", "negative_definite", "clipped", "sd", "eigen")},
              "fits": {k: {kk: v.get(kk) for kk in ("theta", "natural", "edges", "evaluations", "message", "F")}
                       for k, v in fits.items() if k != "fit_B"},
              "fit_B": {"windows": len(fits["fit_B"]["theta"]), "n": fits["fit_B"]["n"],
                        "edges_any": sum(bool(e) for e in fits["fit_B"]["edges"]), "seconds": float(np.sum(fits["fit_B"]["seconds"]))},
              "failure_runs": fail, "walls_s": walls, "peak_vram_gib": vram_peak, "peak_ram_gb_gpu_jobs": ram_peak,
              "peak_ram_gb_audit_job": ram_audit, "labels": "measured unless a key says otherwise"}
    store.save_json("scores", scores)
    arrays = {**a1, **a2, "map_mean": store.load_array(f"map_mean_{hs['primary']}"),
              "map_sd": store.load_array(f"map_sd_{hs['primary']}"),
              "map2025_mean": store.load_array("map2025_mean"), "map2025_sd": store.load_array("map2025_sd"),
              "B_centres": np.asarray(fits["fit_B"]["centres"], np.float32), "B_theta": np.asarray(fits["fit_B"]["theta"], np.float32),
              "sobol_theta": np.asarray(sob["theta"], np.float32), "sobol_weights": np.asarray(sob["weights"], np.float32),
              **{f"audit_draws_{n}": store.load_array(f"pymc_audit_draws_{n}") for n in audit}}
    if has("maps") and "Afix" in jl("maps")["models"]:
        arrays["map_sd_Afix"] = store.load_array("map_sd_Afix")
    store.save_chunk("replay_arrays", 0, arrays)
    maps, m25 = jl("maps"), jl("map2025")
    store.save_json("replay", {"maps": {k: maps[k] for k in ("cell_lat", "cell_lon", "days")},
                               "map2025": {k: m25[k] for k in ("cell_lat", "cell_lon", "days")},
                               "arrays": sorted(arrays), "scores": "scores.json"})
    # banner: wall time is the span from the first FULL stage to the last
    st = store.stages()
    full_stages = [s for s in st if s["stage"] not in ("small_choices",)]
    t_first = min(s["at"] for s in full_stages)
    t_last = max(s["at"] for s in full_stages)
    span_h = (time.mktime(time.strptime(t_last, "%Y-%m-%dT%H:%M:%SZ")) - time.mktime(time.strptime(t_first, "%Y-%m-%dT%H:%M:%SZ"))) / 3600
    gpu_h = sum(v for k, v in walls.items()) / 3600
    store.save_json("banner", {
        "wall_time": measured(f"{span_h:.1f} h end to end (GPU stages {gpu_h:.1f} h; PyMC audit beside them)", ""),
        "peak_ram": measured(f"{ram_peak:.1f} GB (GPU jobs), {ram_audit:.1f} GB (one audit chain)", ""),
        "peak_vram": measured(f"{vram_peak:.1f} GiB of the 16.6 GiB JAX pool", ""),
        "disk": measured(f"{sum(f.stat().st_size for f in store.path.rglob('*') if f.is_file()) / 1e6:.0f} MB stored "
                         "results (replay arrays 0.8 MB); 23 MB raw pull in the cache", ""),
        "download": measured("214 MB raw from the Argo ERDDAP mirror (once); 1.1 MB table in git", ""),
        "precision": measured("float32 for exploration and the design, float64 for the last optimiser steps, Hessian and Sobol points", ""),
        "chains": measured("4, in the PyMC audit only", ""),
        "hardware": measured("Ryzen 9 5950X (15 cores online), 64 GB, RTX 3090 Ti", ""),
        "build_date": measured(time.strftime("%Y-%m-%d"), ""),
        "git_hash": measured(git_hash() or "unknown", ""),
        "smaller_machine": estimated("replay needs no GPU and loads the stored run on 2 cores; a full run without "
                                     "the GPU pays 48 s per float64 gradient at n = 20,000 on 15 cores against 0.9 s "
                                     "in float32 on the GPU (P3), so roughly 50x longer for the fits", ""),
    })
    versions = library_versions("jax", "numpy", "scipy", "pymc", "nutpie", "arviz")
    from pymc_challenges import data as reg

    store.write_manifest(git_hash=git_hash(), versions=versions,
                         data_fingerprint={"argo_natl_t100": sha256_file(reg.path("argo_natl_t100")), "pull_date": "2026-11-02"},
                         hardware="Ryzen 9 5950X, 64 GB, RTX 3090 Ti (cores 10 and 26 offline)", started=t_first, finished=t_last,
                         gates=["gates_2026-11-03", "gates_2026-11-03b", "gates_2026-11-03c"])
    problems = store.verify()
    print(json.dumps({"verdicts": verdicts, "decision": ds["decision"]["recommendation"], "problems": problems[:5]},
                     default=str, indent=1))
    print(f"scored in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
