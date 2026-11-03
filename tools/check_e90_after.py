"""E90 checks added after the FULL run and its audit, 2026-11-03. Not in gates_full: labelled as later.

    python tools/check_e90_after.py

Reads the local vintages (tools/build_e90_data.py), the stored baselines, RKI's cached nowcasts and
data/e90_nowcasts.npz; writes results/E90/checks_after.json. One core, seconds, nothing is sampled.

1. A probabilistic factor: frozen x each of the 28 recent at-40 / frozen ratios (the factor's own
   ratios) x the median at-80 / at-40 tail of the 28 weeks complete at 80 days. Every input is known on
   the as-of date. P(final >= tau) is the share of the 28 products at or above tau. It is scored against
   the nowcast with the bootstrap of score_e90.decide (same 37 clusters, same seed and draws).
2. The width of RKI's 95% interval relative to its mean, against the nowcast's.
3. The share of cases at delays 41-80, which the factor's 40-day ratio leaves out, by state.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_e90_data as b  # noqa: E402
import e90_core as c  # noqa: E402
import pilot_e90 as p  # noqa: E402
import score_e90 as sc  # noqa: E402

from pymc_challenges import data  # noqa: E402
from pymc_challenges.bigrun import sha256_file  # noqa: E402

store = p.store
TAUS = c.TAUS


def factor_draws(triangle, days):
    span = np.arange(c.day_index(days, b.EVALUATION[0]), c.day_index(days, b.EVALUATION[1]) + 1)
    ratios = np.full((16, span.size, 28), np.nan)
    tail = np.ones((16, span.size))
    frozen = np.zeros((16, span.size))
    beyond40 = np.zeros(16)
    for i in range(16):
        tri = np.where(triangle[i + 1, 0, :, :81] == b.MISSING, 0, triangle[i + 1, 0, :, :81]).astype(float)
        cum = tri.cumsum(axis=1)
        fz = lambda t: sum(cum[r, t - r] for r in range(t - 6, t + 1))  # noqa: E731
        for j, a in enumerate(span):
            frozen[i, j] = fz(a)
            for k, t in enumerate(range(a - 40 - 27, a - 40 + 1)):
                then = fz(t)
                if then > 0:
                    ratios[i, j, k] = cum[t - 6 : t + 1, 40].sum() / then
            tails = [cum[t - 6 : t + 1, 80].sum() / cum[t - 6 : t + 1, 40].sum()
                     for t in range(a - 80 - 27, a - 80 + 1) if cum[t - 6 : t + 1, 40].sum() > 0]
            tail[i, j] = float(np.median(tails)) if tails else 1.0
        rows = span[span <= len(days) - 81]
        beyond40[i] = 1 - cum[rows, 40].sum() / cum[rows, 80].sum()
    return frozen, ratios, tail, beyond40


def main():
    t0 = time.perf_counter()
    triangle, days, population = c.load_wide()
    pop = population[1:17, 0].astype(float)
    scale = 1e5 / pop[:, None]
    frozen, ratios, tail, beyond40 = factor_draws(triangle, days)
    signals = store.load_array("baselines_signals")
    final, factor = signals[1] * scale, signals[2] * scale
    assert np.allclose(signals[0], frozen) and np.allclose(signals[2], frozen * np.nanmedian(ratios, 2))
    draws = frozen[:, :, None] * ratios * tail[:, :, None] * scale[:, :, None]
    ok = np.isfinite(draws)
    pf = np.stack([np.where(ok, draws >= t, False).sum(2) / np.maximum(ok.sum(2), 1) for t in TAUS])

    pack = np.load(data.path("e90_nowcasts"), allow_pickle=False)
    pc, q = pack["p_cross"].astype(float), pack["q"].astype(float)
    assert np.allclose(pack["final"], final) and np.allclose(pack["factor"], factor)

    # the bootstrap of score_e90.decide: same clusters, same seed, same order of draws
    n_dates = final.shape[1]
    cut = [(np.datetime64(w[1]) - p.START) / np.timedelta64(1, "D") for w in sc.WAVES]
    wave = np.searchsorted(cut, np.arange(n_dates))
    cells = wave + 3 * np.arange(16)[:, None]
    clusters = set()
    for tau in TAUS:
        clusters.update((s, int(wave[t])) for s, t in sc.crossings(final >= tau))
    rng = np.random.default_rng(90)
    keep = sorted(3 * s + w for s, w in clusters)
    weights = np.zeros((2000, 48))
    weights[:, keep] = rng.dirichlet(np.ones(len(keep)), 2000)
    n = np.bincount(cells.ravel(), minlength=48).astype(float)
    inside = np.isin(cells, keep)
    out = {"what": "checks added after the FULL run and its audit, 2026-11-03; not written before the run",
           "clusters": len(clusters), "state_days_outside_clusters": int((~inside).sum()), "prob_factor": {}}
    for j, tau in enumerate(TAUS):
        truth = final >= tau
        rows = {}
        for pt in sc.NAMED_PT:
            def boot(up):
                tp, fp = sc.net_benefit_counts(up, truth, cells, 48)
                return (weights @ tp - pt / (1 - pt) * (weights @ fp)) / (weights @ n)

            def point(up, mask=None):
                mask = np.ones_like(up) if mask is None else mask
                return float(((up & truth & mask).sum() - (up & ~truth & mask).sum() * pt / (1 - pt)))

            model_up, pf_up, fac_up = pc[j] >= pt, pf[j] >= pt, factor >= tau
            rows[str(pt)] = {"p_model_beats_prob_factor": float((boot(model_up) - boot(pf_up) > 0).mean()),
                             "p_model_beats_factor": float((boot(model_up) - boot(fac_up) > 0).mean()),
                             "nb_model": point(model_up) / final.size, "nb_prob_factor": point(pf_up) / final.size,
                             "nb_factor": point(fac_up) / final.size,
                             "outside_clusters_model_minus_factor": point(model_up, ~inside) - point(fac_up, ~inside)}
            rows[str(pt)]["win"] = rows[str(pt)]["p_model_beats_prob_factor"] >= 0.9
        out["prob_factor"][f"tau{tau}"] = {"rows": rows, "wins": int(sum(r["win"] for r in rows.values()))}
    srt = np.sort(np.where(ok, draws, np.nan), axis=2)
    crps = []
    for s in range(16):
        for t in range(n_dates):
            x = srt[s, t][np.isfinite(srt[s, t])]
            if x.size:
                crps.append(float(sc.crps_draws(x[:, None], np.array([final[s, t]]))[0]))
    lo80, hi80 = np.nanquantile(draws, 0.1, 2), np.nanquantile(draws, 0.9, 2)
    lo95, hi95 = np.nanquantile(draws, 0.025, 2), np.nanquantile(draws, 0.975, 2)
    out["prob_factor_scores"] = {"crps": float(np.mean(crps)), "cover_80": float(((lo80 <= final) & (final <= hi80)).mean()),
                                 "cover_95": float(((lo95 <= final) & (final <= hi95)).mean()),
                                 "median_draws": float(np.median(ok.sum(2)))}

    width = []
    for path in sorted((b.CACHE / "rki").glob("*.csv")):
        found = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["age_group"] != "00+" or row["location"] not in c.STATES or row["target"] != f"-{sc.RKI_LAG} day ahead inc hosp":
                    continue
                found.setdefault(row["location"], {})[row["type"] + row["quantile"]] = float(row["value"])
        width += [(v["quantile0.975"] - v["quantile0.025"]) / v["mean"] for v in found.values()
                  if {"quantile0.975", "quantile0.025", "mean"} <= v.keys() and v["mean"] > 0]
    out["rki_width_95_over_mean"] = {"median": float(np.median(width)), "q10": float(np.quantile(width, 0.1)),
                                     "q90": float(np.quantile(width, 0.9)), "n": len(width), "rki_lag_days": sc.RKI_LAG}
    out["model_width_95_over_median"] = float(np.median((q[6] - q[0]) / q[3]))
    out["share_beyond_40_days"] = dict(zip(c.STATES, map(float, beyond40)))
    # the run's code as it stands after the run (results/E90/manifest.json records no code hash)
    tools = Path(__file__).resolve().parent
    out["code_sha256_after_run"] = {name: sha256_file(tools / name) for name in (
        "e90_core.py", "run_e90_full.py", "pilot_e90.py", "score_e90.py", "pack_e90.py", "build_e90_data.py",
        "check_e90_after.py")}
    out["wall_s"] = time.perf_counter() - t0
    store.save_json("checks_after", out)
    print({k: v for k, v in out.items() if k not in ("share_beyond_40_days", "code_sha256_after_run")})
    print({k: round(v, 4) for k, v in out["share_beyond_40_days"].items()})


if __name__ == "__main__":
    main()
