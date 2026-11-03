"""E90 scores and decisions after FULL: stages 9 to 11 of plans/large/E90.md, lines in gates_full_2026-11-02.json.

    python tools/score_e90.py rki | score | decide | banner LOGDIR

Reads the stored fits only, on one core, seconds to minutes. `score` and `decide` also run on a part of
the fits: they then cover the dates present and say how many. Plain NumPy, nothing is sampled.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_e90_data as b  # noqa: E402
import e90_core as c  # noqa: E402
import pilot_e90 as p  # noqa: E402
import run_e90_full as r  # noqa: E402

from pymc_challenges.bigrun import git_hash, library_versions, sha256_file  # noqa: E402

store = p.store
TAUS = c.TAUS
LEVELS = {50: (2, 4), 80: (1, 5), 95: (0, 6)}
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
NAMED_PT = (0.2, 0.33, 0.5)
ALERT_P = (0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9)
RECAL_START = 93  # 2021-12-17: the first date with 14 dates at least 80 days older
HORIZON = 80
RKI_LAG = 3
WAVES = (("Delta", "2021-12-14"), ("BA.1", "2022-02-28"), ("BA.2", "2022-04-29"))


class Data:
    """Fits and signals on one grid: dates x states, incidence per 100,000."""

    def __init__(self):
        self.triangle, self.days, self.population = c.load_wide()
        self.pop = self.population[1:17, 0].astype(float)
        signals, self.span = store.load_array("baselines_signals"), store.load_array("baselines_span")
        scale = 1e5 / self.pop[:, None]
        self.frozen, self.final, self.factor, self.stale, self.ladder = [x * scale for x in signals]
        self.all_dates = [str(p.START + np.timedelta64(k, "D")) for k in range(self.final.shape[1])]
        have = store._chunk_ids("gpu_fits")
        self.ids = np.array([i for i in have if i < self.final.shape[1]])
        self.q = np.full((7, 16, self.final.shape[1]), np.nan)
        self.pc = np.full((3, 16, self.final.shape[1]), np.nan)
        self.keep = {}
        for i in self.ids:
            fit = store.load_chunk("gpu_fits", int(i))
            self.q[:, :, i], self.pc[:, :, i], self.keep[int(i)] = fit["q"], fit["p_cross"], fit["keep"].astype(float)
        self.has = np.zeros(self.final.shape[1], bool)
        self.has[self.ids] = True
        cut = [(np.datetime64(w[1]) - p.START) / np.timedelta64(1, "D") for w in WAVES]
        self.wave = np.searchsorted(cut, np.arange(self.final.shape[1]))
        self._recal()

    def _recal(self):
        """Section 6's rule: one k per date from state-dates whose truth was known by then."""
        n = self.final.shape[1]
        med, half = self.q[3], (np.log(self.q[5]) - np.log(self.q[1])) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(np.log(self.final) - np.log(med)) / half
        self.k = np.ones(n)
        self.qr, self.pr = self.q.copy(), self.pc.copy()
        for j in range(RECAL_START, n):
            if not self.has[j]:
                continue
            old = ratio[:, : j - HORIZON + 1][:, self.has[: j - HORIZON + 1]]
            old = old[np.isfinite(old)]
            if old.size < 14 * 16 * 0.5:
                continue
            self.k[j] = max(1.0, float(np.quantile(old, 0.8)))
            self.qr[:, :, j] = med[:, j] * (self.q[:, :, j] / med[:, j]) ** self.k[j]
            x = med[:, j] * (self.keep[j] / med[:, j]) ** self.k[j]
            self.pr[:, :, j] = np.stack([(x >= t).mean(0) for t in TAUS])


def crps_draws(x, y):
    """Sample CRPS of draws x (n, ...) against y (...): E|X-y| - 0.5 E|X-X'|, by sorted draws."""
    n = x.shape[0]
    s = np.sort(x, axis=0)
    weight = (2 * np.arange(1, n + 1) - n - 1).reshape((-1,) + (1,) * (x.ndim - 1))
    return np.abs(x - y).mean(0) - (weight * s).sum(0) / n**2


def coverage(q, final, mask):
    return {str(level): float(((q[a][mask] <= final[mask]) & (final[mask] <= q[z][mask])).mean())
            for level, (a, z) in LEVELS.items()}


def mae(x, final, mask):
    ok = mask & (x > 0) & (final > 0)
    return float(np.abs(np.log(x[ok]) - np.log(final[ok])).mean()), int((mask & ~ok).sum())


def fp_fn(up, truth, mask):
    return [int((up & ~truth & mask).sum()), int((~up & truth & mask).sum())]


def held_out(data):
    """Section 6's predictive check: cells revealed after the as-of date, no refit."""
    rng = np.random.default_rng(90)
    daily = (np.arange(c.N_HAZARD) < 15).astype(float)
    inside, total, pit = 0, 0, np.zeros(10)
    by_date, skipped = {}, []
    for i in data.ids:
        date = data.all_dates[int(i)]
        fit = store.load_chunk("gpu_fits", int(i))
        if "u_phi" not in fit:  # 2022-01-13 is P4's fit, stored before these draws were added
            skipped.append(date)
            continue
        d = c.as_of_window(data.triangle, data.days, data.population, date)
        step = max(1, fit["u_phi"].shape[0] // 500)
        phi, base = fit["u_phi"][::step].astype(float), fit["u_base"][::step].astype(float)
        rho, wd = fit["u_rho_d"][::step].astype(float), fit["u_wd_d"][::step].astype(float)
        lvl, grw = fit["u_lvl_end"][::step].astype(float), fit["u_grw_end"][::step].astype(float)
        n = phi.shape[0]
        hits = cells = 0
        for s, state in enumerate(c.STATES):
            block = data.triangle[b.LOCATIONS.index(state), 0]
            later, _ = b.asof_nonnegative(block[d["asof"] - c.T + 1 : d["asof"] + 81], c.T + 79)
            for row in range(c.K):
                t = c.T - c.K + row
                dow = d["dow"][t]
                logit = base[:, s] + rho[:, s, (dow + np.arange(c.N_HAZARD)) % 7] * daily
                h = 1 / (1 + np.exp(-logit))
                keep_going = np.cumprod(1 - h, axis=1)
                probs = np.concatenate([h[:, :1], keep_going[:, :-1] * h[:, 1:], keep_going[:, -1:]], axis=1)
                lam = np.exp(lvl[:, s] - (c.K - 1 - row) * grw[:, s] + wd[:, s, dow])
                for k, (lo, hi) in enumerate(c.BIN_EDGES):
                    if not c.FUTURE[row, k]:
                        continue
                    mu = lam * probs[:, k]
                    draw = rng.poisson(rng.gamma(phi[:, s], mu / phi[:, s]))
                    truth = later[t, lo : hi + 1].sum()
                    low, high = np.quantile(draw, [0.1, 0.9])
                    hits += int(low <= truth <= high)
                    cells += 1
                    pit[min(9, int(10 * ((draw < truth).mean() + 0.5 * (draw == truth).mean())))] += 1
        by_date[date] = hits / cells
        inside, total = inside + hits, total + cells
    return {"coverage_80": inside / total, "cells": total, "pit_deciles": (pit / total).tolist(),
            "worst_dates": sorted(by_date.items(), key=lambda kv: kv[1])[:10], "draws_per_fit": 500, "skipped": skipped}


def rki_table(data):
    """RKI's real-time nowcast per target day: on as-of target + 3, so it saw 3 more days than we did."""
    out = {name: np.full(data.final.shape, np.nan) for name in ("mean", "low", "high")}
    for path in sorted((b.CACHE / "rki").glob("*.csv")):
        forecast = path.stem
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["age_group"] != "00+" or row["location"] not in c.STATES:
                    continue
                if row["target"] != f"-{RKI_LAG} day ahead inc hosp":
                    continue
                j = int((np.datetime64(row["target_end_date"]) - p.START) / np.timedelta64(1, "D"))
                if not 0 <= j < data.final.shape[1]:
                    continue
                s = c.STATES.index(row["location"])
                value = float(row["value"]) * 1e5 / data.pop[s]
                if row["type"] == "mean":
                    out["mean"][s, j] = value
                elif row["quantile"] in ("0.025", "0.975"):
                    out["low" if row["quantile"] == "0.025" else "high"][s, j] = value
    return out


def rki_prob(rki, tau):
    """Label: an approximation from a mean and two quantiles, lognormal."""
    from scipy.stats import norm

    with np.errstate(divide="ignore", invalid="ignore"):
        mu, sd = np.log(rki["mean"]), (np.log(rki["high"]) - np.log(rki["low"])) / (2 * 1.96)
        return 1 - norm.cdf((np.log(tau) - mu) / sd)


def rki():
    """Fetch the benchmark: section 3, about 17 MB, local scoring only."""
    missing = []
    for day in np.arange(np.datetime64("2021-12-01"), np.datetime64("2022-05-03")):
        try:
            b.fetch(b.RKI_URL.format(day), b.CACHE / "rki" / f"{day}.csv")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            missing.append(str(day))
    files = sorted((b.CACHE / "rki").glob("*.csv"))
    print("files", len(files), "missing", len(missing), missing[:10], "bytes", sum(f.stat().st_size for f in files), flush=True)
    store.save_json("rki_fetch", {"files": len(files), "missing": missing, "bytes": sum(f.stat().st_size for f in files),
                                  "sha256": {f.name: sha256_file(f) for f in files}})


def score():
    data = Data()
    full = np.broadcast_to(data.has, data.final.shape)
    final = data.final
    med = data.q[3]
    result = {"dates_scored": int(data.has.sum()), "state_days": int(full.sum()), "signals": {}}
    for name, x in (("frozen", data.frozen), ("revised_7d_stale", data.stale), ("factor", data.factor),
                    ("ladder", data.ladder), ("model", np.where(full, med, 1.0))):
        m, excl = mae(x, final, full)
        row = {"mae_log": m, "excluded": excl, "crps": float(np.abs(x - final)[full].mean())}
        for tau in TAUS:
            row[f"tau{tau}"] = fp_fn(x >= tau, final >= tau, full)
        result["signals"][name] = row
    cr = np.full(final.shape, np.nan)
    for i, draws in data.keep.items():
        cr[:, i] = crps_draws(draws, final[:, i])
    result["signals"]["model"]["crps"] = float(np.nanmean(cr))
    result["signals"]["model"]["fp_fn_at_p"] = {
        str(a): {f"tau{t}": fp_fn(np.nan_to_num(data.pc[j]) >= a, final >= t, full) for j, t in enumerate(TAUS)} for a in (0.2, 0.33, 0.5)}
    later = full & (np.arange(final.shape[1]) >= RECAL_START)
    result["coverage"] = {"whole": coverage(data.q, final, full),
                          "by_state": {s: coverage(data.q[:, i], final[i], full[i]) for i, s in enumerate(c.STATES)},
                          "by_wave": {w[0]: coverage(data.q, final, full & (data.wave == k)) for k, w in enumerate(WAVES)},
                          "before_recal_start": coverage(data.q, final, full & ~later),
                          "from_recal_start_raw": coverage(data.q, final, later) if later.any() else None,
                          "from_recal_start_recalibrated": coverage(data.qr, final, later) if later.any() else None,
                          "k_median": float(np.median(data.k[RECAL_START:][data.has[RECAL_START:]])) if later.any() else None,
                          "k": data.k.tolist()}
    if later.any():
        result["coverage"]["recalibrated_by_state"] = {s: coverage(data.qr[:, i], final[i], later[i]) for i, s in enumerate(c.STATES)}
    rows = []
    for j in np.flatnonzero(data.has):
        one = np.zeros_like(full)
        one[:, j] = True
        rows.append({"date": data.all_dates[j], "mae": mae(med, final, one)[0], "factor": mae(data.factor, final, one)[0],
                     "cover_80": coverage(data.q, final, one)["80"], "crps": float(np.nanmean(cr[:, j]))})
    result["by_date"] = rows
    result["model_wins_dates"] = sum(row["mae"] < row["factor"] for row in rows)
    ca = np.array([row["cover_80"] for row in rows])
    result["cover_80_quantiles_by_date"] = np.quantile(ca, [0, 0.25, 0.5, 0.75, 1]).tolist()
    result["by_state_mae"] = {s: {"model": mae(med[i : i + 1], final[i : i + 1], full[i : i + 1])[0],
                                  "factor": mae(data.factor[i : i + 1], final[i : i + 1], full[i : i + 1])[0]} for i, s in enumerate(c.STATES)}
    result["by_wave_mae"] = {w[0]: {"model": mae(med, final, full & (data.wave == k))[0], "factor": mae(data.factor, final, full & (data.wave == k))[0]} for k, w in enumerate(WAVES)}
    christmas = np.zeros(final.shape[1], bool)
    christmas[(np.datetime64("2021-12-24") - p.START).astype(int) : (np.datetime64("2022-01-06") - p.START).astype(int) + 1] = True
    hol = full & christmas
    if hol.any():
        result["christmas_dates_mae"] = {"model": mae(med, final, hol)[0], "factor": mae(data.factor, final, hol)[0],
                                         "cover": coverage(data.q, final, hol), "dates": "2021-12-24 to 2022-01-06"}
    rk = rki_table(data)
    rmask = full & np.isfinite(rk["mean"]) & np.isfinite(rk["low"])
    if rmask.any():
        rq = lambda x: (rk["low"][x] <= final[x]) & (final[x] <= rk["high"][x])
        result["rki"] = {"state_days": int(rmask.sum()), "note": f"RKI's value on as-of target + {RKI_LAG}: it saw {RKI_LAG} more days than the model. Local scoring, terms unconfirmed",
                         "cover_95": float(rq(rmask).mean()), "model_cover_95_same_days": coverage(data.q, final, rmask)["95"],
                         "mae_log": mae(rk["mean"], final, rmask)[0], "model_mae_log_same_days": mae(med, final, rmask)[0],
                         "factor_mae_log_same_days": mae(data.factor, final, rmask)[0],
                         "by_state_cover_95": {s: float(rq(rmask & (np.arange(16)[:, None] == i)).mean()) for i, s in enumerate(c.STATES)}}
    result["held_out"] = held_out(data)
    result.update(variants(data, cr))
    print(json.dumps({k: v for k, v in result.items() if k not in ("by_date",)}, indent=0, default=float)[:6000], flush=True)
    store.save_json("scores", result)
    return result


def variants(data, cr):
    out = {}
    pick = [r.p.chunk_id(d) for d in r.HARD]
    final = data.final
    hind = [i for i in pick if not store.pending("hindsight", i + 1)[-1:] == [i] and data.has[i]]
    if hind:
        rows, a, z = [], [], []
        for i in hind:
            fit = store.load_chunk("hindsight", i)
            one = crps_draws(fit["keep"].astype(float), final[:, i])
            mask = np.zeros_like(final, bool)
            mask[:, i] = True
            q = fit["q"]
            rows.append({"date": data.all_dates[i], "hindsight_crps": float(one.mean()), "honest_crps": float(cr[:, i].mean()),
                         "hindsight_mae": float(np.abs(np.log(q[3]) - np.log(final[:, i])).mean()), "honest_mae": float(np.abs(np.log(data.q[3, :, i]) - np.log(final[:, i])).mean()),
                         "hindsight_cover_80": float(((q[1] <= final[:, i]) & (final[:, i] <= q[5])).mean()), "honest_cover_80": float(((data.q[1, :, i] <= final[:, i]) & (final[:, i] <= data.q[5, :, i])).mean())})
            a.append(rows[-1]["hindsight_crps"])
            z.append(rows[-1]["honest_crps"])
        gap = 1 - np.mean(a) / np.mean(z)
        out["hindsight"] = {"rows": rows, "hindsight_crps": float(np.mean(a)), "honest_crps": float(np.mean(z)), "crps_lower_by": float(gap),
                            "failure_2_appears": bool(gap > 0.05), "n": len(rows)}
    strat = [i for i in pick if not store.pending("strata", i + 1)[-1:] == [i] and data.has[i]]
    if strat:
        changed, dp, rows, ps = 0, [], [], []
        for i in strat:
            fit = store.load_chunk("strata", i)
            gap = fit["p_cross"] - data.pc[:, :, i]
            n = int(((fit["p_cross"] >= 0.5) != (data.pc[:, :, i] >= 0.5)).sum())
            changed += n
            dp.append(np.abs(gap))
            ps.append({"date": data.all_dates[i], "p_state": data.pc[:, :, i].tolist(), "p_strata": fit["p_cross"].tolist()})
            cs = crps_draws(fit["keep"].astype(float), final[:, i]) if fit["keep"].shape[1] == 16 else None
            rows.append({"date": data.all_dates[i], "alerts_changed_at_0.5": n, "dp_max": float(np.abs(gap).max()),
                         "strata_crps": float(cs.mean()), "state_crps": float(cr[:, i].mean()),
                         "strata_mae": float(np.abs(np.log(fit["q"][3]) - np.log(final[:, i])).mean()), "state_mae": float(np.abs(np.log(data.q[3, :, i]) - np.log(final[:, i])).mean()),
                         "strata_cover_80": float(((fit["q"][1] <= final[:, i]) & (final[:, i] <= fit["q"][5])).mean())})
        dp = np.stack(dp)
        out["strata"] = {"rows": rows, "n_dates": len(rows), "alerts_compared": int(dp.size), "alerts_changed": changed, "dp_max": float(dp.max()), "dp_median": float(np.median(dp)), "slope": ps}
    return out


def crossings(up):
    n = up.shape[1]
    return [(s, t) for s in range(16) for t in range(3, n - 2) if up[s, t : t + 3].all() and not up[s, t - 3 : t].any()]


def lateness(up, cross):
    late, never = [], 0
    for s, t in cross:
        hits = np.flatnonzero(up[s, max(0, t - 10) :])
        if hits.size:
            late.append(int(hits[0]) + max(0, t - 10) - t)
        else:
            never += 1
    return {"n": len(cross), "median_late": float(np.median(late)) if late else None, "never": never, "late": late}


def alert_periods(up, release):
    """Up on the first day the signal is up; down after `release` days in a row below."""
    state = np.zeros_like(up)
    for s in range(up.shape[0]):
        on, below = False, 0
        for t in range(up.shape[1]):
            if up[s, t]:
                on, below = True, 0
            elif on:
                below += 1
                if below >= release:
                    on = False
            state[s, t] = on
    return state


def net_benefit_counts(up, truth, ids, n_cells):
    tp = np.bincount(ids[up & truth], minlength=n_cells).astype(float)
    fp = np.bincount(ids[up & ~truth], minlength=n_cells).astype(float)
    return tp, fp


def decide():
    data = Data()
    final, has = data.final, data.has
    full = np.broadcast_to(has, final.shape)
    alert = {"frozen": data.frozen, "revised_7d_stale": data.stale, "factor": data.factor, "ladder": data.ladder}
    cells = data.wave + 3 * np.arange(16)[:, None]  # 48 state x wave
    rk = rki_table(data)
    rng = np.random.default_rng(90)
    result = {"dates": int(has.sum()), "lateness": {}, "net_benefit": {}, "release": {}, "rules": {}, "fp_fn_tradeoff": {}}
    clusters = set()
    crossing = {}
    for tau in TAUS:
        cross = crossings(final >= tau)
        crossing[tau] = cross
        clusters.update((s, int(data.wave[t])) for s, t in cross)
    result["crossings"] = {str(t): len(v) for t, v in crossing.items()}
    result["clusters"] = len(clusters)
    keep_ids = sorted(3 * s + w for s, w in clusters)
    weights = np.zeros((2000, 48))
    weights[:, keep_ids] = rng.dirichlet(np.ones(len(keep_ids)), 2000)
    days_per = np.bincount(cells[full], minlength=48).astype(float)
    for j, tau in enumerate(TAUS):
        truth = final >= tau
        near = np.abs(final - tau) <= 0.25 * tau
        model_p, recal_p = np.nan_to_num(data.pc[j]), np.nan_to_num(data.pr[j])
        rp = rki_prob(rk, tau)
        late = {}
        for w, wave in enumerate(WAVES):
            part = [x for x in crossing[tau] if data.wave[x[1]] == w and has[x[1]]]
            row = {n: lateness((x >= tau) & full, part) for n, x in alert.items()}
            for a in (0.2, 0.33, 0.5):
                row[f"model_p{a}"] = lateness((model_p >= a) & full, part)
            late[wave[0]] = row
        result["lateness"][f"tau{tau}"] = late
        table = {}
        for sub_name, sub in (("all", full), ("near", full & near)):
            n_days = np.bincount(cells[sub], minlength=48).astype(float)
            denom_b = weights @ n_days
            denom_48 = n_days.sum()
            sub_tab = {}
            for name in (*alert, "model", "model_recalibrated", "rki_approx"):
                rows = []
                for pt in GRID:
                    if name in alert:
                        up = alert[name] >= tau
                    elif name == "model":
                        up = model_p >= pt
                    elif name == "model_recalibrated":
                        up = recal_p >= pt
                    else:
                        up = np.nan_to_num(rp) >= pt
                    if name == "rki_approx":
                        mask = sub & np.isfinite(rp)
                        tp, fp = net_benefit_counts(up & mask, truth, cells, 48)
                        rows.append({"pt": float(pt), "point_48": float((tp.sum() - fp.sum() * pt / (1 - pt)) / max(mask.sum(), 1))})
                        continue
                    tp, fp = net_benefit_counts(up & sub, truth, cells, 48)
                    boots = (weights @ tp - pt / (1 - pt) * (weights @ fp)) / denom_b
                    rows.append({"pt": float(pt), "point_48": float((tp.sum() - fp.sum() * pt / (1 - pt)) / denom_48),
                                 "mean": float(boots.mean()), "q05": float(np.quantile(boots, 0.05)), "q95": float(np.quantile(boots, 0.95))})
                sub_tab[name] = rows
            sub_tab["treat_all"] = [{"pt": float(pt), "point_48": float((truth & sub).sum() / sub.sum() - (~truth & sub).sum() / sub.sum() * pt / (1 - pt))} for pt in GRID]
            table[sub_name] = sub_tab
        by_wave = {}
        for w, wave in enumerate(WAVES):
            sub = full & (data.wave == w)
            by_wave[wave[0]] = {name: [float(((up := (x >= tau)) & truth & sub).sum() / sub.sum() - (up & ~truth & sub).sum() / sub.sum() * pt / (1 - pt)) for pt in GRID] for name, x in alert.items()}
            by_wave[wave[0]]["model"] = [float((((up := model_p >= pt) & truth & sub).sum() - (up & ~truth & sub).sum() * pt / (1 - pt)) / sub.sum()) for pt in GRID]
        table["by_wave_point"] = by_wave
        wins, later = {}, full & (np.arange(final.shape[1]) >= RECAL_START)
        for pt in NAMED_PT:
            def boot(up, mask):
                tp, fp = net_benefit_counts(up & mask, truth, cells, 48)
                n = np.bincount(cells[mask], minlength=48).astype(float)
                return (weights @ tp - pt / (1 - pt) * (weights @ fp)) / (weights @ n)

            fac = data.factor >= tau
            raw = boot(model_p >= pt, full) - boot(fac, full)
            cal = boot(recal_p >= pt, later) - boot(fac, later) if later.any() else np.zeros(1)
            wins[str(pt)] = {"p_model_beats_factor": float((raw > 0).mean()), "gap_mean": float(raw.mean()), "win": bool((raw > 0).mean() >= 0.9),
                             "recalibrated_from_2021_12_17": {"p": float((cal > 0).mean()), "gap_mean": float(cal.mean()), "win": bool((cal > 0).mean() >= 0.9)}}
        table["wins"] = wins
        result["net_benefit"][f"tau{tau}"] = table
        rel = {}
        for name, up in [*[(n, x >= tau) for n, x in alert.items()], ("model_p0.5", model_p >= 0.5), ("model_p0.33", model_p >= 0.33), ("model_p0.2", model_p >= 0.2)]:
            row = {}
            for release in (1, 3, 5):
                state = alert_periods(up & full, release)
                starts = state[:, 1:] & ~state[:, :-1]
                row[str(release)] = {"false_alarm_days": int((state & ~truth & full).sum()), "missed_days": int((~state & truth & full).sum()),
                                     "alert_periods": int(starts.sum() + state[:, 0].sum())}
            rel[name] = row
        result["release"][f"tau{tau}"] = rel
        trade = []
        for a in GRID:
            row = lateness((model_p >= a) & full, [x for x in crossing[tau] if has[x[1]]])
            trade.append({"p": float(a), "fp_days": int(((model_p >= a) & ~truth & full).sum()), "fn_days": int(((model_p < a) & truth & full).sum()), "median_late": row["median_late"], "never": row["never"]})
        result["fp_fn_tradeoff"][f"tau{tau}"] = trade
        best = {}
        for s, state in enumerate(c.STATES):
            mask = full[s]
            rows = []
            options = [(n, None, x[s] >= tau) for n, x in alert.items()] + [("model", a, model_p[s] >= a) for a in ALERT_P]
            for name, a, up in options:
                tp, fp = (up & truth[s] & mask).sum(), (up & ~truth[s] & mask).sum()
                nb = (tp - fp * 0.33 / 0.67) / max(mask.sum(), 1)
                part = [(0, t) for ss, t in crossing[tau] if ss == s and has[t]]
                late_row = lateness((up & mask)[None], part)
                rows.append({"signal": name, "alert_p": a, "net_benefit": float(nb), "false_alarm_days": int(fp), "median_late": late_row["median_late"], "crossings": len(part), "never": late_row["never"]})
            best[state] = max(rows, key=lambda x: x["net_benefit"])
            best[state]["factor_net_benefit"] = next(x["net_benefit"] for x in rows if x["signal"] == "factor")
        result["rules"][f"tau{tau}"] = best
    misses = []
    for j in np.flatnonzero(has):
        for s in range(16):
            for t_i, tau in enumerate(TAUS):
                call, truth_up = bool(data.pc[t_i, s, j] >= 0.5), bool(final[s, j] >= tau)
                if call != truth_up:
                    misses.append({"date": data.all_dates[j], "state": c.STATES[s], "tau": tau, "p": float(data.pc[t_i, s, j]),
                                   "median": float(data.q[3, s, j]), "frozen": float(data.frozen[s, j]), "final": float(final[s, j])})
    misses.sort(key=lambda m: -abs(m["p"] - 0.5))
    result["wrong_calls"] = {"n": len(misses), "worst": misses[:40]}
    bar_nb = {}
    for tau in TAUS:
        bar_nb[f"tau{tau}"] = sum(v["win"] for v in result["net_benefit"][f"tau{tau}"]["wins"].values())
        bar_nb[f"tau{tau}_recal"] = sum(v["recalibrated_from_2021_12_17"]["win"] for v in result["net_benefit"][f"tau{tau}"]["wins"].values())
    result["bar_net_benefit"] = {**bar_nb, "beats_factor": bool(sum(bar_nb[f"tau{t}"] >= 2 for t in TAUS) >= 2),
                                 "beats_factor_recalibrated": bool(sum(bar_nb[f"tau{t}_recal"] >= 2 for t in TAUS) >= 2)}
    store.save_json("decision", result)
    print(json.dumps({k: result[k] for k in ("dates", "crossings", "clusters", "bar_net_benefit")}, default=float), flush=True)
    for tau in TAUS:
        print(tau, json.dumps(result["net_benefit"][f"tau{tau}"]["wins"], default=float), flush=True)
    return result


def banner(logdir):
    """Stage 11: measured rows and the manifest; marks the stages."""
    logs = Path(logdir)
    walls, gates, flagged, cull = [], [], [], []
    for date in r.ALL_DATES:
        j = store.load_json(f"gpu_fit_{date}")
        walls.append(j["wall_s"])
        gates.append(j["gate"])
        cull.append((j["cull1"]["stuck"] + j["cull2"]["stuck"], j["cull1"]["check_mode"] + j["cull2"]["check_mode"]))
        if not j["gate"]["pass"]:
            flagged.append(date)
    failed = sorted(f.stem for f in store.path.glob("failed_*.json"))
    vram = [store.load_json(f"gpu_fit_{d}")["vram"]["peak_gb"] for d in r.ALL_DATES]
    strata = [store.load_json(f"strata_{d}") for d in r.HARD]
    hind = [store.load_json(f"hindsight_{d}") for d in r.HARD]
    finished = store.load_json("finished_fit")
    peaks = {}
    for path in logs.glob("*.log"):
        stage = path.name.split(".")[0]
        found = re.findall(r"\[memguard\] peak memory ([0-9.]+) GB", path.read_text(errors="replace"))
        if found:
            peaks[stage] = max(peaks.get(stage, 0.0), float(found[-1]))
    rows = [line.split(",") for line in (logs / "thermal.csv").read_text().splitlines()]
    rows = [x for x in rows if len(x) >= 7 and x[1] and x[2]]
    result = {"state_fits": {"n": len(walls), "wall_s_mean": float(np.mean(walls)), "wall_s_max": float(np.max(walls)), "wall_s_total": float(np.sum(walls)),
                             "r_hat_max": max(g["r_hat"] for g in gates), "ess_min": min(g["ess"] for g in gates), "divergence_max": max(g["divergence"] for g in gates),
                             "failed_both": flagged, "stuck_chains": sum(x[0] for x in cull), "check_mode_chains": sum(x[1] for x in cull), "vram_peak_gb": max(vram)},
              "failed_runs": failed,
              "hindsight": {"finished": {k: finished[k] for k in ("wall_s", "gate", "dim", "seed", "attempt")}, "walls": [h["wall_s"] for h in hind], "gates": [h["gate"] for h in hind]},
              "strata": {"walls": [s["wall_s"] for s in strata], "gates": [s["gate"] for s in strata], "vram_peak_gb": max(s["vram"]["peak_gb"] for s in strata),
                         "stuck": [s["cull1"]["stuck"] + s["cull2"]["stuck"] for s in strata]},
              "rss_peak_gb_by_stage": peaks,
              "thermal": {"samples": len(rows), "max_tctl": max(float(x[1]) for x in rows), "max_gpu": max(float(x[2]) for x in rows),
                          "max_power_w": max(float(x[5]) for x in rows), "hw_slowdown": sum("Active" in x[3] and "Not" not in x[3] for x in rows)},
              "gates_sha256": {"pilot": sha256_file(store.path / "gates_2026-11-02.json"), "full": sha256_file(store.path / f"{r.GATES}.json")},
              "health": p.health()}
    start = int((logs / "full.start").read_text()) if (logs / "full.start").exists() else None
    end = int((logs / "full.end").read_text()) if (logs / "full.end").exists() else None
    result["wall_s"] = end - start if start and end else None
    store.save_json("banner", result)
    print(json.dumps(result, indent=0, default=float), flush=True)
    for stage, info in (("baselines", {}), ("pilot", {}), ("state_fits", {"fits": len(walls)}), ("hindsight", {"fits": 9}), ("strata", {"fits": 8}),
                        ("failed_runs", {"runs": len(failed)}), ("scores", {}), ("decision", {}), ("banner", {})):
        store.mark_done(stage, **info)
    store.write_manifest(git_hash=git_hash(), versions=library_versions("pymc", "pytensor", "jax", "blackjax", "numpy", "nutpie"),
                         data_fingerprint=store.load_json("data")["sha256"], hardware="AMD 5950X, cores 0-9 and 11-15; RTX 3090 Ti",
                         started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)) if start else None,
                         finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)) if end else None, rung="daily fits, no daily update")
    problems = store.verify()
    print("Store.verify problems:", problems, flush=True)
    return result


JOBS = {"rki": rki, "score": score, "decide": decide, "banner": banner}

if __name__ == "__main__":
    t0 = time.perf_counter()
    JOBS[sys.argv[1]](*sys.argv[2:])
    print(f"{sys.argv[1]} done in {time.perf_counter() - t0:.1f} s", flush=True)
