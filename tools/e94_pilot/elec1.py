"""Pilot: one fit per target cycle. Training races from earlier cycles + target-cycle polls."""
import json, logging, sys, time, warnings
import numpy as np, pandas as pd, pymc as pm, pytensor.tensor as pt, arviz as az
from pymc_challenges import data

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.WARNING)
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"   # full | fund (no polls) | agg (aggregate national only)
races = data.load("e94_races"); polls = data.load("e94_polls"); perr = data.load("e94_pollerr")
log = json.load(open(data.path("e94_build_log")))
nat = {int(k): v for k, v in log["national_house_dem2"].items()}
nat.update({int(k): v for k, v in log.get("national_house_dem2_medsl", {}).items() if int(k) < 2016})
EVE = {2018: "2018-11-05", 2022: "2022-11-07", 2026: "2026-11-02"}
ELEC = {2018: "2018-11-06", 2022: "2022-11-08", 2026: "2026-11-03"}
PRES = {1978: "D", 1982: "R", 1986: "R", 1990: "R", 1994: "D", 1998: "D", 2002: "R", 2006: "R", 2010: "D",
        2014: "D", 2018: "R", 2022: "D", 2026: "R"}
TRAIN = {2018: [2016], 2022: [2016, 2018, 2020], 2026: [2016, 2018, 2020, 2022]}[TARGET]

# ---- midterm prior for the national margin (D minus R, points, two-party House vote)
marg = {y: 100 * (2 * v - 1) for y, v in nat.items()}
sw = []
for y, p in PRES.items():
    if y < TARGET and y in marg and y - 2 in marg:
        s = marg[y] - marg[y - 2]
        sw.append(s if p == "R" else -s)        # swing toward the out-party
sw = np.array(sw)
prev = marg.get(TARGET - 2, np.nan)
if TARGET == 2026:
    prev = 100 * (2 * log["national_house_2024_wikipedia"]["dem2"] - 1)
sign = 1 if PRES[TARGET] == "R" else -1
N_prior_mu = prev + sign * sw.mean()
N_prior_sd = sw.std(ddof=1)
print(f"midterm prior: prev {prev:.1f}, out-party swing {sw.mean():.1f} +- {N_prior_sd:.1f} (n={len(sw)}) -> {N_prior_mu:.1f}")

# ---- historical polling error scales from earlier cycles
pe = perr[perr.cycle < TARGET]
gen_err = pe[pe.generic.astype(bool)].groupby("cycle").err.mean()
race_err = pe[~pe.generic.astype(bool) & (pe.type_simple == "Sen-G")]
cyc_sh = race_err.groupby("cycle").err.mean()
s_bias_nat = float(np.sqrt(np.mean(np.r_[gen_err.values, cyc_sh.values] ** 2)))
s_bias_race = float((race_err.err - race_err.cycle.map(cyc_sh)).std())
print(f"poll error priors: shared {s_bias_nat:.1f} pts (n={len(gen_err)}+{len(cyc_sh)} cycles), race-level {s_bias_race:.1f}")


def money(df, eve_cols):
    d = np.where(eve_cols, df.d_rec_eve, df.d_rec_final)
    r = np.where(eve_cols, df.r_rec_eve, df.r_rec_final)
    d = np.nan_to_num(d.astype(float)); r = np.nan_to_num(r.astype(float))
    return np.clip(np.log((d + 5e4) / (r + 5e4)), -5, 5)


R = races.copy()
R["inc"] = np.where(R.d_inc, 1.0, np.where(R.r_inc, -1.0, 0.0))
R["money"] = money(R, R.cycle.isin([2018, 2022, 2026]))
R["m"] = 100 * (2 * R.dem2 - 1)
R["leanx"] = R.lean / 10
train = R[R.cycle.isin(TRAIN) & R.contested & ~R.split.fillna(False).astype(bool) & R.lean.notna() & R.seat]
tgt = R[(R.cycle == TARGET) & R.seat].copy()
tgt_model = tgt[tgt.contested & ~tgt.split.fillna(False).astype(bool)].reset_index(drop=True)
states = sorted(set(R.state))
s_idx = {s: i for i, s in enumerate(states)}
cycles = TRAIN
print("train races", len(train), "target modelled", len(tgt_model), "of", len(tgt))

# ---- polls
P = polls[polls.cycle == TARGET].copy()
P["end"] = pd.to_datetime(P.end)
elec = pd.Timestamp(ELEC[TARGET])
P["week"] = ((elec - P.end).dt.days // 7).astype(int)
W = 22
P = P[(P.week < W) & (P.week >= 0)]
P["two"] = 100 * (2 * P.dem / (P.dem + P.rep) - 1)
P["nn"] = P.n.fillna(600).clip(200, 5000) * (P.dem + P.rep) / 100
q = P.dem / (P.dem + P.rep)
P["se"] = 200 * np.sqrt(q * (1 - q) / P.nn)
if MODE == "fund":
    P = P.iloc[0:0]
gp = P[P.kind == ("generic_avg" if TARGET == 2026 else "generic")]
if MODE == "agg" and TARGET < 2026:
    # stand-in aggregator: a 14-day mean of the generic polls, once a week (like 2026's weekly snapshots)
    rows = []
    for w in range(W):
        win = gp[(gp.week >= w) & (gp.week <= w + 1)]
        if len(win) >= 3:
            rows.append({"week": w, "two": win.two.mean(), "pollster": "rolling mean", "se": 1.0})
    gp = pd.DataFrame(rows)
if TARGET == 2026:
    gp = gp.copy(); gp["se"] = 1.0
rp = P[P.kind.isin(["house", "senate"]) & P.race.isin(tgt_model.race)]
rp = rp[rp.week < 10]                                  # race polls from the last ~10 weeks
pollsters = sorted(set(gp.pollster) | set(rp.pollster))
pidx = {p: i for i, p in enumerate(pollsters)}
r_idx = {r: i for i, r in enumerate(tgt_model.race)}
polled = sorted(set(rp.race))
print("generic polls", len(gp), "race polls", len(rp), "polled races", len(polled), "pollsters", len(pollsters))

isS_tr = (train.office == "S").to_numpy().astype(int)
isS_tg = (tgt_model.office == "S").to_numpy().astype(int)
X_tr = np.c_[train.leanx, train.inc, train.money]
X_tg = np.c_[tgt_model.leanx, tgt_model.inc, tgt_model.money]
c_tr = train.cycle.map({c: i for i, c in enumerate(cycles)}).to_numpy()
st_tr = train.state.map(s_idx).to_numpy(); st_tg = tgt_model.state.map(s_idx).to_numpy()
natm_tr = np.array([marg[c] for c in cycles])

coords = {"office": ["H", "S"], "x": ["lean", "inc", "money"], "cycle": cycles, "state": states,
          "week": np.arange(W)}
if pollsters:
    coords.update(pollster=pollsters, polled=polled)
with pm.Model(coords=coords) as m:
    beta = pm.Normal("beta", [5.0, 3.0, 2.0], [3.0, 3.0, 2.0], dims=("office", "x"))
    sig = pm.HalfNormal("sig", 10.0, dims="office")
    s_state = pm.HalfNormal("s_state", 4.0)
    gamma = pm.Normal("gamma", 0, 3, dims="office")              # intercept minus national margin
    s_int = pm.HalfNormal("s_int", 2.0)
    # training cycles
    z_int = pm.Normal("z_int", 0, 1, dims=("cycle", "office"))
    icpt_tr = natm_tr[:, None] + gamma[None, :] + s_int * z_int
    z_st = pm.Normal("z_st", 0, 1, dims=("cycle", "state"))
    mu_tr = icpt_tr[c_tr, isS_tr] + (beta[isS_tr] * X_tr).sum(-1) + s_state * z_st[c_tr, st_tr]
    pm.Normal("m_tr", mu_tr, sig[isS_tr], observed=train.m.to_numpy())
    # target cycle: national margin, weekly reverse random walk
    # N (true national margin) and bias (shared polling error) are seen by every poll only as N + bias:
    # rotate the two standard normals so that one direction carries the data and the other is its prior.
    zr = pm.Normal("z_rot", 0, 1, shape=2)
    hyp = np.hypot(N_prior_sd, s_bias_nat)
    cN, sB = N_prior_sd / hyp, s_bias_nat / hyp
    N = pm.Deterministic("N", N_prior_mu + N_prior_sd * (cN * zr[0] - sB * zr[1]))
    bias = pm.Deterministic("bias", s_bias_nat * (sB * zr[0] + cN * zr[1]))
    s_rw = pm.HalfNormal("s_rw", 1.0)
    z_rw = pm.Normal("z_rw", 0, 1, dims="week")
    walk = N + s_rw * pt.concatenate([pt.zeros(1), pt.cumsum(z_rw[1:])])   # week 0 = election week
    if pollsters:
        s_house = pm.HalfNormal("s_house", 2.0)
        house = pm.ZeroSumNormal("house", sigma=s_house, dims="pollster")
    z_int_t = pm.Normal("z_int_t", 0, 1, dims="office")
    icpt_t = N + gamma + s_int * z_int_t
    z_st_t = pm.Normal("z_st_t", 0, 1, dims="state")
    if len(gp):
        s_x = pm.HalfNormal("s_x_gen", 2.0)
        mu_g = walk[gp.week.to_numpy()] + bias + house[gp.pollster.map(pidx).to_numpy()]
        pm.Normal("gen", mu_g, pt.sqrt(gp.se.to_numpy() ** 2 + s_x ** 2), observed=gp.two.to_numpy())
    if len(rp):
        pol_i = np.array([r_idx[r] for r in polled])
        e_pol = pm.Normal("e_pol", 0, 1, dims="polled")
        m_pol = (icpt_t[isS_tg[pol_i]] + (beta[isS_tg[pol_i]] * X_tg[pol_i]).sum(-1)
                 + s_state * z_st_t[st_tg[pol_i]] + sig[isS_tg[pol_i]] * e_pol)
        pm.Deterministic("m_pol", m_pol, dims="polled")
        b_race = pm.Normal("b_race", 0, s_bias_race, dims="polled")
        d_part = pm.Normal("d_part", 2.0, 2.0)
        k = rp.race.map({r: i for i, r in enumerate(polled)}).to_numpy()
        spons = np.where(rp.partisan.fillna("") == "D", 1.0, np.where(rp.partisan.fillna("") == "R", -1.0, 0.0))
        s_xr = pm.HalfNormal("s_x_race", 3.0)
        mu_r = (m_pol[k] + (walk[rp.week.to_numpy()] - N) + bias + b_race[k] + d_part * spons
                + house[rp.pollster.map(pidx).to_numpy()])
        pm.Normal("racepoll", mu_r, pt.sqrt(rp.se.to_numpy() ** 2 + s_xr ** 2), observed=rp.two.to_numpy())
    print("model built", flush=True)
    print({k: v.shape for k, v in m.initial_point().items() if np.ndim(v)}, flush=True)
    import resource
    t0 = time.time(); f = m.compile_logp(); g = m.compile_dlogp(); ip = m.initial_point()
    print("logp", f(ip), "compile", time.time()-t0, "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
    if "--debug" in sys.argv:
        import nutpie
        t0 = time.time(); cm = nutpie.compile_pymc_model(m)
        print("nutpie compile", time.time()-t0, "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
        t0 = time.time(); tr = nutpie.sample(cm, chains=4, cores=2, draws=1000, tune=400, seed=1, progress_bar=False)
        print("4x1000", time.time()-t0, "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
        for kw in ({"chains": 1, "draws": 100, "tune": 100}, {"chains": 2, "draws": 300, "tune": 300}):
            t0 = time.time(); tr = pm.sample(random_seed=1, progressbar=False, **kw)
            print(kw, time.time()-t0, "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
        for dr in ():
            t0 = time.time(); tr = nutpie.sample(cm, chains=1, draws=dr, tune=dr, seed=1, progress_bar=False)
            print(dr, "sample", time.time()-t0, "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
        sys.exit()
    t0 = time.time()
    idata = pm.sample(random_seed=1, progressbar=False, target_accept=0.9)
    print(f"sample {time.time() - t0:.1f}s")

ss = idata.sample_stats
print("divergences", int(ss["diverging"].sum()), flush=True)
rh = az.rhat(idata.posterior); es = az.ess(idata.posterior)
import resource
print("rhat+ess computed, maxrss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, flush=True)
print("rhat max", max(float(rh[v].max()) for v in rh.data_vars), "ess min", min(float(es[v].min()) for v in es.data_vars), flush=True)
print(az.summary(idata, var_names=["beta", "sig", "s_state", "N", "s_int", "gamma"], round_to=2).to_string())

# ---- simulate every target race
post = az.extract(idata, num_samples=2000)
rng = np.random.default_rng(2)
S = post.sizes["sample"]
beta_d = post["beta"].transpose("sample", "office", "x").values
icpt = (post["N"].values[:, None] + post["gamma"].transpose("sample", "office").values
        + post["s_int"].values[:, None] * post["z_int_t"].transpose("sample", "office").values)
zst = post["z_st_t"].transpose("sample", "state").values * post["s_state"].values[:, None]
sig_d = post["sig"].transpose("sample", "office").values
mu = icpt[:, isS_tg] + (beta_d[:, isS_tg, :] * X_tg[None]).sum(-1) + zst[:, st_tg]
sim = mu + sig_d[:, isS_tg] * rng.standard_normal(mu.shape)
if len(rp):
    sim[:, pol_i] = post["m_pol"].transpose("sample", "polled").values
pD = (sim > 0).mean(0)
tgt_model["pD"] = pD
tgt_model["pred"] = np.median(sim, 0)
# seats
for off, total in (("H", 435), ("S", None)):
    sel = (tgt_model.office == off).to_numpy()
    fixedD = ((tgt.office == off) & ~(tgt.contested & ~tgt.split.fillna(False).astype(bool)) & (tgt.winner == "D")).sum() if TARGET < 2026 else \
        ((tgt.office == off) & ~tgt.contested & (tgt.d_name.fillna("") != "")).sum()
    seats = (sim[:, sel] > 0).sum(1) + fixedD
    actual = (tgt[tgt.office == off].winner == "D").sum() if TARGET < 2026 else np.nan
    print(f"{off}: D seats median {np.median(seats):.0f} 90% [{np.quantile(seats, .05):.0f}, {np.quantile(seats, .95):.0f}] actual {actual}")
if TARGET < 2026:
    y = tgt_model.m.to_numpy()
    pit = (sim < y).mean(0)
    win = (y > 0).astype(float)
    brier = np.mean((pD - win) ** 2)
    comp = (pD > 0.05) & (pD < 0.95)
    for lab, sel in (("all", np.ones(len(y), bool)), ("polled", np.isin(tgt_model.race, polled)), ("H", isS_tg == 0), ("S", isS_tg == 1)):
        lo, hi = np.quantile(sim[:, sel], [0.05, 0.95], 0)
        print(f"  {lab:6s} n={sel.sum():3d} 90% cover {np.mean((y[sel] >= lo) & (y[sel] <= hi)):.2f}  brier {np.mean((pD[sel] - win[sel]) ** 2):.4f}  "
              f"MAE {np.mean(np.abs(np.median(sim[:, sel], 0) - y[sel])):.1f}  wrong calls {int(((pD[sel] > .5) != win[sel].astype(bool)).sum())}")
    print("  PIT deciles", np.histogram(pit, bins=10, range=(0, 1))[0])
    print("  N posterior", np.quantile(post["N"].values, [.05, .5, .95]).round(1), "actual national margin", round(marg[TARGET], 1))
    print("  competitive (5-95%):", comp.sum(), "brier", np.mean((pD[comp] - win[comp]) ** 2).round(3))
lo, hi = np.quantile(sim, [0.1, 0.9], 0)
tgt_model["q10"], tgt_model["q90"] = lo, hi
if TARGET == 2026:
    notup = {k: int(v) for k, v in log["senate_2026_not_up"].items()}
    out = {}
    for off in ("H", "S"):
        sel = (tgt_model.office == off).to_numpy()
        fixedD = int(((tgt.office == off) & ~tgt.contested & (tgt.d_name.fillna("") != "")).sum())
        seats = (sim[:, sel] > 0).sum(1) + fixedD
        if off == "S":
            seats = seats + notup["Democratic"] + notup["Independent"]
            p_ctrl = float((seats >= 51).mean())          # VP Vance breaks a 50-50 tie for the Republicans
        else:
            p_ctrl = float((seats >= 218).mean())
        out[off] = {"D_seats_q05_q50_q95": np.quantile(seats, [.05, .5, .95]).tolist(), "P_D_control": p_ctrl}
    out["N_q05_q50_q95"] = np.quantile(post["N"].values, [.05, .5, .95]).round(2).tolist()
    print(json.dumps(out))
    json.dump(out, open(f"/tmp/claude-1000/-home-mattr-Desktop-pymc-challenges/a3268804-1cea-451e-a672-dc5bb3a3bc85/scratchpad/e94pilot/seats_{TARGET}_{MODE}.json", "w"))
tgt_model[["race", "office", "pD", "pred", "q10", "q90"]].to_csv(f"/tmp/claude-1000/-home-mattr-Desktop-pymc-challenges/a3268804-1cea-451e-a672-dc5bb3a3bc85/scratchpad/e94pilot/pred_{TARGET}_{MODE}.csv", index=False)
