import time, logging, warnings
import pandas as pd, numpy as np, pymc as pm, arviz as az
from pymc_challenges import data
logging.getLogger("pymc").setLevel(logging.WARNING)
rep = data.load("e94_reports"); races = data.load("e94_races")
for c in ("cvg_start","cvg_end","receipt"): rep[c]=pd.to_datetime(rep[c])
EVE={2018:"2018-11-05",2022:"2022-11-07",2026:"2026-11-02"}
def cand_table(y):
    r = rep[rep.cycle==y]; eve=pd.Timestamp(EVE[y])
    known = r[r.receipt<=eve]
    k = known.groupby("cand_id").agg(disb_known=("disb","sum"), last_end=("cvg_end","max"), race=("race","first"), office=("office","first"), side=("side","first"))
    p12 = known[known.rpt=="12G"].groupby("cand_id").agg(d12=("disb","sum"), s12=("cvg_start","min"), e12=("cvg_end","max"), coh12=("coh","sum"))
    late = r[(r.receipt>eve) & (r.cvg_start > r[r.rpt=="12G"].cvg_end.max() - pd.Timedelta(days=60))]
    g30 = r[r.rpt=="30G"].groupby("cand_id").agg(d30=("disb","sum"), e30=("cvg_end","max"))
    df = k.join(p12).join(g30)
    df["cycle"]=y
    rr = races[races.cycle==y].set_index("race")
    df["lean"] = df.race.map(rr.lean)
    df["inc"] = [ (rr.d_inc.get(rc,False) if s=="D" else rr.r_inc.get(rc,False)) for rc,s in zip(df.race, df.side)]
    return df.reset_index()
tabs = {y: cand_table(y) for y in (2018,2022,2026)}
def prep(df):
    df = df[df.d12.notna()].copy()
    df["days12"]=(df.e12-df.s12).dt.days+1
    df["x_rate"] = np.log1p((df.d12.clip(0)/df.days12)*20)   # 20 days at the pre-general rate (Oct 15 -> Nov 3)
    df["x_coh"] = np.log1p(df.coh12.clip(0))
    df["comp"] = pd.cut(df.lean.abs().fillna(15), [-1, 8, 20, 200], labels=False).astype(int)
    df["grp"] = (df.office=="S").astype(int)*3 + df.comp
    return df
def fit(train, seed=1):
    t = prep(train); t = t[t.d30.notna()]
    y = np.log1p(t.d30.clip(0)).to_numpy()
    with pm.Model() as m:
        a_mu = pm.Normal("a_mu", 0, 2); a_sd = pm.HalfNormal("a_sd", 1)
        a = pm.Normal("a", a_mu, a_sd, shape=6)
        b1 = pm.Normal("b1", 0.5, 0.5, shape=6); b2 = pm.Normal("b2", 0.3, 0.5)
        binc = pm.Normal("binc", 0, 0.5)
        mu = a[t.grp.values] + b1[t.grp.values]*(t.x_rate.values-10) + b2*(t.x_coh.values-12) + binc*t.inc.values
        sd = pm.HalfNormal("sd", 1, shape=6)
        pm.Normal("y", mu=mu, sigma=sd[t.grp.values], observed=y)
        t0=time.time(); idata = pm.sample(random_seed=seed, progressbar=False); dt=time.time()-t0
    return idata, dt
def predict(idata, df, rng, cycle_sd=0.1):
    d = prep(df); post = az.extract(idata, num_samples=1000)
    a = post["a"].values.T; b1=post["b1"].values.T; b2=post["b2"].values; binc=post["binc"].values
    sd=post["sd"].values.T
    g = d.grp.values
    mu = a[:, g] + b1[:, g]*(d.x_rate.values-10) + b2[:,None]*(d.x_coh.values-12) + binc[:,None]*d.inc.values
    cyc = rng.normal(0, cycle_sd, size=(mu.shape[0],1))
    yrep = mu + cyc + sd[:, g]*rng.standard_normal(size=mu.shape)
    late = np.expm1(yrep).clip(0)
    return d, late
rng = np.random.default_rng(1)
for tr, te in ((2018,2022),(2022,2018)):
    idata, dt = fit(tabs[tr])
    print(f"fit on {tr}: {dt:.1f}s, div={int(idata.sample_stats['diverging'].sum())}, rhat max={float(az.rhat(idata).to_dataset().to_dataarray().max()):.3f}")
    d, late = predict(idata, tabs[te], rng)
    ok = d.d30.notna()
    tot_pred = late[:, ok.values].sum(1); tot_true = d.d30[ok].clip(0).sum()
    print(f"  {te}: late spending true {tot_true/1e6:.0f}M, pred median {np.median(tot_pred)/1e6:.0f}M, 90% [{np.quantile(tot_pred,.05)/1e6:.0f}, {np.quantile(tot_pred,.95)/1e6:.0f}]")
    pit = (late[:, ok.values] < d.d30[ok].clip(0).values).mean(0)
    print("  candidate PIT deciles", np.histogram(pit, bins=10, range=(0,1))[0])
    cov = ((np.quantile(late[:,ok.values],0.05,0) <= d.d30[ok].values) & (d.d30[ok].values <= np.quantile(late[:,ok.values],0.95,0))).mean()
    print(f"  90% interval coverage per candidate {cov:.2f}")
