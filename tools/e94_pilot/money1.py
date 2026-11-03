import time, logging, warnings
import numpy as np, pandas as pd, pymc as pm, arviz as az
from pymc_challenges import data
warnings.filterwarnings("ignore"); logging.getLogger("pymc").setLevel(logging.WARNING)
r = data.load("e94_races")
x = r[r.cycle.isin([2018,2022]) & (r.office=="H") & r.contested & ~r.split.astype(bool) & (r.d_inc | r.r_inc) & r.lean.notna()].copy()
x["inc_share"] = np.where(x.d_inc, x.dem2, 1-x.dem2)*100
x["inc_sp"] = np.log10(np.where(x.d_inc, x.d_disb_final.fillna(0), x.r_disb_final.fillna(0)).astype(float).clip(1e4))
x["ch_sp"] = np.log10(np.where(x.d_inc, x.r_disb_final.fillna(0), x.d_disb_final.fillna(0)).astype(float).clip(1e4))
x["lean_inc"] = np.where(x.d_inc, x.lean, -x.lean)
A = np.c_[np.ones(len(x)), x.inc_sp, x.ch_sp]
b = np.linalg.lstsq(A, x.inc_share, rcond=None)[0]; print("naive OLS: inc_share ~ log10 inc spend + log10 chal spend:", b.round(2), "n", len(x))
A = np.c_[np.ones(len(x)), x.inc_sp, x.ch_sp, x.lean_inc, x.cycle==2022]
b = np.linalg.lstsq(A, x.inc_share, rcond=None)[0]; print(" + lean, cycle:", b.round(2))
print("corr(log inc spend, |margin|)", np.corrcoef(x.inc_sp, np.abs(x.inc_share-50))[0,1].round(2))
rm = data.load("e94_rematch")
rm = rm[(rm.d_disb>0)&(rm.r_disb>0)&(rm.d_disb_prev>0)&(rm.r_disb_prev>0)].copy()
rm["dv"] = 100*(rm.dem2-rm.dem2_prev)
rm["dd"] = np.log(rm.d_disb/rm.d_disb_prev); rm["dr"] = np.log(rm.r_disb/rm.r_disb_prev)
print("rematches", len(rm), rm.groupby("cycle").size().to_dict())
A = np.c_[np.ones(len(rm)), rm.dd, rm.dr]
b, *_ = np.linalg.lstsq(A, rm.dv, rcond=None); print("first-difference OLS (no cycle effects):", b.round(2))
cyc = sorted(rm.cycle.unique()); ci = rm.cycle.map({c:i for i,c in enumerate(cyc)}).values
with pm.Model(coords={"cycle": cyc}) as m:
    a = pm.Normal("a", 0, 5, dims="cycle")          # national swing per cycle
    bd_mu = pm.Normal("bd_mu", 0, 2); br_mu = pm.Normal("br_mu", 0, 2)
    tau = pm.HalfNormal("tau", 1.0)
    bd = pm.Normal("bd", bd_mu, tau, dims="cycle"); br = pm.Normal("br", br_mu, tau, dims="cycle")
    sd = pm.HalfNormal("sd", 5); nu = pm.Gamma("nu", 2, 0.1)
    pm.StudentT("dv", nu=nu, mu=a[ci] + bd[ci]*rm.dd.values + br[ci]*rm.dr.values, sigma=sd, observed=rm.dv.values)
    t0=time.time(); idata = pm.sample(random_seed=1, progressbar=False, target_accept=0.95); print("fit", round(time.time()-t0,1))
print("div", int(idata.sample_stats["diverging"].sum()))
print(az.summary(idata, var_names=["bd_mu","br_mu","tau","sd","nu"], round_to=2).to_string())
