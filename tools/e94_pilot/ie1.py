import time, logging, warnings
import numpy as np, pandas as pd, pymc as pm, arviz as az
from pymc_challenges import data
warnings.filterwarnings("ignore"); logging.getLogger("pymc").setLevel(logging.WARNING)
ie = data.load("e94_ie")
for c in ("date","first_receipt"): ie[c]=pd.to_datetime(ie[c])
EVE={2018:"2018-11-05",2022:"2022-11-07",2026:"2026-11-02"}
rows=[]
for y in (2018,2022):
    f = ie[(ie.cycle==y)&(ie.version=="final")].copy(); eve=pd.Timestamp(EVE[y])
    f["L"]=(eve-f.date).dt.days
    for L in range(0,8):
        g=f[f.L==L]; rows.append({"cycle":y,"L":L,"final":g.amt.sum()/1e5,"seen":g.amt[g.first_receipt<=eve].sum()/1e5})
    rows.append({"cycle":y,"L":-1,"final":f.amt[f.L<0].sum()/1e5,"seen":0.0,
                 "week":f.amt[(f.L>=0)&(f.L<7)].sum()/1e5})
d = pd.DataFrame(rows); print(d.round(1).to_string())
lag = d[d.L>=0]; post_eve = d[d.L<0]
with pm.Model() as m:
    a = pm.Normal("a", 0, 2); b = pm.Normal("b", 2, 2); kappa = pm.Gamma("kappa", 2, 0.05)
    f = pm.math.invlogit(a + b*np.log1p(lag.L.values))
    pm.BetaBinomial("seen", n=np.round(lag.final.values).astype(int), alpha=f*kappa, beta=(1-f)*kappa, observed=np.round(lag.seen.values).astype(int))
    # dollars dated after the eve (election day itself), as a share of the last week's dollars
    r = pm.Beta("r_post", 1, 50)
    pm.Binomial("post", n=np.round(post_eve.week.values).astype(int), p=r, observed=np.round(post_eve.final.values).astype(int))
    t0=time.time(); idata = pm.sample(random_seed=1, progressbar=False); print("fit", round(time.time()-t0,1), "div", int(idata.sample_stats["diverging"].sum()))
print(az.summary(idata, round_to=3).to_string())
