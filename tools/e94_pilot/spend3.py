# Pilot freeze of the 2026 spending nowcast: candidates' late spending (model fitted on 2018+2022) + IE.
import json, time
exec(open("/tmp/claude-1000/-home-mattr-Desktop-pymc-challenges/a3268804-1cea-451e-a672-dc5bb3a3bc85/scratchpad/e94pilot/spend2.py").read().split("rng = np.random.default_rng(1)")[0])
both = pd.concat([tabs[2018], tabs[2022]])
idata, dt = fit(both)
print(f"fit on 2018+2022: {dt:.1f}s div={int(idata.sample_stats['diverging'].sum())}")
rng = np.random.default_rng(7)
d, late = predict(idata, tabs[2026], rng, cycle_sd=0.25)
known = tabs[2026].disb_known.clip(0).sum()
cand_total = known + late.sum(1)
ie = data.load("e94_ie")
for c in ("date","first_receipt"): ie[c]=pd.to_datetime(ie[c])
f = ie[(ie.cycle==2026)&(ie.version=="eve")]
eve = pd.Timestamp(EVE[2026]); L = (eve - f.date).dt.days
seen = f.amt.sum()
# lag completion from the ie1 pilot's posterior means: logit f_L = 0.22 + 2.0 log1p(L); election-day share 1.1% of last week
fL = 1/(1+np.exp(-(0.22 + 2.0*np.log1p(np.arange(8)))))
by = np.array([f.amt[L==k].sum() for k in range(8)])
unseen = (by*(1/fL-1)).sum(); post = 0.011*by[:7].sum()
ie_total = seen + unseen + post
q = lambda x: np.quantile(x, [0.05, 0.5, 0.95]).round(-6).tolist()
out = {"candidates_known_to_12G": round(known, -5), "candidates_late_q05_q50_q95": q(late.sum(1)),
       "candidates_total_q05_q50_q95": q(cand_total), "ie_seen_by_eve": round(seen, -5),
       "ie_unseen_estimate": round(unseen + post, -5), "ie_total_point": round(ie_total, -5),
       "all_q05_q50_q95": q(cand_total + ie_total),
       "n_candidates": int(len(tabs[2026])), "n_with_12G": int(tabs[2026].d12.notna().sum())}
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/claude-1000/-home-mattr-Desktop-pymc-challenges/a3268804-1cea-451e-a672-dc5bb3a3bc85/scratchpad/e94pilot/spend_2026_pilot.json", "w"))
