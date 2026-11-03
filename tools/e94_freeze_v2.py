"""Write E94's second (corrected) 2026 freeze, data/e94_freeze_2026_v2.json, with the notebook's own code.

Why a v2: the v1 freeze (data/e94_freeze_2026.json, commit e9c37e7) was computed from an e94_races.csv in
which 2026 nominees without an FEC ID found by name (50 R, 11 D) had their receipts set to $0, so the
money signal of their races was pinned at the clip (+/-5, about 12 points of margin). v2 uses the rebuilt
data (IDs matched against the FEC candidate master, unmatched money missing and imputed in the model) and
the notebook's code, with separate random streams for the candidate and IE parts of the spending nowcast.
Information cutoff unchanged: 2026-11-02. v1 stays as the pre-registered file; both are scored.

This runs the notebook source (notebook_src/examples/E94_midterms_polls_money.py) up to section 7, then
writes the file. It refuses to overwrite an existing freeze.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" taskset -c 12,13 .venv/bin/python tools/memguard.py --max-gb 8 -- \
        .venv/bin/python tools/e94_freeze_v2.py
"""

import datetime as dt
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook_src/examples/E94_midterms_polls_money.py"
OUT = ROOT / "data/e94_freeze_2026_v2.json"
INPUTS = ("e94_races.csv", "e94_polls.csv", "e94_reports.csv.gz", "e94_ie.csv.gz", "e94_pollerr.csv")


def content_sha(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha256(gzip.decompress(b) if p.suffix == ".gz" else b).hexdigest()


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists: a freeze is never overwritten")
    code = SRC.read_text()
    cut = code.index("# ## 7 · 2026")
    code = code[: code.rindex("# %%", 0, cut)]
    ns = {"__name__": "e94_freeze_v2"}
    exec(compile(code, str(SRC), "exec"), ns)
    np, pd = ns["np"], ns["pd"]
    fit26, build_log = ns["fit26"], ns["build_log"]
    q = lambda v, r=2: [round(float(x), r) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731

    s26 = fit26["seats"]
    notup = {k: int(v) for k, v in build_log["senate_2026_not_up"].items()}
    sen = s26["S"] + notup["Democratic"] + notup["Independent"]
    seats = {"H": {"D_seats_q05_q50_q95": q(s26["H"], 0), "P_D_control": round(float((s26["H"] >= 218).mean()), 4)},
             "S": {"D_seats_q05_q50_q95": q(sen, 0), "P_D_control": round(float((sen >= 51).mean()), 4),
                   "D_seats_up_q05_q50_q95": q(s26["S"], 0)},
             "N_q05_q50_q95": q(fit26["N"])}
    tm, sim = fit26["d"]["tm"], fit26["sim"]
    q10, med, q90 = np.quantile(sim, [0.1, 0.5, 0.9], 0)
    races = [{"race": r, "office": o, "pD": round(float(p), 4), "pred": round(float(m), 3), "q10": round(float(a), 3),
              "q90": round(float(b), 3)}
             for r, o, p, m, a, b in zip(tm.race, tm.office, (sim > 0).mean(0), med, q10, q90)]

    CT, late_fit, ie_fit = ns["CT"], ns["LATE_FIT"]["both -> 2026"], ns["IE_FIT"]["both -> 2026"]
    late = ns["predict_late"](late_fit, CT[2026])
    ie_tot, ie_seen = ns["ie_nowcast"](ie_fit, 2026)
    known = float(CT[2026].disb_known.sum())
    cand = known + late.sum(1)
    allin = cand + ie_tot
    races_, draws = ns["race_totals"](2026, late, CT[2026], ie_tot)
    qd = lambda v: [round(float(x), -5) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    spend = {"candidates_known_to_12G": round(known, -5), "candidates_late_q05_q50_q95": qd(late.sum(1)),
             "candidates_total_q05_q50_q95": qd(cand), "ie_seen_by_eve": round(float(ie_seen), -5),
             "ie_unseen_q05_q50_q95": qd(ie_tot - ie_seen), "ie_total_q05_q50_q95": qd(ie_tot),
             "all_q05_q50_q95": qd(allin), "n_candidates": int(len(CT[2026])),
             "late_per_candidate_q05_q50_q95": {c: qd(late[:, j]) for j, c in enumerate(CT[2026].cand_id)},
             "race_total_q05_q50_q95": {r: qd(draws[:, j]) for j, r in enumerate(races_)}}
    miss = fit26["d"]["miss"]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "what": "E94 election-eve forecast and spending nowcast for 2026-11-03, freeze v2 (corrected), frozen before "
                "any results. Computed by tools/e94_freeze_v2.py, which runs the notebook's code "
                "(notebook_src/examples/E94_midterms_polls_money.py, sections 1-6) on the inputs below.",
        "version": 2,
        "supersedes": "data/e94_freeze_2026.json (v1, commit e9c37e70fda7799543ad8d825164ae14245d7d04, 2026-11-03 19:09:40 UTC)",
        "why_v2": "In v1, 2026 nominees with no FEC ID found by name (50 R, 11 D, among them Ashley Hinson, IA-S, filed by "
                  "the FEC as 'ARENHOLZ, ASHLEY HINSON') had $0 receipts, which pinned their races' money signal at the "
                  "clip and moved them by about 12 points (IA-S P(D) 0.87 at R+1 in the polls). v2: IDs matched "
                  "against the FEC candidate master by state, office, district and name (D 100%, R 99.6% found); the "
                  "money of the 2 still unmatched is missing and imputed; the candidate and IE parts of the spending "
                  "nowcast use separate random streams; the IE part's uncertainty is included.",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit_before_run": head,
        "information_cutoff": "2026-11-02 (polls ending, reports and IE received, Wikipedia revisions on or before this date)",
        "inputs_sha256": {f: hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest() for f in INPUTS},
        "inputs_sha256_content": {f: content_sha(ROOT / "data" / f) for f in INPUTS},
        "money_missing_races": [str(r) for r in tm.race.to_numpy()[miss]],
        "fit_diagnostics": {k: fit26[k] for k in ("divergences", "max r_hat", "worst", "min ESS")},
        "seats": seats,
        "seats_note": "House: D seats incl. uncontested; P_D_control = P(D >= 218). Senate: D-caucus seats incl. 34 not "
                      "up (32 D + 2 I); D_seats_up counts only the 35 seats up; P_D_control = P(>= 51), VP Vance breaks "
                      "ties. N = national House two-party margin, D minus R, points.",
        "spending_nowcast": spend,
        "spending_note": "Dollars. Candidates = general-election D and R nominees' committees: disbursements in reports "
                         "received by 2026-11-02 plus modelled late spending to the 30G coverage end (2026-11-23). IE = "
                         "general-election independent expenditures received by the eve, plus lag completion and an "
                         "election-day share, with posterior and cycle-shift uncertainty.",
        "races": races,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("frozen_at_utc", "seats", "money_missing_races", "fit_diagnostics")}, indent=1))
    print({k: v for k, v in spend.items() if not isinstance(v, dict)})
    r = {x["race"]: x for x in races}
    for k in ("IA-S", "CA-45", "ID-S", "KY-03", "MD-04"):
        print(k, r.get(k))


if __name__ == "__main__":
    main()
