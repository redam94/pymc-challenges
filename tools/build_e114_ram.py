"""Build the E114 dataset: catches, abundance indices and assessment B/Bmsy for 22 fish stocks.

    python tools/build_e114_ram.py      # download (cached), convert with Rscript, write data/e114_ram_stocks.csv

Source: RAM Legacy Stock Assessment Database v4.44 (Ricard, Minto, Jensen & Baum 2012, Fish and
Fisheries 13:380-398), Zenodo record 2542919, DOI 10.5281/zenodo.2542919, licence CC BY 4.0
(checked on the Zenodo record, 2026-11-05):
  https://zenodo.org/records/2542919/files/RLSADB%20v4.44.zip?download=1   (55,491,344 bytes)
Zenodo's /api/.../content route has answered 403 "unusual traffic"; the records route works, and
the download retries politely with backoff. A newer release, v4.66 (2024, DOI
10.5281/zenodo.14043031, 112 MB), exists; this notebook stays on v4.44, whose series end 2001-2016.

The zip (not committed) is cached in ~/.cache/pymc-challenges/e114 with the R workspace
`DB Files With Assessment Data/DBdata.RData` extracted from it. R is needed once: a one-off
`Rscript` call writes the tables we use (timeseries_values_views, timeseries_units_views, stock,
taxonomy, metadata) as CSV into the same cache. pyreadr/rdata are not dependencies of this repo.

Stock selection (see the E114 notes): every stock with >= 25 years of total catch (TCbest) and an
abundance index (CPUE or survey biomass) with >= 15 years whose index is not a copy of the
assessment's own biomass, restricted to finfish (no lobsters, crabs, shrimps, scallops or squid),
plus two catch-only stocks:
  COD2J3KL     northern cod, NAFO 2J3KL (no index in RAM; the 1992 collapse)
  PANCHPERUNC  Peruvian anchoveta: its RAM 'survB' equals the assessment's TBbest in every year
               (ratio 1.000, sd 0), so it is NOT an independent index and is dropped here.
`index_check` prints, for every kept index and for the screened-out candidates, the correlation of
the log index with the log assessed total biomass (TBbest) and the sd of their log ratio, and asserts
that the anchoveta series is a copy (sd 0) and that no kept index is one. Beyond the copy test the
screen was a judgement made while exploring, not a fixed threshold: GHAL23KLMNO (Greenland halibut,
catch-rate correlation 0.03), COD3Pn4RS (17 overlapping years, 0.54) and POLL3Ps (no assessed biomass
to compare with) were left out, while RSOLEHSTR was kept although its catch rate correlates -0.17
with the assessment, because its survey tracks it (0.40) and the official comparison is part of the
lesson.

Resilience (FishBase species summaries, 'Resilience' field, citing FishBase Ref. 120179; read on
2026-11-06) is attached per species, with the CMSY prior range of r for each category from
Froese, Demirel, Coro, Kleisner & Winker (2017, Fish and Fisheries 18:506-526):
  High 0.6-1.5, Medium 0.2-0.8, Low 0.05-0.5, Very low 0.015-0.1.
Redfish (Sebastes spp., S. mentella 'Very low' and S. fasciatus 'Low') gets the union 0.015-0.5.

Output: data/e114_ram_stocks.csv, one row per stock-year from the first to the last catch year:
  stockid, year
  catch_t      total catch, metric tonnes (TCbest; all 22 stocks are in MT)
  cpue         RAM's preferred CPUE series (units in cpue_units), NaN if none
  survey       RAM's preferred survey biomass (units in survey_units), NaN if none or if dropped
  bdivbmsy     the assessment's B/Bmsy (BdivBmsypref: total or spawning biomass, per assessment)
  udivumsy     the assessment's U/Umsy (UdivUmsypref), exploitation rate relative to Umsy
  tb_t         assessment total biomass (TBbest, MT), NaN if not reported
and stock-level columns repeated on each row:
  stocklong, commonname, scientificname, family, region, assessorid, assessyear,
  cpue_units, survey_units, role ('index' or 'catch_only'), resilience, r_lo, r_hi
"""

from __future__ import annotations

import hashlib
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e114_ram_stocks.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e114"
URL = "https://zenodo.org/records/2542919/files/RLSADB%20v4.44.zip?download=1"
ZIP = RAW / "RLSADB_v4.44.zip"
ZIP_MD5 = "482846476626b9208407814dbbb4315b"
MEMBER = "RLSADB v4.44/DB Files With Assessment Data/DBdata.RData"
TABLES = ["timeseries_values_views", "timeseries_units_views", "stock", "taxonomy", "metadata"]

INDEX_STOCKS = [
    "PCODHS", "PCODWCVANI", "PCOD5AB",                       # Pacific cod (Gadidae)
    "RSOLE5AB", "RSOLEHSTR", "YELL3LNO", "YELLGB", "WITFLOUN4RST",   # flatfish (Pleuronectidae)
    "SPHAKECH", "SHAKE4VWX", "SHAKESGBMATL", "SHAKEGOMNGB",  # hakes (Merlucciidae)
    "ANCHOSA", "SARDSA",                                     # small pelagics
    "REDFISHSPP3LN", "NZSNAPNZ7", "TREVALLYTRE7", "NZLINGESE", "BLSHARNPAC", "WHAKE4T",
]
CATCH_ONLY = ["COD2J3KL", "PANCHPERUNC"]
DROP_SURVEY = {"PANCHPERUNC"}            # survB == TBbest exactly: the assessment's own output

# FishBase resilience per species (read 2026-11-06) and the Froese et al. (2017) r ranges
RES_RANGE = {"High": (0.6, 1.5), "Medium": (0.2, 0.8), "Low": (0.05, 0.5), "Very low": (0.015, 0.1),
             "Very low-Low": (0.015, 0.5)}
RESILIENCE = {
    "Gadus macrocephalus": "Low", "Gadus morhua": "Medium", "Lepidopsetta bilineata": "Low",
    "Limanda ferruginea": "Low", "Glyptocephalus cynoglossus": "Medium", "Merluccius gayi": "Medium",
    "Merluccius bilinearis": "Medium", "Engraulis encrasicolus": "Medium", "Engraulis ringens": "High",
    "Sardinops sagax": "Medium", "Sebastes spp": "Very low-Low", "Chrysophrys auratus": "Low",
    "Pagrus auratus": "Low", "Pseudocaranx dentex": "Low", "Genypterus blacodes": "Low",
    "Prionace glauca": "Low", "Urophycis tenuis": "Low",
}


def fetch_zip() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and hashlib.md5(ZIP.read_bytes()).hexdigest() == ZIP_MD5:
        return
    for attempt in range(5):
        try:
            print(f"downloading {URL}")
            req = urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges-e114"})
            ZIP.write_bytes(urllib.request.urlopen(req, timeout=600).read())
            break
        except Exception as err:                       # Zenodo throttles now and then
            wait = 30 * 2 ** attempt
            print(f"  failed ({err}); retrying in {wait} s")
            time.sleep(wait)
    if hashlib.md5(ZIP.read_bytes()).hexdigest() != ZIP_MD5:
        raise RuntimeError("RAM zip checksum mismatch")


def export_tables() -> None:
    if all((RAW / f"{t}.csv").exists() for t in TABLES):
        return
    rdata = RAW / "DBdata.RData"
    if not rdata.exists():
        with zipfile.ZipFile(ZIP) as zf:
            rdata.write_bytes(zf.read(MEMBER))
    tabs = ", ".join(f'"{t}"' for t in TABLES)
    code = (f'load("{rdata}"); for (n in c({tabs})) '
            f'write.csv(get(n), file.path("{RAW}", paste0(n, ".csv")), row.names = FALSE)')
    subprocess.run(["Rscript", "-e", code], check=True)


SCREENED_OUT = ["GHAL23KLMNO", "COD3Pn4RS", "POLL3Ps"]


def index_check(tv: pd.DataFrame) -> None:
    """Compare each candidate index with the assessment's own total biomass (TBbest)."""
    print("index check: log index against log TBbest (copy if the log-ratio sd is ~0)")
    for sid in INDEX_STOCKS + CATCH_ONLY + SCREENED_OUT:
        g = tv[tv.stockid == sid]
        for col in ("CPUE", "survB"):
            if g[col].notna().sum() < 5:
                continue
            ok = (g[col] > 0) & (g.TBbest > 0)
            tag = ("kept" if sid in INDEX_STOCKS else "screened out" if sid in SCREENED_OUT
                   else "catch-only")
            if ok.sum() < 5:
                if sid in DROP_SURVEY and col == "survB":  # fail closed: the copy must be shown
                    raise RuntimeError(f"{sid}: too few years to check that its survey is a copy")
                print(f"  {sid:14s} {col:6s} {tag:12s} no assessed biomass to compare")
                continue
            li, lb = np.log(g.loc[ok, col]), np.log(g.loc[ok, "TBbest"])
            sd = float(np.std(li - lb))
            print(f"  {sid:14s} {col:6s} {tag:12s} {int(ok.sum()):3d} years, corr "
                  f"{np.corrcoef(li, lb)[0, 1]:5.2f}, log-ratio sd {sd:.3f}")
            if sid in DROP_SURVEY and col == "survB" and sd > 1e-6:
                raise RuntimeError(f"{sid}: expected its survey to be a copy of TBbest")
            if sid in INDEX_STOCKS and sd < 0.05:
                raise RuntimeError(f"{sid} {col}: index looks like a copy of the assessment")


def main() -> None:
    fetch_zip()
    export_tables()
    tv = pd.read_csv(RAW / "timeseries_values_views.csv", low_memory=False)
    index_check(tv)
    tu = pd.read_csv(RAW / "timeseries_units_views.csv").set_index("stockid")
    st = pd.read_csv(RAW / "stock.csv").set_index("stockid")
    tx = pd.read_csv(RAW / "taxonomy.csv").drop_duplicates("scientificname").set_index("scientificname")
    md = pd.read_csv(RAW / "metadata.csv").set_index("stockid")
    frames = []
    for sid in INDEX_STOCKS + CATCH_ONLY:
        g = tv[tv.stockid == sid].sort_values("year")
        yrs = g.loc[g.TCbest.notna(), "year"]
        g = g[(g.year >= yrs.min()) & (g.year <= yrs.max())]
        sp = st.loc[sid, "scientificname"]
        res = RESILIENCE.get(sp)
        survey = g.survB * np.nan if sid in DROP_SURVEY else g.survB
        cpue = g.CPUE
        f = pd.DataFrame({
            "stockid": sid, "year": g.year.to_numpy(), "catch_t": g.TCbest.to_numpy(),
            "cpue": cpue.to_numpy(), "survey": survey.to_numpy(),
            "bdivbmsy": g.BdivBmsypref.to_numpy(), "udivumsy": g.UdivUmsypref.to_numpy(),
            "tb_t": g.TBbest.to_numpy(),
            "stocklong": st.loc[sid, "stocklong"], "commonname": st.loc[sid, "commonname"],
            "scientificname": sp, "family": tx.loc[sp, "family"], "region": st.loc[sid, "region"],
            "assessorid": md.loc[sid, "assessorfull"], "assessyear": md.loc[sid, "assessyear"],
            "cpue_units": tu.loc[sid, "CPUE"] if cpue.notna().any() else np.nan,
            "survey_units": tu.loc[sid, "survB"] if survey.notna().any() else np.nan,
            "role": "catch_only" if sid in CATCH_ONLY else "index",
            "resilience": res,
            "r_lo": RES_RANGE[res][0] if res else np.nan, "r_hi": RES_RANGE[res][1] if res else np.nan,
        })
        if f.catch_t.isna().any():
            raise RuntimeError(f"{sid}: gap in the catch series")
        if tu.loc[sid, "TCbest"] != "MT":
            raise RuntimeError(f"{sid}: catch not in MT")
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False, float_format="%.6g")
    print(f"wrote {OUT}: {len(out)} rows, {out.stockid.nunique()} stocks, {OUT.stat().st_size / 1e3:.0f} KB")
    print(out.groupby("stockid").agg(y0=("year", "min"), y1=("year", "max"), n_cpue=("cpue", "count"),
                                     n_survey=("survey", "count"), n_bb=("bdivbmsy", "count"),
                                     res=("resilience", "first")).to_string())


if __name__ == "__main__":
    main()
