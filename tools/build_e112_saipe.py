"""Build the E112 dataset: ACS 5-year child poverty by county, with rurality and the SAIPE benchmark.

    python tools/build_e112_saipe.py     # download (cached), stream, write data/e112_county_child_poverty.csv

Sources (all US federal works, public domain), checked 2026-11-05/06:
  ACS 2019-2023 5-year detailed table B17020 (poverty status in the past 12 months by age),
    table-based summary file, 42 MB pipe-delimited:
    https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b17020.dat
  USDA ERS Rural-Urban Continuum Codes 2023 (long format: FIPS, Attribute, Value):
    https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/53251/Ruralurbancontinuumcodes2023.csv
  Census SAIPE 2023 state and county estimates (fixed width):
    https://www2.census.gov/programs-surveys/saipe/datasets/2023/2023-state-and-county/est23all.txt
  ACS 2014-2018 5-year, the same table B17020 from the older sequence-based summary file
    (sequence 0054, table start position 124; one zip and one geography file per state, ~110 MB
    in all): https://www2.census.gov/programs-surveys/acs/summary_file/2018/data/5_year_seq_by_state/
    The 2014-2018 and 2019-2023 samples do not overlap, so this is an independent second survey of
    the same counties, five years earlier.
The raw files are cached in ~/.cache/pymc-challenges/e112 (not committed).

The ACS file is streamed in chunks and only county rows (GEO_ID 0500000USxxxxx, 50 states + DC;
Puerto Rico dropped because SAIPE does not cover it) are kept. Children under 18 for whom poverty
status is determined are the sum of the under-18 age cells of B17020:
  below poverty   under 6 (003), 6-11 (004), 12-17 (005)
  at or above     under 6 (011), 6-11 (012), 12-17 (013)
B17001 (by sex by age, 118 MB) gives exactly the same counts but needs twelve cells per sum, and
a root sum of squares over more cells understates the margin more (cells of one county are
positively correlated through households): its child-poverty margins came out a median 10%
smaller than B17020's. B17020 is used for that reason.
Margins of error (90%) of a sum follow the ACS handbook (2020, ch. 8): root sum of squares, with
only the LARGEST margin among cells whose estimate is zero (otherwise twelve zero cells in a tiny
county would each add their zero-estimate margin). The margin of the proportion p = x / N uses the
proportion formula sqrt(MOE_x^2 - p^2 MOE_N^2) / N, and the ratio formula (a plus sign) when the
term under the root is negative. SE = MOE / 1.645.

Output: data/e112_county_child_poverty.csv, one row per county (3,143 or so):
  fips            5-digit county FIPS (Connecticut: the nine planning regions 09110-09190)
  state, county   postal code and county name (from SAIPE)
  n_children      ACS children under 18 with poverty status determined (estimate)
  n_poor          ACS children under 18 below poverty (estimate)
  moe_poor, moe_children   90% margins of the two sums
  p_acs, se_acs   direct proportion and its standard error
  rucc            2023 Rural-Urban Continuum Code (1 = metro of 1M+ ... 9 = remote rural)
  pop2020         2020 census population (ERS file)
  saipe_pct, saipe_lo, saipe_hi   SAIPE 2023 model-based under-18 poverty percent and 90% bounds
  n_children_18, n_poor_18, p_acs_18, se_acs_18   the same quantities from ACS 2014-2018 (missing for
                  Connecticut's planning regions and Alaska's 02063/02066, which did not exist then)
SAIPE uses IRS and SNAP records and the ACS single year; it is an external benchmark, not a
covariate. Note the period mismatch: ACS 5-year pools 2019-2023, SAIPE is for 2023.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e112_county_child_poverty.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e112"
ACS_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/"
           "5YRData/acsdt5y2023-b17020.dat")
RUCC_URL = "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/53251/Ruralurbancontinuumcodes2023.csv"
SAIPE_URL = ("https://www2.census.gov/programs-surveys/saipe/datasets/2023/2023-state-and-county/"
             "est23all.txt")
POOR = [3, 4, 5]
ABOVE = [11, 12, 13]


def fetch(url: str) -> Path:
    return fetch_to(url, "")


def fetch_to(url: str, sub: str) -> Path:
    (RAW / sub).mkdir(parents=True, exist_ok=True)
    p = RAW / sub / url.rsplit("/", 1)[1]
    if not p.exists():
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges-e112"})
        tmp = p.with_suffix(p.suffix + ".part")
        for attempt in range(6):          # www2.census.gov answers bursts with HTTP 429: back off and retry
            try:
                with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                break
            except urllib.error.HTTPError as err:
                if err.code != 429 or attempt == 5:
                    raise
                time.sleep(10 * 2 ** attempt)
        tmp.rename(p)
        time.sleep(1.0)
    return p


def sum_moe(est: np.ndarray, moe: np.ndarray) -> np.ndarray:
    """ACS handbook: root sum of squares, counting only the largest zero-estimate margin once."""
    moe = np.where(moe < 0, 0.0, moe)                  # no negative special codes expected here
    zero = est == 0
    nz = np.where(zero, 0.0, moe ** 2).sum(axis=1)
    zmax = np.where(zero, moe, 0.0).max(axis=1)
    return np.sqrt(nz + zmax ** 2)


def acs_counties() -> pd.DataFrame:
    cols = ["GEO_ID"] + [f"B17020_{k}{c:03d}" for c in POOR + ABOVE for k in ("E", "M")]
    parts = []
    for chunk in pd.read_csv(fetch(ACS_URL), sep="|", usecols=cols, dtype={"GEO_ID": str}, chunksize=20000):
        parts.append(chunk[chunk["GEO_ID"].str.startswith("0500000US")])
    d = pd.concat(parts, ignore_index=True)
    d["fips"] = d["GEO_ID"].str[-5:]
    d = d[d["fips"].str[:2] != "72"].reset_index(drop=True)
    E = lambda cs: d[[f"B17020_E{c:03d}" for c in cs]].to_numpy(float)   # noqa: E731
    M = lambda cs: d[[f"B17020_M{c:03d}" for c in cs]].to_numpy(float)   # noqa: E731
    special = (M(POOR + ABOVE) < 0).sum()
    print(f"ACS county rows: {len(d):,}; negative (special-code) margins: {special}")
    ep, ea = E(POOR), E(ABOVE)
    x, n = ep.sum(axis=1), ep.sum(axis=1) + ea.sum(axis=1)
    mx = sum_moe(ep, M(POOR))
    mn = sum_moe(np.c_[ep, ea], np.c_[M(POOR), M(ABOVE)])
    p, se = _prop(x, n, mx, mn)
    return pd.DataFrame({"fips": d["fips"], "n_children": n.astype(int), "n_poor": x.astype(int),
                         "moe_poor": mx.round(1), "moe_children": mn.round(1),
                         "p_acs": p.round(6), "se_acs": se.round(6)})


STATES_2018 = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
               "DistrictOfColumbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
               "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
               "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "NewHampshire", "NewJersey",
               "NewMexico", "NewYork", "NorthCarolina", "NorthDakota", "Ohio", "Oklahoma", "Oregon",
               "Pennsylvania", "RhodeIsland", "SouthCarolina", "SouthDakota", "Tennessee", "Texas", "Utah",
               "Vermont", "Virginia", "Washington", "WestVirginia", "Wisconsin", "Wyoming"]
SEQ_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/2018/data/5_year_seq_by_state"
START = 124   # B17020_001 is column 124 (1-based) of sequence 0054


def acs_2018() -> pd.DataFrame:
    import zipfile
    out = []
    for name in STATES_2018:
        base = f"{SEQ_URL}/{name}/All_Geographies_Not_Tracts_Block_Groups"
        geo = pd.read_csv(fetch_to(f"{base}/g20185{_abbr(name)}.csv", "acs2018"), header=None, dtype=str,
                          encoding="latin-1")
        geo = geo[(geo[2] == "050") & (geo[3] == "00")]
        logrec = dict(zip(geo[4].astype(int), geo[48].str[-5:]))
        z = zipfile.ZipFile(fetch_to(f"{base}/20185{_abbr(name)}0054000.zip", "acs2018"))
        stem = f"20185{_abbr(name)}0054000.txt"
        e = pd.read_csv(z.open("e" + stem), header=None, dtype=str)
        m = pd.read_csv(z.open("m" + stem), header=None, dtype=str)
        keep = e[5].astype(int).isin(logrec).to_numpy()
        e, m = e[keep], m[keep]
        col = lambda df, cs: df[[START + c - 2 for c in cs]].apply(pd.to_numeric, errors="coerce").to_numpy(float)  # noqa: E731
        ep, ea, mp, ma = col(e, POOR), col(e, ABOVE), col(m, POOR), col(m, ABOVE)
        x, n = ep.sum(axis=1), ep.sum(axis=1) + ea.sum(axis=1)
        mx, mn = sum_moe(ep, mp), sum_moe(np.c_[ep, ea], np.c_[mp, ma])
        p, se = _prop(x, n, mx, mn)
        out.append(pd.DataFrame({"fips": e[5].astype(int).map(logrec).to_numpy(), "n_children_18": n.astype(int),
                                 "n_poor_18": x.astype(int), "p_acs_18": p.round(6), "se_acs_18": se.round(6)}))
    return pd.concat(out, ignore_index=True)


ABBR = {"DistrictOfColumbia": "dc", "NewHampshire": "nh", "NewJersey": "nj", "NewMexico": "nm", "NewYork": "ny",
        "NorthCarolina": "nc", "NorthDakota": "nd", "RhodeIsland": "ri", "SouthCarolina": "sc",
        "SouthDakota": "sd", "WestVirginia": "wv"}
POSTAL = {"Alabama": "al", "Alaska": "ak", "Arizona": "az", "Arkansas": "ar", "California": "ca", "Colorado": "co",
          "Connecticut": "ct", "Delaware": "de", "Florida": "fl", "Georgia": "ga", "Hawaii": "hi", "Idaho": "id",
          "Illinois": "il", "Indiana": "in", "Iowa": "ia", "Kansas": "ks", "Kentucky": "ky", "Louisiana": "la",
          "Maine": "me", "Maryland": "md", "Massachusetts": "ma", "Michigan": "mi", "Minnesota": "mn",
          "Mississippi": "ms", "Missouri": "mo", "Montana": "mt", "Nebraska": "ne", "Nevada": "nv", "Ohio": "oh",
          "Oklahoma": "ok", "Oregon": "or", "Pennsylvania": "pa", "Tennessee": "tn", "Texas": "tx", "Utah": "ut",
          "Vermont": "vt", "Virginia": "va", "Washington": "wa", "Wisconsin": "wi", "Wyoming": "wy"}


def _abbr(name: str) -> str:
    return ABBR.get(name) or POSTAL[name]


def _prop(x, n, mx, mn):
    p = np.divide(x, n, out=np.zeros_like(x), where=n > 0)
    rad = mx ** 2 - p ** 2 * mn ** 2
    rad = np.where(rad < 0, mx ** 2 + p ** 2 * mn ** 2, rad)
    moe_p = np.divide(np.sqrt(rad), n, out=np.full_like(x, np.nan), where=n > 0)
    return p, moe_p / 1.645


def rucc() -> pd.DataFrame:
    r = pd.read_csv(fetch(RUCC_URL), dtype={"FIPS": str}, encoding="latin-1")
    r = r[r["Attribute"].isin(["RUCC_2023", "Population_2020"])]
    w = r.pivot_table(index="FIPS", columns="Attribute", values="Value", aggfunc="first")
    w = w.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"fips": w.index.str.zfill(5), "rucc": w["RUCC_2023"].to_numpy(),
                         "pop2020": w["Population_2020"].to_numpy()})


def saipe() -> pd.DataFrame:
    spec = {"st": (0, 2), "co": (3, 6), "saipe_pct": (76, 80), "saipe_lo": (81, 85), "saipe_hi": (86, 90),
            "county": (193, 238), "state": (239, 241)}
    s = pd.read_fwf(fetch(SAIPE_URL), colspecs=list(spec.values()), names=list(spec), dtype=str)
    s = s[s["co"].astype(int) > 0].copy()
    s["fips"] = s["st"].str.zfill(2) + s["co"].str.zfill(3)
    for c in ("saipe_pct", "saipe_lo", "saipe_hi"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s[["fips", "state", "county", "saipe_pct", "saipe_lo", "saipe_hi"]]


def main() -> None:
    a, r, s, old = acs_counties(), rucc(), saipe(), acs_2018()
    print(f"ACS 2014-2018 county rows: {len(old):,}")
    d = a.merge(s, on="fips", how="left").merge(r, on="fips", how="left").merge(old, on="fips", how="left")
    for name, miss in [("SAIPE", d["saipe_pct"].isna()), ("RUCC", d["rucc"].isna()),
                       ("ACS 2014-2018", d["n_children_18"].isna())]:
        print(f"no {name} match: {miss.sum()} {list(d.loc[miss, 'fips'])[:12]}")
    d = d[["fips", "state", "county", "n_children", "n_poor", "moe_poor", "moe_children", "p_acs", "se_acs",
           "rucc", "pop2020", "saipe_pct", "saipe_lo", "saipe_hi",
           "n_children_18", "n_poor_18", "p_acs_18", "se_acs_18"]]
    d["rucc"] = d["rucc"].astype("Int64")
    d["pop2020"] = d["pop2020"].astype("Int64")
    for c in ("n_children_18", "n_poor_18"):
        d[c] = d[c].astype("Int64")
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(d):,} counties, {OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
