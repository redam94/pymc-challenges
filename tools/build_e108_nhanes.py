"""Build the E108 dataset: two 24-hour dietary recalls per US adult, NHANES 2017-2018 (cycle J).

    python tools/build_e108_nhanes.py            # download (cached) and write data/e108_nhanes_recalls.csv.gz

Sources (CDC / National Center for Health Statistics, US federal work, public domain, no login):
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1TOT_J.xpt  day-1 total nutrients
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR2TOT_J.xpt  day-2 total nutrients
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt    demographics
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALQ_J.xpt     alcohol questionnaire (18+)

The raw SAS transport files (22 MB) are cached in ~/.cache/pymc-challenges/e108 and not committed.

Rows kept: adults aged 19 and over (RIDAGEYR) with a reliable day-1 recall (DR1DRSTZ == 1, "reliable
and met the minimum criteria"); their day-2 recall is kept when it is reliable too (DR2DRSTZ == 1).
Day 1 is an in-person interview, day 2 a telephone interview 3-10 days later. One row per person and
reliable recall day (long format).

Facts this script depends on (checked on 2026-11-05):
- md5 DR1TOT_J ad2c80017329d11a3527844f3ec9c7db, DR2TOT_J 841c050684aa2dad6d360a951d357cfb,
  DEMO_J 3d5e48e4255b2ec80578a0eb45480d4c. DR1TOT_J 8,704 x 168, DR2TOT_J 8,704 x 85, DEMO_J 9,254 x 46.
- `pandas.read_sas(format="xport")` reads the transport format's exact zeros as 5.4e-79: alcohol is
  "zero" on 6,481 of the day-1 rows in that form. Values below 1e-9 in absolute value are set to 0.
- One adult (SEQN 93792) reports 0 mg sodium and 0 mg potassium on day 1 (reliable status); the
  row is kept here, and the notebook drops recall days with zero sodium before taking logs.
- 4,863 adults have a reliable day-1 recall; 4,240 of them also have a reliable day-2 recall.
- Day of the week (DRxDAY: 1 = Sunday ... 7 = Saturday) is not uniform: NHANES schedules
  weekend days on purpose (Fri-Sun are 63% of adult day-1 recalls). The dietary weights
  (WTDRD1, WTDR2D) correct for this and for non-response.
- ALQ_J (md5 042926fb111908aaf08e0a0afd496fda, 5,533 x 10): ALQ111 "ever had a drink" (1 yes, 2 no);
  ALQ121 "past 12 months, how often" (0 never in the last year, 1 every day, 2 nearly every day,
  3 three-four times a week, 4 twice a week, 5 once a week, 6 two-three times a month, 7 once a month,
  8 seven-eleven times a year, 9 three-six times a year, 10 once or twice a year; 77/99 refused/don't
  know -> NaN). It is skipped (NaN) when ALQ111 = 2, so `alq_freq` is set to 0 for lifetime abstainers.
  ALQ130 = usual drinks on a drinking day in the past year (777/999 -> NaN).
- INDFMPIR (family income to poverty ratio) is top-coded at 5 and missing for 572 adults.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e108_nhanes_recalls.csv.gz"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e108"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
FILES = {
    "DR1TOT_J": "ad2c80017329d11a3527844f3ec9c7db",
    "DR2TOT_J": "841c050684aa2dad6d360a951d357cfb",
    "DEMO_J": "3d5e48e4255b2ec80578a0eb45480d4c",
    "ALQ_J": "042926fb111908aaf08e0a0afd496fda",
}
UA = "pymc-challenges-e108/1.0 (teaching notebook; https://github.com/redam94/pymc-challenges)"
NUTRIENTS = {"TKCAL": "kcal", "TSODI": "sodium_mg", "TPOTA": "potassium_mg", "TALCO": "alcohol_g"}


def fetch(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.xpt"
    if not path.exists() or path.stat().st_size == 0:
        req = urllib.request.Request(f"{BASE}/{name}.xpt", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != FILES[name]:
        print(f"warning: {name}.xpt md5 {md5} differs from the checked {FILES[name]} (CDC may have revised it)")
    return path


def read_xpt(name: str) -> pd.DataFrame:
    df = pd.read_sas(fetch(name), format="xport")
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-9, 0.0)   # transport-format zeros come back as 5.4e-79
    return df


def main() -> None:
    demo = read_xpt("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "RIDRETH3", "SDMVPSU", "SDMVSTRA"]]
    alq = read_xpt("ALQ_J")[["SEQN", "ALQ111", "ALQ121", "ALQ130"]]
    alq["alq_ever"] = alq["ALQ111"].map({1.0: 1.0, 2.0: 0.0})
    alq["alq_freq"] = alq["ALQ121"].where(alq["ALQ121"] <= 10)
    alq.loc[alq["ALQ111"] == 2, "alq_freq"] = 0.0
    alq["alq_drinks"] = alq["ALQ130"].where(alq["ALQ130"] < 100)
    demo = demo.merge(alq[["SEQN", "alq_ever", "alq_freq", "alq_drinks"]], on="SEQN", how="left")
    days = []
    for d in (1, 2):
        t = read_xpt(f"DR{d}TOT_J")
        cols = {"SEQN": "seqn", f"DR{d}DRSTZ": "status", f"DR{d}DAY": "dow"}
        cols |= {f"DR{d}{k}": v for k, v in NUTRIENTS.items()}
        keep = t[list(cols) + (["WTDRD1", "WTDR2D"] if d == 1 else [])].rename(columns=cols)
        keep = keep[keep["status"] == 1].drop(columns="status")
        keep["day"] = d
        days.append(keep)
    w = days[0][["seqn", "WTDRD1", "WTDR2D"]]
    days[0] = days[0].drop(columns=["WTDRD1", "WTDR2D"])
    long = pd.concat(days, ignore_index=True)
    demo = demo.rename(columns={"SEQN": "seqn", "RIDAGEYR": "age", "RIAGENDR": "sex", "INDFMPIR": "pir",
                                "RIDRETH3": "race_eth", "SDMVPSU": "psu", "SDMVSTRA": "stratum"})
    demo = demo.merge(w.rename(columns={"WTDRD1": "wt_day1", "WTDR2D": "wt_day2"}), on="seqn")
    adults = demo[demo["age"] >= 19]
    long = long.merge(adults, on="seqn", how="inner")
    long = long[long["seqn"].isin(days[0]["seqn"])]          # day 2 only for people with a reliable day 1
    long["sex"] = long["sex"].map({1.0: "male", 2.0: "female"})
    for c in ("seqn", "age", "race_eth", "psu", "stratum", "dow", "day"):
        long[c] = long[c].astype(int)
    for c in ("kcal", "sodium_mg", "potassium_mg"):
        long[c] = long[c].round(0).astype(int)
    long["alcohol_g"] = long["alcohol_g"].round(1)
    long = long[["seqn", "day", "dow", "sodium_mg", "potassium_mg", "alcohol_g", "kcal", "age", "sex", "pir",
                 "race_eth", "alq_ever", "alq_freq", "alq_drinks", "wt_day1", "wt_day2", "psu", "stratum"]].sort_values(["seqn", "day"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    n1 = (long["day"] == 1).sum()
    n2 = (long["day"] == 2).sum()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB): {len(long):,} rows, {n1:,} adults, {n2:,} with day 2")
    print("md5", hashlib.md5(OUT.read_bytes()).hexdigest())
    assert np.isfinite(long["sodium_mg"]).all()


if __name__ == "__main__":
    main()
