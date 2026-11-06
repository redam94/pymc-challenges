"""Build the E113 dataset: annual peak flows at 12 Susquehanna, Potomac and Schuylkill gauges.

    python tools/build_e113_peaks.py      # download (cached), parse, write data/e113_peaks.csv

Source: USGS National Water Information System (NWIS), US federal work in the public domain.
  annual peaks  https://nwis.waterdata.usgs.gov/nwis/peak?site_no=<id>&agency_cd=USGS&format=rdb
  site metadata https://waterservices.usgs.gov/nwis/site/?sites=<ids>&siteOutput=expanded&format=rdb
NWIS is being migrated to https://api.waterdata.usgs.gov (OGC API; the same peaks are served
there). The raw RDB files are cached in ~/.cache/pymc-challenges/e113 (not committed); the
retrieval date is printed in each file's header (first build: 2026-11-06).

Output: data/e113_peaks.csv, one row per recorded annual peak (about 1,400 rows):
  site_no, station, drain_sqmi, lat, lon     gauge and its drainage area (square miles)
  water_year     October-September water year of the peak (a peak on 1786-10-05 is water year 1787)
  peak_date      as published; '-00' marks an unknown day or month
  peak_cfs       peak discharge (cubic feet per second); NaN when the file gives none
  gage_ht_ft     gauge height of the peak (feet), NaN when missing
  peak_cd        the NWIS qualification codes, comma-separated (see below)
  historic       1 for code 7 (a historic peak, outside the systematic record)
  regulated      1 for code 5 or 6 (affected by regulation or diversion)
  estimate       1 for code 2 (discharge is an estimate)
  year_last_pk   "highest since" year when the file gives one
Codes: 1 daily mean, 2 estimate, 5/6 regulation, 7 historic, 8 greater than, 9 snowmelt/hurricane/
ice jam, A year unknown, Bd/Bm day/month unknown, C urbanisation, R revised.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e113_peaks.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e113"
SITES = ["01570500", "01536500", "01531500", "01540500", "01553500", "01567000", "01576000",
         "01503000", "01638500", "01613000", "01646500", "01474500"]
PEAK_URL = "https://nwis.waterdata.usgs.gov/nwis/peak?site_no={}&agency_cd=USGS&format=rdb"
SITE_URL = ("https://waterservices.usgs.gov/nwis/site/?sites={}&siteOutput=expanded&format=rdb")


def fetch(url: str, name: str) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    if not p.exists():
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges-e113"})
        text = urllib.request.urlopen(req, timeout=120).read().decode()
        if "agency_cd" not in text:
            raise RuntimeError(f"{url} did not return an RDB table")
        p.write_text(text)
    return p.read_text()


def read_rdb(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#", dtype=str, keep_default_na=False)
    return df.iloc[1:].reset_index(drop=True)          # the second line holds column widths


def main() -> None:
    sites = read_rdb(fetch(SITE_URL.format(",".join(SITES)), "sites.rdb"))
    sites = sites[["site_no", "station_nm", "drain_area_va", "dec_lat_va", "dec_long_va"]].rename(
        columns={"station_nm": "station", "drain_area_va": "drain_sqmi", "dec_lat_va": "lat",
                 "dec_long_va": "lon"})
    for c in ("drain_sqmi", "lat", "lon"):
        sites[c] = pd.to_numeric(sites[c])
    frames = []
    for s in SITES:
        frames.append(read_rdb(fetch(PEAK_URL.format(s), f"peak_{s}.rdb")))
    pk = pd.concat(frames, ignore_index=True)
    year = pk["peak_dt"].str[:4].astype(int)
    month = pd.to_numeric(pk["peak_dt"].str[5:7], errors="coerce").fillna(0).astype(int)
    codes = pk["peak_cd"].str.split(",").apply(lambda c: {x.strip() for x in c if x.strip()})
    out = pd.DataFrame({
        "site_no": pk["site_no"],
        "water_year": np.where(month >= 10, year + 1, year),
        "peak_date": pk["peak_dt"],
        "peak_cfs": pd.to_numeric(pk["peak_va"], errors="coerce"),
        "gage_ht_ft": pd.to_numeric(pk["gage_ht"], errors="coerce"),
        "peak_cd": pk["peak_cd"],
        "historic": codes.apply(lambda c: int("7" in c)),
        "regulated": codes.apply(lambda c: int(bool(c & {"5", "6"}))),
        "estimate": codes.apply(lambda c: int("2" in c)),
        "year_last_pk": pd.to_numeric(pk["year_last_pk"], errors="coerce").astype("Int64"),
    })
    out = sites.merge(out, on="site_no", how="right")
    out = out.sort_values(["site_no", "water_year"]).reset_index(drop=True)
    assert not out.duplicated(["site_no", "water_year"]).any(), "two peaks in one water year"
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    g = out.groupby("site_no")
    print(pd.DataFrame({"station": g["station"].first(), "n": g.size(), "historic": g["historic"].sum(),
                        "first": g["water_year"].min(), "last": g["water_year"].max(),
                        "regulated": g["regulated"].sum()}).to_string())
    print(f"wrote {OUT} ({len(out)} rows, {OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
