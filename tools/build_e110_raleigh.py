"""Build the E110 dataset: Raleigh traffic stops, searches and hits by officer x race and year x race.

    python tools/build_e110_raleigh.py     # download (cached), stream, write data/e110_raleigh_units.csv

Source: Stanford Open Policing Project (Pierson et al. 2020, Nature Human Behaviour), the Raleigh,
North Carolina city file, standardised release of 2020-04-01:
https://stacks.stanford.edu/file/druid:yg821jf8611/yg821jf8611_nc_raleigh_2020_04_01.csv.zip
(13.5 MB zip, 205 MB CSV, 856,400 vehicular stops by the Raleigh Police Department, 2002-2015).
Licence: Open Data Commons Attribution License (ODC-By) v1.0 - attribute "Stanford Open Policing
Project". The raw zip is cached in ~/.cache/pymc-challenges/e110 (not committed).

The CSV is streamed in 300k-row chunks (peak about 0.5 GB). Every stop in the file is
`type == vehicular`. `contraband_found` is recorded only for searched stops (NaN otherwise; no
stop has contraband without a search). `search_basis` is one of consent, probable cause, other
(other includes protective frisks and searches incident to arrest), or missing.

Output: data/e110_raleigh_units.csv, one row per unit x race, for the four largest groups
(black, white, hispanic, asian/pacific islander; 'other' and 'unknown' race, 8,644 stops, dropped):
  level      'city', 'year' or 'officer'
  unit       'Raleigh', the year (2002-2015), or officer_id_hash (a 10-character hash)
  race       black, white, hispanic, asian
  stops, searches, hits                      all searches
  searches_consent, hits_consent             consent searches
  searches_pc, hits_pc                       probable-cause searches
  searches_other, hits_other                 'other' basis (protective, incident to arrest, ...)
(searches with a missing basis count only in the totals). Officers are kept when they made at
least one stop of anyone; officer x race rows with zero stops are written with zeros.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e110_raleigh_units.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e110"
URL = "https://stacks.stanford.edu/file/druid:yg821jf8611/yg821jf8611_nc_raleigh_2020_04_01.csv.zip"
RACES = {"black": "black", "white": "white", "hispanic": "hispanic", "asian/pacific islander": "asian"}
COLS = ["date", "subject_race", "officer_id_hash", "type", "search_conducted", "contraband_found",
        "search_basis"]
BASIS = {"consent": "consent", "probable cause": "pc", "other": "other"}


def raw_zip() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / URL.rsplit("/", 1)[1]
    if not p.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges-e110"})
        tmp = p.with_suffix(".part")
        tmp.write_bytes(urllib.request.urlopen(req, timeout=300).read())
        tmp.rename(p)
    return p


def chunk_counts(c: pd.DataFrame) -> pd.DataFrame:
    """Per-row indicator columns, summed later by any grouping."""
    s = c["search_conducted"].astype(bool).to_numpy()
    h = (c["contraband_found"] == True).to_numpy() & s           # noqa: E712 - NaN-safe
    out = pd.DataFrame({"race": c["subject_race"].map(RACES), "year": c["date"].str[:4].astype(int),
                        "officer": c["officer_id_hash"].astype(str),
                        "stops": 1, "searches": s.astype(int), "hits": h.astype(int)})
    b = c["search_basis"].map(BASIS)
    for key in BASIS.values():
        m = (b == key).to_numpy() & s
        out[f"searches_{key}"] = m.astype(int)
        out[f"hits_{key}"] = (m & h).astype(int)
    return out[out["race"].notna()]


def main() -> None:
    z = zipfile.ZipFile(raw_zip())
    name = z.namelist()[0]
    keys = ["stops", "searches", "hits"] + [f"{a}_{k}" for k in BASIS.values() for a in ("searches", "hits")]
    by_year, by_off = [], []
    n_rows = 0
    for c in pd.read_csv(z.open(name), usecols=COLS, chunksize=300_000, low_memory=False):
        n_rows += len(c)
        assert (c["type"] == "vehicular").all()
        x = chunk_counts(c)
        by_year.append(x.groupby(["year", "race"])[keys].sum())
        by_off.append(x.groupby(["officer", "race"])[keys].sum())
    year = pd.concat(by_year).groupby(level=[0, 1]).sum().reset_index()
    off = pd.concat(by_off).groupby(level=[0, 1]).sum()
    full = pd.MultiIndex.from_product([off.index.levels[0], list(RACES.values())], names=["officer", "race"])
    off = off.reindex(full, fill_value=0).reset_index()
    city = year.groupby("race")[keys].sum().reset_index()
    rows = [city.assign(level="city", unit="Raleigh"),
            year.rename(columns={"year": "unit"}).assign(level="year"),
            off.rename(columns={"officer": "unit"}).assign(level="officer")]
    df = pd.concat(rows, ignore_index=True)[["level", "unit", "race"] + keys]
    df["unit"] = df["unit"].astype(str)
    order = {r: i for i, r in enumerate(RACES.values())}
    df = df.sort_values(["level", "unit", "race"], key=lambda s: s.map(order) if s.name == "race" else s,
                        kind="stable")
    df.to_csv(OUT, index=False)
    print(f"read {n_rows:,} stops; wrote {OUT} ({len(df):,} rows, {OUT.stat().st_size / 1e3:.0f} KB)")
    print(city.set_index("race")[["stops", "searches", "hits"]])
    n_off = off.groupby("officer")["searches"].sum()
    print(f"officers: {len(n_off):,}; with >= 20 searches: {(n_off >= 20).sum()}")


if __name__ == "__main__":
    main()
