"""Build the E111 dataset: deaths and exposures by single year of age for 30 European countries.

    python tools/build_e111_eurostat.py        # download (cached), parse, write data/e111_eu_mortality.csv.gz

Sources (Eurostat, CC BY 4.0 under Commission Decision 2011/833/EU; no login; checked 2026-11-06):
- demo_magec: deaths by age (single years 0-99, open 100+, plus UNK), sex and country, annual.
- demo_pjan: population on 1 January by age (single years, open 100+), sex and country.
- demo_mlexpec: life expectancy by age and sex (only e0 is kept, as a check on a life-table code).
Dissemination API: https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/<code>/?format=TSV&compressed=true
The raw .tsv.gz files are cached in ~/.cache/pymc-challenges/e111 (not committed).

SDMX-TSV: the first column holds the comma-joined dimension keys, then one column per year; ':' is a
missing value and flags ('p' provisional, 'e' estimated, 'b' break, ...) trail the numbers.

Selection: EU27 + Iceland, Norway and Switzerland (all 30 kept), years 2000-2024, sexes T, F and M,
single ages 0-89 plus an open group 90+. Several countries' early years stop their single ages at 85,
90, 95 or 99 (Eurostat's open interval moved), so only 11 of the 30 have every single age 0-99 for
2000-2025; the 90+ group is the sum of single ages 90..100+ where those exist and TOTAL - UNK -
(ages 0-89) otherwise, which keeps every country. Cells still missing (NaN): Latvia 2000-01 and
Croatia 2000 (no single ages at all), and ages 85-90+ in Cyprus 2000-01, Romania 2000, Croatia
2011-12 and Hungary 2006-11 (open interval at 85+). Norway 2012 has no total-sex deaths: filled as
F + M. Deaths and population of unknown age (UNK) are spread over ages in proportion. Exposure is the mean of the 1 January populations of years t
and t+1 at the same age (the usual approximation to person-years lived).

Breaks in series: Eurostat flags a 1 January population 'b' when its level was reset (mostly by a
census). The flags are kept (column pop_break) and every break is printed with the size of its step;
the exposures are left as published, and E111 corrects the three large ones (CZ 2021, PL 2021,
DE 2011) itself from the pop column.

Output: data/e111_eu_mortality.csv.gz (1.8 MB), one row per country, sex, year and age (0..90, where
90 is 90+): geo, sex, year, age, deaths (float: whole numbers plus the pro-rata share of UNK deaths),
exposure (person-years), pop (1 January population of the year, sex T only, to keep the file small),
e0_eurostat (Eurostat's published life expectancy at birth for that country, sex and year, repeated
on every age row; NaN if Eurostat has none), pop_break (1 if any 1 January population cell of that
country, sex and year carries Eurostat's 'b' flag).
"""

from __future__ import annotations

import gzip
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e111_eu_mortality.csv.gz"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e111"
API = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{}/?format=TSV&compressed=true"
EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE",
        "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
GEOS = EU27 + ["IS", "NO", "CH"]
YEARS = list(range(2000, 2025))
SEXES = ["T", "F", "M"]
OPEN = 90                                            # ages 90+ are one open group
NUM = re.compile(r"^\s*([0-9.]+)\s*([a-z]*)")


def fetch(code: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{code}.tsv.gz"
    if not p.exists():
        req = urllib.request.Request(API.format(code), headers={"User-Agent": "pymc-challenges-e111"})
        p.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return p


def read_tsv(code: str) -> pd.DataFrame:
    """Long table: the key dimensions + year + value (NaN where Eurostat has ':') + flag letters."""
    with gzip.open(fetch(code), "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        keys = header[0].split("\\")[0].split(",")
        years = [int(y) for y in header[1:]]
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            k = parts[0].split(",")
            if k[keys.index("geo")] not in GEOS:
                continue
            for y, v in zip(years, parts[1:]):
                if 2000 <= y <= 2025:
                    m = NUM.match(v)
                    rows.append((*k, y, float(m.group(1)) if m else np.nan, m.group(2) if m else ""))
    return pd.DataFrame(rows, columns=keys + ["year", "value", "flag"])


def age_num(a: pd.Series) -> pd.Series:
    return a.map(lambda s: 0 if s == "Y_LT1" else 100 if s == "Y_OPEN" else
                 int(s[1:]) if re.fullmatch(r"Y\d+", s) else -1)


def to_open(t: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """(geo, sex, year) x age 0..OPEN, where OPEN collects OPEN+, with UNK spread pro rata.

    The open group is TOTAL - UNK - (ages below OPEN), so it exists even in years where Eurostat's
    own open interval starts lower than 100; it is NaN only when a single age below OPEN is missing.
    The total-sex row is filled from F + M where Eurostat lacks it.
    """
    tot = t[t["age"] == "TOTAL"].set_index(["geo", "sex", "year"])["value"]
    unk = t[t["age"] == "UNK"].set_index(["geo", "sex", "year"])["value"]
    t = t.assign(a=age_num(t["age"]))
    t = t[t["a"] >= 0]
    idx = pd.MultiIndex.from_product([GEOS, SEXES, years], names=["geo", "sex", "year"])
    S = t.set_index(["geo", "sex", "year", "a"])["value"].unstack("a").reindex(idx)
    tot, unk = tot.reindex(idx), unk.reindex(idx).fillna(0.0)
    X = S[list(range(OPEN))].copy()
    singles = S[list(range(OPEN, 101))].sum(axis=1, min_count=101 - OPEN)   # all of OPEN..100+ given
    X[OPEN] = singles.fillna(tot - unk - X.sum(axis=1, min_count=OPEN))
    for g in GEOS:                                     # T = F + M where T is missing
        for y in years:
            if X.loc[(g, "T", y)].isna().all() and X.loc[(g, "F", y)].notna().all():
                X.loc[(g, "T", y)] = X.loc[(g, "F", y)] + X.loc[(g, "M", y)]
                tot[(g, "T", y)] = tot[(g, "F", y)] + tot[(g, "M", y)]
                print(f"  filled {g} {y} T from F + M")
    known = X.sum(axis=1)
    X = X.add(X.div(known, axis=0).mul(unk, axis=0))
    return X


def main() -> None:
    d = read_tsv("demo_magec")
    d = d[(d["unit"] == "NR") & d["sex"].isin(SEXES) & (d["year"] <= 2024)]
    print("deaths:")
    D = to_open(d, YEARS)
    p = read_tsv("demo_pjan")
    p = p[(p["unit"] == "NR") & p["sex"].isin(SEXES)]
    print("population:")
    P = to_open(p, YEARS + [2025])
    # Eurostat's 'b' flag marks a break in series (mostly a census that reset the population level);
    # keep it per country, sex and year, and print the size of each step against the years around it
    brk = p[p["flag"].str.contains("b")].groupby(["geo", "sex", "year"]).size().index
    tp = p[(p["age"] == "TOTAL") & (p["sex"] == "T")].set_index(["geo", "year"])["value"]
    print("breaks in series (population on 1 January, sex T):")
    for g, s, y in brk:
        if s == "T":
            v = {t: tp.get((g, t), np.nan) for t in range(y - 2, y + 2)}
            step, usual = v[y] / v[y - 1] - 1, (v[y - 1] / v[y - 2] + v[y + 1] / v[y]) / 2 - 1
            print(f"  {g} {y}: total {step:+.2%} against {usual:+.2%} in the years around it")
    P0 = P.drop(index=2025, level="year")
    P1 = P.drop(index=2000, level="year").rename(index=lambda y: y - 1, level="year")
    E = (P0 + P1.reindex(P0.index)) / 2

    le = read_tsv("demo_mlexpec")
    le = le[(le["unit"] == "YR") & (le["age"] == "Y_LT1") & le["sex"].isin(SEXES) & (le["year"] <= 2024)]
    e0 = le.set_index(["geo", "sex", "year"])["value"]

    out = pd.DataFrame({"deaths": D.stack(), "exposure": E.stack(), "pop": P0.stack()})
    out.index.names = ["geo", "sex", "year", "age"]
    out = out.reset_index()
    miss = out[out["deaths"].isna() | out["exposure"].isna()]
    for (g, y), m in miss.groupby(["geo", "year"]):
        print(f"missing {g} {y}: ages {m['age'].min()}-{m['age'].max()} ({len(m)} cells over sexes)")
    out["e0_eurostat"] = e0.reindex(pd.MultiIndex.from_frame(out[["geo", "sex", "year"]])).to_numpy()
    out["pop_break"] = pd.MultiIndex.from_frame(out[["geo", "sex", "year"]]).isin(brk).astype(int)
    out["deaths"] = out["deaths"].round(2)
    out["pop"] = out["pop"].where(out["sex"] == "T").round(0).astype("Int64")   # sex T only (size)
    out["exposure"] = out["exposure"].round(1)
    assert (out["exposure"].dropna() > 0).all(), "a zero exposure"
    assert (out["deaths"].dropna() >= 0).all(), "negative deaths"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): {len(out):,} rows, "
          f"{out['deaths'].isna().sum()} cells without deaths, {out['exposure'].isna().sum()} without exposure")
    t = out[out["sex"] == "T"]
    tot = t.groupby("year")["deaths"].sum()
    print("total deaths (sex T, 30 countries):", {y: round(tot[y]) for y in (2000, 2019, 2020, 2021, 2022, 2024)})


if __name__ == "__main__":
    main()
