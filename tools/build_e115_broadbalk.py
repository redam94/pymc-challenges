"""Build the E115 dataset: Broadbalk wheat yields 1852-1925 joined to a hand-coded treatment design.

    python tools/build_e115_broadbalk.py   # download (cached), join, check, write data/e115_broadbalk.csv

Yields: agridat ``broadbalk.wheat`` (Kevin Wright's R package, MIT licence), electronic version of
Andrews & Herzberg (1985), *Data*, Springer, from the Rothamsted (Lawes & Gilbert) records:
  https://raw.githubusercontent.com/kwstat/agridat/main/data/broadbalk.wheat.txt
Grain and straw are t/ha at field moisture (about 85% dry matter), per the e-RA Broadbalk page.
The raw file is cached in ~/.cache/pymc-challenges/e115 (not committed).

Design: coded by hand from the e-RA (Electronic Rothamsted Archive) Broadbalk "Early period
1852-1925" factor tables (https://era.rothamsted.ac.uk/Broadbalk, read 2026-11-06) and the agridat
help page. Attribute Rothamsted Research and the e-RA. Every switch is listed in PLOTS below.

  N1 = 48, N2 = 96, N3 = 144, N4 = 192 kg N/ha.
  Ammonium salts (ammonium sulphate/chloride, "N"): all in autumn 1852-77, all in spring 1878-83,
    24 kg in autumn and the rest in spring 1884-1925; strip 15 instead: spring 1873-77, all in
    autumn 1878-1925. Sodium nitrate ("N*"): all in spring.
  Minerals: P 35, K 90, Na 16, Mg 11 kg/ha in autumn; all omitted in 1915, K also 1917-1919 (war).
    Plots 12 and 14 instead get the larger Na* and Mg* dressings in place of K (agridat's "N2 P Na*",
    "N2 P Mg*"; the e-RA gives 57 kg Na and 31 kg Mg for the period before 1974). They are coded
    na = 1 / mg = 1 like the standard dose: only these two plots have Na or Mg without K, so the
    model's w_na and w_mg describe these dressings.
  Farmyard manure 35 t/ha a year on plot 2b since 1843, about 225 kg N/ha.

Output: data/e115_broadbalk.csv, one row per plot and year (1,258 rows):
  year, plot (P02 ... P19), grain, straw (t/ha)
  n_kg        fertiliser N applied for that crop (kg N/ha); 0 for none
  n_form      "amm" (ammonium salts), "nit" (sodium nitrate), "none"
  n_timing    "autumn", "spring", "split" or "none"
  p, k, na, mg  1 if that mineral was applied for that crop (0 also in the war omissions)
  fym         1 on plot 2b (farmyard manure)
  rape        1 on plot 19 (rape cake, about 96 kg organic N/ha)
  minerals_last_year  1 on the alternating plots 17/18 in their nitrogen year (PKNaMg the year before)
  n_last_year fertiliser N applied to the plot the previous year (kg/ha; for the carry-over test)
  use         1 if the row's treatment is known exactly; 0 for plot 9 before 1894 (9a/9b halves
              with different treatments, harvested as one), plot 15 before 1873 (15a/15b halves),
              plot 19 (rape cake of unknown availability)
  note        why use = 0, or the treatment phase
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e115_broadbalk.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e115"
URL = "https://raw.githubusercontent.com/kwstat/agridat/main/data/broadbalk.wheat.txt"
MD5 = "b604da5cb694205935b7e5e9192a6504"   # of the file as downloaded 2026-11-06

FULL = dict(p=1, k=1, na=1, mg=1)
NONE = dict(p=0, k=0, na=0, mg=0)


def amm_timing(year: int, plot: str) -> str:
    """When the ammonium salts were spread, from the e-RA factor notes."""
    if plot == "P15":
        return "spring" if year <= 1877 else "autumn"
    if year <= 1877:
        return "autumn"
    if year <= 1883:
        return "spring"
    return "split"


def design(plot: str, year: int) -> dict:
    """Treatment of one plot for the crop harvested in ``year`` (1852-1925)."""
    d = dict(n_kg=0, n_form="none", n_timing="none", **NONE, fym=0, rape=0, minerals_last_year=0,
             use=1, note="")

    def amm(kg, minerals=FULL):
        d.update(n_kg=kg, n_form="amm", n_timing=amm_timing(year, plot), **minerals)

    if plot == "P02":                                   # 2b: farmyard manure since 1843
        d.update(fym=1, note="FYM 35 t/ha")
    elif plot == "P03":                                 # nil since 1844
        d.update(note="nil")
    elif plot == "P05":
        d.update(**FULL, note="PKNaMg")
    elif plot == "P06":
        amm(48); d["note"] = "N1 PKNaMg"
    elif plot == "P07":
        amm(96); d["note"] = "N2 PKNaMg"
    elif plot == "P08":
        amm(144); d["note"] = "N3 PKNaMg"
    elif plot == "P09":
        if year >= 1894:
            d.update(n_kg=48, n_form="nit", n_timing="spring", **FULL, note="N1* PKNaMg")
        else:
            d.update(use=0, note="9a/9b halves treated differently 1852-93")
    elif plot == "P10":
        amm(96, NONE); d["note"] = "N2"
    elif plot == "P11":
        amm(96, dict(p=1, k=0, na=0, mg=0)); d["note"] = "N2 P"
    elif plot == "P12":
        amm(96, dict(p=1, k=0, na=1, mg=0)); d["note"] = "N2 P Na"
    elif plot == "P13":
        amm(96, dict(p=1, k=1, na=0, mg=0)); d["note"] = "N2 P K"
    elif plot == "P14":
        amm(96, dict(p=1, k=0, na=0, mg=1)); d["note"] = "N2 P Mg"
    elif plot == "P15":
        if year >= 1873:
            amm(96); d["note"] = "N2 PKNaMg, N timing differs from plot 7"
        else:
            d.update(use=0, note="15a/15b halves treated differently 1852-72")
    elif plot == "P16":
        if year <= 1864:
            d.update(n_kg=192, n_form="amm", n_timing="autumn", **FULL, note="N4 PKNaMg")
        elif year <= 1883:
            d.update(note="nil 1865-83 (after N4 PKNaMg)")
        else:
            d.update(n_kg=96, n_form="nit", n_timing="spring", **FULL, note="N2* PKNaMg")
    elif plot in ("P17", "P18"):                        # N2 alone and PKNaMg alone, alternately
        n_year = (year % 2 == 0) if plot == "P17" else (year % 2 == 1)
        if n_year:
            amm(96, NONE); d.update(minerals_last_year=1, note="N2 year (minerals last year)")
        else:
            d.update(**FULL, note="PKNaMg year (N2 last year)")
    elif plot == "P19":
        d.update(rape=1, use=0, note="rape cake (with N1.5 P to 1878)")
    else:
        raise ValueError(plot)
    # war-time omissions of the minerals
    if year == 1915:
        d.update(**NONE)
    if 1917 <= year <= 1919:
        d["k"] = 0
    return d


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / "broadbalk.wheat.txt"
    if not raw.exists():
        print(f"downloading {URL}")
        raw.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    import hashlib
    md5 = hashlib.md5(raw.read_bytes()).hexdigest()
    if md5 != MD5:
        print(f"warning: md5 {md5} differs from the 2026-11-06 download {MD5}")
    y = pd.read_csv(raw, sep=r"\s+")
    assert y.shape == (1258, 4), y.shape
    assert y["year"].between(1852, 1925).all() and y["plot"].nunique() == 17
    assert not y.duplicated(["year", "plot"]).any() and not y.isna().any().any()

    rows = [dict(year=int(r.year), plot=r.plot, grain=r.grain, straw=r.straw, **design(r.plot, int(r.year)))
            for r in y.itertuples()]
    df = pd.DataFrame(rows).sort_values(["plot", "year"]).reset_index(drop=True)
    df["n_last_year"] = df.groupby("plot")["n_kg"].shift(1)
    first = df["year"] == 1852
    # 1851 (not in the table): the alternating strips were in the same rhythm; others as in 1852
    df.loc[first, "n_last_year"] = df.loc[first].apply(
        lambda r: (96 if r.n_kg == 0 else 0) if r.plot in ("P17", "P18") else r.n_kg, axis=1)
    # the first used year of a split strip: both halves had nitrogen the year before (e-RA: 9a and 9b
    # N1*/N2*, so 72 kg on average; 15a N2 and 15b N1.5, so 84 kg), not the 0 coded for the unused rows
    for plot, year, kg in (("P09", 1894, 72), ("P15", 1873, 84)):
        df.loc[(df["plot"] == plot) & (df["year"] == year), "n_last_year"] = kg
    df["n_last_year"] = df["n_last_year"].astype(int)

    # unit tests on the design
    per = df.groupby("plot")
    assert (per.size() == 74).all()
    assert df.query("plot == 'P08' and (year < 1915 or year > 1919)")[["n_kg", "p", "k"]].eq([144, 1, 1]).all().all()
    assert (df.query("plot == 'P17'").set_index("year")["n_kg"].loc[[1852, 1853]] == [96, 0]).all()
    assert (df.query("plot == 'P18'").set_index("year")["n_kg"].loc[[1852, 1853]] == [0, 96]).all()
    assert df.query("plot == 'P16'").groupby("note").size().to_dict() == {
        "N2* PKNaMg": 42, "N4 PKNaMg": 13, "nil 1865-83 (after N4 PKNaMg)": 19}
    assert df["use"].sum() == 1258 - 42 - 21 - 74, df["use"].sum()
    assert df.query("year == 1915")[["p", "k", "na", "mg"]].sum().sum() == 0

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['use'].sum()} usable, {OUT.stat().st_size / 1e3:.0f} KB")
    print(df.groupby(["plot", "note"]).agg(n=("grain", "size"), y0=("year", "min"), y1=("year", "max"),
                                           grain=("grain", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
