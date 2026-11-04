"""Build the E97 datasets: Galveston sea level from a tide gauge, satellite altimetry and GPS.

    uv run python tools/build_e97_data.py            # downloads (about 2 MB) and writes two CSVs

Writes
  data/e97_galveston_sea_land.csv   one table, three instruments (about 100 KB)
  data/e97_noaa2022_scenarios.csv   NOAA 2022 relative sea-level scenarios at the Galveston gauge

Sources (all public, no login; read 2026-11-04):

- PSMSL Revised Local Reference (RLR) monthly means, station 161 GALVESTON II, PIER 21, TX
  https://psmsl.org/data/obtaining/rlr.monthly.data/161.rlrdata
  Permanent Service for Mean Sea Level (PSMSL), 2026, "Tide Gauge Data"; Holgate et al. (2013),
  J. Coastal Research 29(3), 493-504. Free to use with acknowledgement.
- NOAA Laboratory for Satellite Altimetry, mean sea-level anomaly of the Gulf of Mexico ("Gulf of
  America" in the file), annual signals retained, one column per mission
  https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gom_keep_ref_90.csv
  Required acknowledgement: "Altimetry data are provided by NOAA Laboratory for Satellite Altimetry."
- Nevada Geodetic Laboratory (NGL) daily GPS positions, IGS20 frame, stations TXGA (2005-) and GAL2
  (1997-2003), and the NGL equipment-change list
  https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/TXGA.tenv3, .../GAL2.tenv3
  https://geodesy.unr.edu/NGLStationPages/steps.txt
  Blewitt, Hammond & Kreemer (2018), Eos 99, doi:10.1029/2018EO104623.
- NOAA CO-OPS API, 2022 Sea Level Rise Technical Report scenarios at station 8771450 (Galveston Pier 21)
  https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/slr_projections.json?station=8771450
  Sweet et al. (2022), NOAA Technical Report NOS 01. US government work, public domain.

What is changed: GPS daily heights become monthly means (split at antenna changes); altimetry and the
tide gauge are kept at their own resolution. Nothing is smoothed or corrected.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
UA = {"User-Agent": "pymc-challenges"}
URL_TG = "https://psmsl.org/data/obtaining/rlr.monthly.data/161.rlrdata"
URL_ALT = "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gom_keep_ref_90.csv"
URL_GPS = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{}.tenv3"
URL_STEPS = "https://geodesy.unr.edu/NGLStationPages/steps.txt"
URL_SCEN = ("https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/slr_projections.json"
            "?station=8771450&units=metric")


def get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read().decode()


def ngl_date(s: str) -> float:
    d = pd.to_datetime(s, format="%y%b%d")
    start = pd.Timestamp(year=d.year, month=1, day=1)
    return d.year + (d - start).days / (365.25)


def main() -> None:
    rows = []
    # tide gauge: monthly, mm above the RLR datum; -99999 = missing
    tg = pd.read_csv(io.StringIO(get(URL_TG)), sep=";", header=None, names=["t", "h", "miss", "flag"])
    tg = tg[tg.h > -9999]
    for t, h, m in zip(tg.t, tg.h, tg.miss):
        rows.append(("tide_gauge", "PSMSL161", round(t, 4), float(h), 30 - int(m) if m < 99 else 0))
    # altimetry: about every 10 days, mm, one column per mission
    alt = pd.read_csv(io.StringIO(get(URL_ALT)), comment="#")
    for mission in alt.columns[1:]:
        s = alt[["year", mission]].dropna()
        for t, v in zip(s.year, s[mission]):
            rows.append(("altimetry", mission, round(t, 4), float(v), 1))
    # GPS: daily up = u0 + up (m), monthly means, new segment at each NGL antenna change
    steps = [ln.split() for ln in get(URL_STEPS).splitlines()]
    for st in ["GAL2", "TXGA"]:
        ant = sorted(ngl_date(s[1]) for s in steps if s[0] == st and len(s) > 3 and s[3] == "Antenna_Type_Changed")
        g = pd.read_csv(io.StringIO(get(URL_GPS.format(st))), sep=r"\s+")
        t = g["yyyy.yyyy"].to_numpy()
        up = (g["u0(m)"] + g["____up(m)"]).to_numpy() * 1000.0
        seg = np.searchsorted(ant, t)
        month = np.floor((t - np.floor(t)) * 12).clip(0, 11)
        df = pd.DataFrame({"yr": np.floor(t), "mo": month, "seg": seg, "t": t, "up": up})
        for (yr, mo, sg), d in df.groupby(["yr", "mo", "seg"]):
            rows.append(("gps", f"{st}-{sg + 1}", round(d.t.mean(), 4), round(d.up.mean(), 2), len(d)))
        print(st, "antenna changes at", [round(a, 3) for a in ant])
    out = pd.DataFrame(rows, columns=["source", "segment", "t", "value_mm", "n"])
    out.to_csv(OUT / "e97_galveston_sea_land.csv", index=False)
    print(out.groupby(["source", "segment"]).agg(t0=("t", "min"), t1=("t", "max"), rows=("t", "size")))

    sc = pd.DataFrame(json.loads(get(URL_SCEN))["SlrProjections"])
    sc = sc[["scenario", "projectionYear", "projectionRsl", "projectionCiLow", "projectionCiHigh",
             "backgroundRslPerYear"]]
    sc.columns = ["scenario", "year", "rsl_cm", "lo_cm", "hi_cm", "background_cm_per_yr"]
    sc.to_csv(OUT / "e97_noaa2022_scenarios.csv", index=False)
    print(sc.pivot(index="year", columns="scenario", values="rsl_cm"))


if __name__ == "__main__":
    main()
