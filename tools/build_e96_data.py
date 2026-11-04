"""Build the E96 snowpack datasets from the USDA NRCS Air and Water Database (AWDB) REST API.

    .venv/bin/python tools/build_e96_data.py            # writes data/e96_snotel_daily.csv, data/e96_snow_course.csv

No key is needed. The API is public (USDA NRCS National Water and Climate Center, public domain):
https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html

What it fetches:

* SNOTEL daily data, water years 2006-2025, October 1 to June 30, for University Camp (838:CO:SNTL,
  10,330 ft, Boulder Creek headwaters) and four neighbours used only for their snow water equivalent
  (Niwot 663, Lake Eldora 564, Wild Basin 1042, Bear Lake 322). Elements WTEQ (pillow snow water
  equivalent, in), SNWD (ultrasonic snow depth, in), PREC (storage-gauge precipitation accumulated
  since October 1, in) and TAVG (daily mean air temperature, deg F), with the published value, its QC
  flag, and the ORIGINAL value as first reported in real time (`returnOriginalValues=true`), plus
  the 1991-2020 daily median of WTEQ (`centralTendencyType=MEDIAN`).
* The University Camp snow course (05J08:CO:SNOW, 10,310 ft, measured by hand since 1938), duration
  SEMIMONTHLY, which carries the real collection date of each survey. (The MONTHLY values of the same
  course are flagged derivedData and sit on the first of the month; do not use them.)

The API answered 502 to a 20-year request with original values, so the script asks for five years at a
time and retries.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

API = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
OUT = Path(__file__).resolve().parents[1] / "data"
STATIONS = {838: "University Camp", 663: "Niwot", 564: "Lake Eldora", 1042: "Wild Basin", 322: "Bear Lake"}
COURSE = "05J08:CO:SNOW"
FIRST_WY, LAST_WY = 2006, 2025


def get(path: str, **params) -> list:
    for attempt in range(5):
        r = requests.get(f"{API}/{path}", params=params, timeout=180,
                         headers={"User-Agent": "pymc-challenges"})
        if r.ok:
            return r.json()
        print(f"  {r.status_code}, retrying", flush=True)
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    return []


def station_meta() -> dict:
    trip = ",".join(f"{s}:CO:SNTL" for s in STATIONS) + "," + COURSE
    return {m["stationTriplet"]: m for m in get("stations", stationTriplets=trip)}


def daily(sid: int) -> pd.DataFrame:
    rows = []
    for y0 in range(FIRST_WY, LAST_WY + 1, 5):
        js = get("data", stationTriplets=f"{sid}:CO:SNTL", elements="WTEQ,SNWD,PREC,TAVG", duration="DAILY",
                 beginDate=f"{y0 - 1}-10-01", endDate=f"{min(y0 + 4, LAST_WY)}-09-30", returnFlags="true",
                 returnOriginalValues="true", returnSuspectData="true", centralTendencyType="MEDIAN")
        for el in js[0]["data"]:
            code = el["stationElement"]["elementCode"].lower()
            for v in el["values"]:
                rows.append((v["date"], code, v.get("value"), v.get("qcFlag"), v.get("origValue"), v.get("median")))
        time.sleep(0.5)
    long = pd.DataFrame(rows, columns=["date", "el", "value", "flag", "orig", "median"])
    long = long.drop_duplicates(["date", "el"])
    wide = long.pivot(index="date", columns="el", values="value")
    out = pd.DataFrame(index=wide.index)
    out["wteq"] = wide.get("wteq")
    out["wteq_flag"] = long[long.el == "wteq"].set_index("date")["flag"]
    out["wteq_raw"] = long[long.el == "wteq"].set_index("date")["orig"]
    out["wteq_median"] = long[long.el == "wteq"].set_index("date")["median"]
    if sid == 838:
        for el in ("snwd", "prec", "tavg"):
            out[el] = wide.get(el)
            out[el + "_flag"] = long[long.el == el].set_index("date")["flag"]
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out.date.dt.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6])]
    out.insert(0, "station", STATIONS[sid])
    out.insert(0, "station_id", sid)
    return out


def course() -> pd.DataFrame:
    js = get("data", stationTriplets=COURSE, elements="WTEQ,SNWD", duration="SEMIMONTHLY",
             beginDate="1937-10-01", endDate=f"{LAST_WY}-09-30", returnFlags="true")
    rows = []
    for el in js[0]["data"]:
        code = el["stationElement"]["elementCode"].lower()
        for v in el["values"]:
            if "value" in v and "collectionDate" in v:
                rows.append((v["collectionDate"][:10], v["year"], v["month"], code, v["value"]))
    long = pd.DataFrame(rows, columns=["date", "year", "month", "el", "value"])
    wide = long.pivot_table(index=["date", "year", "month"], columns="el", values="value").reset_index()
    wide = wide.rename(columns={"wteq": "swe", "snwd": "depth", "month": "survey_month"})
    wide["survey"] = wide.survey_month.map({1: "Feb 1", 2: "Mar 1", 3: "Apr 1", 4: "May 1", 5: "Jun 1",
                                            12: "Jan 1"})
    wide.insert(0, "course", "University Camp (05J08)")
    return wide[["course", "date", "survey", "swe", "depth"]].sort_values("date")


def main() -> None:
    meta = station_meta()
    frames = []
    for sid in STATIONS:
        print("station", sid, STATIONS[sid], flush=True)
        d = daily(sid)
        m = meta[f"{sid}:CO:SNTL"]
        d.insert(2, "elevation_ft", m["elevation"])
        frames.append(d)
    daily_df = pd.concat(frames, ignore_index=True)
    daily_df.to_csv(OUT / "e96_snotel_daily.csv", index=False, date_format="%Y-%m-%d")
    c = course()
    c.to_csv(OUT / "e96_snow_course.csv", index=False)
    for f in ("e96_snotel_daily.csv", "e96_snow_course.csv"):
        print(f, (OUT / f).stat().st_size // 1024, "KB")
    print(daily_df.groupby("station").size())
    print(c.tail())


if __name__ == "__main__":
    main()
