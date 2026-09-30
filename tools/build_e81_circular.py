"""Build the E81 datasets: hourly wind at Santa Barbara airport and camera-trap times from Barro
Colorado Island.

    .venv/bin/python tools/memguard.py --max-gb 3 -- .venv/bin/python tools/build_e81_circular.py

* data/asos_ksba_2021_2023.csv.gz - routine hourly (METAR) observations of Santa Barbara Municipal
  Airport (ICAO KSBA, FAA SBA) for 2021-2023 from the Iowa Environmental Mesonet ASOS archive:
  valid (UTC), drct (degrees the wind blows FROM, clockwise from true north, reported in steps
  of 10; 0 with sknt = 0 means calm), sknt (knots), tmpf, dwpf (Fahrenheit). Missing values are
  empty. Only the routine hourly reports are kept (report_type=3; no specials).
* data/bci_camera_trap_times.csv - Rowcliffe, Kays, Kranstauber, Carbone & Jansen (2014),
  "Activity level estimation data", figshare, doi:10.6084/m9.figshare.1160536 (CC BY 4.0), file
  BCItime.txt: species and time of day (fraction of 24 h) of 17,820 camera-trap records on Barro
  Colorado Island, Panama, 2008. Written unchanged apart from the delimiter (space -> comma).
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "pymc-challenges"}
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=SBA&data=drct&data=sknt"
       "&data=tmpf&data=dwpf&year1=2021&month1=1&day1=1&year2=2024&month2=1&day2=1&tz=Etc/UTC"
       "&format=onlycomma&latlon=no&missing=empty&trace=empty&direct=no&report_type=3")
BCI = "https://ndownloader.figshare.com/files/3219290"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        return r.read()


def main() -> None:
    w = pd.read_csv(io.BytesIO(fetch(IEM)))
    w = w.drop(columns="station")
    w.to_csv(DATA / "asos_ksba_2021_2023.csv.gz", index=False)
    print(f"KSBA: {len(w):,} rows, {w.valid.min()} .. {w.valid.max()}")

    b = pd.read_csv(io.BytesIO(fetch(BCI)), sep=r"\s+")
    b.to_csv(DATA / "bci_camera_trap_times.csv", index=False)
    print(f"BCI: {len(b):,} records, {b.species.nunique()} species")


if __name__ == "__main__":
    main()
