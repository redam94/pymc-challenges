"""Build the E98 data: US output indicators as they were published on given dates (ALFRED vintages).

    python tools/build_e98_data.py            # about 190 small CSV requests to ALFRED, a few minutes

Source: ALFRED (ArchivaL Federal Reserve Economic Data), Federal Reserve Bank of St. Louis,
https://alfred.stlouisfed.org. The endpoint
    https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=<SERIES>&vintage_date=YYYY-MM-DD
returns the whole series as it stood on that date (the latest vintage on or before it; the column
header repeats the requested date) and needs no API key. Do NOT use
fred.stlouisfed.org/graph/fredgraph.csv: it ignores vintage_date and returns today's data.
The series are compiled by BEA (GDP, GDI, personal income), BLS (payrolls), the Federal Reserve
Board (industrial production) and the Census Bureau (advance retail sales RSAFS, deflated by the
CPI, CPIAUCSL, by FRED / the St. Louis Fed to give RRSFS); US government
statistics in the public domain, cite as "Federal Reserve Bank of St. Louis, ALFRED".

What is written, to data/e98_alfred_vintages.csv (long format, about 3,750 rows, about 150 KB):
  series, vintage, date, value
`value` is the value for period `date` as it stood on date `vintage`.
Two kinds of rows:
* replay: the complete history from 2011-10 to 2022-12 of each of the six series at six request
  dates (2022-07-28, 2022-08-25, 2022-09-29, 2023-10-01, 2024-10-01, 2026-10-01);
* first releases: for the two quarterly series, the value of each quarter 2014Q3-2019Q4 in the first
  monthly request that contains it (a vintage on the 15th of every month from 2014-10, when ALFRED's
  vintages of these two growth-rate series begin, to 2020-12; only quarters new in it are kept).
Downloads are kept in ~/.cache/pymc-challenges/e98/ and parsed as data only.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e98"
OUT = ROOT / "data" / "e98_alfred_vintages.csv"
URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}&vintage_date={v}"

QUARTERLY = {"gdp": "A191RL1Q225SBEA", "gdi": "A261RL1Q225SBEA"}
MONTHLY = {"payrolls": "PAYEMS", "indpro": "INDPRO", "income_ex_transfers": "W875RX1",
           "retail_sales": "RRSFS"}
REPLAY = ["2022-07-28", "2022-08-25", "2022-09-29", "2023-10-01", "2024-10-01", "2026-10-01"]
START, END = "2011-10-01", "2022-12-01"


def fetch(sid: str, vintage: str) -> tuple[str, list[tuple[str, str]]]:
    """(vintage date, [(date, value), ...]) for one series at one request date."""
    f = CACHE / f"{sid}_{vintage}.csv"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(URL.format(sid=sid, v=vintage), timeout=60) as r:
                    f.write_bytes(r.read())
                break
            except OSError:
                time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError(f"could not fetch {sid} {vintage}")
        time.sleep(0.3)
    rows = list(csv.reader(io.StringIO(f.read_text())))
    head = rows[0]
    if head[0] != "observation_date" or not head[1].startswith(sid + "_"):
        raise ValueError(f"unexpected header {head} for {sid} {vintage}")
    return vintage, [(d, v) for d, v in rows[1:] if v not in ("", ".")]


def main() -> None:
    out = []
    for vreq in REPLAY:
        for name, sid in {**QUARTERLY, **MONTHLY}.items():
            served, rows = fetch(sid, vreq)
            out += [(name, served, d, v) for d, v in rows if START <= d <= END]
            print(f"{name:20s} vintage {vreq}  last observation {rows[-1][0]}")
    for name, sid in QUARTERLY.items():
        seen = set()
        n = 0
        for year in range(2014, 2021):
            for month in range(10 if year == 2014 else 1, 13):
                served, rows = fetch(sid, f"{year}-{month:02d}-15")
                first = not seen                # the first request only seeds what was already out
                for d, v in rows:
                    if not first and d not in seen and START <= d <= "2019-10-01":
                        out.append((name, served, d, v))
                        n += 1
                    seen.add(d)
        print(f"{name}: {n} first releases")
    out = sorted(set(out))
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "vintage", "date", "value"])
        w.writerows(out)
    print(f"wrote {len(out)} rows to {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
