"""Build the E100 dataset: Safecast radiation readings around Iitate, Fukushima, 2011-2013.

    python tools/build_e100_data.py pull     # fetch the API pages (resumable) to ~/.cache/pymc-challenges/e100
    python tools/build_e100_data.py derive   # write data/safecast_iitate_2011_2013.csv.gz (3.0 MB)

Source: Safecast (https://safecast.org), volunteer bGeigie mobile radiation logs, released under
CC0 1.0 (public domain dedication). Read through the public API, no key:
https://api.safecast.org/en-US/measurements.json?latitude=..&longitude=..&distance=..&captured_after=..
The API returns at most 1,000 rows a page; pages are fetched until one comes back empty.

Facts this script depends on (checked on 2026-11-04):
- `value` is counts per minute (cpm) of the LND 7317 pancake tube; `unit` is "cpm" for every
  mobile log. Safecast converts with 334 cpm per uSv/h (Cs-137 calibration). We keep cpm.
- `device_id`, `height` and `devicetype_id` are null for the bGeigie imports; the drive is
  identified by `measurement_import_id` (one uploaded log file).
- The API circle uses `distance` in metres around (latitude, longitude).

- Some logs were uploaded twice under two import ids (31,761 rows are exact copies of another row's
  time, position and count). They are KEPT here, so the notebook can show and drop them.
- Pages come back in no particular time order, so a pull must run to the first empty page.

Only rows inside the box below and each circle's time window are kept; rows are not otherwise changed.
The pull of 2026-11-04 read 129 pages for the first circle and 181 for the second.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "safecast_iitate_2011_2013.csv.gz"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e100"
API = "https://api.safecast.org/en-US/measurements.json"
UA = "pymc-challenges-e100 (teaching notebook)"

# Circles (lat, lon, radius m, captured_after, captured_before). The first is Iitate village, the
# survey district, over two years; the second is the western edge of Minamisoma, the neighbouring
# district whose 2012 readings set the hyperparameters (it is driven far more often, so one year).
CIRCLES = [(37.66, 140.72, 12000, "2011-09-01", "2013-07-01"),
           (37.633, 140.913, 6000, "2012-01-01", "2013-01-01")]
BOX = dict(lat_min=37.54, lat_max=37.78, lon_min=140.58, lon_max=141.00)


def fetch(url: str, attempts: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if a == attempts - 1:
                raise
            print(f"  {e}: retry in {10 * 2 ** a} s", file=sys.stderr)
            time.sleep(10 * 2**a)
    raise AssertionError


def pull() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for lat, lon, dist, after, before in CIRCLES:
        page = 1
        while True:
            f = RAW / f"c{lat}_{lon}_{dist}_{after}_{before}_p{page:04d}.json.gz"
            if not f.exists():
                url = (f"{API}?latitude={lat}&longitude={lon}&distance={dist}&per_page=1000"
                       f"&page={page}&captured_after={after}&captured_before={before}")
                body = fetch(url)
                tmp = f.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(body))
                tmp.rename(f)
            n = len(json.loads(gzip.decompress(f.read_bytes())))
            print(f"circle {lat},{lon},{dist}: page {page}: {n} rows", flush=True)
            if n == 0:
                break
            page += 1


def derive() -> None:
    parts = []
    for lat, lon, dist, after, before in CIRCLES:
        rows = []
        for f in sorted(RAW.glob(f"c{lat}_{lon}_{dist}_{after}_{before}_p*.json.gz")):
            rows += json.loads(gzip.decompress(f.read_bytes()))
        d = pd.DataFrame(rows)
        t = pd.to_datetime(d["captured_at"], utc=True)
        parts.append(d[(t >= after) & (t < before)])
    df = pd.concat(parts)
    print(f"{len(df)} rows read, {df['id'].nunique()} unique ids; units {df['unit'].value_counts().to_dict()}")
    df = df.drop_duplicates("id")
    df = df[df["unit"] == "cpm"]
    df = df[df.latitude.between(BOX["lat_min"], BOX["lat_max"])
            & df.longitude.between(BOX["lon_min"], BOX["lon_max"])]
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    out = pd.DataFrame({
        "id": df["id"].astype("int64"),
        "captured_at": df["captured_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latitude": df["latitude"].round(6),
        "longitude": df["longitude"].round(6),
        "cpm": df["value"].astype(float),
        "import_id": df["measurement_import_id"].fillna(-1).astype("int64"),
        "user_id": df["user_id"].fillna(-1).astype("int64"),
    }).sort_values("id")
    out.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT} ({len(out)} rows, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    {"pull": pull, "derive": derive}[sys.argv[1]]()
