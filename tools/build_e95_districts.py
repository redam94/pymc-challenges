"""Build data/e95_cd120_districts.npz: the 2026 House districts (120th Congress lines) for the E95 map.

Source: US Census Bureau TIGER/Line 2026, congressional districts of the 120th Congress, one file per
state (https://www2.census.gov/geo/tiger/TIGER2026/CD/). These are the lines in force for the
November 2026 elections, including the maps redrawn mid-decade. The shapefiles are read with numpy
(no GIS dependency). Only clockwise rings, the outer boundaries, are kept, and holes are dropped. The rings
are thinned by snapping to a 0.02-degree grid and dropping repeated points, and rings smaller than a few
grid cells are discarded unless they are a district's largest piece.

    .venv/bin/python tools/build_e95_districts.py <download dir>
"""
import io
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_{}_cd120.zip"
STATES = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
          "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS",
          "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
          "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY",
          "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC",
          "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
          "55": "WI", "56": "WY"}
GRID = 0.02


def read_dbf(b):
    n, hlen, rlen = struct.unpack("<IHH", b[4:12])
    fields, off = [], 32
    while b[off] != 0x0D:
        name = b[off:off + 11].split(b"\0")[0].decode()
        fields.append((name, b[off + 16]))
        off += 32
    rows = []
    for i in range(n):
        rec, pos, row = b[hlen + i * rlen: hlen + (i + 1) * rlen], 1, {}
        for name, ln in fields:
            row[name] = rec[pos:pos + ln].decode("latin-1").strip()
            pos += ln
        rows.append(row)
    return rows


def read_shp(b):
    out, off = [], 100
    while off < len(b):
        clen = struct.unpack(">I", b[off + 4:off + 8])[0] * 2
        rec = b[off + 8: off + 8 + clen]
        off += 8 + clen
        if struct.unpack("<i", rec[:4])[0] != 5:
            out.append([])
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = np.frombuffer(rec, "<i4", nparts, 44)
        pts = np.frombuffer(rec, "<f8", 2 * npts, 44 + 4 * nparts).reshape(-1, 2)
        out.append(np.split(pts, parts[1:]))
    return out


def signed_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def thin(r):
    q = np.round(r / GRID) * GRID
    keep = np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]
    return q[keep]


def main(dl):
    dl.mkdir(parents=True, exist_ok=True)
    xy, start, code = [], [0], []
    for fips, st in STATES.items():
        z = dl / f"tl_2026_{fips}_cd120.zip"
        if not z.exists():
            z.write_bytes(urllib.request.urlopen(URL.format(fips), timeout=120).read())
        with zipfile.ZipFile(io.BytesIO(z.read_bytes())) as zf:
            nm = {Path(n).suffix: n for n in zf.namelist()}
            rows, shapes = read_dbf(zf.read(nm[".dbf"])), read_shp(zf.read(nm[".shp"]))
        for row, rings in zip(rows, shapes):
            cd = row.get("CD120FP", "")
            if not cd.isdigit() or int(cd) > 60:                  # "ZZ" water-only, 98 non-voting
                continue
            outer = [r for r in rings if signed_area(r) < 0]       # shapefile outer rings run clockwise
            if not outer:
                continue
            big = max(range(len(outer)), key=lambda i: -signed_area(outer[i]))
            for i, r in enumerate(outer):
                t = thin(r)
                if len(t) < 4 or (i != big and -signed_area(t) < 4 * GRID ** 2):
                    continue
                xy.append(t)
                start.append(start[-1] + len(t))
                code.append(f"{st}-{int(cd):02d}")
        print(st, len(set(c for c in code if c.startswith(st))), "districts")
    xy = np.vstack(xy).astype(np.float32)
    code = np.array(code)
    print(len(set(code)), "districts,", len(code), "rings,", len(xy), "points")
    np.savez_compressed(ROOT / "data/e95_cd120_districts.npz", xy=xy, start=np.array(start, np.int64), code=code)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
