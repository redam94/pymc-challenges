"""Build the E109 dataset: capture-history counts for multiple-systems estimation (MSE).

    python tools/build_e109_mse.py        # download (cached), parse, write data/e109_mse_lists.csv

Sources (CRAN package tarballs, GPL >= 2; the data inside come from published reports):
- LCMCR 0.4.14 (Manrique-Vallier), data/kosovo_aggregate.RData: 4,400 killings in Kosovo,
  20 March - 22 June 1999, one row per victim with 0/1 factors EXH (exhumations), ABA (American Bar
  Association / CEELI interviews), OSCE (OSCE reports) and HRW (Human Rights Watch). Analysed in
  Ball, Betts, Scheuren, Dudukovich & Asher (2002), report to the ICTY.
- SparseMSE 2.0.1 (Chan, Silverman & Vincent), data/*.rda: one row per OBSERVED capture history
  with a count column (zero cells omitted).
  UKdat: the UK 2013 National Crime Agency strategic assessment of potential victims of trafficking
  (lists LA local authorities, NG NGOs, PF police forces, GO government organisations, GP general
  public, NCA National Crime Agency), used for the Home Office's 2014 estimate (Silverman 2014).
  Ned: the Netherlands (six lists, Silverman 2020). NewOrl: Greater New Orleans (eight anonymised
  lists, Bales, Murphy & Silverman 2019). Western: a US western site (five anonymised lists,
  Farrell et al. 2019).
CRAN moves superseded tarballs to src/contrib/Archive/<pkg>/, so both URLs are tried. The raw
tarballs are cached in ~/.cache/pymc-challenges/e109 (not committed).

The files are R XDR serialisations (gzip or bzip2); a small pure-Python reader parses them (as in
E23/E24/E28). Checked on 2026-11-05 against R 4 (`load()` + `table()`): Kosovo 15 observed histories
summing to 4,400 (EXH only 1,131; all four 27); UK 25 histories summing to 2,744 (LA only 54, NG only
463); Netherlands 24 cells, 8,234; New Orleans 19 cells, 185; Western 13 cells, 345.

Output: data/e109_mse_lists.csv, long format, one row per observed history:
dataset, lists (list names joined by '|', in history order), history (a 0/1 string), count.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e109_mse_lists.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e109"
CRAN = "https://cran.r-project.org/src/contrib"
PKGS = {"LCMCR": "0.4.14", "SparseMSE": "2.0.1"}
SPARSE = {"uk2013": "UKdat", "netherlands": "Ned", "new_orleans": "NewOrl", "western_us": "Western"}


def tarball(pkg: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{pkg}_{PKGS[pkg]}.tar.gz"
    if not p.exists():
        for url in (f"{CRAN}/{p.name}", f"{CRAN}/Archive/{pkg}/{p.name}"):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges-e109"})
                p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception as err:                       # noqa: BLE001 - try the archive next
                print(f"  {url}: {err}")
        else:
            raise RuntimeError(f"could not download {p.name}")
    return p


def member(pkg: str, name: str) -> bytes:
    with tarfile.open(tarball(pkg)) as tf:
        return tf.extractfile(f"{pkg}/data/{name}").read()


def read_rdata(raw: bytes):
    """Minimal reader for R's XDR serialisation: pairlists, vectors, strings, attributes."""
    if raw[:2] == b"\xfd7":
        buf = lzma.decompress(raw)
    elif raw[:3] == b"BZh":
        buf = bz2.decompress(raw)
    else:
        buf = gzip.decompress(raw)
    assert buf[:7] in (b"RDX2\nX\n", b"RDX3\nX\n"), "not an XDR R data file"
    pos, refs = 7, []

    def i32():
        nonlocal pos
        pos += 4
        return struct.unpack(">i", buf[pos - 4:pos])[0]

    def item():
        nonlocal pos
        flags = i32()
        kind, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
        if kind == 254:
            return None
        if kind == 255:
            return refs[(flags >> 8) - 1]
        if kind == 1:
            refs.append(item())
            return refs[-1]
        if kind == 9:
            n = i32()
            pos += max(n, 0)
            return None if n == -1 else buf[pos - n:pos].decode()
        if kind == 2:
            out = {}
            while kind == 2:
                attr = item() if has_attr else None
                tag = item() if has_tag else None
                out[tag] = item()
                del attr
                flags = i32()
                kind, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
            return out
        n = i32()
        if kind in (10, 13):
            value, pos = np.frombuffer(buf, ">i4", n, pos).astype(int), pos + 4 * n
        elif kind == 14:
            value, pos = np.frombuffer(buf, ">f8", n, pos).astype(float), pos + 8 * n
        elif kind in (16, 19):
            value = [item() for _ in range(n)]
        else:
            raise NotImplementedError(f"R type {kind}")
        attrs = item() if has_attr else None
        if attrs and "levels" in attrs:                    # a factor: codes 1..k -> level labels
            value = np.asarray(attrs["levels"], dtype=object)[np.asarray(value) - 1]
        if kind == 19 and attrs and "names" in attrs:
            value = dict(zip(attrs["names"], value))
        return value

    version = i32()
    pos += 8
    if version == 3:
        pos += 4 + struct.unpack(">i", buf[pos:pos + 4])[0]
    return item()


def main() -> None:
    rows = []
    kos = read_rdata(member("LCMCR", "kosovo_aggregate.RData"))["kosovo_aggregate"]
    lists = list(kos)
    mat = np.column_stack([np.asarray(kos[k]).astype(int) for k in lists])
    hist, cnt = np.unique(["".join(map(str, r)) for r in mat], return_counts=True)
    rows += [("kosovo", "|".join(lists), h, int(c)) for h, c in zip(hist, cnt)]
    for ds, obj in SPARSE.items():
        d = read_rdata(member("SparseMSE", f"{obj}.rda"))[obj]
        names = list(d)
        lists, cname = names[:-1], names[-1]
        mat = np.column_stack([np.asarray(d[k]).astype(int) for k in lists])
        for r, c in zip(mat, np.asarray(d[cname]).astype(int)):
            rows.append((ds, "|".join(lists), "".join(map(str, r)), int(c)))
    df = pd.DataFrame(rows, columns=["dataset", "lists", "history", "count"])
    df = df.sort_values(["dataset", "history"], kind="stable").reset_index(drop=True)
    checks = {"kosovo": (15, 4400), "uk2013": (25, 2744), "netherlands": (24, 8234),
              "new_orleans": (19, 185), "western_us": (13, 345)}
    for ds, (ncell, tot) in checks.items():
        g = df[df.dataset == ds]
        assert (len(g), g["count"].sum()) == (ncell, tot), (ds, len(g), g["count"].sum())
        assert not (g.history.str.count("1") == 0).any()
    uk = df[df.dataset == "uk2013"].set_index("history")["count"]
    assert uk["100000"] == 54 and uk["010000"] == 463
    assert df[(df.dataset == "kosovo") & (df.history == "1000")]["count"].item() == 1131
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(df.groupby("dataset")["count"].agg(["size", "sum"]))


if __name__ == "__main__":
    main()
