"""Build the E120 dataset: every call to the "Anonymous Bank" call centre in 1999.

    python tools/build_e120_calls.py   # download (cached), derive, check, write data/e120_calls.csv.gz
    python tools/build_e120_calls.py --allow-new-source   # accept an archive whose md5 has changed

Source: the "Anonymous Bank" call-centre data of the Technion SEE Center (Service Enterprise
Engineering), documented by Ilan Guedj and Avi Mandelbaum (9 February 2000) and analysed in Brown,
Gans, Mandelbaum, Sakov, Shen, Zeltyn and Zhao (2005), "Statistical analysis of a telephone call
center: a queueing-science perspective", JASA 100:36-50.
https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.rar
Terms, from the documentation: "The data is free for use. If used, please notify Avi Mandelbaum, at
avim@ie.technion.ac.il and do acknowledge the data source in any of your work."

The archive is RAR v4 with twelve tab-separated monthly files; it is unpacked with ``bsdtar``
(libarchive, part of macOS and of ``libarchive-tools`` on Linux). Archive and text files are cached
in ~/.cache/pymc-challenges/e120 (not committed).

One row per call, in order of arrival. What the builder does, and nothing else:
  - drops the 3,599 PHANTOM records, which the documentation says to ignore;
  - drops ``customer_id`` and the VRU line, keeping ``identified`` (the caller gave an id);
  - replaces the agents' first names by codes A01, A02, ... in order of calls served;
  - turns clock times into seconds after midnight.
Cleaning beyond that (opening hours, which calls count as arrivals, the very short "services") is
part of the notebook.

Columns:
  date          YYYY-MM-DD (the day the call entered the VRU)
  vru_entry_s   second of the day the call entered the voice response unit
  vru_time      seconds in the voice response unit (a few are negative or huge: clock errors)
  q_start_s     second of the day the call joined the agents' queue; empty if it never queued
  q_time        seconds in the queue (0 if it never queued)
  outcome       AGENT (served) or HANG (the caller hung up)
  ser_start_s   second of the day service began; empty if the call was not served
  ser_time      seconds of service (0 if not served)
  type          PS regular, PE regular in English, IN internet support, NE stock exchange,
                NW prospective customer, TT returned call put on hold (and 5 rows coded AA)
  priority      0 unidentified or regular, 1 regular, 2 priority (moved up the queue by 90 s)
  identified    1 if the caller gave a customer id
  agent         code of the agent who served the call; empty if none (NO_SERVER in the source)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e120_calls.csv.gz"
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e120"
URL = "https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.rar"
MD5 = "ccb2773446e54a9478baad34b4ec6fb8"   # of the archive as downloaded 2026-11-14 (10,625,021 bytes)
N_RAW, N_PHANTOM = 444_448, 3_599


def fetch(allow_new: bool) -> Path:
    rar = CACHE / "AnonymousBank.rar"
    if not rar.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            rar.write_bytes(resp.read())
    md5 = hashlib.md5(rar.read_bytes()).hexdigest()
    if md5 != MD5 and not allow_new:
        raise SystemExit(f"archive md5 {md5} differs from the recorded {MD5}; pass --allow-new-source")
    raw = CACHE / "raw"
    if len(list(raw.glob("*1999.txt"))) != 12:
        if shutil.which("bsdtar") is None:
            raise SystemExit("bsdtar (libarchive) is needed to unpack the RAR archive")
        raw.mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(raw)], check=True)
    return raw


def seconds(clock: pd.Series) -> np.ndarray:
    hms = clock.str.split(":", expand=True).astype(int)
    return (hms[0] * 3600 + hms[1] * 60 + hms[2]).to_numpy()


def build(raw: Path) -> pd.DataFrame:
    files = sorted(raw.glob("*1999.txt"))
    assert len(files) == 12, files
    # whitespace, not tabs: twelve records have a space where a tab should be
    d = pd.concat([pd.read_csv(f, sep=r"\s+", dtype=str) for f in files], ignore_index=True)
    assert len(d) == N_RAW and d.shape[1] == 17 and d.notna().all().all(), d.shape
    assert (d["outcome"] == "PHANTOM").sum() == N_PHANTOM
    d = d[d["outcome"] != "PHANTOM"].copy()

    out = pd.DataFrame({"date": pd.to_datetime(d["date"], format="%y%m%d").dt.strftime("%Y-%m-%d")})
    out["vru_entry_s"] = seconds(d["vru_entry"])
    out["vru_time"] = d["vru_time"].astype(int).to_numpy()
    q_time = d["q_time"].astype(int).to_numpy()
    q_start = seconds(d["q_start"]).astype(float)
    q_start[(q_start == 0) & (q_time == 0)] = np.nan       # 0:00:00 marks "never queued"
    out["q_start_s"] = q_start
    out["q_time"] = q_time
    out["outcome"] = d["outcome"].to_numpy()
    ser_time = d["ser_time"].astype(int).to_numpy()
    ser_start = seconds(d["ser_start"]).astype(float)
    ser_start[(ser_start == 0) & (ser_time == 0)] = np.nan  # 0:00:00 marks "not served"
    out["ser_start_s"] = ser_start
    out["ser_time"] = ser_time
    out["type"] = d["type"].to_numpy()
    out["priority"] = d["priority"].astype(int).to_numpy()
    out["identified"] = (d["customer_id"] != "0").astype(int).to_numpy()

    served_by = d["server"].where(d["server"] != "NO_SERVER")
    order = served_by.value_counts().index                  # most calls first; ties by first seen
    code = {name: f"A{i + 1:02d}" for i, name in enumerate(order)}
    out["agent"] = served_by.map(code).to_numpy()

    out = out.sort_values(["date", "vru_entry_s"], kind="stable").reset_index(drop=True)
    for c in ("q_start_s", "ser_start_s"):
        out[c] = out[c].astype("Int64")
    return out


def check(out: pd.DataFrame) -> None:
    assert len(out) == N_RAW - N_PHANTOM
    assert set(out["outcome"]) == {"AGENT", "HANG"}
    assert out["date"].min() == "1999-01-01" and out["date"].max() == "1999-12-31"
    assert out["agent"].notna().sum() > 300_000 and out["agent"].nunique() < 60
    assert not {"customer_id", "server"} & set(out.columns)
    assert out["vru_entry_s"].between(0, 86_399).all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-new-source", action="store_true")
    args = ap.parse_args()
    out = build(fetch(args.allow_new_source))
    check(out)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:   # mtime=0: byte-identical rebuilds
        gz.write(out.to_csv(index=False).encode())
    OUT.write_bytes(buf.getvalue())
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out):,} rows, {OUT.stat().st_size / 1e6:.1f} MB, "
          f"md5 {hashlib.md5(buf.getvalue()).hexdigest()}")
    print(out["outcome"].value_counts().to_dict(), "| agents:", out["agent"].nunique(),
          "| queued:", int(out["q_start_s"].notna().sum()))


if __name__ == "__main__":
    main()
