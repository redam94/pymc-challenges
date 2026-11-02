"""Build the E87 flight data from the BTS Reporting Carrier On-Time Performance monthly files.

    # one month, from a zip already on disk (what has been run so far):
    .venv/bin/python tools/build_e87_data.py --months 2024-1 --zip-dir DIR --root DIR2 --no-download

    # the whole build, NOT YET RUN: 24 files, 710 MB to download, about 10 minutes on one core
    # (24 s for one month on a loaded machine, times 24), about 1 GB of month chunks
    .venv/bin/python tools/memguard.py --max-gb 6 -- \
        .venv/bin/python tools/build_e87_data.py --full

Source: US DOT Bureau of Transportation Statistics, TranStats, "Reporting Carrier On-Time Performance
(1987-present)". No licence or terms were found for this table (see plans/large/E87.md): do not
commit monthly files or row-level arrays. The zips are read in place and never unpacked to disk.

Stages, each resumable, and each written atomically through pymc_challenges.bigrun.Store:

    month     one chunk per month, checked against the file's own invariants
    merge     integer row arrays, one sort order, and flight x month cells
    summary   counts, sizes and the source notice

The unit is the "flight" of 14 CFR 399.81(c): a carrier's departures in one city-pair market within
30 minutes of the most frequent scheduled departure time count as one flight. That wording does not
say how to repeat the step, how to treat direction, or how to link a flight across months. Here:
repeat on what is left until nothing is left, keep directions apart, link to the previous month's
flight of the same carrier and market with the nearest modal time within 30 minutes. A late outcome
is ArrDelay > 30 or Cancelled. Diversions are not mentioned in the rule, have no ArrDelay, and stay
at 0, with `diverted` kept so that the choice can be tested. Times are not wrapped at midnight.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before NumPy: see bigrun's docstring

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pymc_challenges.bigrun import Store, atomic_write, sha256_file

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
COLUMNS = ["FlightDate", "Reporting_Airline", "Tail_Number", "OriginCityMarketID",
           "DestCityMarketID", "Origin", "Dest", "CRSDepTime", "ArrDelay", "ArrDel15", "Cancelled",
           "Diverted"]
NOTICE = ("Derived from US DOT Bureau of Transportation Statistics, Reporting Carrier On-Time "
          "Performance (1987-present). Modified: aggregated and re-coded, see "
          "tools/build_e87_data.py. No endorsement by BTS or DOT is implied.")
ALL_MONTHS = [(year, month) for year in (2024, 2025) for month in range(1, 13)]


def fetch(year, month, zip_dir, download):
    """The month's zip, downloaded atomically if missing; a partial download never stays on disk."""
    path = Path(zip_dir) / f"ontime_{year}_{month}.zip"
    if path.exists() and zipfile.is_zipfile(path):
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing and downloads are off")

    def write(out):
        with urllib.request.urlopen(URL.format(year=year, month=month), timeout=120) as response:
            size = int(response.headers["Content-Length"])
            got = 0
            while block := response.read(1 << 20):
                out.write(block)
                got += len(block)
        if got != size:
            raise OSError(f"{year}-{month}: {got} of {size} bytes")

    atomic_write(path, write)
    return path


def read_month(path, year, month):
    """One month's columns, with the checks that have held on 2024-1."""
    with zipfile.ZipFile(path) as archive:
        member = next(i.filename for i in archive.infolist() if i.filename.endswith(".csv"))
        with archive.open(member) as f:
            header = f.readline().decode("utf8").replace('"', "").strip().split(",")
        if missing := sorted(set(COLUMNS) - set(header)):
            raise ValueError(f"{path}: columns gone: {missing}")
        with archive.open(member) as f:
            df = pd.read_csv(f, usecols=COLUMNS, low_memory=False,
                             dtype={"Tail_Number": "str", "Reporting_Airline": "str"})
    dates = pd.to_datetime(df["FlightDate"])
    if not ((dates.dt.year == year) & (dates.dt.month == month)).all():
        raise ValueError(f"{path}: dates outside {year}-{month}")
    no_arrival = int(df["ArrDel15"].isna().sum())
    disrupted = int((df["Cancelled"] + df["Diverted"]).sum())
    if no_arrival != disrupted:
        print(f"  {year}-{month}: ArrDel15 is missing on {no_arrival} rows, "
              f"{disrupted} cancelled or diverted", flush=True)
    return df


def greedy_flights(minutes):
    """Split one carrier-market's scheduled times into 399.81(c)(3) flights; returns ids, modes."""
    ids = np.full(len(minutes), -1, np.int32)
    modes = []
    while (left := ids < 0).any():
        times, counts = np.unique(minutes[left], return_counts=True)
        mode = times[np.argmax(counts)]  # ties: the earliest time
        ids[left & (np.abs(minutes - mode) <= 30)] = len(modes)
        modes.append(mode)
    return ids, np.array(modes, np.int16)


def month_chunk(df, year, month):
    """Everything later stages need from one month; numeric or fixed-width, never objects."""
    dep = df["CRSDepTime"].to_numpy(int)
    minutes = (dep // 100) * 60 + dep % 100
    groups = df.groupby(["Reporting_Airline", "OriginCityMarketID", "DestCityMarketID"],
                        sort=False).indices
    local = np.empty(len(df), np.int32)
    modes = np.empty(len(df), np.int16)
    for rows in groups.values():
        ids, group_modes = greedy_flights(minutes[rows])
        local[rows] = ids
        modes[rows] = group_modes[ids]
    dates = pd.to_datetime(df["FlightDate"])
    late = (df["ArrDelay"].to_numpy(float) > 30) | (df["Cancelled"].to_numpy(float) == 1)
    return {"carrier": df["Reporting_Airline"].to_numpy(dtype="U2"),
            "tail": df["Tail_Number"].fillna("").to_numpy(dtype="U6"),
            "origin": df["Origin"].to_numpy(dtype="U3"), "dest": df["Dest"].to_numpy(dtype="U3"),
            "origin_market": df["OriginCityMarketID"].to_numpy(np.int32),
            "dest_market": df["DestCityMarketID"].to_numpy(np.int32),
            "day": dates.dt.day.to_numpy(np.int8), "hour": (dep // 100).clip(0, 23).astype(np.int8),
            "local": local, "mode": modes, "y": late.astype(np.int8),
            "cancelled": df["Cancelled"].to_numpy(np.int8),
            "diverted": df["Diverted"].to_numpy(np.int8),
            "month": np.full(len(df), (year - 2024) * 12 + month - 1, np.int16)}


def link_flights(chunks):
    """One id per flight across months: nearest modal time of the same carrier and market."""
    known = {}  # (carrier, origin market, dest market) -> [modal minutes, flight id, last month]
    n_flights = 0
    out = []
    for chunk in chunks:
        frame = pd.DataFrame({"carrier": chunk["carrier"], "o": chunk["origin_market"],
                              "d": chunk["dest_market"], "local": chunk["local"],
                              "mode": chunk["mode"]})
        table = frame.drop_duplicates()
        ids = {}
        for carrier, o, d, local, mode in table.itertuples(index=False):
            entries = known.setdefault((carrier, o, d), [])
            near = [e for e in entries if abs(e[0] - mode) <= 30 and e[2] != chunk["month"][0]]
            if near:
                entry = min(near, key=lambda e: abs(e[0] - mode))
                entry[0], entry[2] = mode, chunk["month"][0]
            else:
                entry = [mode, n_flights, chunk["month"][0]]
                entries.append(entry)
                n_flights += 1
            ids[(carrier, o, d, local)] = entry[1]
        out.append(np.fromiter((ids[k] for k in zip(frame["carrier"], frame["o"], frame["d"],
                                                    frame["local"])), np.int32, len(frame)))
    return out, n_flights


def merge(store, months):
    chunks = [store.load_chunk("month", i) for i in range(len(months))]
    flights, n_flights = link_flights(chunks)
    levels = {key: np.unique(np.concatenate([c[key] for c in chunks]))
              for key in ("carrier", "tail", "origin", "dest")}
    airports = np.union1d(levels["origin"], levels["dest"])
    columns = {"flight": np.concatenate(flights)}
    for key, labels, dtype in (("carrier", levels["carrier"], np.int8),
                               ("tail", levels["tail"], np.int16),
                               ("origin", airports, np.int16), ("dest", airports, np.int16)):
        codes = [np.searchsorted(labels, c[key]) for c in chunks]
        columns[key] = np.concatenate(codes).astype(dtype)
    for key in ("month", "day", "hour", "y", "cancelled", "diverted"):
        columns[key] = np.concatenate([c[key] for c in chunks])

    for name, values in columns.items():
        store.save_array(f"rows_{name}", values)
    order = np.argsort(columns["flight"], kind="stable").astype(np.int32)
    store.save_array("rows_order_by_flight", order)

    frame = pd.DataFrame({k: columns[k] for k in ("flight", "month", "y", "cancelled")})
    cells = frame.groupby(["flight", "month"], sort=True).agg(n=("y", "size"), late=("y", "sum"))
    cells = cells.reset_index()
    store.save_array("cells_flight", cells["flight"].to_numpy(np.int32))
    store.save_array("cells_month", cells["month"].to_numpy(np.int16))
    store.save_array("cells_n", cells["n"].to_numpy(np.int16))
    store.save_array("cells_late", cells["late"].to_numpy(np.int16))
    store.save_json("levels", {"carriers": levels["carrier"], "airports": airports,
                               "n_tails": len(levels["tail"]), "n_flights": n_flights,
                               "months": [f"{y}-{m:02d}" for y, m in months]})
    return columns, cells, n_flights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--months", nargs="*", help="YYYY-M values; omit with --full for all 24")
    parser.add_argument("--full", action="store_true", help="really fetch all 24 months")
    parser.add_argument("--zip-dir", default=".scratch/e87_zips", help="where the zips live")
    parser.add_argument("--root", default=None, help="store root, default results/")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    if args.months:
        months = [tuple(int(x) for x in m.split("-")) for m in args.months]
    elif args.full:
        months = ALL_MONTHS
    else:
        parser.error("name --months, or pass --full for all 24 months")

    store = Store("E87", root=args.root)
    stored = store.load_json("months") if (store.path / "months.json").exists() else None
    if stored is not None and stored != [list(m) for m in months]:
        # Chunks are numbered by position in this list, so another list would mix months up.
        raise SystemExit(f"{store.path} was built for other months: use another --root")
    store.save_json("months", months)
    started = time.time()
    for i in store.pending("month", len(months)):
        year, month = months[i]
        path = fetch(year, month, args.zip_dir, download=not args.no_download)
        df = read_month(path, year, month)
        store.save_chunk("month", i, month_chunk(df, year, month))
        print(f"{year}-{month}: {len(df):,} rows, {sha256_file(path)[:12]}, "
              f"{time.time() - started:.1f} s", flush=True)

    columns, cells, n_flights = merge(store, months)
    n = len(columns["y"])
    per_cell = cells["n"].to_numpy()
    eligible = per_cell >= 10
    flagged = eligible & (cells["late"].to_numpy() / per_cell > 0.5)
    summary = {"rows": n, "flights": n_flights, "flight_months": len(cells),
               "rows_per_cell": n / len(cells), "eligible_cells": int(eligible.sum()),
               "eligible_share_of_rows": float(per_cell[eligible].sum() / n),
               "flagged_cells": int(flagged.sum()), "late_share": float(columns["y"].mean()),
               "diverted_not_late": int(((columns["diverted"] == 1) & (columns["y"] == 0)).sum()),
               "notice": NOTICE, "seconds": time.time() - started}
    store.save_json("summary", summary)
    store.mark_done("data", months=len(months), rows=n)
    print(summary)
    print("stored:", {p.name: p.stat().st_size for p in sorted(store.path.glob("cells_*.npy"))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
