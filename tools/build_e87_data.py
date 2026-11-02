"""Build the E87 flight data from the BTS Reporting Carrier On-Time Performance monthly files.

    # the linking rules on invented months, no data needed:
    .venv/bin/python tools/build_e87_data.py --selftest

    # the whole build: 24 files; everything large goes to ~/.cache/pymc-challenges
    .venv/bin/python tools/memguard.py --max-gb 8 -- \
        .venv/bin/python -I tools/build_e87_data.py --full

    # the two small tables that go in data/, from the arrays the build left in the cache
    .venv/bin/python tools/memguard.py --max-gb 3 -- .venv/bin/python tools/build_e87_data.py --tables

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
flight of the same carrier and market, nearest modal time first, within 30 minutes (see
`link_flights`). A late outcome is ArrDelay > 30 or Cancelled. Diversions are not mentioned in the
rule, have no ArrDelay, and stay at 0, with `diverted` kept so that the choice can be tested. Times are not wrapped at midnight.

Flight numbers play no part in the links, which is why they can test them: `summary` compares them.
Large files stay in the cache. The one file written to results/E87/ is `data_summary.json`, and it
names no carrier, airport or tail.

`--tables` writes `data/e87_flights.npz` and `data/e87_month_rows.npz` for the notebook. They hold no
carrier, airport, tail or flight number and no scheduled time. Carrier and airport codes are shuffled
by a permutation drawn from the operating system once and kept in the cache beside `levels.json`, so
the committed code cannot undo it. Flight and tail codes in the month table are numbered by first
appearance in the file, which keeps the file's real index pattern without keeping any sorted order.
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

from pymc_challenges.bigrun import CACHE_DIR, Store, atomic_write, sha256_file

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
COLUMNS = ["FlightDate", "Reporting_Airline", "Tail_Number", "OriginCityMarketID",
           "DestCityMarketID", "Origin", "Dest", "CRSDepTime", "ArrDelay", "ArrDel15", "Cancelled",
           "Diverted", "Flight_Number_Reporting_Airline"]
NOTICE = ("Derived from US DOT Bureau of Transportation Statistics, Reporting Carrier On-Time "
          "Performance (1987-present). Modified: aggregated and re-coded, see "
          "tools/build_e87_data.py. No endorsement by BTS or DOT is implied.")
ALL_MONTHS = [(year, month) for year in (2024, 2025) for month in range(1, 13)]
ZIP_DIR = CACHE_DIR / "e87_zips"
USER_AGENT = "pymc-challenges data build (polite, single download)"
KEY = ["carrier", "o", "d"]
MAX_GAP = 1  # a flight absent for a whole month gets a new id; the plan's pilot results say why
TARGETS = (15, 18, 21)  # 2025-04, 2025-07, 2025-10 as month indices
FIRST_TARGET = TARGETS[0]  # persistence checks stay before the held-out months


def fetch(year, month, zip_dir, download):
    """The month's zip, downloaded atomically if missing; a partial download never stays on disk."""
    path = Path(zip_dir) / f"ontime_{year}_{month}.zip"
    if path.exists() and zipfile.is_zipfile(path):
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing and downloads are off")

    def write(out):
        request = urllib.request.Request(URL.format(year=year, month=month),
                                         headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            size = int(response.headers["Content-Length"])
            got = 0
            while block := response.read(1 << 20):
                out.write(block)
                got += len(block)
        if got != size:
            raise OSError(f"{year}-{month}: {got} of {size} bytes")

    Path(zip_dir).mkdir(parents=True, exist_ok=True)
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
    df.attrs["n_columns"] = len(header)
    dates = pd.to_datetime(df["FlightDate"])
    if not ((dates.dt.year == year) & (dates.dt.month == month)).all():
        raise ValueError(f"{path}: dates outside {year}-{month}")
    return df


def month_checks(df):
    """The file's own invariants, recorded for every month rather than only printed."""
    no_arrival = int(df["ArrDel15"].isna().sum())
    disrupted = int((df["Cancelled"] + df["Diverted"]).sum())
    tails = df["Tail_Number"]
    dep = df["CRSDepTime"]
    return {"rows": len(df), "n_columns": df.attrs["n_columns"], "arrdel15_missing": no_arrival,
            "cancelled_or_diverted": disrupted,
            "gap_share": abs(no_arrival - disrupted) / max(disrupted, 1),
            "cancelled": int(df["Cancelled"].sum()), "diverted": int(df["Diverted"].sum()),
            "tail_missing": int(tails.isna().sum()),
            "tail_longest": int(tails.dropna().str.len().max()),
            "tail_missing_not_cancelled": int((tails.isna() & (df["Cancelled"] == 0)).sum()),
            "arrdelay_missing_not_disrupted": int((df["ArrDelay"].isna() & (df["Cancelled"] == 0)
                                                   & (df["Diverted"] == 0)).sum()),
            "crsdep_range": [int(dep.min()), int(dep.max())],
            "carriers": int(df["Reporting_Airline"].nunique()),
            "late_share": float(((df["ArrDelay"] > 30) | (df["Cancelled"] == 1)).mean())}


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
            "number": df["Flight_Number_Reporting_Airline"].to_numpy(np.int32),
            "cancelled": df["Cancelled"].to_numpy(np.int8),
            "diverted": df["Diverted"].to_numpy(np.int8),
            "month": np.full(len(df), (year - 2024) * 12 + month - 1, np.int16)}


def month_table(chunk):
    """One row per flight of one month: its key, modal time, counts and modal flight number."""
    frame = pd.DataFrame({"carrier": chunk["carrier"], "o": chunk["origin_market"],
                          "d": chunk["dest_market"], "local": chunk["local"],
                          "mode": chunk["mode"], "y": chunk["y"], "number": chunk["number"]})
    table = frame.groupby(KEY + ["local", "mode"], sort=True).agg(n=("y", "size"), late=("y", "sum"))
    top = frame.groupby(KEY + ["local", "number"]).size().reset_index(name="count")
    top = top.sort_values(["count", "number"], ascending=[False, True]).drop_duplicates(KEY + ["local"])
    table = table.reset_index().merge(top[KEY + ["local", "number"]], on=KEY + ["local"])
    table["month"] = int(chunk["month"][0])
    return table


def link_flights(tables, tol=30, max_gap=None):
    """One id per flight across months, in place: adds `flight` to each month's table.

    Within a carrier and market, this month's flights and the flights known so far are paired
    last month's flights first, then nearest modal time, so a stale flight cannot break a live one
    and the result does not depend on row order. No flight takes two partners. A pair must be within
    `tol` minutes. `max_gap` is the largest step in months (1 means last month's flights only, None
    any earlier month). A flight's time is its latest month's, so it may drift by up to `tol` each
    month: the summary measures how far.
    """
    known = {}  # (carrier, origin market, dest market) -> list of [modal minutes, id, last month]
    n_flights = 0
    stats = {"flights_with_two_candidates": 0, "linked": 0, "linked_after_gap": 0}
    for table in tables:
        month = int(table["month"].iloc[0])
        groups = {}
        for row, key in enumerate(zip(table["carrier"], table["o"], table["d"])):
            groups.setdefault(key, []).append(row)
        modes = table["mode"].to_numpy(int)
        flight = np.empty(len(table), np.int32)
        for key, rows in groups.items():
            entries = known.setdefault(key, [])
            pairs = sorted((month - entry[2] > 1, abs(entry[0] - modes[row]), int(modes[row]),
                            e, row)
                           for e, entry in enumerate(entries) for row in rows
                           if abs(entry[0] - modes[row]) <= tol
                           and (max_gap is None or month - entry[2] <= max_gap))
            candidates = {}
            for *_, row in pairs:
                candidates[row] = candidates.get(row, 0) + 1
            stats["flights_with_two_candidates"] += sum(c > 1 for c in candidates.values())
            taken, placed = set(), {}
            for *_, e, row in pairs:
                if e not in taken and row not in placed:
                    taken.add(e)
                    placed[row] = e
            for row in rows:
                if row in placed:
                    entry = entries[placed[row]]
                    stats["linked"] += 1
                    stats["linked_after_gap"] += month - entry[2] > 1
                    entry[0], entry[2] = int(modes[row]), month
                else:
                    entry = [int(modes[row]), n_flights, month]
                    entries.append(entry)
                    n_flights += 1
                flight[row] = entry[1]
        table["flight"] = flight
    return n_flights, {k: int(v) for k, v in stats.items()}


def consecutive(cells):
    """Pairs of one flight's cells in months t and t+1."""
    next_month = cells.assign(month=cells["month"] - 1)
    pairs = cells.merge(next_month, on=["flight", "month"], suffixes=("", "_next"))
    return pairs.assign(month_next=pairs["month"] + 1)


def linking_report(cells):
    """What the links do: continuity, persistence before the first target, and an independent test:
    the modal flight number takes no part in linking, so agreement across a link is evidence."""
    cells = cells.assign(eligible=cells["n"] >= 10)
    cells["flagged"] = cells["eligible"] & (cells["late"] / cells["n"] > 0.5)
    pairs = consecutive(cells)
    later = cells[(cells["month"] > 0) & cells["eligible"]]
    with_history = pairs[pairs["eligible_next"] & pairs["eligible"]]
    training = with_history[with_history["month_next"] < FIRST_TARGET]
    last = training[training["flagged"]]
    streak, best = {}, 0
    for flight, month in cells.loc[cells["flagged"], ["flight", "month"]].sort_values(
            ["flight", "month"]).itertuples(index=False):
        run, seen = streak.get(flight, (0, -9))
        run = run + 1 if month == seen + 1 else 1
        streak[flight] = (run, month)
        best = max(best, run)
    report = {"flights": int(cells["flight"].nunique()), "flight_months": len(cells),
              "months_per_flight_median": float(cells.groupby("flight").size().median()),
              "eligible_cells_with_eligible_prior_month": float(len(with_history) / len(later)),
              "training_pairs": len(training),
              "flagged_last_month": len(last), "flagged_next_month": int(training["flagged_next"].sum()),
              "flagged_both": int(last["flagged_next"].sum()),
              "persistence_precision": float(last["flagged_next"].mean()),
              "persistence_recall": float(training.loc[training["flagged_next"], "flagged"].mean()),
              "base_rate_next_month": float(training["flagged_next"].mean()),
              "longest_flagged_streak_months": best}
    report["training_windows"] = {}
    for target in TARGETS:
        window = cells[(cells["month"] >= target - 12) & (cells["month"] < target) & cells["eligible"]]
        counts = window.groupby("flight").size()
        targets = cells[(cells["month"] == target) & cells["eligible"]]
        report["training_windows"][str(target)] = {
            "eligible_cells": len(window), "flights": len(counts),
            "flights_and_cells_by_least_cells": {
                k: [int((counts >= k).sum()), int(counts[counts >= k].sum())] for k in (2, 3, 6, 12)},
            "target_flights": len(targets), "target_flagged": int(targets["flagged"].sum()),
            "target_flights_with_history": int(targets["flight"].isin(counts.index).sum())}
    drift = cells.sort_values(["flight", "month"]).groupby("flight")["mode"]
    drift = (drift.max() - drift.min())[cells.groupby("flight").size() > 1]
    report["time_range_within_flight_min"] = {
        q: float(drift.quantile(float(q))) for q in ("0.5", "0.9", "0.99", "1.0")}
    keyed = cells[cells["eligible"]]
    same = consecutive(keyed)
    report["linked_pairs_same_flight_number"] = float((same["number"] == same["number_next"]).mean())
    report["linked_pairs_tested"] = len(same)
    ordered = cells.sort_values(["flight", "month"])
    steps = ordered.groupby("flight").shift(1)
    ordered = ordered[(ordered["month"] - steps["month"] > 1) & ordered["eligible"] & (steps["n"] >= 10)]
    report["after_gap_pairs_same_flight_number"] = (
        float((ordered["number"] == steps.loc[ordered.index, "number"]).mean())
        if len(ordered) else None)
    report["after_gap_pairs_tested"] = len(ordered)
    rivals = keyed.merge(keyed.assign(month=keyed["month"] - 1), on=KEY + ["month"],
                         suffixes=("", "_next"))
    rivals = rivals[rivals["flight"] != rivals["flight_next"]]
    report["unlinked_rivals_same_flight_number"] = float(
        (rivals["number"] == rivals["number_next"]).mean())
    report["unlinked_rival_pairs_tested"] = len(rivals)
    starts = keyed[keyed["flight"].map(cells.groupby("flight")["month"].min()) == keyed["month"]]
    starts = starts[starts["month"] > 0]
    before = keyed.assign(month=keyed["month"] + 1)
    missed = starts.merge(before, on=KEY + ["month", "number"], suffixes=("", "_before"))
    report["new_eligible_flights"] = len(starts)
    report["new_flights_with_same_number_a_month_before"] = int(missed["flight"].nunique())
    return report


def merge(store, months):
    chunks = [store.load_chunk("month", i) for i in range(len(months))]
    tables = [month_table(chunk) for chunk in chunks]
    n_flights, link_stats = link_flights(tables, max_gap=MAX_GAP)
    flights = []
    for chunk, table in zip(chunks, tables):
        frame = pd.DataFrame({"carrier": chunk["carrier"], "o": chunk["origin_market"],
                              "d": chunk["dest_market"], "local": chunk["local"]})
        joined = frame.merge(table[KEY + ["local", "flight"]], on=KEY + ["local"], how="left",
                             validate="many_to_one")
        flights.append(joined["flight"].to_numpy(np.int32))
    levels = {key: np.unique(np.concatenate([np.unique(c[key]) for c in chunks]))
              for key in ("carrier", "tail", "origin", "dest")}
    airports = np.union1d(levels["origin"], levels["dest"])
    columns = {"flight": np.concatenate(flights)}
    for key, labels, dtype in (("carrier", levels["carrier"], np.int8),
                               ("tail", levels["tail"], np.int16),
                               ("origin", airports, np.int16), ("dest", airports, np.int16)):
        codes = [np.searchsorted(labels, c[key]) for c in chunks]
        columns[key] = np.concatenate(codes).astype(dtype)
    for key in ("month", "day", "hour", "y", "cancelled", "diverted", "mode", "number"):
        columns[key] = np.concatenate([c[key] for c in chunks])

    for name, values in columns.items():
        store.save_array(f"rows_{name}", values)
    order = np.argsort(columns["flight"], kind="stable").astype(np.int32)
    store.save_array("rows_order_by_flight", order)

    cells = pd.concat(tables, ignore_index=True)[KEY + ["flight", "month", "mode", "n", "late",
                                                         "number"]]
    cells = cells.sort_values(["flight", "month"], ignore_index=True)
    store.save_array("cells_flight", cells["flight"].to_numpy(np.int32))
    store.save_array("cells_month", cells["month"].to_numpy(np.int16))
    store.save_array("cells_n", cells["n"].to_numpy(np.int16))
    store.save_array("cells_late", cells["late"].to_numpy(np.int16))

    # What R2 needs per flight: most-used airports, as codes. Carrier names stay in levels.json.
    rows = pd.DataFrame({k: columns[k] for k in ("flight", "carrier", "origin", "dest")})
    modal = lambda column: rows.groupby(["flight", column]).size().reset_index(name="c").sort_values(
        ["flight", "c", column], ascending=[True, False, True]).drop_duplicates("flight")
    store.save_array("flights_carrier", modal("carrier")["carrier"].to_numpy(np.int8))
    store.save_array("flights_origin", modal("origin")["origin"].to_numpy(np.int16))
    store.save_array("flights_dest", modal("dest")["dest"].to_numpy(np.int16))
    store.save_json("levels", {"carriers": levels["carrier"], "airports": airports,
                               "n_tails": len(levels["tail"]), "n_flights": n_flights,
                               "months": [f"{y}-{m:02d}" for y, m in months]})
    return columns, cells, tables, n_flights, link_stats


def sensitivity(tables):
    """The same counts under other linking rules; carrier-free."""
    out = {}
    for tol, gap in ((15, 1), (30, 1), (45, 1), (30, 2), (30, 3), (30, None)):
        tried = [t.drop(columns="flight") for t in tables]
        n_flights, stats = link_flights(tried, tol=tol, max_gap=gap)
        cells = pd.concat(tried, ignore_index=True)
        out[f"tol {tol}, gap {gap}"] = {**linking_report(cells), **stats, "flights": n_flights}
    return out


def selftest():
    """The linking rules on small invented months."""
    def table(month, *rows):
        return pd.DataFrame([dict(carrier="A", o=1, d=2, local=i, mode=m, month=month)
                             for i, m in enumerate(rows)])

    def link(*tables, **kw):
        n, _ = link_flights(list(tables), **kw)
        return n, [t["flight"].tolist() for t in tables]

    assert link(table(0, 600), table(1, 620)) == (1, [[0], [0]])  # within 30
    assert link(table(0, 600), table(1, 630))[0] == 1  # the boundary counts
    assert link(table(0, 600), table(1, 631))[0] == 2  # past it
    # Nearest first, whatever the row order: 615 belongs with 620, so 590 starts a new flight.
    assert link(table(0, 620), table(1, 590, 615)) == (2, [[0], [1, 0]])
    assert link(table(0, 620), table(1, 615, 590)) == (2, [[0], [0, 1]])
    # Two flights of one month never share an id, even if both are near one old flight.
    assert len(set(link(table(0, 600), table(1, 590, 625))[1][1])) == 2
    # A gap: joined with max_gap=None, or 2, and not with 1.
    assert link(table(0, 600), table(2, 600))[0] == 1
    assert link(table(0, 600), table(2, 600), max_gap=2)[0] == 1
    assert link(table(0, 600), table(2, 600), max_gap=1)[0] == 2
    # Drift follows the latest month, so 600, 630, 660 is one flight.
    assert link(table(0, 600), table(1, 630), table(2, 660))[0] == 1
    # A live flight beats a nearer stale one: 585 joins last month's 560, not the older 600.
    assert link(table(0, 560, 600), table(1, 560), table(2, 585))[1][2] == [0]
    # Another market is another flight.
    other = table(1, 600).assign(d=3)
    assert link(table(0, 600), other)[0] == 2
    print("link self-test passed")


def write_tables(store, out_dir):
    """The two committed tables. See the module docstring for what they leave out."""
    load = lambda name: np.load(store.path / f"{name}.npy", mmap_mode="r")
    carrier, origin, dest = (np.asarray(load(f"flights_{k}")) for k in ("carrier", "origin", "dest"))
    if not (store.path / "anon_carrier.npy").exists():
        fresh = np.random.default_rng()  # from the operating system: no seed in this file to undo it
        store.save_array("anon_carrier", fresh.permutation(int(carrier.max()) + 1).astype(np.int8))
        store.save_array("anon_airport", fresh.permutation(
            int(max(origin.max(), dest.max())) + 1).astype(np.int16))
        store.save_array("anon_flight_seed", fresh.integers(2**62, size=1))
    anon_carrier, anon_airport = store.load_array("anon_carrier"), store.load_array("anon_airport")

    cell_flight, cell_month = np.asarray(load("cells_flight")), np.asarray(load("cells_month"))
    cell_n, cell_late = np.asarray(load("cells_n")), np.asarray(load("cells_late"))
    keep = cell_n >= 10  # eligible cells: the only ones the rule or the models use
    kept = np.unique(cell_flight[keep])
    shuffle = np.random.default_rng(int(store.load_array("anon_flight_seed")[0])).permutation(kept.size)
    code = shuffle[np.searchsorted(kept, cell_flight[keep])]
    order = np.lexsort((cell_month[keep], code))
    by_code = np.empty(kept.size, np.int64)
    by_code[shuffle] = kept
    tables = {"cell_flight": code[order].astype(np.int32),
              "cell_month": cell_month[keep][order].astype(np.int8),
              "cell_n": cell_n[keep][order].astype(np.int16),
              "cell_late": cell_late[keep][order].astype(np.int16),
              "flight_carrier": anon_carrier[carrier[by_code]],
              "flight_origin": anon_airport[origin[by_code]],
              "flight_dest": anon_airport[dest[by_code]],
              "notice": np.array(NOTICE)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    atomic_write(Path(out_dir) / "e87_flights.npz", lambda f: np.savez_compressed(f, **tables))

    rows = np.flatnonzero(np.asarray(load("rows_month")) == 0)  # 2024-01, in file order
    first_seen = lambda values: np.unique(values, return_index=True)[1].argsort().argsort()[
        np.unique(values, return_inverse=True)[1]]
    month = {"y": np.asarray(load("rows_y"))[rows].astype(np.int8),
             "day": np.asarray(load("rows_day"))[rows].astype(np.int8),
             "carrier": anon_carrier[np.asarray(load("rows_carrier"))[rows]],
             "origin": anon_airport[np.asarray(load("rows_origin"))[rows]],
             "dest": anon_airport[np.asarray(load("rows_dest"))[rows]],
             "flight": first_seen(np.asarray(load("rows_flight"))[rows]).astype(np.uint16),
             "tail": first_seen(np.asarray(load("rows_tail"))[rows]).astype(np.uint16),
             "notice": np.array(NOTICE)}
    atomic_write(Path(out_dir) / "e87_month_rows.npz", lambda f: np.savez_compressed(f, **month))
    print({p.name: p.stat().st_size for p in sorted(Path(out_dir).glob("e87_*.npz"))})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--months", nargs="*", help="YYYY-M values; omit with --full for all 24")
    parser.add_argument("--full", action="store_true", help="really fetch all 24 months")
    parser.add_argument("--zip-dir", default=str(ZIP_DIR), help="where the zips live")
    parser.add_argument("--root", default=str(CACHE_DIR), help="store root for all large files")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="test the linking and stop")
    parser.add_argument("--tables", action="store_true", help="write the tables in data/ and stop")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.tables:
        return write_tables(Store("E87", root=args.root), Path(__file__).resolve().parents[1] / "data")
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
        began = time.time()
        path = fetch(year, month, args.zip_dir, download=not args.no_download)
        fetched = time.time()
        df = read_month(path, year, month)
        store.save_json(f"check_{i:02d}", {
            **month_checks(df), "month": f"{year}-{month:02d}", "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "download_s": fetched - began})
        store.save_chunk("month", i, month_chunk(df, year, month))
        store.save_json(f"check_{i:02d}_time", {"parse_s": time.time() - fetched})
        print(f"{year}-{month}: {len(df):,} rows, {fetched - began:.1f} s fetch, "
              f"{time.time() - fetched:.1f} s parse", flush=True)

    merged = time.time()
    columns, cells, tables, n_flights, link_stats = merge(store, months)
    merged = time.time() - merged
    n = len(columns["y"])
    per_cell = cells["n"].to_numpy()
    eligible = per_cell >= 10
    flagged = eligible & (cells["late"].to_numpy() / per_cell > 0.5)
    checks = [store.load_json(f"check_{i:02d}") for i in range(len(months))]
    times = [store.load_json(f"check_{i:02d}_time") for i in range(len(months))]
    by_month = cells.assign(eligible=eligible, flagged=flagged).groupby("month").agg(
        cells=("n", "size"), eligible=("eligible", "sum"), flagged=("flagged", "sum"),
        rows=("n", "sum"), late=("late", "sum"))
    summary = {"rows": n, "flights": n_flights, "flight_months": len(cells),
               "rows_per_cell": n / len(cells), "eligible_cells": int(eligible.sum()),
               "eligible_share_of_rows": float(per_cell[eligible].sum() / n),
               "flagged_cells": int(flagged.sum()), "late_share": float(columns["y"].mean()),
               "diverted_not_late": int(((columns["diverted"] == 1) & (columns["y"] == 0)).sum()),
               "carriers": len(np.unique(columns["carrier"])),
               "airports": len(np.union1d(columns["origin"], columns["dest"])),
               "tails_with_missing_code": len(np.unique(columns["tail"])),
               "link_rule": {"tol_min": 30, "max_gap_months": MAX_GAP, **link_stats},
               "by_month": by_month.reset_index().to_dict("list"),
               "months": [{k: v for k, v in c.items()} | t for c, t in zip(checks, times)],
               "linking": linking_report(cells),
               "sensitivity": sensitivity(tables), "merge_s": merged,
               "seconds_this_run": time.time() - started, "notice": NOTICE,
               "labels": "all measured on the files, except where a key says otherwise"}
    day = columns["month"].astype(np.int64) * 40 + columns["day"]
    summary["origin_day_cells"] = len(np.unique(columns["origin"].astype(np.int64) * 4000 + day))
    summary["dest_day_cells"] = len(np.unique(columns["dest"].astype(np.int64) * 4000 + day))
    # No carrier, airport or tail appears in this file: it is the one that goes in git.
    store.save_json("summary", summary)
    Store("E87").save_json("data_summary", summary)
    store.mark_done("data", months=len(months), rows=n)
    print({k: v for k, v in summary.items() if k not in ("months", "by_month", "sensitivity")})
    print("stored:", {p.name: p.stat().st_size for p in sorted(store.path.glob("cells_*.npy"))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
