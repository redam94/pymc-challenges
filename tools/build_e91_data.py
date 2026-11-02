"""Build the E91 datasets from Low Carbon London SmartMeter Energy Consumption Data.

SKELETON. Tested on three sample partitions only; it has never been run on the full data. Plan and
reasons: plans/large/E91.md.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_e91_data.py calendar --tariffs Tariffs.xlsx
    ... parse --members DIR_OR_ZIP      # one checkpoint per member, safe to kill and re-run
    ... assemble                        # homes x half-hours float32 matrix
    ... small --clock CONVENTION        # the cached laptop subset; needs the pilot's clock decision

A full run also needs `fetch` (one resumable download, 796 MB) and
PYMC_CHALLENGES_CONFIRM_FULL=yes. `parse` peaked at 0.41 GB over three partitions; `assemble` holds
the whole matrix (about 1 GB, extrapolated). Run both through tools/memguard.py.

Source: UK Power Networks, Low Carbon London, on the London Datastore, CC BY 4.0:
https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
Do NOT mix in UK Data Service study 7857 (the same trial with surveys): its terms forbid sharing,
and its household keys were randomly regenerated, so the two releases cannot be linked anyway.

Real format, read from sample partitions 0, 84 and 167:

* 168 members `Small LCL Data/LCL-June2015v2_<k>.csv`, CRLF, sorted by group then LCLid (ToU homes
  fill roughly the last 40 members), so a home may straddle two members.
* Columns `LCLid,stdorToU,DateTime,KWH/hh (per half hour) ` - the last header has a trailing space
  and its values are space-padded. There is no Acorn, location or survey column.
* Nearly every home has one row with kWh `Null` at an off-grid stamp (102 of 104 sampled); no other
  rows were Null or off-grid. Exact duplicate rows appear at 00:00 on 27 dates of the sample.

Tariffs.xlsx: one sheet, 17,520 rows of (Excel serial date, Normal|Low|High) for 2013 on a naive
wall-clock grid: the same event hours in winter and summer, 01:00 present on the day the clocks go
forward. It reads with the standard library, so openpyxl is not needed.

The clock trap: tariffs are wall-clock. The meter stamps do not look wall-clock: 01:00 exists on 31
March 2013, and the morning rise moves against the stamps when BST starts. The UK Data Service
release of this feed documents GMT, stamped at the END of each half-hour, but that is a different
release. Until the pilot settles it on the full ToU group, `small` refuses to run without an
explicit `--clock`, because a wrong join silently smears every summer event by an hour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pymc_challenges.bigrun import PreflightError, Store, atomic_write, sha256_file  # noqa: E402

ZIP_URL = ("https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/"
           "Partitioned%20LCL%20Data.zip")
TARIFF_URL = ("https://data.london.gov.uk/download/vqm0d/14855047-44c2-4856-8a48-e5649200e6ce/"
              "Tariffs.xlsx")
ZIP_BYTES = 795_722_689
N_MEMBERS = 168
RAW_DIR = Path.home() / ".cache" / "pymc_challenges" / "lcl"
USER_AGENT = "pymc-challenges data build (polite, single download)"

GRID_START = pd.Timestamp("2011-11-01 00:00")  # before the first reading; the page says Nov 2011
GRID_END = pd.Timestamp("2014-03-01 00:00")  # after the last reading seen, 2014-02-28
N_SLOTS = int((GRID_END - GRID_START) / pd.Timedelta(minutes=30))
SMALL_START = pd.Timestamp("2012-07-01 00:00")  # the published baseline's training window
TRIAL_DAY_START_HOUR = 5  # trial days run 05:00 to 05:00, wall clock
TARIFF_COUNTS = {"Normal": 15072, "Low": 1660, "High": 788}  # measured from Tariffs.xlsx

# Dates of the 21 CM events in A3 Table 3.3 / thesis Table 4.5. The file and the table disagree.
TABLE_CM = """01-19 01-20 01-29 02-09 02-20 02-21 03-16 03-17 03-21 03-27 03-28 03-29 04-23 05-01
              05-02 05-03 10-13 10-20 11-26 11-27 12-15""".split()
CLOCKS = ("gmt_end", "gmt_start", "wall_end", "wall_start")


class ClockUndecided(RuntimeError):
    """The meter clock convention must be decided before meters and tariffs are joined."""


def confirm_full(what):
    if os.environ.get("PYMC_CHALLENGES_CONFIRM_FULL") != "yes":
        raise PreflightError(f"{what} touches the full data. Set PYMC_CHALLENGES_CONFIRM_FULL=yes.")


def fetch(raw):
    """One resumable GET, not per-member range requests: those were rate limited after ~25 calls."""
    confirm_full("fetch")
    raw.mkdir(parents=True, exist_ok=True)
    target, part = raw / "lcl_partitioned.zip", raw / "lcl_partitioned.zip.part"
    if not target.exists():
        done = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(ZIP_URL, headers={"User-Agent": USER_AGENT,
                                                           "Range": f"bytes={done}-"})
        with urllib.request.urlopen(request, timeout=60) as response, part.open("ab") as out:
            while block := response.read(1 << 22):
                out.write(block)
        if part.stat().st_size != ZIP_BYTES:
            raise RuntimeError(f"{part}: {part.stat().st_size} bytes, expected {ZIP_BYTES}: re-run")
        os.replace(part, target)
    tariffs = raw / "Tariffs.xlsx"
    if not tariffs.exists():
        request = urllib.request.Request(TARIFF_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        atomic_write(tariffs, lambda f: f.write(payload))
    return {"zip_sha256": sha256_file(target), "tariffs_sha256": sha256_file(tariffs)}


# Calendar ---------------------------------------------------------------------------------------


def read_tariffs(path):
    """Tariffs.xlsx as a wall-clock frame, with the standard library (no openpyxl)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        strings = [item.find(f"{ns}t").text
                   for item in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rows = list(ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{ns}row"))[1:]
    serial = np.array([float(r[0].find(f"{ns}v").text) for r in rows])
    band = [strings[int(r[1].find(f"{ns}v").text)] for r in rows]
    minutes = np.round(serial * 48).astype(np.int64) * 30
    wall = pd.Timestamp("1899-12-30") + pd.to_timedelta(minutes, unit="m")
    frame = pd.DataFrame({"wall": wall, "band": band})
    if frame.band.value_counts().to_dict() != TARIFF_COUNTS:
        raise ValueError(f"Tariffs.xlsx changed: {frame.band.value_counts().to_dict()}")
    return frame


def calendar(frame):
    """One row per price run: wall start, half-hours, band, day class, and `disputed` days."""
    frame = frame.copy()
    frame["day"] = (frame.wall - pd.Timedelta(hours=TRIAL_DAY_START_HOUR)).dt.strftime("%Y-%m-%d")
    bands = frame.groupby("day").band.agg(lambda s: frozenset(s))
    day_class = bands.map(lambda s: "both" if {"High", "Low"} <= s else
                          "all_low" if s == {"Low"} else
                          "none" if s == {"Normal"} else "single")
    table = {f"2013-{d}" for d in TABLE_CM}
    runs = (frame.band != frame.band.shift()).cumsum()
    events = []
    for _, run in frame[frame.band != "Normal"].groupby(runs):
        # A run can cross 05:00 (two events back to back), so split it on the trial day.
        for day, part in run.groupby("day"):
            events.append({"start_wall": part.wall.iloc[0].isoformat(), "slots": len(part),
                           "band": part.band.iloc[0], "day": day, "class": day_class[day]})
    file_cm = {d for d, c in day_class.items() if c == "both"}
    disputed = sorted(file_cm ^ table)
    for event in events:
        event["disputed"] = event["day"] in disputed
    return {"events": events, "disputed_days": disputed,
            "day_counts": day_class.value_counts().to_dict()}


def event_slots(events, clock):
    """Grid slot of each event's first half-hour, under a stated meter clock convention.

    `*_end`: a reading stamped T covers the half-hour ending at T, so the half-hour starting at the
    event's start carries the stamp 30 minutes later. `gmt_*`: stamps do not follow BST.
    """
    if clock not in CLOCKS:
        raise ClockUndecided(f"Pick --clock from {CLOCKS} after the pilot's clock test; "
                             "the data do not say on their own (plans/large/E91.md, P1).")
    start = pd.DatetimeIndex([e["start_wall"] for e in events])
    if clock.startswith("gmt"):
        start = (start.tz_localize("Europe/London", ambiguous="raise", nonexistent="raise")
                 .tz_convert("UTC").tz_localize(None))
    if clock.endswith("end"):
        start = start + pd.Timedelta(minutes=30)
    return np.asarray((start - GRID_START) / pd.Timedelta(minutes=30), dtype=np.int64)


# Meter data -------------------------------------------------------------------------------------


def members(source):
    """(name, opener) for a zip of partitions, or a directory of already extracted CSV samples."""
    source = Path(source)
    if source.is_dir():
        paths = sorted(source.glob("*.csv"))
        return [(p.name, (lambda p=p: p.open("rb"))) for p in paths]
    archive = zipfile.ZipFile(source)
    names = sorted((n for n in archive.namelist() if n.endswith(".csv")),
                   key=lambda n: int(n.rsplit("_", 1)[1].split(".")[0]))
    return [(n, (lambda n=n: archive.open(n))) for n in names]


def parse_member(handle):
    """Long arrays (home number, group, slot, kWh as float32) and what was dropped, for one file."""
    frame = pd.read_csv(handle, dtype=str, skipinitialspace=True)
    frame.columns = ["id", "group", "stamp", "kwh"]  # the raw last header ends with a space
    n_rows = len(frame)
    stamp = pd.to_datetime(frame.stamp.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    kwh = pd.to_numeric(frame.kwh.str.strip(), errors="coerce")
    nulls = int(kwh.isna().sum())
    on_grid = (stamp.dt.minute % 30 == 0) & (stamp.dt.second == 0)
    keep = kwh.notna() & on_grid
    frame = pd.DataFrame({"home": frame.id.str.slice(3).astype(np.int32)[keep],
                          "tou": (frame.group == "ToU").astype(np.int8)[keep],
                          "slot": ((stamp[keep] - GRID_START) // pd.Timedelta(minutes=30))
                          .astype(np.int32), "kwh": kwh[keep].astype(np.float32)})
    if not frame.slot.between(0, N_SLOTS - 1).all():
        raise ValueError("a reading falls outside the grid: widen GRID_START or GRID_END")
    n_exact = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    n_conflict = int(frame.duplicated(["home", "slot"]).sum())
    frame = frame.drop_duplicates(["home", "slot"], keep=False)  # two values, neither is trusted
    return ({c: frame[c].to_numpy() for c in frame.columns},
            {"rows": n_rows, "null": nulls, "off_grid": int((~on_grid).sum()),
             "exact_duplicates": n_exact, "conflicting_slots": n_conflict})


def parse(store, source, stats_name="parse_stats"):
    """Each member becomes one checksummed chunk, so a crash costs at most one member."""
    files = members(source)
    if len(files) > 8:
        confirm_full("parse")
    stats = store.load_json(stats_name) if (store.path / f"{stats_name}.json").exists() else {}
    for i in store.pending("readings", len(files)):
        name, opener = files[i]
        started = time.perf_counter()
        with opener() as handle:
            arrays, counts = parse_member(handle)
        store.save_chunk("readings", i, arrays)
        stats[name] = {**counts, "seconds": round(time.perf_counter() - started, 2)}
        store.save_json(stats_name, stats)
    store.mark_done("parse", members=len(files))
    return stats


def assemble(store, n_members):
    """Homes x half-hours float32 matrix (NaN = no reading), written once and then memory-mapped."""
    ids = []
    for i in range(n_members):
        chunk = store.load_chunk("readings", i)
        ids.append(np.unique(chunk["home"] * 2 + chunk["tou"]))
    code = np.unique(np.concatenate(ids))
    homes, tou = code // 2, (code % 2).astype(np.int8)
    if len(np.unique(homes)) != len(homes):
        raise ValueError("a home appears in both groups")
    grid = np.full((len(homes), N_SLOTS), np.nan, dtype=np.float32)
    for i in range(n_members):
        chunk = store.load_chunk("readings", i, check=False)
        grid[np.searchsorted(homes, chunk["home"]), chunk["slot"]] = chunk["kwh"]
    store.save_array("grid_kwh", grid)
    store.save_array("homes", homes)
    store.save_array("tou", tou)
    store.mark_done("assemble", homes=len(homes), tou=int(tou.sum()))
    return homes, tou


def pick(homes, n):
    """A fixed subset chosen by a hash of the id alone, never by anything about the outcome."""
    return sorted(homes, key=lambda h: hashlib.sha256(f"MAC{h:06d}".encode()).hexdigest())[:n]


def small(store, clock, n_tou, n_std, out):
    """The laptop subset: hourly mean kW from July 2012, with the events in the same frame."""
    if clock not in CLOCKS:
        raise ClockUndecided(f"--clock must be one of {CLOCKS}, decided by pilot P1.")
    grid = store.load_array("grid_kwh", mmap=True)
    homes, tou = store.load_array("homes"), store.load_array("tou")
    # Hour boundaries must be half-hour STARTS in the meter's frame: an end stamp moves one slot.
    shift = 1 if clock.endswith("end") else 0
    first = int((SMALL_START - GRID_START) / pd.Timedelta(minutes=30)) + shift
    trial = [int((pd.Timestamp(d) - GRID_START) / pd.Timedelta(minutes=30))
             for d in ("2013-01-01", "2014-01-01")]
    covered = np.isfinite(grid[:, trial[0]:trial[1]]).mean(axis=1) >= 0.95
    chosen = sorted(pick(homes[covered & (tou == 1)], n_tou)
                    + pick(homes[covered & (tou == 0)], n_std))
    rows = np.searchsorted(homes, chosen)
    n_hours = (N_SLOTS - first) // 2
    half = np.asarray(grid[rows, first:first + 2 * n_hours]).reshape(len(rows), n_hours, 2)
    kw = half.sum(axis=2)  # kWh in an hour = mean kW; NaN if either half-hour is missing
    payload = dict(kw=kw, homes=np.asarray(chosen), tou=tou[rows], clock=np.array(clock),
                   first_hour=np.array(str(SMALL_START)))
    out = Path(out)
    atomic_write(out, lambda f: np.savez_compressed(f, **payload))
    return {"file": str(out), "bytes": out.stat().st_size, "shape": kw.shape,
            "tou": int(tou[rows].sum()), "sha256": sha256_file(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fetch", "calendar", "parse", "assemble", "small"))
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="where the zip is kept")
    parser.add_argument("--root", type=Path, help="Store root; the default is results/")
    parser.add_argument("--tariffs", type=Path)
    parser.add_argument("--members", type=Path, help="a directory of sample CSVs, or the zip")
    parser.add_argument("--clock", choices=CLOCKS, help="decided by pilot P1; no default")
    parser.add_argument("--n-tou", type=int, default=200)
    parser.add_argument("--n-std", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("data/lcl_small.npz"))
    args = parser.parse_args(argv)

    store = Store("E91", root=args.root)
    source = args.members or args.raw / "lcl_partitioned.zip"
    if args.step == "fetch":
        print(json.dumps(fetch(args.raw), indent=1))
    elif args.step == "calendar":
        result = calendar(read_tariffs(args.tariffs or args.raw / "Tariffs.xlsx"))
        store.save_json("calendar", result)
        store.mark_done("calendar", events=len(result["events"]))
        print(result["day_counts"], len(result["events"]), "runs; disputed:",
              result["disputed_days"])
    elif args.step == "parse":
        stats = parse(store, source)
        print({k: sum(s[k] for s in stats.values())
               for k in ("rows", "null", "off_grid", "exact_duplicates", "conflicting_slots")})
    elif args.step == "assemble":
        homes, tou = assemble(store, len(members(source)))
        print(len(homes), "homes,", int(tou.sum()), "on ToU")
    else:
        print(small(store, args.clock, args.n_tou, args.n_std, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
