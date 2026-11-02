"""Build the E91 datasets from Low Carbon London SmartMeter Energy Consumption Data.

fetch, calendar, parse, assemble, pilot, table and small have all run on the full data
(plans/large/E91.md, Pilot results).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_e91_data.py calendar --tariffs Tariffs.xlsx
    ... parse --members DIR_OR_ZIP      # one checkpoint per member, safe to kill and re-run
    ... assemble                        # homes x half-hours float32 matrix
    ... pilot [--clock CONVENTION]      # P1 and P3; P2 too once a clock is stated
    ... table --clock gmt_end           # data/e91_hours.npz, data/e91_event_table.npz, clock_cost.json
    ... small --clock gmt_end           # data/lcl_small.npz, hourly meters of 200 ToU homes

A full run also needs `fetch` (one resumable download, 796 MB) and
PYMC_CHALLENGES_CONFIRM_FULL=yes. Measured on 2 cores, on a shared machine: `parse` 277 s, peak 0.48
GB; `assemble` 17 s, peak 1.12 GB. Run both through tools/memguard.py. Chunks and the grid stay
under ~/.cache/pymc-challenges/lcl/store (2.9 GB); only small JSON goes to results/E91.

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

The clock, settled by pilot P1 on all ToU homes: tariffs are wall-clock, meter stamps are GMT all
year and mark the END of each half-hour (`gmt_end`). A naive join is 30 minutes off in both seasons,
in opposite directions. `small` and P2 still ask for `--clock` by name, so the choice stays visible
in the command and the output.

The calendar: the file is not the record of what was sent. P2 found six High+Low days in the file
with no response, and two all-Low days that behave like the table's evening CM days. Plan section 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sparse  # noqa: E402

from pymc_challenges.bigrun import (  # noqa: E402
    PreflightError,
    Store,
    atomic_write,
    git_hash,
    library_versions,
    sha256_file,
)

ZIP_URL = ("https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/"
           "Partitioned%20LCL%20Data.zip")
TARIFF_URL = ("https://data.london.gov.uk/download/vqm0d/14855047-44c2-4856-8a48-e5649200e6ce/"
              "Tariffs.xlsx")
ZIP_BYTES = 795_722_689
N_MEMBERS = 168
RAW_DIR = Path.home() / ".cache" / "pymc-challenges" / "lcl"
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

# The owner's calendar defaults (plan, Pilot results). 29 March is reinstated as a CM day on the pattern
# of 27-28 March. 3 May and the six file-only days leave the events and the baseline. The three true
# all-Low days have no effect term, so they leave the baseline too. 28 December stays in.
OUT_DAYS = ("2013-05-03", "2013-11-19", "2013-12-06", "2013-12-08", "2013-12-10", "2013-12-21",
            "2013-12-27")
ALL_LOW_DAYS = ("2013-02-07", "2013-06-20", "2013-12-19")
REINSTATED = (("2013-03-29 05:00", 12, "Low"), ("2013-03-29 17:00", 6, "High"),
              ("2013-03-29 23:00", 6, "Low"))  # wall start, hours, band
BANDS = ("morning", "day", "evening", "night")  # wall clock: 05-11, 11-17, 17-23, 23-05
EFFECTS = (*(f"SF High {b}" for b in BANDS), *(f"SF Low {b}" for b in BANDS),
           "CM High evening", "CM High other", "CM Low")
CELL_PRIOR_SD = 5.0  # kW: an event cell must not be shrunk, or its between-event spread is lost
FLEET_COVERAGE = 0.5  # of 2013, as pilot P6
PANEL_COVERAGE = 0.8  # of the whole window, for the N panel
PLACEBO_COVERAGE = 0.8  # of Jul-Dec 2012, as pilot P5
PLACEBO_SHIFT_DAYS = 364


class ClockUndecided(RuntimeError):
    """The meter clock convention must be decided before meters and tariffs are joined."""


def confirm_full(what):
    if os.environ.get("PYMC_CHALLENGES_CONFIRM_FULL") != "yes":
        raise PreflightError(f"{what} touches the full data. Set PYMC_CHALLENGES_CONFIRM_FULL=yes.")


def download(url, part, size):
    """Append to `part` by a range request, starting again if the server ignores the range."""
    done = part.stat().st_size if part.exists() else 0
    if done >= size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Range": f"bytes={done}-"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with part.open("ab" if response.status == 206 else "wb") as out:
            while block := response.read(1 << 22):
                out.write(block)


def fetch(raw):
    """One resumable GET, not per-member range requests: those were rate limited after ~25 calls."""
    confirm_full("fetch")
    raw.mkdir(parents=True, exist_ok=True)
    target, part = raw / "lcl_partitioned.zip", raw / "lcl_partitioned.zip.part"
    if not target.exists():
        download(ZIP_URL, part, ZIP_BYTES)
        if part.stat().st_size != ZIP_BYTES:
            raise RuntimeError(f"{part}: {part.stat().st_size} bytes, expected {ZIP_BYTES}: re-run")
        os.replace(part, target)
    tariffs = raw / "Tariffs.xlsx"
    if not tariffs.exists():
        request = urllib.request.Request(TARIFF_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        atomic_write(tariffs, lambda f: f.write(payload))
    return {"zip_bytes": target.stat().st_size, "zip_sha256": sha256_file(target),
            "tariffs_sha256": sha256_file(tariffs)}


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
    groups = frame.group.value_counts().to_dict()
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
             "exact_duplicates": n_exact, "conflicting_slots": n_conflict,
             "rows_by_group": groups})


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
    """Homes x half-hours float32 matrix (NaN = no reading), written once and then memory-mapped.

    A home can straddle two members, so a slot already filled by an earlier member is counted:
    the same value is harmless, a different one is a conflict that P3 must read.
    """
    ids = []
    for i in range(n_members):
        chunk = store.load_chunk("readings", i)
        ids.append(np.unique(chunk["home"] * 2 + chunk["tou"]))
    code = np.unique(np.concatenate(ids))
    homes, tou = code // 2, (code % 2).astype(np.int8)
    if len(np.unique(homes)) != len(homes):
        raise ValueError("a home appears in both groups")
    grid = np.full((len(homes), N_SLOTS), np.nan, dtype=np.float32)
    across, conflicts = 0, 0
    for i in range(n_members):
        chunk = store.load_chunk("readings", i, check=False)
        row = np.searchsorted(homes, chunk["home"])
        before = grid[row, chunk["slot"]]
        filled = np.isfinite(before)
        across += int(filled.sum())
        conflicts += int((before[filled] != chunk["kwh"][filled]).sum())
        grid[row, chunk["slot"]] = chunk["kwh"]
    store.save_array("grid_kwh", grid)
    store.save_array("homes", homes)
    store.save_array("tou", tou)
    store.mark_done("assemble", homes=len(homes), tou=int(tou.sum()), across_members=across,
                    across_member_conflicts=conflicts)
    return homes, tou


# Pilots P1-P3 ------------------------------------------------------------------------------------
#
# Everything works on one series: mean W of ToU homes minus mean W of Std homes, per stamp slot, so
# the shared day and the shared daily shape cancel. A response is that difference at a slot minus
# its mean at the same slot of nearby unpriced days (same side of a clock change, same day type).
# An event's noise is measured between events, so the standard errors do not assume homes are
# independent. These are checks of timing, never effect estimates: D vs N is not a causal contrast.

OFFSETS = range(-6, 7)
BASE_DAYS = 28  # days either side searched for the baseline of a slot
CLOCK_CHANGE_EVES = ("2013-03-30", "2013-03-31", "2013-10-26", "2013-10-27")  # no wall map here
CLOCK_STEP = {"wall_start": (0, 0), "wall_end": (1, 1), "gmt_start": (0, -2), "gmt_end": (1, -1)}


def wall_slots(stamps):
    return np.asarray((pd.DatetimeIndex(stamps) - GRID_START) / pd.Timedelta(minutes=30),
                      dtype=np.int64)


def is_bst(stamps):
    stamps = pd.DatetimeIndex(stamps)
    local = stamps.tz_localize("Europe/London", ambiguous=np.ones(len(stamps), dtype=bool),
                              nonexistent="shift_forward")
    return np.asarray([bool(t.dst()) for t in local])


def group_series(grid, tou):
    """Mean W per slot of each group, reading the grid in blocks of rows to stay small."""
    total = {0: np.zeros(N_SLOTS), 1: np.zeros(N_SLOTS)}
    count = {0: np.zeros(N_SLOTS), 1: np.zeros(N_SLOTS)}
    for group in (0, 1):
        rows = np.flatnonzero(tou == group)
        for start in range(0, len(rows), 256):
            block = np.asarray(grid[rows[start:start + 256]], dtype=np.float32)
            seen = np.isfinite(block)
            total[group] += np.where(seen, block, 0).sum(axis=0, dtype=np.float64)
            count[group] += seen.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        watts = {g: 2000 * total[g] / count[g] for g in (0, 1)}  # kWh per half hour to mean W
    return watts[1] - watts[0], count[1], count[0]


class Baseline:
    """Subtracts each slot's level on nearby unpriced days from the ToU-Std difference."""

    def __init__(self, diff, frame):
        self.diff = diff
        wall = wall_slots(frame.wall)
        priced = np.zeros(N_SLOTS, dtype=bool)
        priced[wall[(frame.band != "Normal").to_numpy()]] = True
        # An unpriced baseline slot must be clear in the wall frame and in a one-hour-earlier frame.
        wide = np.convolve(priced, np.ones(13, dtype=int), mode="same") > 0
        self.dirty = wide | np.roll(wide, -2) | np.roll(wide, 2)
        self.days = pd.date_range(GRID_START, GRID_END, freq="D", inclusive="left")
        self.bst = is_bst(self.days + pd.Timedelta(hours=12))

    def _same_kind(self, day):
        weekend = day >= 5
        return lambda other: (other >= 5) == weekend

    def level(self, slot):
        """Mean of the difference at `slot` on comparable unpriced days within BASE_DAYS."""
        day = slot // 48
        kind, values = self._same_kind(self.days[day].weekday()), []
        for shift in range(-BASE_DAYS, BASE_DAYS + 1):
            other = day + shift
            if shift == 0 or not 0 <= other < len(self.days):
                continue
            target = slot + 48 * shift
            if (self.bst[other] == self.bst[day] and kind(self.days[other].weekday())
                    and not self.dirty[target] and np.isfinite(self.diff[target])):
                values.append(self.diff[target])
        return (np.mean(values), len(values)) if values else (np.nan, 0)

    def response(self, slot):
        return self.diff[slot] - self.level(slot)[0]


def aligned_steps(base, slots, signs):
    """(events x offsets) step at each stamp offset, positive where the price says it should be."""
    response = np.array([[base.response(s + k) for k in range(OFFSETS[0] - 1, OFFSETS[-1] + 1)]
                         for s in slots])
    return signs[:, None] * np.diff(response, axis=1), response[:, 1:]


def step_table(steps):
    """Mean step and between-event se per offset; the peak and its margin over each neighbour."""
    mean = np.nanmean(steps, axis=0)
    n = np.isfinite(steps).sum(axis=0)
    se = np.nanstd(steps, axis=0, ddof=1) / np.sqrt(n)
    peak = int(np.argmax(mean))
    margins = []
    for near in (peak - 1, peak + 1):
        if 0 <= near < len(mean):
            gap = steps[:, peak] - steps[:, near]
            margins.append(float(np.nanmean(gap) / (np.nanstd(gap, ddof=1)
                                                    / np.sqrt(np.isfinite(gap).sum()))))
    return {"offset": list(OFFSETS), "mean_w": np.round(mean, 2), "se_w": np.round(se, 2),
            "n_events": n, "peak_offset": OFFSETS[peak], "peak_w": round(float(mean[peak]), 2),
            "peak_z": round(float(mean[peak] / se[peak]), 2),
            "min_margin_over_neighbours_se": round(min(margins), 2)}


def clean_edges(events, frame):
    """SF events split into starts and ends with six Normal slots on the outer side."""
    wall = wall_slots(frame.wall)
    band = pd.Series(frame.band.to_numpy(), index=wall)
    rows = []
    for e in events:
        if e["class"] != "single":
            continue
        start = int(wall_slots([e["start_wall"]])[0])
        stop = start + e["slots"]
        before = band.reindex(range(start - 6, start)).fillna("Normal")
        after = band.reindex(range(stop, stop + 6)).fillna("Normal")
        sign = -1 if e["band"] == "High" else 1
        rows.append((e, "start", start, sign, (before == "Normal").all()))
        rows.append((e, "end", stop, -sign, (after == "Normal").all()))
    return rows


def pilot_clock(heavy, frame, events):
    """P1: where does the ToU-Std step sit, by season? Offsets count stamps from the wall time."""
    grid = heavy.load_array("grid_kwh", mmap=True, check=False)
    tou = heavy.load_array("tou")
    diff, n_tou, n_std = group_series(grid, tou)
    heavy.save_array("diff_w", diff)
    base = Baseline(diff, frame)
    edges = [row for row in clean_edges(events, frame) if row[4]]
    wall = pd.Timestamp("2011-11-01") + pd.to_timedelta(np.array([r[2] for r in edges]) * 30, "m")
    bst = is_bst(wall)
    signs = np.array([r[3] for r in edges], dtype=float)
    slots = np.array([r[2] for r in edges])
    kinds = np.array([r[1] for r in edges])
    bands = np.array([r[0]["band"] for r in edges])
    steps, response = aligned_steps(base, slots, signs)
    out = {"homes_tou": int((tou == 1).sum()), "homes_std": int((tou == 0).sum()),
           "note": "offset = stamp minus wall time of the edge, in half-hours; positive step = "
                   "the direction the price asks for"}
    tables = {}
    for season, in_season in (("gmt", ~bst), ("bst", bst)):
        for edge in ("start", "end", "both"):
            take = in_season & (kinds == edge if edge != "both" else True)
            tables[f"{season}_{edge}"] = step_table(steps[take])
            for b in ("High", "Low"):
                tables[f"{season}_{edge}_{b}"] = step_table(steps[take & (bands == b)])
    out["tables"] = tables
    a, b = tables["gmt_both"]["peak_offset"], tables["bst_both"]["peak_offset"]
    out["peak_offsets"] = {"gmt": a, "bst": b}
    out["convention_matching"] = [c for c, want in CLOCK_STEP.items() if want == (a, b)]
    out["expected_peaks"] = CLOCK_STEP
    out["gate_margin_se"] = min(tables["gmt_both"]["min_margin_over_neighbours_se"],
                                tables["bst_both"]["min_margin_over_neighbours_se"])
    out["passes"] = bool(out["convention_matching"] and out["gate_margin_se"] >= 3)
    out["per_edge"] = [{"day": r[0]["day"], "edge": r[1], "band": r[0]["band"],
                        "slots": r[0]["slots"], "wall": str(w), "bst": bool(x),
                        "response_w": np.round(row, 1)}
                       for r, w, x, row in zip(edges, wall, bst, response, strict=True)]
    out["homes_per_slot_2013"] = {"tou_median": float(np.median(n_tou[trial_slice()])),
                                  "std_median": float(np.median(n_std[trial_slice()]))}
    return out


def trial_slice():
    return slice(*wall_slots(["2013-01-01", "2014-01-01"]))


def pilot_counts(heavy, stats):
    """P3 and the direct clock check: which stamps exist on the two clock-change days?"""
    grid = heavy.load_array("grid_kwh", mmap=True, check=False)
    homes, tou = heavy.load_array("homes"), heavy.load_array("tou")
    totals = {k: sum(s[k] for s in stats.values())
              for k in ("rows", "null", "off_grid", "exact_duplicates", "conflicting_slots")}
    groups = {}
    for s in stats.values():
        for key, value in s["rows_by_group"].items():
            groups[key] = groups.get(key, 0) + value
    stage = next(s for s in heavy.stages() if s["stage"] == "assemble")
    filled = np.zeros(N_SLOTS)
    for start in range(0, len(homes), 256):
        filled += np.isfinite(np.asarray(grid[start:start + 256])).sum(axis=0)
    trial = trial_slice()
    in_2013 = np.zeros(len(homes))
    for start in range(0, len(homes), 256):
        in_2013[start:start + 256] = np.isfinite(np.asarray(grid[start:start + 256, trial])).mean(1)
    days = {}
    for day in ("2013-03-30", "2013-03-31", "2013-10-26", "2013-10-27"):
        first = int(wall_slots([day])[0])
        days[day] = {"on_grid_slots_with_data": int((filled[first:first + 48] > 0).sum()),
                     "homes_at_00:00_to_03:00": [int(x) for x in filled[first:first + 7]]}
    tou_homes = int((tou == 1).sum())
    within = abs(totals["rows"] / 167.9e6 - 1)
    return {"zip_members": len(stats), **totals, "rows_by_group": groups,
            "cells_in_grid": int(filled.sum()), "homes": len(homes), "homes_tou": tou_homes,
            "homes_std": int((tou == 0).sum()),
            "first_slot": str(GRID_START + pd.Timedelta(minutes=30) * int(np.argmax(filled > 0))),
            "last_slot": str(GRID_START + pd.Timedelta(minutes=30)
                             * int(N_SLOTS - 1 - np.argmax(filled[::-1] > 0))),
            "across_members": stage["across_members"],
            "across_member_conflicts": stage["across_member_conflicts"],
            "homes_95pct_2013": {"tou": int((in_2013[tou == 1] >= 0.95).sum()),
                                 "std": int((in_2013[tou == 0] >= 0.95).sum())},
            "clock_change_days": days, "rows_vs_extrapolated": round(float(within), 5),
            "passes": bool(1000 <= tou_homes <= 1125 and within <= 0.01
                           and totals["conflicting_slots"] == 0
                           and stage["across_member_conflicts"] == 0)}


def pilot_calendar(heavy, frame, events, disputed, clock):
    """P2: response in the evening High window and in the daytime Low flank, per day.

    A real CM day should fall in the evening and rise in the day. The plan's gate reads the evening
    z alone; the flank and their contrast were added after the evening result was seen, so they are
    supporting evidence and not a new gate.
    """
    diff = heavy.load_array("diff_w")
    base = Baseline(diff, frame)

    def window_response(day, start, periods):
        wall = pd.date_range(f"{day} {start}", periods=periods, freq="30min")
        slots = event_slots([{"start_wall": t.isoformat()} for t in wall], clock)
        return float(np.nanmean([base.response(s) for s in slots]))

    measures = {"evening": lambda d: window_response(d, "17:00", 12),
                "daytime": lambda d: window_response(d, "05:00", 24)}
    measures["contrast"] = lambda d: measures["daytime"](d) - measures["evening"](d)
    daily = pd.date_range("2013-01-01", "2013-12-31", freq="D").strftime("%Y-%m-%d")
    classes = {e["day"]: e["class"] for e in events}
    free = [d for d in daily if d not in classes]  # no price event on this trial day
    free_bst = is_bst(pd.DatetimeIndex(free) + pd.Timedelta(hours=12))
    null = {}
    for name, measure in measures.items():
        values = np.array([measure(d) for d in free])
        null[name] = {"gmt": (float(np.nanmedian(values[~free_bst])),
                              float(np.nanstd(values[~free_bst], ddof=1))),
                      "bst": (float(np.nanmedian(values[free_bst])),
                              float(np.nanstd(values[free_bst], ddof=1)))}

    def judge(day):
        season = "bst" if is_bst([f"{day} 12:00"])[0] else "gmt"
        row = {"day": day, "weekday": pd.Timestamp(day).day_name(), "season": season,
               "file_class": classes.get(day, "none"), "in_table": day[5:] in TABLE_CM}
        for name, measure in measures.items():
            value = measure(day)
            row[f"{name}_w"] = round(value, 1)
            row[f"{name}_z"] = round((value - null[name][season][0]) / null[name][season][1], 2)
        return row

    evening_cm = sorted({e["day"] for e in events if e["class"] == "both" and e["band"] == "High"
                         and e["start_wall"][11:16] == "17:00" and not e["disputed"]})
    table = {"clock": clock,
             "windows": {"evening": "17:00-23:00 wall", "daytime": "05:00-17:00 wall"},
             "null_median_sd_w": null, "n_null_days": len(free),
             "disputed": [judge(d) for d in disputed],
             "undisputed_evening_cm": [judge(d) for d in evening_cm],
             "all_low_days": [judge(d) for d in sorted(d for d, c in classes.items()
                                                       if c == "all_low")]}
    # The file was wrong about whole days, so ask the same of each SF run, by its own window.
    runs, spreads = [], {}
    for e in events:
        if e["class"] != "single":
            continue
        day, start = e["start_wall"][:10], e["start_wall"][11:16]
        season = "bst" if is_bst([e["start_wall"]])[0] else "gmt"
        key = (start, e["slots"], season)
        if key not in spreads:
            wanted = free_bst if season == "bst" else ~free_bst
            spreads[key] = float(np.nanstd([window_response(d, start, e["slots"])
                                            for d, keep in zip(free, wanted, strict=True)
                                            if keep and d not in CLOCK_CHANGE_EVES],
                                           ddof=1))
        sign = -1 if e["band"] == "High" else 1
        value = sign * window_response(day, start, e["slots"])
        runs.append({"day": day, "start": start, "slots": e["slots"], "band": e["band"],
                     "season": season, "w_in_price_direction": round(value, 1),
                     "z": round(value / spreads[key], 2)})
    table["sf_runs"] = runs
    table["days_clearing_evening_z3"] = [d["day"] for d in table["disputed"]
                                         if abs(d["evening_z"]) > 3]
    table["passes"] = len(table["days_clearing_evening_z3"]) == len(disputed)
    return table


def pick(homes, n):
    """A fixed subset chosen by a hash of the id alone, never by anything about the outcome."""
    return sorted(homes, key=lambda h: hashlib.sha256(f"MAC{h:06d}".encode()).hexdigest())[:n]


def small(store, clock, n_tou, n_std, out):
    """The laptop subset: hourly Wh from July 2012 as uint16, 65535 where a half-hour is missing."""
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
    wh = np.where(np.isfinite(kw), np.round(np.nan_to_num(kw) * 1000), 65535).astype(np.uint16)
    payload = dict(wh=wh, homes=np.asarray(chosen), tou=tou[rows], clock=np.array(clock),
                   first_hour=np.array(str(SMALL_START)))
    out = Path(out)
    atomic_write(out, lambda f: np.savez_compressed(f, **payload))
    return {"file": str(out), "bytes": out.stat().st_size, "shape": kw.shape,
            "tou": int(tou[rows].sum()), "sha256": sha256_file(out)}


# Derived tables for the notebook -----------------------------------------------------------------


class Hours:
    """The hourly frame: an hour is two half-hours in the meter's frame, and its wall time."""

    def __init__(self, clock):
        self.first = int((SMALL_START - GRID_START) / pd.Timedelta(minutes=30))
        self.first += 1 if clock.endswith("end") else 0
        self.n = (N_SLOTS - self.first) // 2
        self.gmt = pd.date_range(SMALL_START, periods=self.n, freq="h")
        self.wall = self.gmt.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)
        hour = self.wall.hour.to_numpy()
        self.band = np.select([(hour >= 5) & (hour < 11), (hour >= 11) & (hour < 17),
                               (hour >= 17) & (hour < 23)], [0, 1, 2], 3)
        self.weekend = self.wall.dayofweek.to_numpy() >= 5
        self.level = hour * 2 + self.weekend  # s(t): 48 levels, wall hour x weekday or weekend
        self.trial_day = (self.wall - pd.Timedelta(hours=TRIAL_DAY_START_HOUR)).strftime(
            "%Y-%m-%d").to_numpy().astype(str)
        self.in_2013 = (self.gmt >= "2013-01-01") & (self.gmt < "2014-01-01")
        self.pre_trial = (self.gmt >= "2012-07-01") & (self.gmt < "2013-01-01")

    def read(self, grid, rows):
        """Mean kW per hour for these grid rows; NaN if either half-hour is missing."""
        half = np.asarray(grid[rows, self.first:self.first + 2 * self.n])
        return half.reshape(len(rows), self.n, 2).sum(axis=2)


def clean_events(table):
    """The calendar under the owner's defaults, as wall starts, hours, band, day and SF or CM."""
    out = [{"start": pd.Timestamp(e["start_wall"]), "hours": e["slots"] // 2, "band": e["band"],
            "day": e["day"], "kind": "SF" if e["class"] == "single" else "CM"}
           for e in table["events"]
           if e["day"] not in (*OUT_DAYS, *ALL_LOW_DAYS, "2013-03-29")]
    out += [{"start": pd.Timestamp(start), "hours": hours, "band": band, "day": start[:10],
             "kind": "CM"} for start, hours, band in REINSTATED]
    return sorted(out, key=lambda e: e["start"])


def event_cells(hours, events, shift_days=0, first=None, last=None):
    """Per hour: effect number and event-cell number (-1 for none), plus the cells themselves.

    A cell is one event in one time band, so a 12-hour event that crosses 23:00 makes two cells.
    """
    where = {t: i for i, t in enumerate(hours.wall)}
    effect, cell = np.full(hours.n, -1), np.full(hours.n, -1)
    cells = []
    for number, e in enumerate(events):
        if first and not first <= e["day"] <= last:
            continue
        stamps = pd.date_range(e["start"] - pd.Timedelta(days=shift_days), periods=e["hours"], freq="h")
        split = {}
        for stamp in stamps:
            if stamp not in where:
                continue
            t = where[stamp]
            b = int(hours.band[t])
            if e["kind"] == "SF":
                k = (0 if e["band"] == "High" else 4) + b
            else:
                k = 10 if e["band"] == "Low" else 8 if b == 2 else 9
            split.setdefault(k, []).append(t)
        for k, ts in split.items():
            cell[ts] = len(cells)
            effect[ts] = k
            cells.append({"event": number, "day": e["day"], "effect": EFFECTS[k], "hours": len(ts),
                          "start_hour": e["start"].hour, "event_hours": e["hours"]})
    return effect, cell, cells


def out_of_baseline(hours, shift_days=0):
    days = pd.DatetimeIndex([*OUT_DAYS, *ALL_LOW_DAYS]) - pd.Timedelta(days=shift_days)
    return np.isin(hours.trial_day, days.strftime("%Y-%m-%d").to_numpy().astype(str))


def month_band_terms(hours, keep):
    """One indicator per month x time band: the post-hoc local baseline (plan, Pilot results)."""
    stamp = hours.gmt.strftime("%Y-%m").to_numpy().astype(str)
    key = pd.factorize(np.char.add(stamp, hours.band.astype(str)))[0]
    key = np.where(keep, key, -1)
    used = np.unique(key[key >= 0])
    number = -np.ones(key.max() + 2, dtype=int)
    number[used] = np.arange(len(used))
    k = number[key]
    on = k >= 0
    return sparse.csr_matrix((np.ones(on.sum()), (np.flatnonzero(on), k[on])),
                             shape=(hours.n, len(used)))


def deviation(n_panel, hours, keep):
    """z[t]: the N panel in kW against its own mean at that wall hour and day type."""
    z = np.zeros(hours.n)
    good = keep & np.isfinite(n_panel)
    for s in range(48):
        at = hours.level == s
        z[at] = n_panel[at] - n_panel[at & good].mean()
    return np.nan_to_num(z)


def m1_cells(kw, hours, z, cell, keep, extra=None):
    """M1 home by home: conjugate Gaussian with plug-in sigma, one column per event cell.

    Returns W per home x used cell, NaN where a home has no reading in a cell, and sigma in W.
    Columns: 48 levels, z, the cells, then any month x band terms. Priors as in the plan.
    """
    used = np.unique(cell[keep & (cell >= 0)])
    number = -np.ones(cell.max() + 2, dtype=int)
    number[used] = np.arange(len(used))
    c = number[cell]
    n, m = hours.n, len(used)
    rows = np.arange(n)
    on = c >= 0
    parts = [sparse.csr_matrix((np.ones(n), (rows, hours.level)), shape=(n, 48)),
             sparse.csr_matrix(z[:, None]),
             sparse.csr_matrix((np.ones(on.sum()), (rows[on], c[on])), shape=(n, m))]
    q = 0
    if extra is not None:
        parts.append(extra)
        q = extra.shape[1]
    design = sparse.hstack(parts, format="csr")
    p = 49 + m + q
    prior_mean = np.r_[np.full(48, 0.4), 1.0, np.zeros(m + q)]
    prior_prec = 1 / np.r_[np.full(48, 0.5), 1.0, np.full(m, CELL_PRIOR_SD), np.full(q, 0.5)] ** 2
    response = np.full((len(kw), m), np.nan, dtype=np.float32)
    sigma = np.full(len(kw), np.nan, dtype=np.float32)
    for h in range(len(kw)):
        ok = np.isfinite(kw[h]) & keep
        if ok.sum() < 200:
            continue
        x, y = design[ok], kw[h][ok].astype(float)
        gram = (x.T @ x).toarray()
        moment = x.T @ y
        variance = y.var()
        for _ in range(2):
            theta = np.linalg.solve(gram / variance + np.diag(prior_prec),
                                    moment / variance + prior_prec * prior_mean)
            variance = float(np.mean((y - x @ theta) ** 2))
        seen = np.bincount(c[ok & on], minlength=m) > 0
        response[h, seen] = 1000 * theta[49:49 + m][seen]
        sigma[h] = 1000 * np.sqrt(variance)
    return response, sigma, used


def clock_cost(heavy, frame, events):
    """Mean ToU-Std response inside SF runs, price direction, under a naive join and the settled one."""
    base = Baseline(heavy.load_array("diff_w"), frame)
    out = {}
    runs = [e for e in events if e["class"] == "single"]
    for clock in ("wall_start", "gmt_end"):
        firsts = event_slots(runs, clock)
        values = {"gmt": [], "bst": []}
        for run, first in zip(runs, firsts, strict=True):
            season = "bst" if is_bst([run["start_wall"]])[0] else "gmt"
            sign = -1 if run["band"] == "High" else 1
            values[season].append(sign * np.nanmean([base.response(first + k)
                                                     for k in range(run["slots"])]))
        out[clock] = {season: {"mean_w": float(np.nanmean(v)), "runs": int(np.isfinite(v).sum())}
                      for season, v in values.items()}
    return out


def tables(store, heavy, raw, clock, out_dir):
    """data/e91_hours.npz and data/e91_event_table.npz: all a notebook needs from the 2.9 GB grid."""
    if clock not in CLOCKS:
        raise ClockUndecided(f"--clock must be one of {CLOCKS}, decided by pilot P1.")
    grid = heavy.load_array("grid_kwh", mmap=True, check=False)
    homes, tou = heavy.load_array("homes"), heavy.load_array("tou")
    hours = Hours(clock)
    events = clean_events(store.load_json("calendar"))

    std = np.flatnonzero(tou == 0)
    fleet = np.flatnonzero(tou == 1)
    def block_mean(rows):
        total, count = np.zeros(hours.n), np.zeros(hours.n)
        for start in range(0, len(rows), 256):
            kw = hours.read(grid, rows[start:start + 256])
            seen = np.isfinite(kw)
            total += np.where(seen, kw, 0).sum(axis=0, dtype=np.float64)
            count += seen.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return total / count, count
    coverage = np.concatenate([np.isfinite(hours.read(grid, std[s:s + 256])).mean(axis=1)
                               for s in range(0, len(std), 256)])
    panel = std[coverage >= PANEL_COVERAGE]
    n_panel, n_panel_homes = block_mean(panel)

    kw = hours.read(grid, fleet)
    on_2013 = np.isfinite(kw[:, hours.in_2013]).mean(axis=1)
    keep_rows = on_2013 >= FLEET_COVERAGE
    kw_fleet = kw[keep_rows]
    fleet_homes = np.isfinite(kw_fleet).sum(axis=0)

    effect, cell, cells = event_cells(hours, events)
    keep_all = ~out_of_baseline(hours) & np.isfinite(n_panel)
    plan, sigma_plan, used = m1_cells(kw_fleet, hours, deviation(n_panel, hours, keep_all), cell, keep_all)
    keep_2013 = keep_all & hours.in_2013
    local, sigma_local, used_2013 = m1_cells(kw_fleet, hours, deviation(n_panel, hours, keep_2013), cell,
                                             keep_2013, month_band_terms(hours, keep_2013))
    if not np.array_equal(used, used_2013):
        raise ValueError("the two baselines must cover the same event cells")

    # Gate P5 failed, so CM must not carry the estimate. A cell's own column absorbs its mean, but the shape
    # of a CM day inside its cells still reaches the hour levels and beta, by up to 0.7 W on an SF evening
    # cell. So every SF column comes from a fit with the CM hours out, and only CM columns keep them.
    cm_hours = effect >= EFFECTS.index("CM High evening")

    def free_of_cm(response, keep, month_terms):
        clean = keep & ~cm_hours
        extra = month_band_terms(hours, clean) if month_terms else None
        free, free_sigma, free_used = m1_cells(kw_fleet, hours, deviation(n_panel, hours, clean), cell, clean, extra)
        response[:, np.searchsorted(used, free_used)] = free
        return response, free_sigma

    plan, sigma_plan = free_of_cm(plan, keep_all, False)
    local, sigma_local = free_of_cm(local, keep_2013, True)
    key = pd.DataFrame([cells[i] for i in used])
    key["hours"] = np.bincount(cell[keep_all & (cell >= 0)], minlength=len(cells))[used]
    key["season"] = np.where(is_bst(pd.DatetimeIndex(key["day"]) + pd.Timedelta(hours=12)), "bst", "gmt")

    # Pre-trial: the 2013 calendar moved back 364 days, so that every weekday lines up.
    p_effect, p_cell, p_cells = event_cells(hours, events, PLACEBO_SHIFT_DAYS, "2013-06-30", "2013-12-30")
    keep_pre = hours.pre_trial & ~out_of_baseline(hours, PLACEBO_SHIFT_DAYS) & np.isfinite(n_panel)
    kw_pre = kw[np.isfinite(kw[:, hours.pre_trial]).mean(axis=1) >= PLACEBO_COVERAGE]
    pre_homes = homes[fleet[np.isfinite(kw[:, hours.pre_trial]).mean(axis=1) >= PLACEBO_COVERAGE]]
    placebo, sigma_placebo, p_used = m1_cells(kw_pre, hours, deviation(n_panel, hours, keep_pre), p_cell,
                                              keep_pre)
    p_key = pd.DataFrame([p_cells[i] for i in p_used])
    p_key["hours"] = np.bincount(p_cell[keep_pre & (p_cell >= 0)], minlength=len(p_cells))[p_used]
    p_key["season"] = np.where(is_bst(pd.DatetimeIndex(p_key["day"]) + pd.Timedelta(hours=12)), "bst", "gmt")

    evening = hours.pre_trial & (hours.band == 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a home with no reading in a period stays NaN
        fleet_kw = np.nanmean(kw_fleet, axis=0)
        pre_mean = np.nanmean(kw_fleet[:, hours.pre_trial], axis=1) * 1000
        pre_evening = np.nanmean(kw_fleet[:, evening], axis=1) * 1000

    def whole_watts(a):
        return np.where(np.isfinite(a), np.clip(np.round(a), -32767, 32767), -32768).astype(np.int16)

    Path(out_dir).mkdir(exist_ok=True)
    hours_file = Path(out_dir) / "e91_hours.npz"
    table_file = Path(out_dir) / "e91_event_table.npz"
    atomic_write(hours_file, lambda f: np.savez_compressed(
        f, fleet_kw=fleet_kw.astype(np.float32), fleet_homes=fleet_homes.astype(np.int16),
        panel_kw=n_panel.astype(np.float32), panel_homes=n_panel_homes.astype(np.int16),
        effect=effect.astype(np.int8), cell=cell.astype(np.int16),
        out_of_baseline=out_of_baseline(hours), placebo_effect=p_effect.astype(np.int8),
        placebo_out=out_of_baseline(hours, PLACEBO_SHIFT_DAYS), effects=np.array(EFFECTS),
        first_hour=np.array(str(SMALL_START)), clock=np.array(clock), n_panel=np.array(len(panel))))
    def as_arrays(frame, prefix):
        return {f"{prefix}{c}": frame[c].to_numpy(dtype=str if c in ("day", "effect", "season") else int)
                for c in frame.columns}
    columns, p_columns = as_arrays(key, "cell_"), as_arrays(p_key, "placebo_")
    atomic_write(table_file, lambda f: np.savez_compressed(
        f, homes=homes[fleet][keep_rows], local_w=whole_watts(local), plan_w=whole_watts(plan),
        sigma_w=whole_watts(sigma_local), pre_mean_w=whole_watts(pre_mean),
        pre_evening_w=whole_watts(pre_evening), placebo_homes=pre_homes,
        placebo_w=whole_watts(placebo), placebo_sigma_w=whole_watts(sigma_placebo),
        **columns, **p_columns))

    cost = clock_cost(heavy, read_tariffs(raw / "Tariffs.xlsx"), store.load_json("calendar")["events"])
    store.save_json("clock_cost", cost)
    summary = {"homes": int(keep_rows.sum()), "cells": len(key), "placebo_homes": len(pre_homes),
               "placebo_cells": len(p_key), "panel_homes": len(panel),
               "hours_bytes": hours_file.stat().st_size, "table_bytes": table_file.stat().st_size,
               "hours_sha256": sha256_file(hours_file), "table_sha256": sha256_file(table_file)}
    store.save_json("table_summary", summary)
    store.mark_done("tables", **{k: v for k, v in summary.items() if not k.endswith("sha256")})
    return summary


def pilot(store, heavy, raw, clock):
    """P3, then P1, then P2, each stored as small JSON, then the manifest. P2 needs a clock."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frame = read_tariffs(raw / "Tariffs.xlsx")
    table = store.load_json("calendar")
    counts = pilot_counts(heavy, store.load_json("parse_stats"))
    store.save_json("pilot_counts", counts)
    store.mark_done("pilot_counts", passes=counts["passes"])
    result = pilot_clock(heavy, frame, table["events"])
    store.save_json("pilot_clock", result)
    store.mark_done("pilot_clock", passes=result["passes"], peaks=result["peak_offsets"])
    print(json.dumps({k: result[k] for k in ("peak_offsets", "convention_matching",
                                             "gate_margin_se", "passes")}))
    if clock is None:
        print("No --clock: P2 skipped.")
        return
    calendar_check = pilot_calendar(heavy, frame, table["events"], table["disputed_days"], clock)
    store.save_json("pilot_calendar", calendar_check)
    store.mark_done("pilot_calendar", passes=calendar_check["passes"],
                    clear=calendar_check["days_clearing_evening_z3"])
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store.write_manifest(git_hash=git_hash(), versions=library_versions("numpy", "pandas"),
                         data_fingerprint=store.load_json("fetch"), hardware="2 cores, CPU only",
                         started=started, finished=finished, clock=clock,
                         calendar_rule="undecided: P2 failed its gate, see the plan")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fetch", "calendar", "parse", "assemble", "pilot", "table",
                                         "small"))
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="where the zip is kept")
    parser.add_argument("--root", type=Path, help="Store root; the default is results/")
    parser.add_argument("--tariffs", type=Path)
    parser.add_argument("--members", type=Path, help="a directory of sample CSVs, or the zip")
    parser.add_argument("--clock", choices=CLOCKS, help="decided by pilot P1; no default")
    parser.add_argument("--n-tou", type=int, default=200)
    parser.add_argument("--n-std", type=int, default=0, help="the notebook uses no Std meters")
    parser.add_argument("--out", type=Path, default=Path("data/lcl_small.npz"))
    args = parser.parse_args(argv)

    store = Store("E91", root=args.root)  # small JSON, in results/E91 and in git
    heavy = Store("E91", root=args.raw / "store")  # chunks and the grid, never in the repo
    source = args.members or args.raw / "lcl_partitioned.zip"
    if args.step == "fetch":
        checks = fetch(args.raw)
        store.save_json("fetch", checks)
        store.mark_done("fetch", **checks)
        print(json.dumps(checks, indent=1))
    elif args.step == "calendar":
        result = calendar(read_tariffs(args.tariffs or args.raw / "Tariffs.xlsx"))
        store.save_json("calendar", result)
        store.mark_done("calendar", events=len(result["events"]))
        print(result["day_counts"], len(result["events"]), "runs; disputed:",
              result["disputed_days"])
    elif args.step == "parse":
        stats = parse(heavy, source)
        store.save_json("parse_stats", stats)
        store.mark_done("parse", members=len(stats))
        print({k: sum(s[k] for s in stats.values())
               for k in ("rows", "null", "off_grid", "exact_duplicates", "conflicting_slots")})
    elif args.step == "assemble":
        homes, tou = assemble(heavy, len(members(source)))
        print(len(homes), "homes,", int(tou.sum()), "on ToU")
    elif args.step == "pilot":
        pilot(store, heavy, args.raw, args.clock)
    elif args.step == "table":
        print(tables(store, heavy, args.raw, args.clock, args.out.parent))
    else:
        print(small(heavy, args.clock, args.n_tou, args.n_std, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
