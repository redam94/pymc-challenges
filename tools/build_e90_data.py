"""Build the E90 data: the unprocessed German hospitalisation triangle, with its true vintages.

    uv run python tools/build_e90_data.py                  # about 30 MB of downloads, see below
    uv run python tools/build_e90_data.py --benchmark      # also RKI's real-time nowcasts, 17 MB
    uv run python tools/build_e90_data.py --out data       # only once plan question Q3 is answered
    python tools/build_e90_data.py --sample DIR --out DIR  # test on a range-request sample

Why E38's file is not enough. `data/covid_hosp_triangle_de.csv` is the hub's *preprocessed* triangle:
negative corrections were pushed back to earlier delays using reports that came later, so cutting
it at a diagonal leaks the future. The hub's *unprocessed* `COVID-19_hospitalizations.csv` keeps the
negative cells, and its cumulative sum up to a diagonal is the value published that day. Measured on
four dates, 17 locations each, all ages: unprocessed 68 of 68 exact against the hub's daily
snapshots; preprocessed 63 of 68, worst Bremen on 2022-01-20 (published 136, preprocessed 94).

Sources
* Robert Koch-Institut, "COVID-19-Hospitalisierungen in Deutschland", CC BY 4.0: the underlying data.
  github.com/robert-koch-institut/COVID-19-Hospitalisierungen_in_Deutschland
* KIT, hospitalization-nowcast-hub, reformatted triangle and population file. The hub repository has
  no licence file; see plans/large/E90.md before publishing anything taken only from the hub.
* RKI-weekly_report in the hub's data-processed (--benchmark): RKI's own nowcast as submitted in
  real time. Kept in the cache, not `data/`, until its redistribution terms are settled.
* RKI's dated `Archiv/` CSVs, as captured by the Internet Archive in 2021-22: the only source found
  that holds published counts and incidences together, used to check the hub's populations (P2).
* Statistisches Bundesamt, "Bundeslaender mit Hauptstaedten nach Flaeche, Bevoelkerung und
  Bevoelkerungsdichte am 31.12.2020": the all-age populations, copied into `DESTATIS_2020`.

Downloads go to ~/.cache/pymc-challenges/e90/, in their own directories, and are read as data only.
Snapshots are 2 MB each, but each starts with the rows of its own date, so a range request for the
first few KB gets all 119 cells: the 16 snapshots cost a few dozen KB rather than 32 MB.

What we write, to `--out` (default the cache, because Q3 in the plan has not been answered)
* `covid_hosp_vintages_de.npz` - `triangle` (location, age, date, delay) int32 with `MISSING`
  for unknown cells, delays 0..80 then ">80"; `locations`, `ages`, `dates`; `population`
  (location, age). The all-age column is Destatis 31.12.2020, which is the denominator RKI used; the
  age columns are the hub's file, which is not (pilot P2, see plans/large/E90.md).
* `covid_hosp_vintages_de.json` - sources, SHA-256 of each input, build date and check results.
* `results/E90/data.json`, `results/E90/pilot.json` - the same checks and the P0-P2 verdicts, written
  even when a check fails, so a refusal leaves a record.
Registering the file in `src/pymc_challenges/data.py` is a separate step.

Nothing here reads a fitted result; this is arithmetic on published counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import urllib.error
import urllib.request
from itertools import product
from pathlib import Path

import numpy as np

from pymc_challenges.bigrun import Store, atomic_write, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e90"
HUB = "https://raw.githubusercontent.com/KITmetricslab/hospitalization-nowcast-hub/main"
RAW_URL = f"{HUB}/data-truth/COVID-19/COVID-19_hospitalizations.csv"
POP_URL = f"{HUB}/nowcast_viz_de/plot_data/population_sizes.csv"
SNAPSHOT_URL = f"{HUB}/data-truth/COVID-19/rolling-sum/{{}}_COVID-19_hospitalization.csv"
RKI_URL = f"{HUB}/data-processed/RKI-weekly_report/{{0}}-RKI-weekly_report.csv"
RKI_ARCHIVE = ("https://raw.githubusercontent.com/robert-koch-institut/"
               "COVID-19-Hospitalisierungen_in_Deutschland/master/Archiv/"
               "{0}_Deutschland_COVID-19-Hospitalisierungen.csv")
WAYBACK = "https://web.archive.org/web/{0}id_/{1}"

LOCATIONS = ("DE", "DE-BB", "DE-BE", "DE-BW", "DE-BY", "DE-HB", "DE-HE", "DE-HH", "DE-MV", "DE-NI",
             "DE-NW", "DE-RP", "DE-SH", "DE-SL", "DE-SN", "DE-ST", "DE-TH")
AGES = ("00+", "00-04", "05-14", "15-34", "35-59", "60-79", "80+")
# Earliest fit starts 70 days before 2021-09-15; the last as-of date needs 80 further days of truth.
FIRST, LAST = "2021-07-01", "2022-07-20"
N_DELAYS = 82  # 0..80 and >80
MISSING = np.iinfo(np.int32).min
CHECK_DATES = ("2021-12-15", "2022-01-20", "2022-02-10", "2022-03-15")
# P1, chosen before any of them was compared: Christmas, New Year, ordinary weekends.
HARD_DATES = ("2021-12-24", "2021-12-25", "2021-12-26", "2021-12-27",
              "2021-12-31", "2022-01-01", "2022-01-02", "2022-01-03",
              "2021-10-16", "2021-11-21", "2022-02-19", "2022-03-27")
# P2: RKI file date -> Internet Archive capture. Each file's rows for its own date are as published.
ARCHIVED = {"2021-11-20": "20211207114326", "2022-01-30": "20220130094555",
            "2022-02-08": "20220208221327"}
STATE_IDS = dict(zip(("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
                      "13", "14", "15", "16"),
                     ("DE", "DE-SH", "DE-HH", "DE-NI", "DE-HB", "DE-NW", "DE-HE", "DE-RP", "DE-BW",
                      "DE-BY", "DE-SL", "DE-BE", "DE-BB", "DE-MV", "DE-SN", "DE-ST", "DE-TH")))
# Copied from Destatis 02-bundeslaender.xlsx, Internet Archive capture 20211028170955. The build checks
# them against the range each RKI archive file allows, so a wrong figure here fails P2.
DESTATIS_2020 = {"DE": 83155031, "DE-BB": 2531071, "DE-BE": 3664088, "DE-BW": 11103043,
                 "DE-BY": 13140183, "DE-HB": 680130, "DE-HE": 6293154, "DE-HH": 1852478,
                 "DE-MV": 1610774, "DE-NI": 8003421, "DE-NW": 17925570, "DE-RP": 4098391,
                 "DE-SH": 2910875, "DE-SL": 983991, "DE-SN": 4056941, "DE-ST": 2180684,
                 "DE-TH": 2120237}
EVALUATION = ("2021-09-15", "2022-04-29")
SNAPSHOT_BYTES = 8192  # 119 rows of about 24 bytes
INCIDENCE_TOLERANCE = 0.01  # per 100,000, gate P2
FETCHED = {}  # bytes on the wire this run, by file name, printed


def fetch(url, path):
    """Download once; a file already on disk is reused."""
    path = Path(path)
    if not path.exists():
        request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
            packed = response.headers.get("Content-Encoding") == "gzip"
        FETCHED[path.name] = len(body)
        text = gzip.decompress(body) if packed else body
        atomic_write(path, lambda f: f.write(text))
    return path


def fetch_head(url, path, n=SNAPSHOT_BYTES):
    """Download the first `n` bytes once, ending on a whole line."""
    path = Path(path)
    if not path.exists():
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read(n)
        FETCHED[path.name] = len(body)
        whole = body[: body.rfind(b"\n") + 1]
        atomic_write(path, lambda f: f.write(whole))
    return path


def date_axis():
    days = np.arange(np.datetime64(FIRST), np.datetime64(LAST) + np.timedelta64(1, "D"))
    return days, {str(d): i for i, d in enumerate(days)}


def parse_rows(lines, triangle, index):
    """Fill `triangle` from CSV lines; partial lines from a range request simply fail the checks."""
    for line in lines:
        cells = line.rstrip("\n").split(",")
        if len(cells) != 3 + N_DELAYS or cells[1] not in LOCATIONS or cells[2] not in AGES:
            continue
        t = index.get(cells[0])
        if t is None:
            continue
        values = [int(float(c)) if c else MISSING for c in cells[3:]]
        triangle[LOCATIONS.index(cells[1]), AGES.index(cells[2]), t] = values


def read_triangle(source):
    """`source` is the whole CSV, or a directory of partial files from range requests."""
    days, index = date_axis()
    triangle = np.full((len(LOCATIONS), len(AGES), len(days), N_DELAYS), MISSING, np.int32)
    source = Path(source)
    for path in sorted(source.glob("*.part")) if source.is_dir() else [source]:
        with open(path, encoding="utf-8", errors="replace") as f:
            parse_rows(f, triangle, index)
    return triangle, days


def vintage_row_totals(triangle_2d, asof):
    """Each row's total as known on day `asof`: cumulative sum up to the diagonal, negatives kept.

    `triangle_2d` is (date, delay) for one stratum. Row t knows delays 0..asof-t. The >80 column is
    the hub's latest file, so it never enters a vintage.
    """
    known = np.where(triangle_2d[: asof + 1, :81] == MISSING, 0, triangle_2d[: asof + 1, :81])
    delays = np.arange(81)
    mask = delays[None, :] <= (asof - np.arange(asof + 1))[:, None]
    return (known * mask).sum(axis=1)


def frozen_7d(triangle_2d, asof):
    """The 7-day count published on day `asof`: the frozen value."""
    return int(vintage_row_totals(triangle_2d, asof)[asof - 6 : asof + 1].sum())


def final_7d(triangle_2d, asof, max_delay=80):
    """The same 7 days reported within `max_delay` days: the truth used for scoring."""
    cells = triangle_2d[asof - 6 : asof + 1, : max_delay + 1]
    return int(np.where(cells == MISSING, 0, cells).sum())


def asof_nonnegative(triangle_2d, asof):
    """Non-negative counts as known on `asof`, using nothing reported after it.

    This is the hub's rule (walk from the longest delay down, carrying each negative to the previous
    delay), applied to the cut triangle instead of the finished one, so a correction made after
    `asof` cannot move a cell. Unknown cells come back as -1. Returns the counts and, per row, any
    negative left over after delay 0 (a row whose running total was below zero).
    """
    rows = asof + 1
    out = np.full((rows, 81), -1, np.int64)
    leftover = np.zeros(rows, np.int64)
    for t in range(rows):
        top = min(asof - t, 80)
        cells = np.where(triangle_2d[t, : top + 1] == MISSING, 0, triangle_2d[t, : top + 1]).astype(np.int64)
        carry = 0
        for k in range(top, -1, -1):
            value = cells[k] + carry
            carry = min(value, 0)
            cells[k] = max(value, 0)
        out[t, : top + 1] = cells
        leftover[t] = carry
    return out, leftover


def read_population(path):
    population = np.zeros((len(LOCATIONS), len(AGES)))
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            _, location, age, value = (c.strip('"') for c in line.strip().split(","))
            if location in LOCATIONS and age in AGES:
                population[LOCATIONS.index(location), AGES.index(age)] = float(value)
    return population


def read_snapshot(path, date):
    """{(location, age): published 7-day value} for `date` from one daily hub snapshot."""
    published = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            cells = line.strip().split(",")
            if cells[0] == date and cells[3]:
                published[cells[1], cells[2]] = int(float(cells[3]))
    return published


def check_vintages(triangle, days, snapshots):
    """Compare frozen values with the hub's daily snapshots, cell by cell.

    A cell that cannot be compared is listed as absent, never skipped: `expected` is every location
    x age x date, so a short file cannot pass.
    """
    index = {str(d): i for i, d in enumerate(days)}
    by_date, absent, failures = {}, [], []
    for date, path in snapshots.items():
        published = read_snapshot(path, date)
        t = index[date]
        exact = checked = 0
        for location, age in product(LOCATIONS, AGES):
            block = triangle[LOCATIONS.index(location), AGES.index(age)]
            value = published.get((location, age))
            if value is None or (block[t - 6 : t + 1, 0] == MISSING).any():
                absent.append([date, location, age])
                continue
            rebuilt = frozen_7d(block, t)
            checked += 1
            exact += rebuilt == value
            if rebuilt != value:
                failures.append({"date": date, "location": location, "age": age,
                                 "published": value, "rebuilt": rebuilt})
        by_date[date] = {"checked": checked, "exact": exact}
    all_age = [f for f in failures if f["age"] == "00+"]
    return {"expected": len(snapshots) * len(LOCATIONS) * len(AGES),
            "checked": sum(d["checked"] for d in by_date.values()),
            "exact": sum(d["exact"] for d in by_date.values()),
            "all_age_failures": len(all_age),
            "worst_gap": max((abs(f["published"] - f["rebuilt"]) for f in failures), default=0),
            "by_date": by_date, "absent": absent, "failures": failures}


def check_hard_dates(triangle, days, snapshots, missing):
    """P1: every hard date must be a real publication, and must match the rebuilt vintage.

    A date counts as filled in if its snapshot lacks rows of that date, or if nothing at all was
    reported that day: no all-age cell on any location's diagonal changed.
    """
    index = {str(d): i for i, d in enumerate(days)}
    result = check_vintages(triangle, days, snapshots)
    filled, unchanged = list(missing), {}
    for date in snapshots:
        t = index[date]
        rows = np.arange(t - 80, t + 1)
        cells = triangle[:, 0][:, rows, t - rows]  # (location, row): what arrived on this day
        reported = int(np.abs(np.where(cells == MISSING, 0, cells)).sum())
        same = sum(frozen_7d(triangle[i, 0], t) == frozen_7d(triangle[i, 0], t - 1)
                   for i in range(len(LOCATIONS)))
        unchanged[date] = {"all_age_cases_on_diagonal": reported,
                           "locations_same_as_day_before": same}
        if reported == 0 or not result["by_date"][date]["checked"]:
            filled.append(date)
    result.update(filled_in=filled, publication=unchanged, missing_files=list(missing))
    result["exact_dates"] = [d for d, v in result["by_date"].items()
                             if v["checked"] == len(LOCATIONS) * len(AGES) == v["exact"]]
    return result


def read_archive(path, date):
    """{(location, age): (count, incidence)} for rows of `date` in one RKI archive file."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            cells = line.strip().split(",")
            if len(cells) == 6 and cells[0] == date and cells[2] in STATE_IDS and cells[3] in AGES:
                rows[STATE_IDS[cells[2]], cells[3]] = (int(cells[4]), float(cells[5]))
    return rows


def check_population(archives, triangle, days, population):
    """P2: published incidence against published count over the hub's population.

    Published incidences have 2 decimals, so rounding alone can give a gap of 0.005. The same files
    are a separate source for the frozen counts, so those are compared too.
    """
    index = {str(d): i for i, d in enumerate(days)}
    gaps, counts_differ = {"00+": [], "strata": []}, []
    for date, path in archives.items():
        rows = read_archive(path, date)
        for (location, age), (count, incidence) in rows.items():
            l, a = LOCATIONS.index(location), AGES.index(age)
            gap = incidence - count / population[l, a] * 1e5
            kind = "00+" if age == "00+" else "strata"
            gaps[kind].append({"date": date, "location": location, "age": age, "count": count,
                               "incidence": incidence, "gap": round(gap, 4)})
            rebuilt = frozen_7d(triangle[l, a], index[date])
            if rebuilt != count:
                counts_differ.append({"date": date, "location": location, "age": age,
                                      "published": count, "rebuilt": rebuilt})
    largest = lambda rows: max((abs(r["gap"]) for r in rows), default=None)
    return {"dates": list(archives), "all_age_values": len(gaps["00+"]),
            "strata_values": len(gaps["strata"]), "largest_gap_all_age": largest(gaps["00+"]),
            "largest_gap_strata": largest(gaps["strata"]),
            "all_age_over_tolerance": [r for r in gaps["00+"]
                                       if abs(r["gap"]) > INCIDENCE_TOLERANCE],
            "strata_over_tolerance": sum(abs(r["gap"]) > INCIDENCE_TOLERANCE
                                         for r in gaps["strata"]),
            "germany": [r for r in gaps["00+"] if r["location"] == "DE"],
            "archive_counts_compared": len(gaps["00+"]) + len(gaps["strata"]),
            "archive_counts_differ": counts_differ}


def population_bounds(path, year):
    """{location: (low, high)}: the populations that fit every all-age row of `year` in one RKI file.

    Each row has a count and an incidence rounded to 2 decimals, so it allows a narrow range.
    """
    low, high = {}, {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            cells = line.strip().split(",")
            if (len(cells) != 6 or not cells[0].startswith(year) or cells[2] not in STATE_IDS
                    or cells[3] != "00+" or "NA" in cells[4:]):
                continue
            count, incidence = int(cells[4]), float(cells[5])
            if count and incidence > 0.005:
                location = STATE_IDS[cells[2]]
                low[location] = max(low.get(location, 0.0), count * 1e5 / (incidence + 0.005))
                high[location] = min(high.get(location, np.inf), count * 1e5 / (incidence - 0.005))
    return {location: (low[location], high[location]) for location in low}


def check_destatis(archives, hub_all_age):
    """The no-go action of P2: do the Destatis figures reproduce what RKI published?"""
    bounds = {f"{date} rows of {year}": population_bounds(path, year)
              for date, path in archives.items() for year in {"2021", date[:4]}}
    outside = [[name, location, *map(round, range_)] for name, ranges in bounds.items()
               for location, range_ in ranges.items()
               if not range_[0] <= DESTATIS_2020[location] <= range_[1]]
    population = np.zeros((len(LOCATIONS), len(AGES)))
    population[:, 0] = [DESTATIS_2020[location] for location in LOCATIONS]
    largest = 0.0
    for date, path in archives.items():
        for (location, age), (count, incidence) in read_archive(path, date).items():
            if age == "00+":
                value = count / population[LOCATIONS.index(location), 0] * 1e5
                largest = max(largest, abs(incidence - value))
    return {"go": not outside and largest <= INCIDENCE_TOLERANCE,
            "source": "Destatis 02-bundeslaender.xlsx, 31.12.2020, capture 20211028170955",
            "ranges_checked": sum(len(r) for r in bounds.values()), "outside_range": outside,
            "largest_gap_all_age": round(largest, 4),
            "hub_minus_destatis_percent": {
                location: round(100 * (hub / DESTATIS_2020[location] - 1), 3)
                for location, hub in zip(LOCATIONS, hub_all_age)}}


def threshold_sensitivity(triangle, days, hub, destatis, taus=(3, 6, 9)):
    """What the population choice does to the decision: section 4's counts under each denominator.

    Nothing is fitted. `hub` must reproduce the plan's section 4 table, which is the control.
    """
    index = {str(d): i for i, d in enumerate(days)}
    span = range(index[EVALUATION[0]], index[EVALUATION[1]] + 1)
    frozen = np.array([[frozen_7d(triangle[l, 0], t) for t in span] for l in range(1, 17)], float)
    final = np.array([[final_7d(triangle[l, 0], t) for t in span] for l in range(1, 17)], float)

    def crossings(above):
        n = above.shape[1]
        return sum(above[s, t : t + 3].all() and not above[s, t - 3 : t].any()
                   for s in range(16) for t in range(3, n - 2))

    out = {}
    for tau in taus:
        row, flags = {}, {}
        for name, population in (("hub", hub), ("destatis", destatis)):
            frozen_up = frozen * 1e5 / population[1:17, 0, None] >= tau
            final_up = final * 1e5 / population[1:17, 0, None] >= tau
            flags[name] = frozen_up, final_up
            row[name] = {"frozen_fp": int((frozen_up & ~final_up).sum()),
                         "frozen_fn": int((~frozen_up & final_up).sum()),
                         "crossings": int(crossings(final_up))}
        row["state_days_frozen_moved"] = int((flags["hub"][0] != flags["destatis"][0]).sum())
        row["state_days_final_moved"] = int((flags["hub"][1] != flags["destatis"][1]).sum())
        out[f"tau_{tau}"] = row
    out["state_days"] = int(frozen.size)
    return out


def check_asof(triangle, days, first=EVALUATION[0], last=EVALUATION[1]):
    """The as-of counts must add up to the published value; report where they cannot.

    Also counts rows with a negative left after delay 0, over the whole triangle known on each day
    and within the 7 days that make the published value.
    """
    index = {str(d): i for i, d in enumerate(days)}
    checked = differ = rows = leftover_rows = leftover_in_7d = 0
    for loc in range(len(LOCATIONS)):
        block = triangle[loc, 0]
        for asof in range(index[first], index[last] + 1, 7):
            cells, leftover = asof_nonnegative(block, asof)
            kept = np.where(cells < 0, 0, cells)[asof - 6 : asof + 1].sum()
            checked += 1
            differ += int(kept) != frozen_7d(block, asof)
            rows += leftover.size
            leftover_rows += int((leftover < 0).sum())
            leftover_in_7d += int((leftover[asof - 6 : asof + 1] < 0).sum())
    return {"checked": checked, "differ": differ, "rows": rows, "rows_negative_left": leftover_rows,
            "rows_negative_left_in_7d": leftover_in_7d}


def describe(triangle, days):
    """What the triangle holds, in the window used and in the cells that exist."""
    known = triangle != MISSING
    return {"shape": list(triangle.shape), "first": str(days[0]), "last": str(days[-1]),
            "rows_without_data": int((~known.any(axis=-1)).sum()),
            "cells_unknown": int((~known).sum()),
            "negative_cells": int((triangle[known] < 0).sum())}


def write_outputs(out, triangle, days, population, provenance):
    out = Path(out)

    def save(f):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, triangle=triangle, dates=days.astype("datetime64[D]").astype(str),
                            locations=np.array(LOCATIONS), ages=np.array(AGES), population=population)
        f.write(buffer.getvalue())

    atomic_write(out / "covid_hosp_vintages_de.npz", save)
    text = json.dumps(provenance, indent=1).encode()
    atomic_write(out / "covid_hosp_vintages_de.json", lambda f: f.write(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", help="directory with *.part, population and roll_<date>.csv files")
    parser.add_argument("--out", default=CACHE / "out")
    parser.add_argument("--benchmark", action="store_true", help="fetch RKI's real-time nowcasts")
    args = parser.parse_args()

    work = Path(args.sample) if args.sample else CACHE
    hard, missing, archives, archive_error = {}, [], {}, None
    if args.sample:
        source, pop_path = work / "raw", work / "hub_pop.csv"
        snapshots = {d: work / f"roll_{d}.csv" for d in CHECK_DATES}
    else:
        source = fetch(RAW_URL, work / "hub" / "COVID-19_hospitalizations.csv")
        pop_path = fetch(POP_URL, work / "hub" / "population_sizes.csv")
        snapshots = {d: fetch_head(SNAPSHOT_URL.format(d), work / "snapshots" / f"{d}.csv")
                     for d in CHECK_DATES}
        for d in HARD_DATES:
            try:
                hard[d] = fetch_head(SNAPSHOT_URL.format(d), work / "snapshots" / f"{d}.csv")
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                missing.append(d)
        try:
            for d, capture in ARCHIVED.items():
                url = WAYBACK.format(capture, RKI_ARCHIVE.format(d))
                archives[d] = fetch(url, work / "archive" / f"{d}.csv")
        except OSError as error:  # the Internet Archive goes offline at times: P2 not run, not passed
            archive_error = repr(error)

    triangle, days = read_triangle(source)
    population = read_population(pop_path)
    if (population <= 0).any():
        raise RuntimeError("a location x age group has no population")

    vintages = check_vintages(triangle, days, snapshots)
    asof = check_asof(triangle, days)
    print("vintages against snapshots:",
          {k: vintages[k] for k in ("expected", "checked", "exact", "worst_gap")})
    print("as-of non-negative counts against vintages:", asof)
    p0_go = vintages["exact"] == vintages["checked"] == vintages["expected"] and not asof["differ"]

    p1 = check_hard_dates(triangle, days, hard, missing) if hard else None
    p2 = None
    if len(archives) == len(ARCHIVED):
        p2 = check_population(archives, triangle, days, population)
        destatis = population.copy()
        destatis[:, 0] = [DESTATIS_2020[location] for location in LOCATIONS]
        p2["destatis"] = check_destatis(archives, population[:, 0])
        p2["sensitivity"] = threshold_sensitivity(triangle, days, population, destatis)
        print("P2 largest gap, hub:", p2["largest_gap_all_age"],
              "| Destatis:", p2["destatis"]["largest_gap_all_age"],
              "| outside range:", p2["destatis"]["outside_range"])
        print("sensitivity:", p2["sensitivity"])
        population = destatis  # what the npz carries: see the docstring
    if not args.sample:
        p1_go = bool(p1) and not (p1["failures"] or p1["absent"] or p1["filled_in"])
        # A capture that holds no rows (an error page) is not a pass.
        p2_compared = p2 is not None and p2["all_age_values"] == len(ARCHIVED) * len(LOCATIONS)
        p2_go = (not p2["all_age_over_tolerance"]) if p2_compared else None
        results = {"P0": {"go": p0_go, "vintages": vintages, "asof": asof},
                   "P1": {"go": p1_go, "dates": list(HARD_DATES), **(p1 or {})},
                   "P2": {"go": p2_go, "tolerance": INCIDENCE_TOLERANCE, "not_run": archive_error,
                          **(p2 or {})}}
        inputs = [source, pop_path, *snapshots.values(), *hard.values(), *archives.values()]
        store = Store("E90")
        store.save_json("pilot", results)
        store.save_json("data", {"triangle": describe(triangle, days), "window": [FIRST, LAST],
                                 "sha256": {p.name: sha256_file(p) for p in map(Path, inputs)}})
        print("P0 go:", p0_go, "| P1 go:", p1_go, "| P2 go:", p2_go)
        print("bytes fetched this run (0 when all were cached):", sum(FETCHED.values()))
    if not p0_go:  # nothing is written; the record above says why
        raise RuntimeError(f"P0 failed: {vintages['exact']} of {vintages['expected']} exact, "
                           f"as-of {asof['differ']} differ")

    if args.benchmark:  # one file per date, roughly 110 KB each
        for day in np.arange(np.datetime64("2021-12-01"), np.datetime64("2022-04-30")):
            name = str(day)
            fetch(RKI_URL.format(name), work / "rki" / f"{name}.csv")

    inputs = [Path(p) for p in ([pop_path] if source.is_dir() else [source, pop_path])]
    write_outputs(args.out, triangle, days, population, {
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sample_only": bool(args.sample),
        "sources": {"triangle": RAW_URL, "population": POP_URL, "data_owner": "Robert Koch-Institut"},
        "licence": "RKI data CC BY 4.0; hub reformatting carries no licence file",
        "sha256": {p.name: sha256_file(p) for p in inputs},
        "checks": {"vintages": vintages, "asof": asof},
        "window": [FIRST, LAST],
    })
    if not args.sample:
        Store("E90").mark_done("data", checked=vintages["checked"], exact=vintages["exact"])


if __name__ == "__main__":
    main()
