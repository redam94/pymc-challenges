"""Build the E90 data: the unprocessed German hospitalisation triangle, with its true vintages.

    uv run python tools/build_e90_data.py                  # about 30 MB of downloads
    uv run python tools/build_e90_data.py --benchmark      # also RKI's real-time nowcasts, 17 MB
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
  real time. Kept in `.scratch/`, not `data/`, until its redistribution terms are settled.

What we write
* `data/covid_hosp_vintages_de.npz` - `triangle` (location, age, date, delay) int32 with `MISSING`
  for unknown cells, delays 0..80 then ">80"; `locations`, `ages`, `dates`; `population`
  (location, age) from the hub's file, not RKI's current file (see plans/large/E90.md).
* `data/covid_hosp_vintages_de.json` - sources, SHA-256 of each input, build date and check results.
Registering the file in `src/pymc_challenges/data.py` is a separate step.

Nothing here reads a fitted result; this is arithmetic on published counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from pymc_challenges.bigrun import atomic_write, sha256_file

ROOT = Path(__file__).resolve().parents[1]
HUB = "https://raw.githubusercontent.com/KITmetricslab/hospitalization-nowcast-hub/main"
RAW_URL = f"{HUB}/data-truth/COVID-19/COVID-19_hospitalizations.csv"
POP_URL = f"{HUB}/nowcast_viz_de/plot_data/population_sizes.csv"
SNAPSHOT_URL = f"{HUB}/data-truth/COVID-19/rolling-sum/{{}}_COVID-19_hospitalization.csv"
RKI_URL = f"{HUB}/data-processed/RKI-weekly_report/{{0}}-RKI-weekly_report.csv"

LOCATIONS = ("DE", "DE-BB", "DE-BE", "DE-BW", "DE-BY", "DE-HB", "DE-HE", "DE-HH", "DE-MV", "DE-NI",
             "DE-NW", "DE-RP", "DE-SH", "DE-SL", "DE-SN", "DE-ST", "DE-TH")
AGES = ("00+", "00-04", "05-14", "15-34", "35-59", "60-79", "80+")
# Earliest fit starts 70 days before 2021-09-15; the last as-of date needs 80 further days of truth.
FIRST, LAST = "2021-07-01", "2022-07-20"
N_DELAYS = 82  # 0..80 and >80
MISSING = np.iinfo(np.int32).min
CHECK_DATES = ("2021-12-15", "2022-01-20", "2022-02-10", "2022-03-15")


def fetch(url, path):
    """Download once; a file already on disk is reused."""
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read()
        atomic_write(path, lambda f: f.write(body))
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
    """Compare frozen values with the hub's daily snapshots. Refuses to pass silently on no data."""
    index = {str(d): i for i, d in enumerate(days)}
    checked = exact = worst = 0
    for date, path in snapshots.items():
        published = read_snapshot(path, date)
        for (location, age), value in published.items():
            if location not in LOCATIONS or age not in AGES:
                continue
            block = triangle[LOCATIONS.index(location), AGES.index(age)]
            if (block[index[date] - 6 : index[date] + 1, 0] == MISSING).any():
                continue
            gap = abs(frozen_7d(block, index[date]) - value)
            checked += 1
            exact += gap == 0
            worst = max(worst, gap)
    if not checked:
        raise RuntimeError("no snapshot cell could be compared")
    return {"checked": checked, "exact": exact, "worst_gap": worst}


def check_asof(triangle, days, first="2021-09-15", last="2022-04-29"):
    """The as-of counts must add up to the published value; report where they cannot."""
    index = {str(d): i for i, d in enumerate(days)}
    checked = differ = 0
    for loc in range(len(LOCATIONS)):
        block = triangle[loc, 0]
        for asof in range(index[first], index[last] + 1, 7):
            cells, _ = asof_nonnegative(block, asof)
            kept = np.where(cells < 0, 0, cells)[asof - 6 : asof + 1].sum()
            checked += 1
            differ += int(kept) != frozen_7d(block, asof)
    return {"checked": checked, "differ": differ}


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
    parser.add_argument("--out", default=ROOT / "data")
    parser.add_argument("--benchmark", action="store_true", help="fetch RKI's real-time nowcasts")
    args = parser.parse_args()

    work = Path(args.sample) if args.sample else ROOT / ".scratch" / "e90"
    if args.sample:
        source, pop_path = work / "raw", work / "hub_pop.csv"
        snapshots = {d: work / f"roll_{d}.csv" for d in CHECK_DATES}
    else:
        source = fetch(RAW_URL, work / "COVID-19_hospitalizations.csv")
        pop_path = fetch(POP_URL, work / "population_sizes.csv")
        snapshots = {d: fetch(SNAPSHOT_URL.format(d), work / f"roll_{d}.csv") for d in CHECK_DATES}

    triangle, days = read_triangle(source)
    population = read_population(pop_path)
    if (population <= 0).any():
        raise RuntimeError("a location x age group has no population")

    vintages = check_vintages(triangle, days, snapshots)
    if vintages["exact"] != vintages["checked"]:
        raise RuntimeError(f"reconstructed vintages disagree with the snapshots: {vintages}")
    asof = check_asof(triangle, days)
    print("vintages against snapshots:", vintages)
    print("as-of non-negative counts against vintages:", asof)

    if args.benchmark:  # one file per date, roughly 110 KB each
        for day in np.arange(np.datetime64("2021-12-01"), np.datetime64("2022-04-30")):
            name = str(day)
            fetch(RKI_URL.format(name), ROOT / ".scratch" / "e90" / "rki" / f"{name}.csv")

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


if __name__ == "__main__":
    main()
