"""Build the E122 datasets: MBTA bus route 111 (Woodlawn - Haymarket), September to November 2024.

    python tools/build_e122_buses.py   # fetch (cached), derive, check, write data/e122_*.csv.gz

Two tables, both from the open-data portal of the Massachusetts Bay Transportation Authority
(https://mbta-massdot.opendata.arcgis.com), both marked "CC0" in the licence field of their items:

1. "MBTA Bus Arrival Departure Times 2024"
   https://www.arcgis.com/sharing/rest/content/items/96c77138c3144906bce93d0257531b6a/data
   A 315 MB zip of twelve monthly CSV files (about 245 MB each, every bus route). "This file contains
   the arrival and departure events for buses ... Due to data collection issues, data is not
   guaranteed to be complete for any stop or date."
2. "MBTA Bus Ridership by Trip, Season, Route, Line, and Stop"
   https://www.arcgis.com/sharing/rest/content/items/8daf4a33925a4df59183f860826d29ee/data
   A 73 MB zip with one CSV per season: average boardings, alightings and load per scheduled trip and
   stop, from the automatic passenger counters. We take Fall 2024.
Both items carry the same notice: "MassDOT/MBTA shall not be held liable for any errors in this data."

Neither archive is downloaded whole. The members we need are read through HTTP range requests,
decompressed as a stream and filtered to route 111; the filtered rows are cached, untouched, in
~/.cache/pymc-challenges/e122 (about 6 MB, not committed).

What the first table is, and is not. A row is one bus passing one TIMEPOINT (ten of the route's
stops or checkpoints). There is one clock reading per row, ``actual``: "The time the trip actually
departed the timepoint" (the arrival, at the last one). So the time a bus stands at a stop is not
recorded separately, stops between timepoints are not recorded at all, and there are no passenger
counts per bus. Clock times come stamped ``1900-01-01T10:06:00Z``: a time of day on a dummy date,
running past 24 h for trips late at night. Despite the Z they are not UTC: the first scheduled inbound
trip of a weekday is stamped 09:36 before and after the clocks changed on 3 November 2024, and the
ridership table gives that trip as 4:36. The stamps are local time plus five hours all year, so the
builder subtracts five hours, and the check below holds it to the ridership table.

e122_bus_events.csv.gz, one row per bus and timepoint, in order of date, direction, trip, timepoint:
  date          service date, YYYY-MM-DD
  direction     Inbound (Woodlawn to Haymarket) or Outbound
  trip          the agency's id of the one-way trip (half_trip_id)
  seq           order of the timepoint in the trip, 1 to 10
  timepoint     code of the timepoint (woodc, garfi, smore, wacry, belsq, thche, tobin, ruthf, cacom,
                hayms inbound; bwybc replaces thche outbound)
  stop_id       GTFS stop id of the timepoint
  standard      Headway or Schedule: the standard the agency judges the trip by
  sched_s       scheduled time, seconds after local midnight (above 86,400 after midnight)
  actual_s      actual departure (arrival at seq 10), same clock; empty if not recorded
  sched_headway scheduled gap to the previous trip at this timepoint, seconds (Headway trips only)
  headway       the agency's actual gap to the previous trip, seconds (empty at the last timepoint)

e122_bus_ridership.csv.gz, one row per scheduled weekday trip and stop, Fall 2024:
  direction     Inbound or Outbound (direction_id 1 and 0 of the source)
  trip_start    scheduled start of the trip, seconds after midnight
  stop_seq, stop_id, stop_name   the stop and its order in the trip
  boardings, alightings, load    averages per trip over the season (load: on board when leaving)
  n_trips       number of counted trips behind the average
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_EVENTS = ROOT / "data" / "e122_bus_events.csv.gz"
OUT_RIDERS = ROOT / "data" / "e122_bus_ridership.csv.gz"
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e122"
ITEM = "https://www.arcgis.com/sharing/rest/content/items/{}/data"
EVENTS_ITEM, RIDERS_ITEM = "96c77138c3144906bce93d0257531b6a", "8daf4a33925a4df59183f860826d29ee"
MONTHS = ("2024-09", "2024-10", "2024-11")
ROUTE = "111"
UA = {"User-Agent": "pymc-challenges"}
N_RAW = {"2024-09": 106_230, "2024-10": 113_850, "2024-11": 106_170, "riders": 20_446}   # route 111 rows


class HttpFile(io.RawIOBase):
    """A remote file read through HTTP range requests: enough for ``zipfile`` to open one member."""

    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=120) as r:
            self.size, self.url = int(r.headers["Content-Length"]), r.geturl()
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rng = {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers={**UA, **rng}), timeout=300) as r:
            chunk = r.read()
        b[: len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def fetch(item: str, member_part: str, route_ids: set[str], key: str) -> Path:
    """Stream one CSV member of a remote zip and cache the rows of our route, as they are."""
    out = CACHE / f"raw_{key}.csv.gz"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(io.BufferedReader(HttpFile(ITEM.format(item)), buffer_size=4 << 20))
    (name,) = [i.filename for i in archive.infolist() if member_part in i.filename]
    buf = io.BytesIO()
    with archive.open(name) as raw, gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        reader, writer = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig")), csv.writer(text)
        head = next(reader)
        writer.writerow(head)
        col = head.index("route_id")
        for row in reader:
            if row[col] in route_ids:
                writer.writerow(row)
        text.flush()
    out.write_bytes(buf.getvalue())
    return out


def local_seconds(stamp: pd.Series) -> np.ndarray:
    """Time of day stamped on a dummy date, five hours ahead of local time -> seconds after local midnight."""
    ahead = (pd.to_datetime(stamp, format="%Y-%m-%dT%H:%M:%SZ") - pd.Timestamp("1900-01-01")).dt.total_seconds()
    return (ahead - 5 * 3600.0).to_numpy()


def build_events() -> pd.DataFrame:
    parts = []
    for month in MONTHS:
        d = pd.read_csv(fetch(EVENTS_ITEM, f"_{month}.csv", {ROUTE}, month), dtype=str)
        assert len(d) == N_RAW[month], (month, len(d))
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({
        "date": d["service_date"].str[:10],
        "direction": d["direction_id"],
        "trip": d["half_trip_id"].astype(int),
        "seq": d["time_point_order"].astype(int),
        "timepoint": d["time_point_id"],
        "stop_id": d["stop_id"].astype(int),
        "standard": d["standard_type"],
    })
    out["sched_s"] = local_seconds(d["scheduled"])
    out["actual_s"] = local_seconds(d["actual"])
    out["sched_headway"] = pd.to_numeric(d["scheduled_headway"])
    out["headway"] = pd.to_numeric(d["headway"])
    out = out.sort_values(["date", "direction", "trip", "seq"], kind="stable").reset_index(drop=True)
    for c in ("sched_s", "actual_s", "sched_headway", "headway"):
        out[c] = out[c].round().astype("Int64")
    return out


def build_riders() -> pd.DataFrame:
    d = pd.read_csv(fetch(RIDERS_ITEM, "Fall_2024.csv", {ROUTE}, "riders_fall2024"), dtype=str)
    assert len(d) == N_RAW["riders"], len(d)
    d = d[d["day_type_name"] == "weekday"]
    hms = d["trip_start_time"].str.split(":", expand=True).astype(int)
    out = pd.DataFrame({
        "direction": d["direction_id"].map({"1": "Inbound", "0": "Outbound"}),
        "trip_start": hms[0] * 3600 + hms[1] * 60 + hms[2],
        "stop_seq": d["stop_sequence"].astype(int),
        "stop_id": d["stop_id"].astype(int),
        "stop_name": d["stop_name"].str.title(),
        "boardings": d["boardings"].astype(float).round(2),
        "alightings": d["alightings"].astype(float).round(2),
        "load": d["load"].astype(float).round(2),
        "n_trips": d["sample_count"].astype(int),
    })
    return out.sort_values(["direction", "trip_start", "stop_seq"], kind="stable").reset_index(drop=True)


def check(ev: pd.DataFrame, rd: pd.DataFrame) -> None:
    assert set(ev["direction"]) == {"Inbound", "Outbound"}
    assert ev["date"].min() == "2024-09-01" and ev["date"].max() == "2024-11-30"
    assert ev["seq"].between(1, 10).all() and ev.groupby(["date", "direction", "trip"])["seq"].count().eq(10).all()
    inbound = ev[ev["direction"] == "Inbound"]
    assert inbound.loc[inbound["seq"] == 1, "timepoint"].eq("woodc").all()
    assert inbound.loc[inbound["seq"] == 10, "timepoint"].eq("hayms").all()
    assert 0.01 < ev["actual_s"].isna().mean() < 0.06
    # the clock: the first scheduled inbound trip of a weekday is the 4:36 of the ridership table, all autumn
    first = inbound[inbound["seq"] == 1].groupby("date")["sched_s"].min()
    weekday = pd.to_datetime(first.index).dayofweek < 5
    before, after = first[weekday & (first.index < "2024-11-03")], first[weekday & (first.index >= "2024-11-03")]
    first_counted = rd.loc[(rd["direction"] == "Inbound") & (rd["trip_start"] > 3 * 3600), "trip_start"].min()
    assert before.median() == after.median() == first_counted == 4 * 3600 + 36 * 60, (before.median(), after.median())
    seg = inbound.pivot(index=["date", "trip"], columns="seq", values="actual_s").astype(float).diff(axis=1)
    assert (seg.iloc[:, 1:] < 0).mean().max() < 0.002          # a bus does not go back in time
    assert set(rd["direction"]) == {"Inbound", "Outbound"} and rd["boardings"].ge(0).all()
    per_trip = rd[rd["direction"] == "Inbound"].groupby("trip_start")["boardings"].sum()
    assert 150 < len(per_trip) < 260 and 10 < per_trip.mean() < 40


def write(out: pd.DataFrame, path: Path) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:      # mtime=0: byte-identical rebuilds
        gz.write(out.to_csv(index=False).encode())
    path.write_bytes(buf.getvalue())
    print(f"wrote {path.relative_to(ROOT)}: {len(out):,} rows, {path.stat().st_size / 1e6:.2f} MB, "
          f"md5 {hashlib.md5(buf.getvalue()).hexdigest()}")


def main() -> None:
    ev, rd = build_events(), build_riders()
    check(ev, rd)
    write(ev, OUT_EVENTS)
    write(rd, OUT_RIDERS)
    inbound = ev[ev["direction"] == "Inbound"]
    print(f"trips: {ev.groupby(['date', 'direction', 'trip']).ngroups:,} on {ev['date'].nunique()} days "
          f"({inbound.groupby(['date', 'trip']).ngroups:,} inbound); clock readings missing: "
          f"{ev['actual_s'].isna().mean():.1%}; ridership rows: {len(rd):,}")


if __name__ == "__main__":
    main()
