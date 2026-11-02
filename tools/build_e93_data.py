"""Build the E93 dataset: annual temperature maxima on the land cells of a 1-degree global grid.

SKELETON. Nothing here has been run on the full data. See plans/large/E93.md for the plan, the
sizes and the go/no-go thresholds. What has been run is `self-check`, on a few sampled chunks.
`population`, `stations` and `cities` still raise NotImplementedError.

    .venv/bin/python tools/build_e93_data.py plan                 # prints the stages; no network
    .venv/bin/python tools/build_e93_data.py self-check DIR       # offline, on sampled chunks
    .venv/bin/python tools/build_e93_data.py stream --first 2021-01 --months 12   # pilot P0
    PYMC_CHALLENGES_RUN=full PYMC_CHALLENGES_CONFIRM_FULL=yes \
        .venv/bin/python tools/build_e93_data.py stream           # about 155 GB through the network

Run it through tools/memguard.py with no GPU job's timing at stake (it does not use the GPU, and
imports neither JAX nor PyMC). A killed run resumes: each month is one checkpointed chunk.

Sources (both anonymous HTTPS, Zarr v2, Blosc lz4, little-endian float32, K):
  ARCO-ERA5, 1 degree hourly, conservative, 1959-01-01T00 to 2021-12-31T23: (time, lon, lat), lat
      ascending, chunks of 8 hours.
  ARCO-ERA5, 0.25 degree hourly, from 2022 on: (time, lat, lon), lat descending, 1 hour per chunk;
      time index 0 is 1900-01-01T00, and the shape runs past the data (the store's `valid_time_stop`
      attribute says how far final ERA5 goes: 2026-06-30 when this was written).

Why hourly: ERA5 has no daily maximum in the 1.5 degree WeatherBench 2 store, and sampling at 00,
06, 12 and 18 UTC misses the afternoon peak by an amount that depends on longitude. The stream
keeps both the hourly maximum and the maximum of those four hours, so that error is measured.

Days are LOCAL SOLAR days, not UTC days: the sampled "29 June 2021 UTC" maximum near Sea-Tac is the
afternoon of 28 June there. Each month is read with 12 extra hours on both sides so that every
local day is complete at every longitude.

Licence: Copernicus Products licence (see plans/large/E93.md). Anything published from this script
must carry COPERNICUS_NOTICE.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NOLOCK", "1")

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GCS = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar"
ONE_DEGREE = f"{GCS}/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
QUARTER_DEGREE = f"{GCS}/full_37-1h-0p25deg-chunk-1.zarr-v3"

COPERNICUS_NOTICE = (
    "Contains modified Copernicus Climate Change Service information {year}. Neither the European "
    "Commission nor ECMWF is responsible for any use that may be made of the Copernicus "
    "information or data it contains."
)

HOUR = np.timedelta64(1, "h")
EPOCH_1959 = np.datetime64("1959-01-01T00", "h")
EPOCH_1900 = np.datetime64("1900-01-01T00", "h")
LAST_ONE_DEGREE_HOUR = np.datetime64("2021-12-31T23", "h")
FIRST_MONTH = "1979-01"
LAST_MONTH = "2026-06"  # re-read valid_time_stop before a full run
LAND_MIN = 0.25  # keep coastal cells here; the model uses >= 0.5 and the city rule needs the rest
LAT_MIN = -60.0
GRID_LON = np.arange(360.0)
GRID_LAT = np.arange(-90.0, 91.0)
LOCAL_SHIFT_H = ((GRID_LON + 180.0) % 360.0 - 180.0) / 15.0  # local solar hour = UTC hour + shift
RETRIES = 6


def decode(blob, shape, dtype="<f4"):
    """One Zarr v2 Blosc chunk to an array. The caller gives the shape, so a short read raises."""
    from numcodecs import Blosc

    return np.frombuffer(Blosc().decode(blob), dtype=dtype).reshape(shape)


def fetch(session, url):
    """GET with retries and exponential backoff. A 404 is a bug in the index, so it raises."""
    for attempt in range(RETRIES):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 200:
                return response.content
            if response.status_code == 404:
                raise FileNotFoundError(url)
        except OSError as err:
            if isinstance(err, FileNotFoundError) or attempt == RETRIES - 1:
                raise
        time.sleep(2.0**attempt)
    raise RuntimeError(f"gave up on {url}")


def coarsen_quarter_to_one(field):
    """0.25 degree (lat N to S, lon) to the 1 degree grid (lon, lat S to N), area-weighted.

    Weights 1/2, 1, 1, 1, 1/2 on both axes. This reproduced the conservative 1 degree store to
    0.0095 K rms (largest 0.23 K) on the one hour compared, 2021-06-29T00.
    """
    x = field[::-1].astype(np.float64)
    weight = np.cos(np.deg2rad(np.arange(-90.0, 90.125, 0.25))).clip(1e-4)
    x = x * weight[:, None]
    kernel = (0.5, 1.0, 1.0, 1.0, 0.5)
    across = sum(k * x[:, (np.arange(360) * 4 + o) % 1440] for o, k in zip(range(-2, 3), kernel))
    out = np.empty((360, 181), np.float32)
    for j in range(181):
        rows = [(j * 4 + o, k) for o, k in zip(range(-2, 3), kernel) if 0 <= j * 4 + o < 721]
        total = sum(k * weight[i] * 4.0 for i, k in rows)
        out[:, j] = sum(k * across[i] for i, k in rows) / total
    return out


class Hours:
    """Yields (hour, field[lon, lat]) in time order, fetching with a thread pool."""

    def __init__(self, workers=8):
        import requests

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        self.session.mount("https://", adapter)
        self.pool = ThreadPoolExecutor(workers)

    def _one_degree(self, chunk):
        url = f"{ONE_DEGREE}/2m_temperature/{chunk}.0.0"
        return decode(fetch(self.session, url), (8, 360, 181))

    def _quarter(self, hour):
        index = int((hour - EPOCH_1900) / HOUR)
        url = f"{QUARTER_DEGREE}/2m_temperature/{index}.0.0"
        return coarsen_quarter_to_one(decode(fetch(self.session, url), (721, 1440)))

    def between(self, start, stop):
        hours = np.arange(start, stop, HOUR)
        old = hours[hours <= LAST_ONE_DEGREE_HOUR]
        new = hours[hours > LAST_ONE_DEGREE_HOUR]
        if old.size:
            offsets = ((old - EPOCH_1959) / HOUR).astype(np.int64)
            chunks = np.unique(offsets // 8)
            for chunk, block in zip(chunks, self.pool.map(self._one_degree, chunks.tolist())):
                base = int(chunk) * 8
                for offset in offsets[offsets // 8 == chunk]:
                    yield EPOCH_1959 + int(offset) * HOUR, block[int(offset) - base]
        if new.size:
            for hour, field in zip(new, self.pool.map(self._quarter, new)):
                yield hour, field


def month_days(month):
    first = np.datetime64(month, "M")
    return np.arange(first.astype("datetime64[D]"), (first + 1).astype("datetime64[D]"))


def reduce_hours(items, days, cell_lon, cell_lat):
    """Local-solar-day maxima of (items) on the listed cells: all hours, and the 00/06/12/18 UTC."""
    first_day = days[0].astype("datetime64[h]")
    every = np.full((days.size, cell_lon.size), -np.inf, np.float32)
    six = every.copy()
    cell = np.arange(cell_lon.size)
    shift = LOCAL_SHIFT_H[cell_lon]
    for hour, field in items:
        if not np.isfinite(field).all():
            raise ValueError(f"non-finite temperature at {hour}: past the end of the store?")
        since = (hour - first_day) / HOUR
        index = np.floor((since + shift) / 24.0).astype(np.int64)
        keep = (index >= 0) & (index < days.size)
        values = field[cell_lon, cell_lat]
        d, c, v = index[keep], cell[keep], values[keep]
        every[d, c] = np.maximum(every[d, c], v)
        if (hour.astype("datetime64[h]").astype(int) % 6) == 0:
            six[d, c] = np.maximum(six[d, c], v)
    if not np.isfinite(every).all() or not np.isfinite(six).all():
        raise ValueError("a local day was not covered: check the 12-hour margin")
    return every, six


def hot_year(days, cell_lat):
    """Year label per (day, cell): calendar year north of the equator, July to June south of it.

    A southern summer straddles the new year, so calendar years would count one heatwave twice.
    """
    months = days.astype("datetime64[M]").astype(int) % 12 + 1
    years = days.astype("datetime64[Y]").astype(int) + 1970
    south = years - (months < 7).astype(int)
    return np.where(cell_lat[None, :] < 0, south[:, None], years[:, None])


def annual_maxima(daily, days, cell_lat, window=1):
    """(year, cell) maxima of a `window`-day running mean of daily maxima, plus their day index."""
    if window > 1:
        cumulative = np.cumsum(np.vstack([np.zeros_like(daily[:1]), daily.astype(np.float64)]), 0)
        values = (cumulative[window:] - cumulative[:-window]) / window
        days = days[window - 1 :]
    else:
        values = daily
    labels = hot_year(days, cell_lat)
    years = np.arange(1979, 2026)
    out = np.full((years.size, daily.shape[1]), np.nan, np.float32)
    when = np.zeros_like(out, dtype=np.int32)
    for i, year in enumerate(years):
        block = np.ma.masked_array(values, labels != year)
        out[i] = block.max(axis=0).filled(np.nan)
        when[i] = block.argmax(axis=0)
    return out, when, years


def land_cells(lsm):
    """Indices (lon, lat) of the cells kept. `lsm` is land_sea_mask on the 1 degree grid."""
    lat = np.broadcast_to(GRID_LAT, lsm.shape)
    return np.nonzero((lsm >= LAND_MIN) & (lat > LAT_MIN))


def stage_static(args):
    import requests

    from pymc_challenges.bigrun import Store

    store = Store("E93")
    fields = {}
    for name in ("land_sea_mask", "geopotential_at_surface"):
        blob = fetch(requests, f"{ONE_DEGREE}/{name}/0.0")
        fields[name] = decode(blob, (360, 181))
    for name, array in fields.items():
        store.save_array(name, array)
    store.mark_done("s0_static")


def stage_stream(args):
    from pymc_challenges.bigrun import Store, run_level

    level = run_level(needs=dict(ram_gb=6.0, disk_gb=10.0))
    if args.months > 12 and level != "full":
        sys.exit("more than 12 months needs PYMC_CHALLENGES_RUN=full and the confirm variable")
    store = Store("E93")
    lsm = store.load_array("land_sea_mask")
    cell_lon, cell_lat = land_cells(lsm)
    months = np.arange(np.datetime64(args.first, "M"), np.datetime64(LAST_MONTH, "M") + 1)
    months = months[: args.months] if args.months else months
    reader = Hours(args.workers)
    name = f"daily_{args.first}"  # one series per starting month, so a pilot never mixes with a run
    for i in store.pending(name, months.size):
        days = month_days(str(months[i]))
        start = days[0].astype("datetime64[h]") - 12 * HOUR
        stop = (days[-1] + 1).astype("datetime64[h]") + 12 * HOUR
        began = time.time()
        every, six = reduce_hours(reader.between(start, stop), days, cell_lon, cell_lat)
        store.save_chunk(name, i, {"tx_hourly": every, "tx_6hourly": six})
        print(f"{months[i]}: {time.time() - began:.1f} s", flush=True)
    store.mark_done("s1_stream", months=int(months.size), first=args.first)


def stage_annual(args):
    raise NotImplementedError(
        "Gather the daily chunks of Store('E93') one product at a time (about 1.1 GB each), call "
        "annual_maxima with window 1 and 3, and save year x cell arrays as s3_annual."
    )


def stage_population(args):
    raise NotImplementedError(
        "GHS-POP R2023A, 2025 epoch, 30 arc-second, 483,694,490 bytes (HEAD only so far). Sum to the "
        "1 degree grid. Reading the TIFF needs tifffile and imagecodecs, as a build-time extra in "
        "the style of build_e84_margarine.py: uv run --no-project --with ... Fallback with no extra "
        "package: Natural Earth populated places. See the plan's open questions."
    )


def stage_stations(args):
    raise NotImplementedError(
        "GHCN-Daily by_station CSVs for stations with TMAX on at least 90% of days from 1979: fetch "
        "ghcnd-stations.txt and ghcnd-inventory.txt first. Match a station's calendar day to the "
        "cell's LOCAL SOLAR day; never compare levels, only exceedance of its own 1980s threshold."
    )


def stage_cities(args):
    raise NotImplementedError(
        "Natural Earth ne_10m_populated_places_simple.geojson and ne_50m_admin_0_countries.geojson "
        "from the natural-earth-vector repository (both answered HEAD). Apply the region rule in "
        "the plan: nearest cell with land_sea_mask >= 0.5, cities in touching cells merge."
    )


def stage_package(args):
    raise NotImplementedError(
        "Write data/era5_heat_extremes.npz through bigrun.atomic_write: year x cell maxima for both "
        "samplings, cell lon/lat, land fraction, elevation, population, and COPERNICUS_NOTICE."
    )


def stage_plan(args):
    print(__doc__)
    for stage in STAGES:
        print(stage)


def stage_self_check(args):
    """Offline: decode sampled chunks and check the code paths the sample can reach."""
    sample = Path(args.sample)
    read = lambda path, shape: decode((sample / path).read_bytes(), shape)
    block = read("h1/t2m.68475", (8, 360, 181))
    arco = read("arco/t2m.1064976", (721, 1440))
    diff = coarsen_quarter_to_one(arco) - block[0]
    rms = float(np.sqrt(np.mean(diff**2)))
    print(f"coarsen vs 1 degree store: rms {rms:.4f} K, largest {np.abs(diff).max():.3f} K")
    assert rms < 0.02

    lsm = read("h1/land_sea_mask", (360, 181))
    cell_lon, cell_lat = land_cells(lsm)
    print(f"kept cells at lsm >= {LAND_MIN}: {cell_lon.size}")

    first = 68475 * 8
    blocks = [read(f"h1/t2m.{c}", (8, 360, 181)) for c in range(68475, 68484)]
    series = np.concatenate(blocks)
    items = [(EPOCH_1959 + (first + i) * HOUR, series[i]) for i in range(series.shape[0])]
    # 72 hours cover the middle local day, 2021-06-30, at every longitude.
    days = np.arange(np.datetime64("2021-06-30"), np.datetime64("2021-07-01"))
    every, six = reduce_hours(items, days, cell_lon, cell_lat)
    assert (every >= six).all()
    print(f"one local day on land: 6-hourly deficit mean {(every - six).mean():.2f} K")

    hit = np.nonzero((cell_lon == 238) & (cell_lat == 47 + 90))[0]
    local = LOCAL_SHIFT_H[238]
    peak = int(series[:, 238, 47 + 90].argmax())
    print(f"Sea-Tac cell peak: {items[peak][0]} UTC, solar hour {(peak + local) % 24:.1f}")
    print(f"Sea-Tac cell on 2021-06-30 local: {every[0, hit][0] - 273.15:.2f} C")

    labels = hot_year(np.array(["2021-01-15", "2021-07-15"], "datetime64[D]"), np.array([10, -10.0]))
    assert labels.tolist() == [[2021, 2020], [2021, 2021]]
    daily = np.array([[1.0], [3.0], [2.0], [9.0]], np.float32)
    dates = np.arange(np.datetime64("2021-06-01"), np.datetime64("2021-06-05"))
    out, _, years = annual_maxima(daily, dates, np.array([10.0]), window=3)
    assert abs(out[years == 2021, 0][0] - 14.0 / 3.0) < 1e-5
    print("self-check passed")


STAGES = {
    "plan": stage_plan,
    "self-check": stage_self_check,
    "static": stage_static,
    "stream": stage_stream,
    "annual": stage_annual,
    "population": stage_population,
    "stations": stage_stations,
    "cities": stage_cities,
    "package": stage_package,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("sample", nargs="?", help="directory of sampled chunks, for self-check")
    parser.add_argument("--first", default=FIRST_MONTH, help="first month, YYYY-MM")
    parser.add_argument("--months", type=int, default=0, help="how many months; 0 means all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
