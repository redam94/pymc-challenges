"""Build the E92 dataset: Argo temperature at 100 dbar in the open North Atlantic, one row a profile.

All stages ran on 2026-11-02 and 2026-11-03 (pull, then derive and climatology under the gates in
results/E92/gates_2026-11-03*.json; see plans/large/E92.md, "Pilot results"). Run each through
tools/e92_job.sh. Nothing here needs a new dependency.

    uv run python tools/build_e92_data.py plan                   # print the requests; no network
    uv run python tools/build_e92_data.py check SAMPLE_DIR       # run the derive step on samples
    uv run python tools/build_e92_data.py land                   # fetch the Natural Earth land
    uv run python tools/build_e92_data.py meta --go              # one metadata file per year
    uv run python tools/build_e92_data.py summary                # results/E92/pilot_meta.json
    uv run python tools/build_e92_data.py pull --go              # one CSV per month, resumable
    uv run python tools/build_e92_data.py derive                 # reduce rows; base years to the cache
    uv run python tools/build_e92_data.py climatology            # write data/argo_natl_t100*.{csv.gz,npz}

Source: Argo float data and metadata from the Global Data Assembly Centre (Argo GDAC), SEANOE,
https://doi.org/10.17882/42182, CC BY 4.0, read through the Ifremer ERDDAP mirror of the GDAC. The
mirror is live: delayed-mode corrections arrive for years, so a re-pull differs. Record the pull
date and the SHA-256 of the derived table in the notebook, and quote the Argo sentence wherever the
table goes:

    These data were collected and made freely available by the International Argo Program and the
    national programs that contribute to it (https://argo.ucsd.edu, https://www.ocean-ops.org).
    The Argo Program is part of the Global Ocean Observing System.

Facts this script depends on, each confirmed on samples on 2026-11-01 (plans/large/E92.md):

- A QC filter needs its value quoted and encoded: temp_qc=%221%22.
- Real-time (R) profiles have NaN `temp_adjusted`, so a `temp_adjusted_qc` filter drops them all:
  25% of 2023 and 41% of 2025. Filter on `pres` at the server and choose the columns here.
- Pressure bins give several rows a profile, so every profile is reduced to one value here.
- The dataset's stated coverage ends in December 2026 and its metadata hold fill values, so
  ranges are checked here and never taken from the metadata.

Land boundaries: Natural Earth 1:50m land, public domain by its terms of use (read 2026-11-01), so
no credit is required; the page suggests "Made with Natural Earth."

Raw pulls and the land rings go to ~/.cache/pymc-challenges/e92/ (not in git). Every file is written
atomically and the machine may crash at any point: rerunning a stage skips what is finished.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

from pymc_challenges.bigrun import CACHE_DIR, Store, atomic_write, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = CACHE_DIR / "e92"
OUT = DATA / "argo_natl_t100.csv.gz"
LAND_ZIP = RAW / "ne_50m_land.zip"
LAND_OUT = RAW / "natural_earth_50m_land_natl.npz"
CLIM_OUT = DATA / "argo_natl_t100_climatology.npz"
BASE = RAW / "base_years.csv.gz"
DERIVED = RAW / "derived_2023_2025.csv.gz"

ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
LAND_URL = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
USER_AGENT = "pymc-challenges-e92 (teaching notebook)"

BASE_YEARS = range(2015, 2023)  # climatology only; never scored
FIT_YEARS = (2023, 2024)  # the GP is fitted here
HINDCAST_YEAR = 2025  # the network and the scores of the deployment hindcast
BOX = dict(lat_min=20.0, lat_max=55.0, lon_min=-80.0, lon_max=0.0)
TARGET_DBAR = 100.0
WINDOW_DBAR = 15.0  # rows requested either side of the target
MAX_GAP_DBAR = 20.0  # bracketing levels further apart than this are not interpolated

KEY = ["platform_number", "cycle_number", "direction"]
PROFILE_VARS = ["platform_number", "cycle_number", "direction", "data_mode", "latitude",
                "longitude", "time", "position_qc"]
ROW_VARS = [*PROFILE_VARS, "time_qc", "pres", "pres_qc", "temp", "temp_qc", "pres_adjusted",
            "pres_adjusted_qc", "temp_adjusted", "temp_adjusted_qc"]

# Water that the land mask keeps but a basin-wide kernel should not see: (lon, lat) vertices.
EXCLUDE = {
    "mediterranean": [(-5.6, 30.0), (0.0, 30.0), (0.0, 46.0), (-5.6, 38.0)],
    "gulf_of_st_lawrence": [(-71.0, 45.0), (-59.3, 45.8), (-55.5, 52.5), (-71.0, 52.5)],
    "caribbean_and_straits": [(-80.0, 20.0), (-60.0, 17.0), (-72.0, 21.3), (-79.0, 25.5),
                              (-80.0, 27.0)],
    "north_sea": [(-3.0, 50.9), (0.0, 50.9), (0.0, 55.0), (-3.0, 55.0)],
}


def fetch(url: str, attempts: int = 5, timeout: int = 600) -> bytes:
    """GET with backoff. The mirror is shared: one request at a time, and never in parallel."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return reply.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            wait = 30 * 2**attempt
            print(f"  {error}: retry in {wait} s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def query(variables: list[str], *constraints: str, distinct: bool = False) -> str:
    """An ERDDAP tabledap URL, with every constraint and quoted QC value percent-encoded."""
    tail = "".join("&" + urllib.parse.quote(c, safe="=") for c in constraints)
    return f"{ERDDAP}?{','.join(variables)}{tail}" + ("&distinct()" if distinct else "")


def space(pad_south: float = 0.0) -> list[str]:
    return [f"latitude>={BOX['lat_min'] - pad_south}", f"latitude<={BOX['lat_max']}",
            f"longitude>={BOX['lon_min']}", f"longitude<={BOX['lon_max']}"]


def meta_url(year: int) -> str:
    """All profile positions in a year: about 1.7 MB and under 10 s when measured."""
    return query(PROFILE_VARS, f"time>={year}-01-01T00:00:00Z", f"time<{year + 1}-01-01T00:00:00Z",
                 *space(), distinct=True)


def month_url(year: int, month: int) -> str:
    """Rows near the target level for one month. Quality is chosen in `reduce_profiles`."""
    end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
    return query(ROW_VARS, f"time>={year}-{month:02d}-01T00:00:00Z", f"time<{end}T00:00:00Z",
                 *space(), f"pres>={TARGET_DBAR - WINDOW_DBAR}", f"pres<={TARGET_DBAR + WINDOW_DBAR}",
                 'position_qc="1"')


def read_erddap(source) -> pd.DataFrame:
    """Parse ERDDAP CSV, whose second line holds units."""
    return pd.read_csv(source, skiprows=[1], dtype={"platform_number": str, "data_mode": str,
                                                    "direction": str, "position_qc": str,
                                                    "time_qc": str, "pres_qc": str, "temp_qc": str,
                                                    "pres_adjusted_qc": str,
                                                    "temp_adjusted_qc": str})


def _good(flag: pd.Series) -> np.ndarray:
    return flag.astype(str).str.strip().isin(["1", "1.0"]).to_numpy()


def choose_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Adjusted values for D and A profiles, raw for R, each only with a good flag.

    The raw fallback is not free of risk: real-time pressure is uncorrected. `data_mode` stays in
    the table so that the notebook can check that the choice does not move the answer.
    """
    use_adjusted = rows["data_mode"].isin(["D", "A"]).to_numpy()
    pres = np.where(use_adjusted, rows["pres_adjusted"], rows["pres"])
    temp = np.where(use_adjusted, rows["temp_adjusted"], rows["temp"])
    good = np.where(use_adjusted, _good(rows["pres_adjusted_qc"]) & _good(rows["temp_adjusted_qc"]),
                    _good(rows["pres_qc"]) & _good(rows["temp_qc"]))
    return rows.assign(p=pres, t=temp)[good & np.isfinite(pres) & np.isfinite(temp)]


def reduce_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    """One value per ascending profile: linear interpolation to the target from its two neighbours.

    A profile without a good level on both sides of the target, or with those levels further apart
    than MAX_GAP_DBAR, is dropped rather than extrapolated.
    """
    rows = rows[(rows["direction"] == "A") & _good(rows["position_qc"]) & _good(rows["time_qc"])]
    rows = choose_columns(rows).sort_values([*KEY, "p"])
    rows = rows.drop_duplicates([*KEY, "p"])
    below = rows[rows["p"] <= TARGET_DBAR].groupby(KEY, sort=False).tail(1)
    above = rows[rows["p"] >= TARGET_DBAR].groupby(KEY, sort=False).head(1)
    pair = below.merge(above[[*KEY, "p", "t"]], on=KEY, suffixes=("_lo", "_hi"))
    gap = pair["p_hi"] - pair["p_lo"]
    pair = pair[gap <= MAX_GAP_DBAR].copy()
    span = (pair["p_hi"] - pair["p_lo"]).to_numpy()
    weight = np.divide(TARGET_DBAR - pair["p_lo"].to_numpy(), span, out=np.zeros(len(pair)),
                       where=span > 0)
    pair["t100"] = (pair["t_lo"] + weight * (pair["t_hi"] - pair["t_lo"])).round(3)
    pair["gap_dbar"] = (pair["p_hi"] - pair["p_lo"]).round(2)
    out = pair.rename(columns={"platform_number": "float_id", "cycle_number": "cycle",
                               "latitude": "lat", "longitude": "lon", "data_mode": "mode"})
    return out[["float_id", "cycle", "mode", "time", "lat", "lon", "t100", "gap_dbar"]]


def read_land(zip_bytes: bytes) -> list[np.ndarray]:
    """Rings of a polygon shapefile as (lon, lat) arrays; a small reader, so no new dependency."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".shp"))
        blob = archive.read(name)
    rings, at = [], 100
    while at < len(blob):
        _, words = struct.unpack(">ii", blob[at:at + 8])
        record = blob[at + 8:at + 8 + 2 * words]
        at += 8 + 2 * words
        kind = struct.unpack("<i", record[:4])[0]
        if kind == 0:
            continue
        if kind != 5:
            raise ValueError(f"Expected polygon records (5), found {kind}")
        n_parts, n_points = struct.unpack("<ii", record[36:44])
        parts = np.frombuffer(record, "<i4", n_parts, 44)
        points = np.frombuffer(record, "<f8", 2 * n_points, 44 + 4 * n_parts).reshape(-1, 2)
        rings.extend(np.split(points, parts[1:]))
    return rings


def crop_land(rings: list[np.ndarray]) -> list[np.ndarray]:
    """Rings touching a margin around the box, for the mask and for drawing the coast."""
    keep = []
    for ring in rings:
        lon, lat = ring[:, 0], ring[:, 1]
        if (lon.max() >= BOX["lon_min"] - 15 and lon.min() <= BOX["lon_max"] + 15
                and lat.max() >= BOX["lat_min"] - 10 and lat.min() <= BOX["lat_max"] + 10):
            keep.append(ring)
    return keep


def open_atlantic(lat, lon, rings: list[np.ndarray]) -> np.ndarray:
    """True for points in the box, at sea, and outside the excluded basins."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    points = np.c_[lon, lat]
    inside = ((lat >= BOX["lat_min"]) & (lat <= BOX["lat_max"])
              & (lon >= BOX["lon_min"]) & (lon <= BOX["lon_max"]))
    for ring in rings:
        inside &= ~Polygon(ring).contains_points(points)
    for vertices in EXCLUDE.values():
        inside &= ~Polygon(np.array(vertices)).contains_points(points)
    return inside


EARTH_KM = 6371.0
KM_PER_DEG = np.pi * EARTH_KM / 180
CELL_LAT = np.arange(BOX["lat_min"] + 0.5, BOX["lat_max"], 1.0)  # 1 degree cell centres
CELL_LON = np.arange(BOX["lon_min"] + 0.5, BOX["lon_max"], 1.0)
CLIM_MIN_PROFILES = 200  # a cell with fewer base profiles inside the bandwidth is left empty
CLIM_MIN_WEIGHT = 50.0  # ... or with a smaller tricube weight sum
N_COEF = 12  # 1, u, v, u^2, uv, v^2, then sin and cos of 1, 2, 3 cycles a year


def year_phase(time: pd.Series) -> np.ndarray:
    """Fraction of the calendar year elapsed, 0 to 1, from ERDDAP's ISO times."""
    t = pd.to_datetime(time, utc=True)
    start = pd.to_datetime(t.dt.year.astype(str) + "-01-01", utc=True)
    length = np.where(t.dt.is_leap_year, 366.0, 365.0)
    return ((t - start).dt.total_seconds() / 86400.0 / length).to_numpy()


def _unit(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def _design(lat, lon, phase, lat0, lon0) -> np.ndarray:
    """Local quadratic in position (units of 1,000 km about the cell centre) and 3 annual harmonics."""
    u = (np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * KM_PER_DEG / 1000
    v = (np.asarray(lat) - lat0) * KM_PER_DEG / 1000
    w = 2 * np.pi * np.asarray(phase)
    return np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v,
                            np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w),
                            np.sin(3 * w), np.cos(3 * w)])


def climatology(base: pd.DataFrame, bandwidth_km: float = 1000.0, adaptive_k: int | None = None,
                h_min: float = 150.0) -> dict:
    """The mean field the anomalies are taken from, fitted on BASE_YEARS rows only.

    At each 1 degree cell centre: weighted least squares on base profiles within ``bandwidth_km``
    (great-circle), tricube weight in distance, with a local quadratic in position and three annual
    harmonics, as in Kuusela and Stein (2018). With ``adaptive_k`` the bandwidth of each cell is the
    distance to its k-th nearest base profile, clipped to ``h_min``-``bandwidth_km`` (gates_2026-11-03b:
    fixed 1,000 and 500 km left a mean error along the Gulf Stream). Returns per-cell coefficients (NaN where the cell has
    too little data) and the counts behind them. The 2023-2025 anomalies then include the warming
    since the base period, so the model keeps a marginalised intercept and trend.
    """
    from scipy.spatial import cKDTree

    lat, lon = base["lat"].to_numpy(), base["lon"].to_numpy()
    phase, y = year_phase(base["time"]), base["t100"].to_numpy()
    tree = cKDTree(_unit(lat, lon))
    coef = np.full((len(CELL_LAT), len(CELL_LON), N_COEF), np.nan)
    count = np.zeros((len(CELL_LAT), len(CELL_LON)), int)
    weight_sum = np.zeros_like(count, dtype=float)
    bandwidth = np.full(count.shape, float(bandwidth_km))
    for i, lat0 in enumerate(CELL_LAT):
        for j, lon0 in enumerate(CELL_LON):
            centre = _unit([lat0], [lon0])[0]
            h = bandwidth_km
            if adaptive_k:
                kth = tree.query(centre, adaptive_k)[0][-1]
                h = float(np.clip(EARTH_KM * 2 * np.arcsin(min(kth / 2, 1.0)), h_min, bandwidth_km))
                bandwidth[i, j] = h
            chord = 2 * np.sin(h / EARTH_KM / 2)
            idx = np.asarray(tree.query_ball_point(centre, chord), int)
            count[i, j] = len(idx)
            if len(idx) < CLIM_MIN_PROFILES:
                continue
            dist = EARTH_KM * 2 * np.arcsin(np.clip(np.linalg.norm(tree.data[idx] - centre, axis=1) / 2, 0, 1))
            w = (1 - np.clip(dist / h, 0, 1) ** 3) ** 3
            weight_sum[i, j] = w.sum()
            if w.sum() < CLIM_MIN_WEIGHT:
                continue
            X = _design(lat[idx], lon[idx], phase[idx], lat0, lon0)
            sw = np.sqrt(w)
            coef[i, j], *_ = np.linalg.lstsq(X * sw[:, None], y[idx] * sw, rcond=None)
    return {"coef": coef, "count": count, "weight_sum": weight_sum, "cell_lat": CELL_LAT,
            "cell_lon": CELL_LON, "bandwidth_km": bandwidth, "adaptive_k": np.int64(adaptive_k or 0)}


def evaluate_climatology(clim: dict, lat, lon, phase) -> np.ndarray:
    """Bilinear blend of the four surrounding cells' local fits, each evaluated at the point itself.

    Weights are renormalised over non-empty cells; NaN where all four are empty.
    """
    lat, lon, phase = (np.asarray(a, float) for a in (lat, lon, phase))
    clat, clon, coef = clim["cell_lat"], clim["cell_lon"], clim["coef"]
    fi = np.clip((lat - clat[0]), 0, len(clat) - 1.000001)
    fj = np.clip((lon - clon[0]), 0, len(clon) - 1.000001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    di, dj = fi - i0, fj - j0
    total, wsum = np.zeros(len(lat)), np.zeros(len(lat))
    for a, b, w in ((0, 0, (1 - di) * (1 - dj)), (1, 0, di * (1 - dj)), (0, 1, (1 - di) * dj),
                    (1, 1, di * dj)):
        ii, jj = np.minimum(i0 + a, len(clat) - 1), np.minimum(j0 + b, len(clon) - 1)
        c = coef[ii, jj]
        ok = np.isfinite(c[:, 0]) & (w > 0)
        if not ok.any():
            continue
        X = _design(lat[ok], lon[ok], phase[ok], clat[ii[ok]], clon[jj[ok]])
        total[ok] += w[ok] * np.einsum("ij,ij->i", X, c[ok])
        wsum[ok] += w[ok]
    return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1), np.nan)


def write_csv_gz(frame: pd.DataFrame, path: Path) -> str:
    return atomic_write(path, lambda f: f.write(gzip.compress(frame.to_csv(index=False).encode(),
                                                              mtime=0)))


def stage_plan() -> None:
    print("metadata:", meta_url(HINDCAST_YEAR))
    print("one month:", month_url(HINDCAST_YEAR, 6))
    years = [*BASE_YEARS, *FIT_YEARS, HINDCAST_YEAR]
    print(f"{len(years)} years, {12 * len(years)} monthly requests, in sequence")


def record(name: str, entries: dict) -> None:
    """Merge one stage's facts into a small JSON file under results/E92/, keeping earlier ones."""
    store = Store("E92")
    old = store.load_json(name) if (store.path / f"{name}.json").exists() else {}
    store.save_json(name, {**old, **entries})


def stage_land() -> None:
    t0 = time.perf_counter()
    blob = fetch(LAND_URL)
    seconds = time.perf_counter() - t0
    atomic_write(LAND_ZIP, lambda f: f.write(blob))
    rings = crop_land(read_land(blob))
    arrays = {f"ring_{i}": ring.round(4) for i, ring in enumerate(rings)}
    atomic_write(LAND_OUT, lambda f: np.savez_compressed(f, **arrays))
    size = LAND_OUT.stat().st_size
    print(f"{len(rings)} rings -> {LAND_OUT} ({size / 1e3:.0f} kB)")
    record("pilot_land", {"url": LAND_URL, "fetched": datetime.date.today().isoformat(),
                          "zip_bytes": len(blob), "fetch_s": round(seconds, 2),
                          "zip_sha256": hashlib.sha256(blob).hexdigest(), "rings_near_box": len(rings),
                          "npz_bytes": size, "npz_sha256": sha256_file(LAND_OUT),
                          "labels": "all measured"})


def load_land() -> list[np.ndarray]:
    with np.load(LAND_OUT, allow_pickle=False) as npz:
        return [npz[k] for k in npz.files]


def stage_meta(go: bool) -> None:
    for year in [*BASE_YEARS, *FIT_YEARS, HINDCAST_YEAR]:
        path = RAW / f"meta_{year}.csv.gz"
        if path.exists():
            continue
        print(meta_url(year))
        if go:
            t0 = time.perf_counter()
            blob = fetch(meta_url(year))
            seconds = time.perf_counter() - t0
            table = read_erddap(io.BytesIO(blob))  # refuse a truncated reply before it is cached
            missing = set(PROFILE_VARS) - set(table.columns)
            years = table["time"].str[:4].unique().tolist()
            if missing or not len(table) or years != [str(year)]:
                raise ValueError(f"{year}: bad reply: missing {missing}, {len(table)} rows, {years}")
            atomic_write(path, lambda f, b=blob: f.write(gzip.compress(b, mtime=0)))
            print(f"  {len(table)} rows, {len(blob) / 1e6:.2f} MB in {seconds:.1f} s")
            record("pilot_meta_fetch", {str(year): {
                "fetched": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                "rows": len(table), "csv_bytes": len(blob), "fetch_s": round(seconds, 2)}})
            time.sleep(2)


def summarise_year(meta: pd.DataFrame, rings: list[np.ndarray]) -> dict:
    """What one year's metadata says, with the same filters as `stage_check` and the plan's table."""
    good = meta[(meta["direction"] == "A") & _good(meta["position_qc"])]
    inside = good[open_atlantic(good["latitude"], good["longitude"], rings)]
    months = inside["time"].str[5:7].value_counts().sort_index()
    return {
        "rows": len(meta), "duplicate_keys": int(meta.duplicated(KEY).sum()),
        "bad_times": int(pd.to_datetime(meta["time"], errors="coerce").isna().sum()),
        "time_min": meta["time"].min(), "time_max": meta["time"].max(),
        "lat_range": [float(meta["latitude"].min()), float(meta["latitude"].max())],
        "lon_range": [float(meta["longitude"].min()), float(meta["longitude"].max())],
        "direction": meta["direction"].value_counts().to_dict(),
        "position_qc": meta["position_qc"].value_counts().to_dict(),
        "ascending_good": len(good), "in_open_atlantic": len(inside),
        "floats": inside["platform_number"].nunique(),
        "floats_cycle_0_or_1": inside[inside["cycle_number"] <= 1]["platform_number"].nunique(),
        "mode": inside["data_mode"].value_counts().to_dict(),
        "by_month": {k: int(v) for k, v in months.items()},
    }


def stage_summary() -> None:
    """Counts from the cached metadata, stored as small JSON. Needs no network."""
    rings = load_land()
    years = {str(y): summarise_year(read_erddap(RAW / f"meta_{y}.csv.gz"), rings)
             for y in [*BASE_YEARS, *FIT_YEARS, HINDCAST_YEAR]}
    Store("E92").save_json("pilot_meta", {
        "labels": "all measured from metadata; a profile here need not reach 100 dbar", "years": years,
        "gz_bytes": {y: (RAW / f"meta_{y}.csv.gz").stat().st_size for y in years}})
    for year, s in years.items():
        print(year, s["rows"], s["ascending_good"], s["in_open_atlantic"], s["floats"],
              s["floats_cycle_0_or_1"], s["mode"], "dup", s["duplicate_keys"], "bad", s["bad_times"])


def stage_pull(go: bool) -> None:
    """Sequential monthly pulls; a finished month is never fetched again."""
    for year in [*BASE_YEARS, *FIT_YEARS, HINDCAST_YEAR]:
        for month in range(1, 13):
            path = RAW / f"rows_{year}_{month:02d}.csv.gz"
            if path.exists():
                continue
            print(month_url(year, month))
            if go:
                t0 = time.perf_counter()
                blob = fetch(month_url(year, month))
                read_erddap(io.BytesIO(blob))  # refuse a truncated reply before it is cached
                atomic_write(path, lambda f, b=blob: f.write(gzip.compress(b, mtime=0)))
                print(f"  {len(blob) / 1e6:.1f} MB in {time.perf_counter() - t0:.0f} s")
                time.sleep(2)


def stage_derive() -> None:
    rings = load_land()
    tables = []
    for path in sorted(RAW.glob("rows_*.csv.gz")):
        table = reduce_profiles(read_erddap(path))
        tables.append(table[open_atlantic(table["lat"], table["lon"], rings)])
    table = pd.concat(tables).drop_duplicates(["float_id", "cycle"]).sort_values("time")
    table["lat"], table["lon"] = table["lat"].round(4), table["lon"].round(4)
    base = table["time"].str[:4].astype(int).isin(BASE_YEARS)
    write_csv_gz(table[base], BASE)  # for `climatology`; not published
    write_csv_gz(table[~base], DERIVED)  # `climatology` adds the mean and writes the published table
    years = table.groupby(table["time"].str[:4]).size()
    print(years.to_string())
    raw_profiles = {}
    for path in sorted(RAW.glob("rows_*.csv.gz")):
        rows = read_erddap(path)
        year = path.name[5:9]
        asc = rows[(rows["direction"] == "A") & _good(rows["position_qc"])]
        raw_profiles[year] = raw_profiles.get(year, 0) + int(asc.groupby(KEY).ngroups)
    record("pilot_data", {"derive": {
        "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "profiles_by_year": {k: int(v) for k, v in years.items()},
        "ascending_good_position_profiles_in_rows_by_year": raw_profiles,
        "raw_files": len(list(RAW.glob("rows_*.csv.gz"))),
        "raw_gz_bytes": sum(p.stat().st_size for p in RAW.glob("rows_*.csv.gz")),
        "labels": "all measured"}})


def mode_offset(table: pd.DataFrame, seed: int = 0) -> dict:
    """R minus D+A mean anomaly within 5 x 5 degree x year cells holding at least 20 of each."""
    t = table.assign(year=table["time"].str[:4], ci=(table["lat"] // 5).astype(int),
                     cj=(table["lon"] // 5).astype(int), r=table["mode"].eq("R"))
    rows = []
    for _, g in t.groupby(["year", "ci", "cj"]):
        n_r, n_da = int(g["r"].sum()), int((~g["r"]).sum())
        if n_r >= 20 and n_da >= 20:
            rows.append((g.loc[g["r"], "anom"].mean() - g.loc[~g["r"], "anom"].mean(), min(n_r, n_da)))
    d, w = np.array(rows).T if rows else (np.array([]), np.array([]))
    est = float(np.sum(d * w) / np.sum(w)) if len(d) else float("nan")
    rng = np.random.default_rng(seed)
    boots = [np.sum((d * w)[k]) / np.sum(w[k]) for k in rng.integers(0, len(d), (2000, len(d)))]
    raw = float(t.loc[t["r"], "anom"].mean() - t.loc[~t["r"], "anom"].mean())
    return {"difference_c": est, "bootstrap_se_c": float(np.std(boots)), "cells": len(d),
            "raw_difference_c": raw, "n_r": int(t["r"].sum()), "n_da": int((~t["r"]).sum())}


def gulf_stream_excess(base: pd.DataFrame) -> dict:
    """Excess rms of base-year residual means in 2 degree cells of 30-45N, 75-45W, 100+ profiles."""
    b = base[(base["lat"] >= 30) & (base["lat"] < 45) & (base["lon"] >= -75) & (base["lon"] < -45)]
    g = b.assign(ci=(b["lat"] // 2).astype(int), cj=(b["lon"] // 2).astype(int)).groupby(["ci", "cj"])
    s = g["resid"].agg(["mean", "var", "count"])
    s = s[s["count"] >= 100]
    excess = np.sum(s["count"] * (s["mean"] ** 2 - s["var"] / s["count"])) / s["count"].sum()
    return {"excess_rms_c": float(np.sqrt(max(0.0, excess))), "cells": len(s),
            "max_abs_cell_mean_c": float(s["mean"].abs().max()),
            "rms_cell_mean_c": float(np.sqrt(np.average(s["mean"] ** 2, weights=s["count"])))}


def _climatology_checks(base: pd.DataFrame, table: pd.DataFrame, **rule) -> tuple[dict, pd.DataFrame, dict]:
    t0 = time.perf_counter()
    clim = climatology(base, **rule)
    fit_s = time.perf_counter() - t0
    b = base.assign(clim=evaluate_climatology(clim, base["lat"], base["lon"], year_phase(base["time"])))
    b["resid"] = b["t100"] - b["clim"]
    t = table.assign(clim=evaluate_climatology(clim, table["lat"], table["lon"], year_phase(table["time"])))
    t["anom"] = t["t100"] - t["clim"]
    fit = t[t["time"].str[:4].astype(int).isin(FIT_YEARS)]
    early = base["time"].str[:4].astype(int) <= 2020  # held-out check: fit 2015-2020, score 2021-2022
    clim_early = climatology(base[early], **rule)
    late = base[~early]
    late = late.assign(resid=late["t100"] - evaluate_climatology(clim_early, late["lat"], late["lon"],
                                                                 year_phase(late["time"])))
    check = {
        "rule": {k: float(v) for k, v in rule.items()}, "fit_s": round(fit_s, 1),
        "cells_fitted": int(np.isfinite(clim["coef"][..., 0]).sum()),
        "bandwidth_km_quantiles": np.nanquantile(np.where(np.isfinite(clim["coef"][..., 0]),
                                                          clim["bandwidth_km"], np.nan), [0, .1, .5, .9, 1]),
        "base_profiles": len(b), "base_missing_clim": int(b["clim"].isna().sum()),
        "base_resid_sd_c": float(b["resid"].std()),
        "missing_clim_by_year": t[t["clim"].isna()].groupby(t["time"].str[:4]).size().to_dict(),
        "fit_profiles": int(fit["anom"].notna().sum()),
        "anomaly_sd_fit_c": float(fit["anom"].std()), "anomaly_mean_fit_c": float(fit["anom"].mean()),
        "anomaly_mean_by_year_c": t.groupby(t["time"].str[:4])["anom"].mean().to_dict(),
        "mode_offset": mode_offset(fit.dropna(subset=["anom"])),
        "gulf_stream": gulf_stream_excess(b.dropna(subset=["resid"])),
        "gulf_stream_heldout_2021_2022": gulf_stream_excess(late.dropna(subset=["resid"]))}
    print(rule, check, flush=True)
    return clim, t, check


def stage_climatology() -> None:
    """Fit the base-period mean, check it against the P2 lines, write the published table.

    The first two rules are gates_2026-11-03's (1,000 km, and 500 km as its fallback); both failed
    the Gulf Stream line. The third is gates_2026-11-03b's adaptive rule, which is the one published.
    """
    import shutil

    store = Store("E92")
    store.load_json("gates_2026-11-03"), store.load_json("gates_2026-11-03b")  # refuse if altered
    base, table = pd.read_csv(BASE, dtype={"float_id": str}), pd.read_csv(DERIVED, dtype={"float_id": str})
    rules = {"fixed_1000km": dict(bandwidth_km=1000.0), "fixed_500km": dict(bandwidth_km=500.0),
             "adaptive_400": dict(bandwidth_km=1000.0, adaptive_k=400, h_min=150.0)}
    results = {name: _climatology_checks(base, table, **rule) for name, rule in rules.items()}
    first = results["fixed_1000km"][2]
    verdict_first = {"fit_profiles": first["fit_profiles"] >= 24000,
                     "mode_offset": abs(first["mode_offset"]["difference_c"]) < 0.02,
                     "anomaly_sd": first["anomaly_sd_fit_c"] < 1.5,
                     "gulf_stream_mean": first["gulf_stream"]["excess_rms_c"] < 0.3}
    clim, t, c = results["adaptive_400"]
    verdict = {"fit_profiles": c["fit_profiles"] >= 24000,
               "mode_offset": abs(c["mode_offset"]["difference_c"]) < 0.02,
               "anomaly_sd": c["anomaly_sd_fit_c"] < 1.5,
               "gulf_stream_mean": c["gulf_stream"]["excess_rms_c"] < 0.3,
               "gulf_stream_heldout": c["gulf_stream_heldout_2021_2022"]["excess_rms_c"] < 0.4,
               "beats_1000km_on_both": (c["gulf_stream"]["excess_rms_c"] < first["gulf_stream"]["excess_rms_c"]
                                        and c["gulf_stream_heldout_2021_2022"]["excess_rms_c"]
                                        < first["gulf_stream_heldout_2021_2022"]["excess_rms_c"])}
    t = t.dropna(subset=["clim"]).copy()
    t["clim"], t["anom"] = t["clim"].round(3), t["anom"].round(3)
    digest = write_csv_gz(t[["float_id", "cycle", "mode", "time", "lat", "lon", "t100", "clim", "anom",
                             "gap_dbar"]], OUT)
    atomic_write(CLIM_OUT, lambda f: np.savez_compressed(f, **{k: np.asarray(v) for k, v in clim.items()}))
    shutil.copyfile(LAND_OUT, DATA / LAND_OUT.name)
    record("pilot_data", {"climatology": {
        "gates": ["gates_2026-11-03.json P2_data", "gates_2026-11-03b.json"], "published_rule": "adaptive_400",
        "checks": {name: r[2] for name, r in results.items()},
        "verdict_gates_2026-11-03_fixed_1000km": verdict_first, "verdict_gates_2026-11-03b_adaptive": verdict,
        "published_rows": len(t), "published_by_year": t.groupby(t["time"].str[:4]).size().to_dict(),
        "table_sha256": digest, "clim_sha256": sha256_file(CLIM_OUT),
        "land_sha256": sha256_file(DATA / LAND_OUT.name), "labels": "all measured"}})
    print("verdict (first gates, 1000 km):", verdict_first)
    print("verdict (gates b, adaptive):", verdict)
    print(f"{len(t)} rows -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), sha256 {digest}")


def stage_check(sample_dir: Path) -> None:
    """Run the derive step on the small samples and print the counts the plan quotes."""
    rings = crop_land(read_land((sample_dir / "ne_50m_land.zip").read_bytes()))
    print(f"{len(rings)} land rings near the box")
    for name in ("week.csv", "days2025.csv", "oneday_script_url.csv"):
        rows = read_erddap(sample_dir / name)
        if "pres_qc" not in rows:  # these two samples predate the final column list
            rows = rows.assign(pres_qc="1")
        table = reduce_profiles(rows)
        keep = open_atlantic(table["lat"], table["lon"], rings)
        print(f"{name}: {rows.groupby(KEY).ngroups} profiles, {len(table)} reduced, "
              f"{int(keep.sum())} in the open Atlantic box; modes "
              f"{table['mode'].value_counts().to_dict()}")
    for year in (2023, 2024, 2025):
        meta = read_erddap(sample_dir / f"profiles{year}.csv")
        meta = meta[(meta["direction"] == "A") & _good(meta["position_qc"])]
        keep = open_atlantic(meta["latitude"], meta["longitude"], rings)
        inside = meta[keep]
        first = inside[inside["cycle_number"] <= 1]["platform_number"].nunique()
        print(f"{year}: {len(meta)} ascending, position good; {len(inside)} in the open Atlantic "
              f"box, {inside['platform_number'].nunique()} floats, {first} floats at cycle 0 or 1, "
              f"mode {inside['data_mode'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["plan", "check", "land", "meta", "summary", "pull",
                                            "derive", "climatology"])
    parser.add_argument("sample_dir", nargs="?", type=Path)
    parser.add_argument("--go", action="store_true", help="without this, only print the requests")
    args = parser.parse_args()
    if args.stage == "plan":
        stage_plan()
    elif args.stage == "check":
        stage_check(args.sample_dir)
    elif args.stage == "land":
        stage_land()
    elif args.stage == "meta":
        stage_meta(args.go)
    elif args.stage == "summary":
        stage_summary()
    elif args.stage == "pull":
        stage_pull(args.go)
    elif args.stage == "derive":
        stage_derive()
    elif args.stage == "climatology":
        stage_climatology()


if __name__ == "__main__":
    main()
