"""Build the E92 dataset: Argo temperature at 100 dbar in the open North Atlantic, one row a profile.

SKELETON. The stages below are written and were tested on small samples only (`check`). The full
pull has not been run, and `climatology` is a specification until the pilot in plans/large/E92.md
reaches it. Nothing here needs a new dependency.

    uv run python tools/build_e92_data.py plan                   # print the requests; no network
    uv run python tools/build_e92_data.py check SAMPLE_DIR       # run the derive step on samples
    uv run python tools/build_e92_data.py land                   # fetch the Natural Earth land
    uv run python tools/build_e92_data.py meta --go              # one metadata file per year
    uv run python tools/build_e92_data.py pull --go              # one CSV per month, resumable
    uv run python tools/build_e92_data.py derive                 # write data/argo_natl_t100.csv.gz

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

Raw pulls go to .scratch/e92/ (not committed). Every file is written atomically and the machine may
crash at any point: rerunning a stage skips what is finished.
"""

from __future__ import annotations

import argparse
import gzip
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

from pymc_challenges.bigrun import atomic_write, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / ".scratch" / "e92"
OUT = DATA / "argo_natl_t100.csv.gz"
LAND_OUT = DATA / "natural_earth_50m_land_natl.npz"

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


def climatology(rows: pd.DataFrame) -> pd.DataFrame:
    """SPECIFICATION, not written yet: the mean field the anomalies are taken from.

    From BASE_YEARS rows only, so that no fitted or scored value enters its own mean: at each 1
    degree cell, weighted least squares on points within 1,000 km (tricube weight), with a local
    quadratic in position and three annual harmonics, as in Kuusela and Stein. Store the coefficients
    per cell and evaluate at each profile's place and date. The 2023-2025 anomalies then include
    the warming since the base period, so the model keeps a marginalised intercept and trend.
    """
    raise NotImplementedError("Write this at pilot P2 of plans/large/E92.md, against real rows.")


def write_csv_gz(frame: pd.DataFrame, path: Path) -> str:
    return atomic_write(path, lambda f: f.write(gzip.compress(frame.to_csv(index=False).encode(),
                                                              mtime=0)))


def stage_plan() -> None:
    print("metadata:", meta_url(HINDCAST_YEAR))
    print("one month:", month_url(HINDCAST_YEAR, 6))
    years = [*BASE_YEARS, *FIT_YEARS, HINDCAST_YEAR]
    print(f"{len(years)} years, {12 * len(years)} monthly requests, in sequence")


def stage_land() -> None:
    rings = crop_land(read_land(fetch(LAND_URL)))
    arrays = {f"ring_{i}": ring.round(4) for i, ring in enumerate(rings)}
    atomic_write(LAND_OUT, lambda f: np.savez_compressed(f, **arrays))
    print(f"{len(rings)} rings -> {LAND_OUT} ({LAND_OUT.stat().st_size / 1e3:.0f} kB)")


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
            blob = fetch(meta_url(year))
            atomic_write(path, lambda f, b=blob: f.write(gzip.compress(b, mtime=0)))
            time.sleep(2)


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
    write_csv_gz(table[base], RAW / "base_years.csv.gz")  # for `climatology`; not published
    table = table[~base]
    digest = write_csv_gz(table, OUT)
    print(f"{len(table)} profiles -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), sha256 {digest}")
    print(table.groupby(table["time"].str[:4]).size().to_string())
    print("SHA-256 of the file on disk:", sha256_file(OUT))


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
    parser.add_argument("stage", choices=["plan", "check", "land", "meta", "pull", "derive"])
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
    elif args.stage == "pull":
        stage_pull(args.go)
    elif args.stage == "derive":
        stage_derive()


if __name__ == "__main__":
    main()
