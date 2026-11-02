"""Build E89's site and county tables from EPA AirData and Census files.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/memguard.py --max-gb 3 -- \
        .venv/bin/python tools/build_e89_data.py --cache .scratch/e89 --out data

SKELETON. It has been run on the single 2023 file only (see plans/large/E89.md, section 2). The
full three-year build has not been run. It downloads about 27 MB of daily files, the 7 MB monitor
list and two small Census files. Each year is checkpointed, so a crash loses at most that year.

Rules that cost someone an hour, all measured on the 2023 file:

- Read with ``keep_default_na=False``. AQS writes the Event Type as the string "None", which pandas
  otherwise reads as NaN, and a filter on it then silently keeps 4% of the data.
- Original T640 data (method 236, 238) is dropped, as EPA's advisory recommends. It reads about
  1.1 ug/m3 above the corrected data (736, 738).
- The corrected data has its own POC (3 becomes 23 at the site checked), and from mid-2023 the same
  instrument reports as 636/638 under the old POC. So one instrument is split across two POCs, and
  a site-day is the mean over surviving POCs. NEVER take a maximum over POCs: that picks the
  original series wherever it survives.
- Annual files cannot apply this filter: their rows say "Multiple Methods Used" and still hold the
  original data (median gap 1.52 ug/m3 from the clean mean on those rows). So annual values are
  built here from daily files and the annual files are used only to check.
"""

import argparse
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pymc_challenges.bigrun import atomic_write, sha256_file

AIRDATA = "https://aqs.epa.gov/aqsweb/airdata"
CENSUS = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals"
CENPOP = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
GEOJSON = Path(__file__).resolve().parents[1] / "data" / "us_counties.geojson"

ORIGINAL_T640 = {"236", "238"}  # EPA advisory of 26 August 2024
NOT_US = {"80"}  # state code of the Mexican and Canadian border sites
NOT_CONUS = {"02", "15", "60", "66", "69", "72", "78"} | NOT_US
QUARTER_DAYS = {1: 90, 2: 91, 3: 92, 4: 92}  # 91 for Q1 in a leap year: too small to matter here
MIN_DAYS_PER_QUARTER = 11  # 75% of a 1-in-6 schedule. NOT the regulatory rule: pilot step 1 checks it
SMALL_CROP = ("17", "18", "19", "21", "26", "27", "29", "39", "42", "47", "54", "55",  # Midwest, Ohio V.
              "08", "20", "30", "31", "38", "46", "56")  # Plains and Rockies, for the empty counties
FIPS_FIXES = {"46113": "46102", "51515": "51019"}  # geojson's Shannon SD; Bedford city VA merged
DAILY_COLUMNS = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude",
                 "Sample Duration", "Date Local", "Event Type", "Arithmetic Mean", "Method Code",
                 "Method Name"]


def fetch(url, cache, offline):
    """Download once into ``cache``, atomically, and return the path."""
    path = Path(cache) / url.rsplit("/", 1)[1]
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"{path} is missing and --offline is set")
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed hosts above)
            payload = response.read()
        atomic_write(path, lambda f: f.write(payload))
    return path


def method_family(names):
    """FRM, T640, BAM or other. FDMS and TEOM say 'Gravimetric' but are continuous FEMs."""
    filter_based = names.str.contains("gravimetric", case=False) & ~names.str.contains("FDMS|TEOM")
    return np.select([filter_based, names.str.contains("T640"), names.str.contains("BAM")],
                     ["FRM", "T640", "BAM"], "other")


def clean_daily(path, log):
    """Daily rows down to one value per site and day, keeping the decision log's counts."""
    daily = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=DAILY_COLUMNS)
    log["rows"] = len(daily)
    daily = daily[~daily["State Code"].isin(NOT_US)]
    log["rows_after_border_sites"] = len(daily)
    keep = daily["Sample Duration"] != "1 HOUR"  # same instrument: 98.6% have a 24-hour row in 2023
    log["rows_one_hour"] = int((~keep).sum())
    original = daily["Method Code"].isin(ORIGINAL_T640)
    log["rows_original_t640"] = int((original & keep).sum())
    excluded = daily["Event Type"] == "Excluded"
    log["rows_event_excluded"] = int((excluded & keep & ~original).sum())
    daily = daily[keep & ~original & ~excluded].copy()
    log["rows_kept"] = len(daily)

    daily["site"] = daily["State Code"] + daily["County Code"] + daily["Site Num"]
    daily["value"] = daily["Arithmetic Mean"].astype(float)
    daily["event"] = daily["Event Type"] == "Included"
    daily["family"] = method_family(daily["Method Name"])
    days = daily.groupby(["site", "Date Local"]).agg(
        value=("value", "mean"), event=("event", "max"), n_poc=("POC", "nunique")).reset_index()
    log["site_days"] = len(days)
    log["sites"] = int(days["site"].nunique())
    log["site_days_negative"] = int((days["value"] < 0).sum())

    shares = daily.groupby(["site", "family"]).size().unstack(fill_value=0)
    shares = shares.reindex(columns=["FRM", "T640", "BAM", "other"], fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0).add_prefix("share_")
    where = daily.groupby("site")[["Latitude", "Longitude"]].first().astype(float)
    return days, shares.join(where)


def site_year(days, year):
    """Annual mean as the mean of quarterly means, and its sampling variance.

    The variance is what a sparse schedule costs: a finite-population term per quarter from the
    site's own day-to-day spread. It goes into the annual model as known measurement error, which
    is how that model carries most of what a daily model would add.
    """
    dates = pd.to_datetime(days["Date Local"])
    work = days.assign(quarter=dates.dt.quarter)
    per = work.groupby(["site", "quarter"])["value"].agg(["mean", "count", "var"]).reset_index()
    per["N"] = per["quarter"].map(QUARTER_DAYS)
    fraction = (1 - per["count"] / per["N"]).clip(lower=0)
    per["var_of_mean"] = per["var"].fillna(0) / per["count"] * fraction
    per["good"] = per["count"] >= MIN_DAYS_PER_QUARTER
    out = per.groupby("site").agg(mean=("mean", "mean"), quarters=("mean", "size"),
                                  good_quarters=("good", "sum"), var_sum=("var_of_mean", "sum"),
                                  days=("count", "sum"))
    out["var_mean"] = out.pop("var_sum") / 16  # the variance of the mean of four means
    out["complete"] = out["good_quarters"] == 4
    out["event_days"] = work.groupby("site")["event"].sum()
    out["year"] = year
    return out.reset_index()


def smoke_ribbon(days, where, year, bands=25):
    """Share of site-days above 35 ug/m3, by day and latitude band: one small array for the ribbon."""
    lat = days["site"].map(where["Latitude"])
    band = pd.cut(lat, np.linspace(24, 50, bands + 1), labels=False)
    day = pd.to_datetime(days["Date Local"]).dt.dayofyear
    high = (days["value"] > 35).groupby([day, band]).mean().unstack()
    return high.reindex(index=range(1, 367), columns=range(bands)).assign(year=year)


def checkpoint(path, make):
    """Run ``make`` once. A finished year survives a crash, because the write is atomic."""
    if path.exists():
        return pd.read_csv(path, dtype={"site": str})
    frame = make()
    atomic_write(path, lambda f: f.write(frame.to_csv(index=False).encode()))
    return frame


def monitor_table(path):
    """Per site: first year of 88101 data, spatial scale, and whether it is registered without data.

    A blank first year marks a registered monitor with no reported data. What that means is not
    documented in anything read so far (open question 6), so it is only flagged here.
    """
    columns = ["State Code", "County Code", "Site Number", "Parameter Code", "First Year of Data",
               "Measurement Scale"]
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as member:
        monitors = pd.read_csv(member, dtype=str, keep_default_na=False, usecols=columns)
    monitors = monitors[monitors["Parameter Code"] == "88101"].copy()
    monitors["site"] = monitors["State Code"] + monitors["County Code"] + monitors["Site Number"]
    monitors["first"] = pd.to_numeric(monitors["First Year of Data"], errors="coerce")
    grouped = monitors.groupby("site")
    table = pd.DataFrame({"first_year": grouped["first"].min(),
                          "scale": grouped["Measurement Scale"].agg(lambda s: s.mode().iloc[0])})
    table["registered_only"] = table["first_year"].isna()
    return table


def window(years, shares):
    """One row per site for the decision window, from complete years only."""
    complete = years[years["complete"]]
    grouped = complete.groupby("site")
    out = pd.DataFrame({"mean": grouped["mean"].mean(), "n_years": grouped["mean"].size(),
                        "days": grouped["days"].sum(), "year_sd": grouped["mean"].std(),
                        "event_days": grouped["event_days"].sum()})
    # Year flags: the model needs them, because 2023 ran 1.2 ug/m3 above 2024 at common sites, so a
    # site that misses 2023 (a new site) or 2025 (the Bay Area) is not comparable without an offset.
    flags = complete.pivot_table(index="site", columns="year", values="mean", aggfunc="size")
    out = out.join(flags.notna().add_prefix("has_"))
    out["var_mean"] = grouped["var_mean"].sum() / out["n_years"] ** 2
    out["log_mean"] = np.log(out["mean"].clip(lower=0.5))
    out["var_log"] = out["var_mean"] / out["mean"].clip(lower=0.5) ** 2  # delta method
    out = out.join(shares)
    out["conus"] = ~out.index.str[:2].isin(NOT_CONUS)
    out["small_crop"] = out.index.str[:2].isin(SMALL_CROP)
    return out.reset_index()


def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5, radius=6371.0):
    """Albers equal-area conic in km, written out so that no GIS library is needed."""
    lon, lat, lon0, lat0, p1, p2 = map(np.radians, (lon, lat, lon0, lat0, p1, p2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = radius * np.sqrt(c - 2 * n * np.sin(lat)) / n
    rho0 = radius * np.sqrt(c - 2 * n * np.sin(lat0)) / n
    theta = n * (lon - lon0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def polygon_stats(geometry):
    """Area in km2 and area-weighted centroid in lon, lat, from outer rings in Albers coordinates."""
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    rings = [polygon[0] for polygon in polygons]
    area, sum_lon, sum_lat = 0.0, 0.0, 0.0
    for ring in rings:
        lon, lat = np.asarray(ring, float).T[:2]
        x, y = albers(lon, lat)
        a = abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])) / 2
        area += a
        sum_lon, sum_lat = sum_lon + a * lon.mean(), sum_lat + a * lat.mean()
    return area, sum_lon / area, sum_lat / area  # the mean of a ring's vertices: fine for a fallback


def county_table(census_path, cenpop_path, monitored, log):
    """One row per county: population, density, and a location that a monitor could go to.

    Census 2024 has Connecticut's nine planning regions, while AQS and the geojson have its eight
    old counties, so Connecticut takes its population from the 2020 centre-of-population file.
    """
    census = pd.read_csv(census_path, dtype=str, encoding="latin-1", keep_default_na=False,
                         usecols=["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME",
                                  "POPESTIMATE2024"])
    census = census[census["SUMLEV"] == "050"]
    census = pd.DataFrame({"fips": census["STATE"] + census["COUNTY"],
                           "state": census["STNAME"].to_numpy(dtype=object),
                           "county": census["CTYNAME"].to_numpy(dtype=object),
                           "population": census["POPESTIMATE2024"].astype(int)}).set_index("fips")

    rows = []
    for feature in json.loads(GEOJSON.read_text())["features"]:
        fips = FIPS_FIXES.get(feature["id"], feature["id"])
        if fips[:2] in NOT_CONUS or fips == "51019" and feature["id"] == "51515":
            continue
        area, lon, lat = polygon_stats(feature["geometry"])
        rows.append((fips, area, lon, lat))
    shapes = pd.DataFrame(rows, columns=["fips", "area_km2", "lon", "lat"]).set_index("fips")
    table = shapes.join(census, how="left")

    centres = "polygon"
    if cenpop_path is not None:
        # Column names are from memory of this file: it was NOT downloaded (pilot step 1).
        pop = pd.read_csv(cenpop_path, dtype=str, encoding="utf-8-sig")
        pop["fips"] = pop["STATEFP"] + pop["COUNTYFP"]
        pop = pop.set_index("fips")
        table.loc[table.index.intersection(pop.index), ["lat", "lon"]] = \
            pop[["LATITUDE", "LONGITUDE"]].astype(float)
        connecticut = table.index.str.startswith("09")
        table.loc[connecticut, "population"] = pop["POPULATION"].astype(float)
        centres = "population"
    log["county_location"] = centres
    log["counties_without_population"] = table.index[table["population"].isna()].tolist()
    table = table.dropna(subset=["population"]).copy()
    table["x_km"], table["y_km"] = albers(table["lon"].to_numpy(), table["lat"].to_numpy())
    table["log_density"] = np.log(table["population"] / table["area_km2"])
    table["monitored"] = table.index.isin(monitored)
    table["small_crop"] = table.index.str[:2].isin(SMALL_CROP)
    log["counties"], log["counties_unmonitored"] = len(table), int((~table["monitored"]).sum())
    return table.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True, help="where raw downloads and checkpoints live")
    parser.add_argument("--out", default="data", help="where the two final tables go")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--offline", action="store_true", help="never download")
    parser.add_argument("--no-cenpop", action="store_true",
                        help="testing only: polygon centres, and Connecticut is dropped")
    args = parser.parse_args()
    cache, out = Path(args.cache), Path(args.out)

    log, sources, years, ribbons, shares = {}, {}, [], [], []
    for year in args.years:
        path = fetch(f"{AIRDATA}/daily_88101_{year}.zip", cache, args.offline)
        sources[path.name] = sha256_file(path)
        log[str(year)] = {}

        def make(year=year, path=path):
            with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as member:
                days, site_info = clean_daily(io.BytesIO(member.read()), log[str(year)])
            atomic_write(cache / f"e89_sites_{year}.csv",
                         lambda f: f.write(site_info.reset_index().to_csv(index=False).encode()))
            ribbon = smoke_ribbon(days, site_info, year)
            atomic_write(cache / f"e89_ribbon_{year}.csv",
                         lambda f: f.write(ribbon.to_csv().encode()))
            return site_year(days, year)

        years.append(checkpoint(cache / f"e89_site_year_{year}.csv", make))
        shares.append(pd.read_csv(cache / f"e89_sites_{year}.csv", dtype={"site": str}))

    site_info = pd.concat(shares).groupby("site").mean(numeric_only=True)
    sites = window(pd.concat(years, ignore_index=True), site_info)
    monitors_path = fetch(f"{AIRDATA}/aqs_monitors.zip", cache, args.offline)
    sources[monitors_path.name] = sha256_file(monitors_path)
    monitors = monitor_table(monitors_path)
    sites = sites.merge(monitors[["first_year", "scale"]], how="left", left_on="site",
                        right_index=True)
    log["sites_in_window"] = len(sites)
    log["sites_by_complete_years"] = {str(k): int(v) for k, v in
                                      sites["n_years"].value_counts().sort_index().items()}

    census = fetch(f"{CENSUS}/co-est2024-alldata.csv", cache, args.offline)
    cenpop = None if args.no_cenpop else fetch(CENPOP, cache, args.offline)
    sources[census.name] = sha256_file(census)
    if cenpop is not None:
        sources[cenpop.name] = sha256_file(cenpop)
    monitored = set(sites["site"].str[:5])
    counties = county_table(census, cenpop, monitored, log)
    registered = set(monitors.index[monitors["registered_only"]].str[:5])
    counties["registered_only"] = counties["fips"].isin(registered - monitored)
    log["counties_registered_only"] = int(counties["registered_only"].sum())

    atomic_write(out / "e89_sites.csv",
                 lambda f: f.write(sites.to_csv(index=False, float_format="%.5g").encode()))
    atomic_write(out / "e89_counties.csv",
                 lambda f: f.write(counties.to_csv(index=False, float_format="%.6g").encode()))
    record = json.dumps({"sources": sources, "log": log}, indent=1).encode()
    atomic_write(out / "e89_build_log.json", lambda f: f.write(record))
    assert math.isfinite(sites["log_mean"].sum()), "a site has a non-finite log mean"
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
