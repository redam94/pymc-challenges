"""Build E89's site and county tables from EPA AirData and Census files.

    OPENBLAS_NUM_THREADS=2 .venv/bin/python tools/memguard.py --max-gb 3 -- \
        .venv/bin/python tools/build_e89_data.py --cache ~/.cache/pymc-challenges/e89/raw --out data

Run on 2023-25 on 2026-11-01 (plans/large/E89.md, Pilot results): 12.5 s, 1.08 GB peak on 2 cores.
It downloads about 27 MB of daily files, the 7 MB monitor list and two small Census files, and with
``--pilot`` the 12 MB of annual files and EPA's 1.84 MB design-value workbook. Each year is
checkpointed, so a crash loses at most that year. ``--pilot`` writes pilots P1 and P2 and the plan's
arithmetic checks to ``results/E89/`` as small JSON. It fits no model.

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
import re
import urllib.request
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pymc_challenges.bigrun import Store, atomic_write, sha256_file

AIRDATA = "https://aqs.epa.gov/aqsweb/airdata"
CENSUS = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals"
CENPOP = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
DESIGN_VALUES = ("https://www.epa.gov/system/files/documents/2026-06/"
                 "pm25_designvalues_2023_2025_final_06_12_26.xlsx")
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
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed hosts)
            payload = response.read()
        atomic_write(path, lambda f: f.write(payload))
    return path


def method_family(names):
    """FRM, T640, BAM or other. FDMS and TEOM say 'Gravimetric' but are continuous FEMs."""
    filter_based = names.str.contains("gravimetric", case=False) & ~names.str.contains("FDMS|TEOM")
    return np.select([filter_based, names.str.contains("T640"), names.str.contains("BAM")],
                     ["FRM", "T640", "BAM"], "other")


def clean_rows(path, log):
    """The monitor-day rows that survive the filters, keeping the decision log's counts."""
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
    return daily


def clean_daily(path, log):
    """Daily rows down to one value per site and day."""
    daily = clean_rows(path, log)
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


def t640_pairs(cache, offline, year=2023):
    """Site means of the original and corrected T640 series on the days both report (finding 1).

    One row per site and pair of methods, from the raw rows, before ``clean_rows`` drops the original.
    The notebook draws these, so that the failure it describes can be seen, not only counted.
    """
    path = fetch(f"{AIRDATA}/daily_88101_{year}.zip", cache, offline)
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as member:
        daily = pd.read_csv(member, dtype=str, keep_default_na=False, usecols=DAILY_COLUMNS)
    daily = daily[(daily["Sample Duration"] != "1 HOUR") & ~daily["State Code"].isin(NOT_US)
                  & (daily["Event Type"] != "Excluded")].copy()
    daily["site"] = daily["State Code"] + daily["County Code"] + daily["Site Num"]
    daily["value"] = daily["Arithmetic Mean"].astype(float)
    rows = []
    for original, corrected in (("236", "736"), ("238", "738")):
        part = daily[daily["Method Code"].isin([original, corrected])]
        wide = part.groupby(["site", "Date Local", "Method Code"])["value"].mean().unstack().dropna()
        site = wide.groupby("site").agg(days=(original, "size"), original=(original, "mean"),
                                        corrected=(corrected, "mean"))
        rows.append(site.reset_index().assign(pair=f"{original} vs {corrected}"))
    return pd.concat(rows, ignore_index=True)


def ribbon_table(cache, years):
    """Share of site-days over 35 ug/m3, by day of year (one row) and latitude band."""
    frames = [pd.read_csv(cache / f"e89_ribbon_{year}.csv") for year in years]
    return pd.concat(frames, ignore_index=True).rename(columns={"Date Local": "day"})


def check_table():
    """What the notebook quotes from pilots P1, P2 and P4, none of it site-level EPA data."""
    store = Store("E89")
    tables, epa = store.load_json("p1_tables"), store.load_json("p2_design_values")
    excluded = store.load_json("p4_excluded_sites")
    return {"excluded_sites": excluded["conus_sites"], "excluded_rule": excluded["rule"],
            "schedule_error": {year: tables[year]["schedule"] for year in ("2023", "2024", "2025")},
            "epa_comparison": {name: epa[name] for name in (
                "gate_no_concurred_events", "all_valid_sites", "frm_and_t640_mixed")}}


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
        pop = pd.read_csv(cenpop_path, dtype=str, encoding="utf-8-sig")
        pop["fips"] = pop["STATEFP"] + pop["COUNTYFP"]
        pop = pop.set_index("fips")
        found = table.index.intersection(pop.index)
        log["counties_on_polygon_centre"] = table.index.difference(pop.index).tolist()
        # .to_numpy(): assigning a frame aligns on column names, and these differ, so every
        # coordinate silently became NaN in the first full run.
        table.loc[found, ["lat", "lon"]] = pop.loc[found, ["LATITUDE", "LONGITUDE"]].astype(
            float).to_numpy()
        connecticut = table.index[table.index.str.startswith("09")]
        table.loc[connecticut, "population"] = pop.loc[connecticut, "POPULATION"].astype(
            float).to_numpy()
        table.loc[connecticut, "state"] = pop.loc[connecticut, "STNAME"].to_numpy(dtype=object)
        table.loc[connecticut, "county"] = pop.loc[connecticut, "COUNAME"].to_numpy(dtype=object)
        centres = "population"
    log["county_location"] = centres
    log["counties_without_population"] = table.index[table["population"].isna()].tolist()
    table = table.dropna(subset=["population"]).copy()
    assert table[["lat", "lon"]].notna().all(axis=None), "a county has no location"
    table["x_km"], table["y_km"] = albers(table["lon"].to_numpy(), table["lat"].to_numpy())
    table["log_density"] = np.log(table["population"] / table["area_km2"])
    table["monitored"] = table.index.isin(monitored)
    table["small_crop"] = table.index.str[:2].isin(SMALL_CROP)
    log["counties"], log["counties_unmonitored"] = len(table), int((~table["monitored"]).sum())
    return table.reset_index()


def read_xlsx(path, sheet):
    """One sheet as a list of {column index: text} rows. An xlsx is a zip of XML: no openpyxl."""
    main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_id = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    def parse(archive, name):
        text = archive.read(name)
        if b"<!DOCTYPE" in text:  # entity expansion is the one way XML can hurt this reader
            raise ValueError(f"{name} declares a DOCTYPE; a real workbook does not")
        return ElementTree.fromstring(text)

    def column(reference):
        letters = re.match(r"[A-Z]+", reference).group()
        return sum((ord(c) - 64) * 26 ** i for i, c in enumerate(reversed(letters))) - 1

    with zipfile.ZipFile(path) as archive:
        shared = ["".join(t.text or "" for t in item.iter(main + "t"))
                  for item in parse(archive, "xl/sharedStrings.xml")]
        book = parse(archive, "xl/workbook.xml")
        target = next(s.get(rel_id) for s in book.find(main + "sheets") if s.get("name") == sheet)
        links = {r.get("Id"): r.get("Target") for r in parse(archive, "xl/_rels/workbook.xml.rels")}
        root = parse(archive, "xl/" + links[target])
    rows = []
    for row in root.iter(main + "row"):
        cells = {}
        for cell in row:
            value = cell.find(main + "v")
            if value is not None:
                text = shared[int(value.text)] if cell.get("t") == "s" else value.text
                cells[column(cell.get("r"))] = text
        rows.append(cells)
    return rows


def design_values(path):
    """EPA's site table: valid and invalid 2023-25 design values and each year's annual mean."""
    rows = read_xlsx(path, "Table5a. Site Status Annual")
    head = next(row for row in rows if "AQS Site ID" in row.values())
    names = {key: " ".join(value.split()) for key, value in head.items()}
    site = next(key for key, name in names.items() if name == "AQS Site ID")
    table = pd.DataFrame([{names[k]: v for k, v in row.items() if k in names} for row in rows
                          if row.get(site, "").strip().isdigit()])
    out = pd.DataFrame({"site": table["AQS Site ID"]})
    for name, prefix in (("dv", "Valid"), ("dv_invalid", "Invalid")):
        out[name] = pd.to_numeric(table[next(c for c in table if c.startswith(prefix))],
                                  errors="coerce")
    for year in (2023, 2024, 2025):
        out[f"epa_{year}"] = pd.to_numeric(table[f"{year} Annual Mean Value (µg/m3)"],
                                           errors="coerce")
    return out


def annual_rows(cache, year, offline):
    """The annual file's rows for the 2024 standard, one per monitor and event type."""
    path = fetch(f"{AIRDATA}/annual_conc_by_monitor_{year}.zip", cache, offline)
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as member:
        rows = pd.read_csv(member, dtype=str, keep_default_na=False)
    rows = rows[(rows["Parameter Code"] == "88101")
                & (rows["Pollutant Standard"] == "PM25 Annual 2024")].copy()
    rows["site"] = rows["State Code"] + rows["County Code"] + rows["Site Num"]
    rows["value"] = rows["Arithmetic Mean"].astype(float)
    return rows


def event_effect(rows):
    """Events Included minus Concurred Events Excluded, leaving out mixed monitors (finding 14).

    A blank method is kept: that is the corrected T640 POC. "Multiple Methods Used" rows still hold
    the original T640 data in 2023.
    """
    keys = ["site", "POC"]
    with_events = rows[(rows["Event Type"] == "Events Included")
                       & (rows["Method Name"] != "Multiple Methods Used")]
    without = rows[rows["Event Type"] == "Concurred Events Excluded"]
    gap = with_events.merge(without, on=keys, suffixes=("", "_out"))
    gap = gap["value"] - gap["value_out"]
    return {"monitors": len(gap), "median": gap.median(), "p90": gap.quantile(0.9),
            "p99": gap.quantile(0.99), "over_0.5": int((gap.abs() > 0.5).sum())}


def compare(group, line=9.05):
    """How well clean window means match EPA's values: the P2 gate's three numbers, and the bias."""
    gap = group["mean"] - group["dv"]
    near = group[group["dv"].between(8.0, 10.0)]
    disagree = (group["mean"] >= line) != (group["dv"] >= line)
    disagree_near = (near["mean"] >= line) != (near["dv"] >= line)
    return {"sites": len(group), "median_abs": gap.abs().median(), "p90_abs": gap.abs().quantile(0.9),
            "p99_abs": gap.abs().quantile(0.99), "max_abs": gap.abs().max(),
            "mean_signed": gap.mean(), "agree_9.05": 1 - disagree.mean(),
            "disagree": int(disagree.sum()), "disagree_we_above": int(
                (disagree & (group["mean"] >= line)).sum()),
            "sites_dv_8_10": len(near), "agree_9.05_dv_8_10": 1 - disagree_near.mean()}


def design_value_check(sites, table, concurred):
    """Pilot P2. Window means use three complete years here, and EPA's value is rounded to 0.1."""
    both = sites.merge(table, on="site", how="outer", indicator=True)
    valid = both[both["dv"].notna() & (both["n_years"] == 3)]
    clean = valid[~valid["site"].isin(concurred)]
    mixed = (valid["share_FRM"] > 0.02) & (valid["share_T640"] > 0.02)
    one_family = valid[["share_FRM", "share_T640", "share_BAM", "share_other"]].max(axis=1) > 0.98
    gate = compare(clean)
    gate["passes"] = bool(gate["median_abs"] <= 0.15 and gate["p90_abs"] <= 0.5
                          and gate["agree_9.05"] >= 0.95)
    epa_mean = valid[["epa_2023", "epa_2024", "epa_2025"]].mean(axis=1)
    return {
        "workbook_sites": len(table), "valid": int(table["dv"].notna().sum()),
        "invalid": int(table["dv_invalid"].notna().sum()),
        "valid_not_in_window": int((both["dv"].notna() & both["n_years"].isna()).sum()),
        "valid_by_our_complete_years": {
            str(int(k)): int(v) for k, v in
            both.loc[both["dv"].notna(), "n_years"].value_counts().sort_index().items()},
        "invalid_with_our_three_years": int((both["dv_invalid"].notna()
                                             & (both["n_years"] == 3)).sum()),
        "workbook_only": int((both["_merge"] == "right_only").sum()),
        "window_only": int((both["_merge"] == "left_only").sum()),
        "window_only_conus": int(((both["_merge"] == "left_only") & both["conus"].eq(True)).sum()),
        "gate_no_concurred_events": gate, "all_valid_sites": compare(valid),
        "frm_and_t640_mixed": compare(valid[mixed]), "one_method_family": compare(valid[one_family]),
        "one_family_no_concurred": compare(valid[one_family & ~valid["site"].isin(concurred)]),
        "unrounded_epa_means": {"median_abs": (valid["mean"] - epa_mean).abs().median(),
                                "p90_abs": (valid["mean"] - epa_mean).abs().quantile(0.9)},
    }


def schedule_error(days, year):
    """Thin near-daily sites to each schedule (finding 12): realised error against the formula."""
    full = site_year(days, year).set_index("site")
    keep = full.index[full["complete"] & (full["days"] >= 330)]
    days = days[days["site"].isin(keep)]
    day = pd.to_datetime(days["Date Local"]).dt.dayofyear
    out = {"sites": len(keep)}
    for every in (3, 6, 12):
        parts = []
        for phase in range(every):
            part = site_year(days[day % every == phase], year).set_index("site")
            part = part[(part["quarters"] == 4) & (part["var_mean"] > 0)]
            parts.append(pd.DataFrame({"error": part["mean"] - full.loc[part.index, "mean"],
                                       "var": part["var_mean"]}))
        all_parts = pd.concat(parts)
        rms = float(np.sqrt((all_parts["error"] ** 2).mean()))
        out[f"1_in_{every}"] = {"rms": rms, "formula_rms": float(np.sqrt(all_parts["var"].mean())),
                                "z_rms": float(np.sqrt((all_parts["error"] ** 2
                                                        / all_parts["var"]).mean()))}
    return out


def method_pairs(daily):
    """Each family against FRM on the same site-day, sites with 30 pairs or more (finding 13)."""
    day = daily.groupby(["site", "Date Local", "family"])["value"].mean().unstack("family")
    out = {}
    for family in ("T640", "BAM", "other"):
        gap = (day[family] - day["FRM"]).dropna().groupby("site").agg(["mean", "size"])
        gap = gap.loc[gap["size"] >= 30, "mean"]
        out[family] = {"sites": len(gap), "median": gap.median(), "p05": gap.quantile(0.05),
                       "p95": gap.quantile(0.95)}
    return out


def paired_method(cache, years, crop):
    """Each method family against FRM on the same site and day, in the crop, as a log ratio.

    The model's method term is learned across sites, where it can be confused with rural siting. These
    pairs compare two instruments on one day's air. Sites with 30 pairs or more; one mean per site.
    """
    frames = []
    for year in years:
        daily = clean_rows(Path(cache) / f"daily_88101_{year}.zip", {})
        daily = daily[daily["site"].isin(crop)]
        frames.append(daily.groupby(["site", "Date Local", "family"])["value"].mean().unstack("family"))
    day = pd.concat(frames)
    out = {}
    for family in ("T640", "BAM", "other"):
        both = day.loc[(day["FRM"] > 2) & (day[family] > 2), ["FRM", family]]
        each = both.groupby("site").agg(pairs=("FRM", "size"), frm=("FRM", "mean"), other=(family, "mean"))
        each = each[each["pairs"] >= 30]
        ratio = np.log(each["other"] / each["frm"])
        out[family] = {"sites": len(each), "site_days": int(each["pairs"].sum()), "mean": float(ratio.mean()),
                       "se": float(ratio.std() / np.sqrt(len(ratio))), "median": float(ratio.median())}
    return out


def new_site_table(sites, years, line=(8.0, 10.0)):
    """One definition of a new site, used by finding 17, F3 and P7, so they cannot disagree.

    New: first data in 2023 or later, by the monitor list, with a complete clean year in the window
    and in CONUS. Older: the same, first data before 2023. Distance: Albers km to the nearest older.
    The difference is same-year, over complete years that the pair share.
    """
    conus = sites[sites["conus"]].set_index("site")
    new, old = conus[conus["first_year"] >= 2023], conus[conus["first_year"] < 2023]
    gap_x = new["x_km"].to_numpy()[:, None] - old["x_km"].to_numpy()[None]
    gap_y = new["y_km"].to_numpy()[:, None] - old["y_km"].to_numpy()[None]
    distance = np.hypot(gap_x, gap_y)
    means = years[years["complete"]].pivot(index="site", columns="year", values="mean")
    nearest = old.index[distance.argmin(axis=1)]
    gaps = (means.loc[new.index].to_numpy() - means.loc[nearest].to_numpy())
    has_pair = np.isfinite(gaps).any(axis=1)
    table = pd.DataFrame({"km": distance.min(axis=1), "mean": new["mean"].to_numpy()},
                         index=new.index)
    table["gap"] = np.nan
    table.loc[has_pair, "gap"] = np.nanmean(gaps[has_pair], axis=1)
    out = {"new_sites": len(table), "older_sites": len(old), "median_km": table["km"].median(),
           "over_25": int((table["km"] > 25).sum()), "over_50": int((table["km"] > 50).sum()),
           "over_100": int((table["km"] > 100).sum()), "with_same_year_gap": int(has_pair.sum()),
           "over_25_in_8_10": int(((table["km"] > 25) & table["mean"].between(*line)).sum())}
    for name, low, high in (("under_25", 0, 25), ("25_50", 25, 50), ("over_50", 50, 1e9),
                            ("over_25", 25, 1e9)):
        gap = table.loc[(table["km"] > low) & (table["km"] <= high), "gap"].dropna()
        out[f"gap_{name}"] = {"sites": len(gap), "mean": gap.mean(), "median": gap.median(),
                              "se": gap.std() / len(gap) ** 0.5}
    return out


def local_counties(counties, sites, radii=(555.0, 800.0, 1200.0, 2590.0)):
    """How many counties sit near a candidate: the term that the greedy loop's cost scales with."""
    candidates = counties[~counties["monitored"] & ~counties["registered_only"]]

    def within(source, target, radius):
        gap_x = source["x_km"].to_numpy()[:, None] - target["x_km"].to_numpy()[None]
        gap_y = source["y_km"].to_numpy()[:, None] - target["y_km"].to_numpy()[None]
        return (gap_x ** 2 + gap_y ** 2 < radius ** 2).sum(axis=1)

    out = {"candidates": len(candidates),
           "candidates_small": int(candidates["small_crop"].sum())}
    crop = counties[counties["small_crop"]]
    for radius in radii:
        national, small = within(candidates, counties, radius), within(
            candidates[candidates["small_crop"]], crop, radius)
        out[f"within_{radius:.0f}_km"] = {
            "mean": float(national.mean()), "median": float(np.median(national)),
            "small_mean": float(small.mean()), "small_median": float(np.median(small))}
    # The p filter cannot be measured without a fit. Proxy: nearest site's window mean, with an
    # ASSUMED county sd, gives the share of counties whose p could lie between 0.01 and 0.99.
    conus = sites[sites["conus"]]
    gap_x = counties["x_km"].to_numpy()[:, None] - conus["x_km"].to_numpy()[None]
    gap_y = counties["y_km"].to_numpy()[:, None] - conus["y_km"].to_numpy()[None]
    proxy = conus["mean"].to_numpy()[(gap_x ** 2 + gap_y ** 2).argmin(axis=1)]
    out["share_inside_p_filter_estimated"] = {
        f"sd_{sd}": {str(line): float((np.abs(proxy - line) < 2.326 * sd).mean())
                     for line in (9.05, 12.05)} for sd in (0.5, 1.0)}
    return out


def hsgp_variance(half, c, modes, ell, where):
    """2-D Matern 3/2 HSGP prior variance at a point, for a unit-variance field (E86 S1)."""
    box = np.asarray(half, float) * c
    j1, j2 = np.meshgrid(*(np.arange(1, m + 1) for m in modes), indexing="ij")
    omega2 = (np.pi * j1 / (2 * box[0])) ** 2 + (np.pi * j2 / (2 * box[1])) ** 2
    density = 2 ** 2 * np.pi * math.gamma(2.5) * 3 ** 1.5 / (math.gamma(1.5) * ell ** 3) \
        * (3 / ell ** 2 + omega2) ** -2.5
    shape = [np.sin(np.pi * j * (x + b) / (2 * b)) ** 2 / b
             for j, x, b in zip((j1, j2), where, box)]
    return float((density * shape[0] * shape[1]).sum())


def hsgp_table(half):
    """Edge variance by boundary factor, for the mode counts the plan names. Arithmetic, no fit."""
    return [{"c": c, "modes": list(modes), "ell_km": ell,
             "centre": hsgp_variance(half, c, modes, ell, (0.0, 0.0)),
             "east_west_edge": hsgp_variance(half, c, modes, ell, (half[0], 0.0)),
             "corner": hsgp_variance(half, c, modes, ell, tuple(half))}
            for modes in ((22, 14), (40, 25)) for c in (1.25, 1.5, 2.0) for ell in (185, 400, 863)]


def pilot(cache, args, sites, counties, years, sources):
    """P1's tables and P2's comparison, as small JSON. Each number here is measured on the files."""
    store = Store("E89")
    tables, concurred = {}, set()
    for year in args.years:
        rows = annual_rows(cache, year, args.offline)
        concurred |= set(rows.loc[rows["Event Type"] == "Concurred Events Excluded", "site"])
        with zipfile.ZipFile(cache / f"daily_88101_{year}.zip") as archive, \
                archive.open(archive.namelist()[0]) as member:
            daily = clean_rows(io.BytesIO(member.read()), {})
        days = daily.groupby(["site", "Date Local"]).agg(
            value=("value", "mean"), event=("event", "max")).reset_index()
        tables[str(year)] = {
            "schedule": schedule_error(days, year), "method_vs_frm": method_pairs(daily),
            "events": event_effect(rows), "site_days_over_35": int((days["value"] > 35).sum()),
            "site_days": len(days), "max": days["value"].max(),
            "negative": int((days["value"] < 0).sum())}
    complete = years[years["complete"]].pivot(index="site", columns="year", values="mean")
    common = complete.dropna()
    conus = sites[sites["conus"]]
    three = conus[conus["n_years"] == 3].assign(county=lambda f: f["site"].str[:5])
    top = three.groupby("county")["mean"].max()
    by_year = years.groupby("year")["site"].apply(set)
    lost = by_year[2024] - by_year[2025]
    unmonitored = counties[~counties["monitored"]]
    tables["network"] = {
        "sites_in_window": len(sites), "conus_in_window": len(conus),
        "conus_by_complete_years": {str(k): int(v) for k, v in
                                    conus["n_years"].value_counts().sort_index().items()},
        "small_crop_sites": int((conus["small_crop"]).sum()),
        "small_crop_counties": int(counties["small_crop"].sum()),
        "small_crop_unmonitored": int((counties["small_crop"] & ~counties["monitored"]).sum()),
        "small_crop_population_m": counties.loc[counties["small_crop"], "population"].sum() / 1e6,
        "complete_site_years": complete.notna().sum().to_dict(),
        "common_sites": len(common), "common_means": common.mean().to_dict(),
        "within_site_year_sd": float(np.sqrt(common.var(axis=1).mean())),
        "in_2024_not_2025": len(lost), "of_which_california": sum(s[:2] == "06" for s in lost),
        "counties": len(counties), "unmonitored": len(unmonitored),
        "unmonitored_population_m": unmonitored["population"].sum() / 1e6,
        "total_population_m": counties["population"].sum() / 1e6,
        "unmonitored_over_50k": int((unmonitored["population"] > 5e4).sum()),
        "unmonitored_over_100k": int((unmonitored["population"] > 1e5).sum()),
        "unmonitored_over_500k": int((unmonitored["population"] > 5e5).sum()),
        "counties_with_three_year_site": len(top),
        "at_or_above_9.05": int((top >= 9.05).sum()), "within_1_of_9.05": int(
            ((top - 9.05).abs() <= 1).sum()), "at_or_above_12.05": int((top >= 12.05).sum()),
        "within_1_of_12.05": int(((top - 12.05).abs() <= 1).sum()),
        "half_width_km": [(counties["x_km"].max() - counties["x_km"].min()) / 2,
                          (counties["y_km"].max() - counties["y_km"].min()) / 2]}
    store.save_json("p1_tables", tables)
    workbook = fetch(DESIGN_VALUES, cache, args.offline)
    sources[workbook.name] = sha256_file(workbook)
    store.save_json("p2_design_values", {
        "workbook": workbook.name, "sha256": sources[workbook.name],
        "sites_with_concurred_events": len(concurred),
        **design_value_check(sites, design_values(workbook), concurred)})
    half = tables["network"]["half_width_km"]
    store.save_json("plan_checks", {"new_sites": new_site_table(sites, years),
                                    "local_counties": local_counties(counties, sites),
                                    "hsgp_edge_variance": hsgp_table(half)})
    store.save_json("sources", sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True, help="where raw downloads and checkpoints live")
    parser.add_argument("--out", default="data", help="where the two final tables go")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--offline", action="store_true", help="never download")
    parser.add_argument("--pilot", action="store_true",
                        help="also run pilots P1 and P2, and write results/E89/*.json")
    parser.add_argument("--no-cenpop", action="store_true",
                        help="testing only: polygon centres, and Connecticut is dropped")
    args = parser.parse_args()
    cache, out = Path(args.cache), Path(args.out)

    log, sources, years, ribbons, shares = {}, {}, [], [], []
    for year in args.years:
        path = fetch(f"{AIRDATA}/daily_88101_{year}.zip", cache, args.offline)
        sources[path.name] = sha256_file(path)
        log_path = cache / f"e89_log_{year}.json"
        log[str(year)] = json.loads(log_path.read_text()) if log_path.exists() else {}

        def make(year=year, path=path, log_path=log_path):
            with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as member:
                days, site_info = clean_daily(io.BytesIO(member.read()), log[str(year)])
            text = json.dumps(log[str(year)], indent=1).encode()
            atomic_write(log_path, lambda f: f.write(text))
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

    # The model needs a location and the density covariate at each site, not only at counties.
    sites["fips"] = sites["site"].str[:5].replace(FIPS_FIXES)
    sites["x_km"], sites["y_km"] = albers(sites["Longitude"].to_numpy(), sites["Latitude"].to_numpy())
    sites = sites.merge(counties[["fips", "log_density"]], how="left", on="fips")
    log["conus_sites_without_density"] = sites.loc[
        sites["conus"] & sites["log_density"].isna(), "site"].tolist()

    atomic_write(out / "e89_sites.csv",
                 lambda f: f.write(sites.to_csv(index=False, float_format="%.6g").encode()))
    atomic_write(out / "e89_counties.csv",
                 lambda f: f.write(counties.to_csv(index=False, float_format="%.6g").encode()))
    pairs = t640_pairs(cache, args.offline)
    atomic_write(out / "e89_t640_pairs.csv",
                 lambda f: f.write(pairs.to_csv(index=False, float_format="%.6g").encode()))
    ribbon = ribbon_table(cache, args.years)
    atomic_write(out / "e89_ribbon.csv",
                 lambda f: f.write(ribbon.to_csv(index=False, float_format="%.4g").encode()))
    if (Store("E89").path / "p4_excluded_sites.json").exists():
        crop = sites.loc[sites["conus"] & sites["small_crop"], "site"]
        checks = json.dumps({**check_table(), "paired_method": paired_method(cache, args.years, set(crop))},
                            indent=1).encode()
        atomic_write(out / "e89_checks.json", lambda f: f.write(checks))
    record = json.dumps({"sources": sources, "log": log}, indent=1).encode()
    atomic_write(out / "e89_build_log.json", lambda f: f.write(record))
    assert math.isfinite(sites["log_mean"].sum()), "a site has a non-finite log mean"
    print(json.dumps(log, indent=1))
    if args.pilot:
        pilot(cache, args, sites, counties, pd.concat(years, ignore_index=True), sources)


if __name__ == "__main__":
    main()
