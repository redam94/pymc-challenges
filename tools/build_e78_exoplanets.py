"""Build the E78 datasets: Kepler-10 light curves, Exoplanet Archive parameters, and the astrometry
and observer geometry of the interstellar comet 2I/Borisov.

    uv run --no-project --with astropy --with pandas --with numpy --with truststore \
        python tools/build_e78_exoplanets.py [kepler archive mpc horizons sbdb]

astropy is only needed here, to read the Kepler FITS files once; the notebook reads the CSVs.

* data/kepler10_llc.csv.gz - Kepler long-cadence (29.4 min) light curves of Kepler-10 (KIC 11904151)
  from MAST, all 15 quarters on the archive (Q0-Q17 minus the quarters the star fell on the failed
  CCD module 3): quarter, time (BKJD = BJD_TDB - 2454833), flux and flux_err (PDCSAP, e-/s),
  quality (SAP_QUALITY bit mask). Rows with a non-finite time are dropped; everything else is kept.
* data/kepler10_archive.csv - the NASA Exoplanet Archive Planetary Systems (ps) rows for Kepler-10 b
  and c (TAP query), selected columns, reference names without HTML.
* data/borisov_mpc_obs.csv - Minor Planet Center optical astrometry of 2I/Borisov (ADES fields
  from the MPC observations API) from discovery (2019-08-30) to 2019-10-15 UTC: obstime (ISO, UTC),
  ra, dec (degrees, as reported), stn (MPC observatory code), mag, band, astcat, rmsra, rmsdec
  (arcsec, where the observer reported them), mode, notes, and for observations from a satellite
  (NEOSSat, C53) its geocentric position: sys (ICRF_KM), pos1, pos2, pos3 (km).
* data/borisov_obscodes.csv - MPC observatory codes used above: stn, longitude (deg E),
  rhocosphi, rhosinphi (parallax constants, Earth radii), name.
* data/borisov_horizons.csv - JPL Horizons geometric heliocentric states (ICRF, au and au/day,
  TDB): Earth (399) hourly, Jupiter (599) and Saturn (699) barycentres every 6 h, and JPL's own
  orbit solution for 2I (JPL#54, with non-gravitational terms) every 6 h, 2019-08-29 to 2019-10-17.
* data/borisov_sbdb.csv - the JPL Small-Body Database orbit of 2I/Borisov: element, value, sigma,
  units (plus orbit id and data arc as rows).
"""

from __future__ import annotations

import html
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
KIC = "011904151"
MAST = f"https://archive.stsci.edu/missions/kepler/lightcurves/{KIC[:4]}/{KIC}/"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
MPC_OBS = "https://data.minorplanetcenter.net/api/get-obs"
MPC_CODES = "https://data.minorplanetcenter.net/api/obscodes"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=2I&full-prec=1"
UA = {"User-Agent": "pymc-challenges"}

try:  # use the operating system's certificate store (some networks re-sign TLS traffic)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def fetch(url: str, data: bytes | None = None, method: str | None = None) -> bytes:
    headers = dict(UA)
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def kepler() -> None:
    from astropy.io import fits

    index = fetch(MAST).decode()
    names = sorted(set(re.findall(rf"kplr{KIC}-\d+_llc\.fits", index)))
    rows = []
    for name in names:
        with fits.open(io.BytesIO(fetch(MAST + name))) as hdu:
            q = int(hdu[0].header["QUARTER"])
            d = hdu[1].data
            t = np.asarray(d["TIME"], float)
            ok = np.isfinite(t)
            rows.append(pd.DataFrame({
                "quarter": q, "time": t[ok],
                "flux": np.asarray(d["PDCSAP_FLUX"], float)[ok],
                "flux_err": np.asarray(d["PDCSAP_FLUX_ERR"], float)[ok],
                "quality": np.asarray(d["SAP_QUALITY"], int)[ok]}))
        print(name, q, ok.sum())
    df = pd.concat(rows).sort_values("time")
    df.to_csv(DATA / "kepler10_llc.csv.gz", index=False, float_format="%.7f")
    print("kepler10_llc", len(df))


def archive() -> None:
    cols = ("pl_name,default_flag,pl_orbper,pl_orbpererr1,pl_tranmid,pl_rade,pl_radeerr1,pl_ratror,"
            "pl_ratrorerr1,pl_imppar,pl_imppareerr1,pl_trandur,pl_trandep,pl_ratdor,pl_orbeccen,"
            "st_rad,st_raderr1,st_raderr2,st_mass,st_masserr1,st_masserr2,st_dens,st_denserr1,st_denserr2,st_teff,"
            "st_logg,pl_refname,disc_year")
    cols = cols.replace("pl_imppareerr1", "pl_impparerr1")
    q = f"select {cols} from ps where hostname='Kepler-10'"
    url = TAP + "?" + urllib.parse.urlencode({"query": q, "format": "csv"})
    df = pd.read_csv(io.BytesIO(fetch(url)))
    df["pl_refname"] = (df["pl_refname"].str.replace(r"<[^>]+>", "", regex=True).str.strip()
                        .map(html.unescape))
    df.sort_values(["pl_name", "default_flag"], ascending=[True, False]).to_csv(
        DATA / "kepler10_archive.csv", index=False)
    print("kepler10_archive", len(df))


def mpc() -> None:
    body = json.dumps({"desigs": ["2I"], "output_format": ["ADES_DF"]}).encode()
    ades = json.loads(fetch(MPC_OBS, data=body, method="GET"))[0]["ADES_DF"]
    df = pd.DataFrame(ades)
    df = df[df["Obstype"] == "optical"]
    df["t"] = pd.to_datetime(df["obstime"], format="ISO8601", utc=True)
    df = df[(df["t"] >= pd.Timestamp("2019-08-30", tz="UTC")) & (df["t"] < pd.Timestamp("2019-10-16", tz="UTC"))]
    df = df[df["deprecated"].isna()]
    keep = ["obstime", "ra", "dec", "stn", "mag", "band", "astcat", "rmsra", "rmsdec", "mode", "notes",
            "sys", "pos1", "pos2", "pos3"]
    df = df.sort_values("t")[keep]
    df.to_csv(DATA / "borisov_mpc_obs.csv", index=False)
    print("borisov_mpc_obs", len(df), df["stn"].nunique(), "stations")
    codes = []
    for stn in sorted(df["stn"].unique()):
        c = json.loads(fetch(MPC_CODES, data=json.dumps({"obscode": stn}).encode(), method="GET"))
        codes.append({"stn": stn, "longitude": c["longitude"], "rhocosphi": c["rhocosphi"],
                      "rhosinphi": c["rhosinphi"], "name": c["name"].strip()})
    pd.DataFrame(codes).to_csv(DATA / "borisov_obscodes.csv", index=False)
    print("borisov_obscodes", len(codes))


def horizons_vectors(command: str, step: str) -> pd.DataFrame:
    params = {"format": "text", "COMMAND": f"'{command}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
              "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@10'", "START_TIME": "'2019-08-29'",
              "STOP_TIME": "'2019-10-17'", "STEP_SIZE": f"'{step}'", "REF_PLANE": "'FRAME'",
              "REF_SYSTEM": "'ICRF'", "VEC_TABLE": "'2'", "CSV_FORMAT": "'YES'", "OUT_UNITS": "'AU-D'"}
    text = fetch(HORIZONS + "?" + urllib.parse.urlencode(params)).decode()
    block = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    rows = [[float(p) for i, p in enumerate(line.split(",")[:-1]) if i != 1] for line in block]
    return pd.DataFrame(rows, columns=["jd_tdb", "x", "y", "z", "vx", "vy", "vz"])


def horizons() -> None:
    out = []
    for body, cmd, step in [("earth", "399", "1h"), ("jupiter", "5", "6h"), ("saturn", "6", "6h"),
                            ("2I_jpl", "DES=1003639;", "6h")]:
        df = horizons_vectors(cmd, step)
        df.insert(0, "body", body)
        out.append(df)
        print(body, len(df))
    pd.concat(out).to_csv(DATA / "borisov_horizons.csv", index=False, float_format="%.12g")


def sbdb() -> None:
    d = json.loads(fetch(SBDB))
    orb = d["orbit"]
    rows = [{"element": e["name"], "value": e["value"], "sigma": e["sigma"], "units": e["units"]}
            for e in orb["elements"]]
    rows += [{"element": p["name"], "value": p["value"], "sigma": p["sigma"], "units": p["units"]}
             for p in orb.get("model_pars", [])]
    for k in ["orbit_id", "epoch", "data_arc", "first_obs", "last_obs", "n_obs_used", "producer"]:
        rows.append({"element": k, "value": orb[k], "sigma": None, "units": None})
    pd.DataFrame(rows).to_csv(DATA / "borisov_sbdb.csv", index=False)
    print("borisov_sbdb", len(rows))


if __name__ == "__main__":
    import sys

    steps = {"kepler": kepler, "archive": archive, "mpc": mpc, "horizons": horizons, "sbdb": sbdb}
    for step in sys.argv[1:] or steps:
        steps[step]()
