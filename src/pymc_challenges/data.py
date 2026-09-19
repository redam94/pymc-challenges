"""Real-world datasets used by the examples and challenges.

Every dataset is a real, publicly available file. Files are cached in the
repository's ``data/`` directory; if one is missing it is downloaded on demand.

    from pymc_challenges import data
    df = data.load("golf")
    data.describe("golf")      # where it came from
    data.fetch_all()           # pre-download everything (or: python -m pymc_challenges)

``load`` only parses the file -- cleaning and feature engineering are part of
the challenges.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"
_PYMC_EXAMPLES = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data"
_RETHINKING = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data"
_STAN_GOLF = "https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/golf"


@dataclass(frozen=True)
class Dataset:
    filename: str
    url: str
    source: str
    description: str
    read_kwargs: dict = field(default_factory=dict)
    zip_member: str | None = None  # extract this member when the URL is a zip archive


REGISTRY: dict[str, Dataset] = {
    # ---- examples -------------------------------------------------------
    "howell": Dataset(
        "howell.csv",
        f"{_RETHINKING}/Howell1.csv",
        "Nancy Howell's Dobe !Kung census (1960s), via McElreath's `rethinking`.",
        "Height (cm), weight (kg), age and sex of 544 individuals.",
        {"sep": ";"},
    ),
    "radon": Dataset(
        "radon.csv",
        f"{_PYMC_EXAMPLES}/radon.csv",
        "US EPA State Residential Radon Survey (Minnesota subset), via Gelman & Hill (2007).",
        "Log radon measurements in 919 Minnesota homes across 85 counties.",
    ),
    "wells": Dataset(
        "wells.csv",
        f"{_RDATASETS}/carData/Wells.csv",
        "Bangladesh arsenic survey, Gelman & Hill (2007), via R package `carData`.",
        "Whether 3020 households switched wells after learning theirs was unsafe.",
        {"index_col": 0},
    ),
    "coal": Dataset(
        "coal.csv",
        f"{_RDATASETS}/boot/coal.csv",
        "Jarrett (1979), UK coal-mining disasters 1851-1962, via R package `boot`.",
        "Dates (decimal years) of 191 explosions that killed 10 or more miners.",
        {"index_col": 0},
    ),
    "mcycle": Dataset(
        "mcycle.csv",
        f"{_RDATASETS}/MASS/mcycle.csv",
        "Silverman (1985), simulated motorcycle crash experiment, via R package `MASS`.",
        "Head acceleration (g) against time after impact (ms), 133 readings.",
        {"index_col": 0},
    ),
    "mastectomy": Dataset(
        "mastectomy.csv",
        f"{_PYMC_EXAMPLES}/mastectomy.csv",
        "Survival of breast-cancer patients after mastectomy, via R package `HSAUR`.",
        "Months of follow-up, death indicator and metastasis status for 44 women.",
    ),
    # ---- challenges -----------------------------------------------------
    "golf": Dataset(
        "golf.txt",
        f"{_STAN_GOLF}/golf_data.txt",
        "Berry (1996), professional golf putting; used in Gelman & Nolan (2002).",
        "Putts attempted and made by distance (feet), 19 distances.",
        {"sep": r"\s+", "skiprows": 2},
    ),
    "golf_new": Dataset(
        "golf_new.txt",
        f"{_STAN_GOLF}/golf_data_new.txt",
        "Mark Broadie's modern PGA putting data, via Gelman's golf case study.",
        "Putts attempted and made by distance (feet), 31 distances, far more putts.",
        {"sep": r"\s+", "skiprows": 2},
    ),
    "epl_2425": Dataset(
        "epl_2425.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "football-data.co.uk, English Premier League 2024/25.",
        "All 380 match results with goals, shots and bookmaker odds.",
    ),
    "epl_2526": Dataset(
        "epl_2526.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
        "football-data.co.uk, English Premier League 2025/26.",
        "All 380 match results with goals, shots and bookmaker odds.",
    ),
    "bike_day": Dataset(
        "bike_day.csv",
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "Fanaee-T & Gama (2013), Capital Bikeshare (Washington DC), UCI ML Repository.",
        "Daily rental counts 2011-2012 with weather and calendar covariates.",
        {"parse_dates": ["dteday"]},
        zip_member="day.csv",
    ),
    "bike_hour": Dataset(
        "bike_hour.csv",
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "Fanaee-T & Gama (2013), Capital Bikeshare (Washington DC), UCI ML Repository.",
        "Hourly rental counts 2011-2012 with weather and calendar covariates.",
        {"parse_dates": ["dteday"]},
        zip_member="hour.csv",
    ),
    "telco": Dataset(
        "telco_churn.csv",
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "IBM sample data, Telco customer churn.",
        "Tenure (months), churn flag and contract details for 7043 customers.",
    ),
    "trolley": Dataset(
        "trolley.csv",
        f"{_RETHINKING}/Trolley.csv",
        "Cushman, Young & Hauser (2006) moral-judgement experiment, via `rethinking`.",
        "9930 ordinal (1-7) permissibility ratings from 331 participants.",
        {"sep": ";"},
    ),
    "galaxies": Dataset(
        "galaxies.csv",
        f"{_RDATASETS}/MASS/galaxies.csv",
        "Postman, Huchra & Geller (1986) via Roeder (1990) and R package `MASS`.",
        "Velocities (km/s) of 82 galaxies in the Corona Borealis region.",
        {"index_col": 0},
    ),
    "sp500": Dataset(
        "sp500.csv",
        f"{_PYMC_EXAMPLES}/SP500.csv",
        "S&P 500 index daily closes (2008-2019), via pymc-examples.",
        "Daily close and log-return ('change').",
        {"parse_dates": ["Date"], "index_col": "Date"},
    ),
    "co2": Dataset(
        "co2_mauna_loa.csv",
        "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        "NOAA Global Monitoring Laboratory, Mauna Loa monthly mean CO2.",
        "Monthly mean atmospheric CO2 (ppm) from March 1958 to the present.",
        {"comment": "#"},
    ),
    "nsw": Dataset(
        "nsw_experiment.csv",
        f"{_RDATASETS}/causaldata/nsw_mixtape.csv",
        "LaLonde (1986) / Dehejia & Wahba (1999), National Supported Work experiment.",
        "Randomised job-training experiment: 185 treated, 260 controls, earnings in 1978.",
        {"index_col": 0},
    ),
    "cps": Dataset(
        "cps_controls.csv",
        f"{_RDATASETS}/causaldata/cps_mixtape.csv",
        "Current Population Survey comparison sample used by LaLonde (1986).",
        "15,992 non-experimental 'controls' with the same covariates as `nsw`.",
        {"index_col": 0},
    ),
    "nba_fouls": Dataset(
        "nba_fouls.csv",
        f"{_PYMC_EXAMPLES}/item_response_nba.csv",
        "NBA Last Two Minute reports 2015-2021, via pymc-examples.",
        "Foul calls / non-calls in close games, with committing and disadvantaged player.",
        {"index_col": 0},
    ),
    # ---- advanced examples -----------------------------------------------
    "oklahoma_quakes": Dataset(
        "oklahoma_quakes.csv",
        "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2005-01-01&endtime=2020-01-01"
        "&minmagnitude=3&minlatitude=33.6&maxlatitude=37.1&minlongitude=-100.1&maxlongitude=-94.4&orderby=time-asc",
        "USGS ComCat earthquake catalogue (FDSN event web service).",
        "Every magnitude 3+ earthquake in and around Oklahoma, 2005-2019: time, location, depth, magnitude.",
        {"parse_dates": ["time"]},
    ),
    "supernovae": Dataset(
        "supernovae_union21.txt",
        "https://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt",
        "Supernova Cosmology Project, Union2.1 compilation (Suzuki et al. 2012).",
        "580 type Ia supernovae: redshift z, distance modulus mu and its error.",
        {"sep": "\t", "comment": "#", "header": None, "names": ["name", "z", "mu", "mu_err", "p_low_mass_host"]},
    ),
    "lynx_hare": Dataset(
        "lynx_hare.csv",
        "https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/lotka-volterra/hudson-bay-lynx-hare.csv",
        "Hudson's Bay Company pelt records 1900-1920 (thousands), via the Stan Lotka-Volterra case study.",
        "Annual lynx and snowshoe-hare pelts: the classic predator-prey cycle.",
        {"comment": "#", "skipinitialspace": True},
    ),
    "lynx_long": Dataset(
        "lynx_long.csv",
        f"{_RDATASETS}/datasets/lynx.csv",
        "Brockwell & Davis (1991) / Campbell & Walker (1977), via R `datasets::lynx`.",
        "Annual Canadian lynx trappings on the Mackenzie River, 1821-1934 (114 years, predator only).",
        {"index_col": 0},
    ),
    "seatbelts": Dataset(
        "seatbelts.csv",
        f"{_RDATASETS}/datasets/Seatbelts.csv",
        "Harvey & Durbin (1986), UK Department of Transport, via R `datasets::Seatbelts`.",
        "Monthly UK road casualties 1969-1984 with petrol price, distance driven and the Jan 1983 seat-belt law.",
        {"index_col": 0},
    ),
    "penguins_raw": Dataset(
        "penguins_raw.csv",
        f"{_RDATASETS}/palmerpenguins/penguins_raw.csv",
        "Gorman, Williams & Fraser (2014), Palmer Station LTER, via R `palmerpenguins` (raw field records).",
        "344 penguins as recorded in the field: messy column names, dates, comments, missing measurements.",
        {"index_col": 0},
    ),
    "waffle_divorce": Dataset(
        "waffle_divorce.csv",
        f"{_RETHINKING}/WaffleDivorce.csv",
        "US Census / American Community Survey state statistics, via McElreath's `rethinking`.",
        "Divorce and marriage rates for 50 US states WITH their standard errors, plus median age at marriage.",
        {"sep": ";"},
    ),
    "gw150914_h1": Dataset(
        "gw150914_h1.hdf5",
        "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        "LIGO Scientific Collaboration / Gravitational Wave Open Science Center (gwosc.org).",
        "32 s of LIGO Hanford strain at 4096 Hz around GW150914 (GPS 1126259462.4). HDF5: use data.path().",
    ),
    "gw150914_l1": Dataset(
        "gw150914_l1.hdf5",
        "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        "LIGO Scientific Collaboration / Gravitational Wave Open Science Center (gwosc.org).",
        "32 s of LIGO Livingston strain at 4096 Hz around GW150914 (GPS 1126259462.4). HDF5: use data.path().",
    ),
}


def _download(ds: Dataset, path: Path) -> None:
    resp = requests.get(ds.url, timeout=60, headers={"User-Agent": "pymc-challenges"})
    resp.raise_for_status()
    content = resp.content
    if ds.zip_member is not None:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            content = zf.read(ds.zip_member)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def path(name: str) -> Path:
    """Local path of a dataset, downloading it first if necessary."""
    try:
        ds = REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}") from None
    p = DATA_DIR / ds.filename
    if not p.exists():
        print(f"Downloading {name} from {ds.url}")
        _download(ds, p)
    return p


def load(name: str) -> pd.DataFrame:
    """Load a dataset as a DataFrame (parsed, but otherwise untouched)."""
    p = path(name)
    if p.suffix == ".hdf5":
        raise ValueError(f"{name!r} is an HDF5 file, not a table: open data.path({name!r}) with h5py.")
    return pd.read_csv(p, **REGISTRY[name].read_kwargs)


def describe(name: str) -> None:
    """Print what a dataset is and where it came from."""
    ds = REGISTRY[name]
    print(f"{name}: {ds.description}\n  source: {ds.source}\n  url:    {ds.url}")


def fetch_all() -> None:
    """Download every dataset that is not already cached."""
    for name in REGISTRY:
        p = path(name)
        print(f"  ok  {name:<12} {p.stat().st_size / 1024:>8.0f} KB  {p.name}")

