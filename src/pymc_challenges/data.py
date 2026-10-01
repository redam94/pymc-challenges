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
_METADAT = "https://raw.githubusercontent.com/wviechtb/metadat/master/data-raw"


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
    "bcg": Dataset(
        "bcg_trials.txt",
        f"{_METADAT}/dat.bcg.txt",
        "Colditz et al. (1994), BCG vaccine against tuberculosis; the `metafor`/`metadat` benchmark.",
        "13 trials as 2x2 tables (TB cases and non-cases in vaccinated / control arms), year, latitude, allocation.",
        {"sep": r"\s+"},
    ),
    "magnesium": Dataset(
        "magnesium_trials.txt",
        f"{_METADAT}/dat.egger2001.txt",
        "Egger, Davey Smith & Altman (2001), intravenous magnesium after myocardial infarction, via `metadat`.",
        "16 trials: deaths and patients per arm, including the 58,050-patient ISIS-4 mega-trial.",
        {"sep": r"\s+"},
    ),
    "konstantopoulos": Dataset(
        "konstantopoulos2011.txt",
        f"{_METADAT}/dat.konstantopoulos2011.txt",
        "Konstantopoulos (2011), modified school calendars and achievement, via `metadat`.",
        "56 studies nested in 11 school districts: standardised mean difference, its variance, year.",
        {"sep": r"\s+"},
    ),
    "us_counties_geojson": Dataset(
        "us_counties.geojson",
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        "US Census Bureau county boundaries (simplified), as distributed in plotly/datasets.",
        "GeoJSON of every US county keyed by 5-digit FIPS code. Not a table: use data.path().",
    ),
    "scotland_lip_geojson": Dataset(
        "scotland_lip.geojson",
        "https://geodacenter.github.io/data-and-lab/data/scotlip.zip",
        "Clayton & Kaldor (1987) male lip cancer 1975-80 via Cressie (1993); expected counts and AFF from "
        "Lawson et al. (1999); WinBUGS boundaries prepared by Luc Anselin for the GeoDa Center data-and-lab.",
        "56 Scottish districts (British National Grid polygons, metres) with observed cases CANCER, expected "
        "cases CEXP, population POP and % in agriculture/fishing/forestry AFF. GeoJSON: use data.path().",
        zip_member="scotlip/scotlips.geojson",
    ),
    "scotland_lip_adjacency": Dataset(
        "scotland_lip_adjacency.csv",
        f"{_PYMC_EXAMPLES}/scotland_lips_cancer.csv",
        "Scottish lip cancer data with the WinBUGS GeoBUGS adjacency lists, via pymc-examples.",
        "56 districts: CANCER, CEXP, AFF and the published (hand-edited, islands linked) neighbour lists ADJ.",
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
    "sachs": Dataset(
        "sachs2005.txt",
        "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/"
        "sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt",
        "Sachs et al. (2005, Science 308:523), single-cell flow cytometry of 11 phosphoproteins/lipids in "
        "human T cells, via the CMU example-causal-datasets repository.",
        "7466 cells from 9 conditions, ALL conditions pooled. Columns 1-11 are log(x + 10) of the raw "
        "intensities (raf mek plc pip2 pip3 erk akt pka pkc p38 jnk); columns 12-20 are 0/1 condition "
        "indicators (cd3_cd28 stimulation, icam2, and the reagents aktinhib g0076 psitect u0126 ly pma "
        "b2camp). The 853 cells with cd3_cd28 = 1 and all reagents 0 are the purely observational set.",
        {"sep": "\t"},
    ),
    "sachs_consensus": Dataset(
        "sachs2005_consensus.txt",
        "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/ground.truth/"
        "sachs.2005.ground.truth.graph.txt",
        "Consensus signalling network of Sachs et al. (2005, Fig. 2/3), as curated in CMU example-causal-datasets.",
        "20 directed edges between the 11 `sachs` variables, lines like '1. erk --> akt'. Not a table: "
        "parse data.path().",
    ),
    "lazega_cowork": Dataset(
        "lazega_cowork.txt",
        "https://www.stats.ox.ac.uk/~snijders/siena/LazegaLawyers.zip",
        "Lazega (2001), The Collegial Phenomenon (Oxford UP): lawyers of a New England corporate law firm, "
        "1988-1991; distributed by T. Snijders' SIENA data page (file ELwork.dat).",
        "71 x 71 symmetric 0/1 strong-coworker matrix (378 ties) among the firm's 36 partners (rows 1-36) "
        "and 35 associates, row i = lawyer i of `lazega_attributes`.",
        {"sep": r"\s+", "header": None},
        zip_member="ELwork.dat",
    ),
    "lazega_attributes": Dataset(
        "lazega_attributes.txt",
        "https://www.stats.ox.ac.uk/~snijders/siena/LazegaLawyers.zip",
        "Lazega (2001), via T. Snijders' SIENA data page (file ELattr.dat).",
        "Attributes of the 71 lawyers: seniority rank, status (1 partner, 2 associate), gender (1 man, "
        "2 woman), office (1 Boston, 2 Hartford, 3 Providence), years with the firm, age, practice "
        "(1 litigation, 2 corporate), law school (1 Harvard/Yale, 2 UConn, 3 other).",
        {"sep": r"\s+", "header": None,
         "names": ["seniority", "status", "gender", "office", "years", "age", "practice", "school"]},
        zip_member="ELattr.dat",
    ),
    "primates301": Dataset(
        "primates301.csv",
        f"{_RETHINKING}/Primates301.csv",
        "Street, Navarrete, Reader & Laland (2017, PNAS 114:7908), primate life history and brain size, "
        "via McElreath's `rethinking`.",
        "301 primate species: brain volume (cc), body mass (g), mean group size, gestation, longevity, "
        "social learning counts and research effort; many NAs (e.g. 114 species lack group size).",
        {"sep": ";"},
    ),
    "primates301_tree": Dataset(
        "primates301_tree.rda",
        f"{_RETHINKING}/Primates301_nex.rda",
        "10kTrees consensus phylogeny (Arnold, Matthews & Nunn 2010, Evol. Anthropol. 19:114) of the 301 "
        "`primates301` species, as an R `ape::phylo` object in McElreath's `rethinking`.",
        "Rooted ultrametric tree (branch lengths in Myr, depth 73): edge matrix, edge.length, Nnode, "
        "tip.label. Not a table: an R serialisation (gzip XDR) - parse data.path() (see E23).",
    ),
    "ff_industries12_daily": Dataset(
        "ff_12_industries_daily.csv",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip",
        "Kenneth R. French Data Library (Fama & French, CRSP), 12 industry portfolios; updated monthly.",
        "Daily % returns of 12 US industry portfolios since July 1926 (NoDur ... Other). The file holds TWO "
        "blocks, value-weighted then equal-weighted, each headed by a text line: keep the rows before the "
        "first non-date index. -99.99 marks missing.",
        {"skiprows": 9, "index_col": 0, "low_memory": False},
        zip_member="12_Industry_Portfolios_Daily.csv",
    ),
    "ff_industries30_daily": Dataset(
        "ff_30_industries_daily.csv",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_daily_CSV.zip",
        "Kenneth R. French Data Library (Fama & French, CRSP), 30 industry portfolios; updated monthly.",
        "Daily % returns of 30 US industry portfolios since July 1926, same two-block layout as "
        "`ff_industries12_daily`.",
        {"skiprows": 9, "index_col": 0, "low_memory": False},
        zip_member="30_Industry_Portfolios_Daily.csv",
    ),
    "world_quakes_m7": Dataset(
        "world_quakes_m7.csv",
        "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=1900-01-01&endtime=2025-01-01"
        "&minmagnitude=7&orderby=time-asc",
        "USGS ComCat earthquake catalogue (FDSN event web service); pre-1976 events come from the "
        "ISC-GEM / Centennial catalogues. The classic Poisson-HMM series of Zucchini, MacDonald & "
        "Langrock (2016) is the 1900-2006 count of this catalogue as it stood then.",
        "Every magnitude 7+ earthquake worldwide, 1900-2024 (1591 events): time, location, depth, "
        "magnitude and magnitude type.",
        {"parse_dates": ["time"]},
    ),
    "elk": Dataset(
        "elk_data.RData",
        "https://raw.githubusercontent.com/TheoMichelot/moveHMM/master/data/elk_data.RData",
        "Morales, Haydon, Frair, Holsinger & Fryxell (2004, Ecology 85:2436), GPS tracks of four elk released in "
        "east-central Ontario, as shipped with the R package `moveHMM` (Michelot, Langrock & Patterson 2016).",
        "735 locations of 4 elk (ID 1-4): UTM Easting/Northing (m) and dist_water, the distance (m) to "
        "the nearest water. Not a table: an R serialisation (gzip XDR) - parse data.path() (see E24).",
    ),
    "pbcseq": Dataset(
        "pbcseq.csv",
        f"{_RDATASETS}/survival/pbcseq.csv",
        "Mayo Clinic trial in primary biliary cholangitis (cirrhosis), 1974-1984, D-penicillamine vs "
        "placebo; Murtaugh et al. (1994, Hepatology 20:126), Fleming & Harrington (1991), via R package "
        "`survival`.",
        "1945 visits of the 312 randomised patients: serum bilirubin (mg/dl) and other labs at each "
        "`day`, plus follow-up `futime` (days) and `status` (0 censored, 1 liver transplant, 2 death); "
        "`trt` 1 = D-penicillamine, 0 = placebo.",
        {"index_col": 0},
    ),
    "rg_mixtures": Dataset(
        "richardson_green_mixdata.txt",
        "https://people.maths.bris.ac.uk/~mapjg/mixdata",
        "Richardson & Green (1997, JRSS B 59:731), 'On Bayesian analysis of mixtures with an unknown number "
        "of components'; data page of P. J. Green. Enzyme: Bechtel et al. (1993); acidity: Crawford et "
        "al. (1992); galaxy: Roeder (1990).",
        "Three univariate benchmark samples in one text file: enzymatic activity in the blood of 245 "
        "people, log acidity index of 155 north-eastern US lakes, velocities (1000 km/s) of 82 "
        "galaxies. Each block is a heading, a description, the sample size, then the values. Not a "
        "table: parse data.path() (see E26).",
    ),
    "prostate": Dataset(
        "prostate.data",
        "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data",
        "Stamey et al. (1989, J. Urology 141:1076), prostate cancer before radical prostatectomy; the "
        "benchmark of Hastie, Tibshirani & Friedman, The Elements of Statistical Learning (2009), ch. 3.",
        "97 men: log PSA (lpsa) and 8 clinical predictors (log cancer volume lcavol, log prostate weight "
        "lweight, age, lbph, seminal vesicle invasion svi, lcp, gleason, pgg45), plus the book's "
        "train/test flag (67 T / 30 F).",
        {"sep": "\t", "index_col": 0},
    ),
    "diabetes": Dataset(
        "diabetes_lars.tsv",
        "https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data",
        "Efron, Hastie, Johnstone & Tibshirani (2004, Ann. Statist. 32:407), 'Least angle regression'; "
        "the LARS diabetes data.",
        "442 diabetes patients: AGE, SEX, BMI, blood pressure BP, six blood serum measurements S1-S6 "
        "(S1 total cholesterol, S2 LDL, S3 HDL, S4 total/HDL ratio, S5 log triglycerides, S6 glucose) "
        "and Y, a measure of disease progression one year after baseline (raw, unstandardised).",
        {"sep": "\t"},
    ),
    "crossbill": Dataset(
        "crossbill.rda",
        "https://raw.githubusercontent.com/cran/unmarked/master/data/crossbill.rda",
        "Swiss breeding bird survey MHB (Schmid, Zbinden & Keller 2004, Swiss Ornithological Institute), "
        "European crossbill Loxia curvirostra, as shipped with the R package `unmarked` (Fiske & Chandler "
        "2011); analysed in Kery & Royle, Applied Hierarchical Modeling in Ecology (2016).",
        "267 1-km2 quadrats surveyed 2-3 times a season in 1999-2007: detection/non-detection det<yy><k>, "
        "day of season date<yy><k>, elevation ele (m), forest cover (%). Not a table: an R serialisation "
        "(xz XDR), NA = -2147483648 - parse data.path() (see E28).",
    ),
    "mallard": Dataset(
        "mallard.RData",
        "https://raw.githubusercontent.com/cran/unmarked/master/data/mallard.RData",
        "Kery, Royle & Schmid (2005, Ecol. Appl. 15:1450), mallard counts from the Swiss breeding bird "
        "survey (2002), as shipped with the R package `unmarked`.",
        "239 sites x 3 repeated counts (mallard.y), site covariates elev, length (route), forest and "
        "survey covariates ivel (survey intensity), date - all covariates standardised by the source. "
        "Not a table: an R serialisation (gzip XDR) - parse data.path() (see E28).",
    ),
    "snowshoe_hare": Dataset(
        "hare.rda",
        "https://raw.githubusercontent.com/cran/Rcapture/master/data/hare.rda",
        "Snowshoe hare live-trapping (Otis, Burnham, White & Anderson 1978, Wildlife Monographs 62; "
        "Cormack 1989, Biometrics 45:395), via the R package `Rcapture` (Baillargeon & Rivest 2007).",
        "Capture histories of 68 hares over 6 trapping occasions (a 68 x 6 0/1 matrix stored column by "
        "column). Not a table: an R serialisation (gzip XDR) - parse data.path() (see E28).",
    ),
    "concrete": Dataset(
        "concrete.csv",
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv",
        "Yeh (1998, Cement and Concrete Research 28:1797), 'Modeling of strength of high-performance "
        "concrete using artificial neural networks'; UCI Machine Learning Repository (CSV copy from the "
        "datasets of Lantz, Machine Learning with R); a standard Bayesian-neural-network benchmark "
        "(Hernandez-Lobato & Adams 2015).",
        "1030 laboratory concrete cylinders: mix ingredients in kg per m3 of concrete (cement, blast-"
        "furnace slag, fly ash, water, superplasticizer, coarse and fine aggregate), age at testing "
        "(1-365 days) and compressive strength (MPa).",
    ),
    "moby_dick": Dataset(
        "moby_dick.txt",
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "Herman Melville, Moby-Dick; or, The Whale (1851), Project Gutenberg eBook #2701 (public domain).",
        "Plain UTF-8 text of the novel with the Project Gutenberg header and licence. Not a table: read "
        "data.path() and tokenise (see E30).",
    ),
    "bci_trees": Dataset(
        "bci_trees.rda",
        "https://raw.githubusercontent.com/cran/vegan/master/data/BCI.rda",
        "Condit et al. (2002, Science 295:666), Barro Colorado Island 50-ha forest plot, Panama; as shipped "
        "with the R package `vegan` (Oksanen et al.).",
        "Counts of 225 tree species (stems >= 10 cm diameter at breast height) in 50 contiguous 1-ha plots "
        "(21457 trees). Not a table: an R serialisation (gzip XDR) of a 50 x 225 data frame - parse "
        "data.path() (see E30).",
    ),
    "optdigits_train": Dataset(
        "optdigits_train.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        "Alpaydin & Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning "
        "Repository (NIST forms preprocessed at Bogazici University).",
        "3823 handwritten digits by 30 writers: each 32x32 bitmap is counted in 4x4 blocks, giving 64 "
        "integer pixels 0-16 (8x8, row by row) followed by the digit 0-9. No header row.",
        {"header": None},
    ),
    "optdigits_test": Dataset(
        "optdigits_test.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
        "Alpaydin & Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning "
        "Repository: the writer-independent test set.",
        "1797 digits by 13 writers who are NOT in optdigits_train; same 64 pixels (0-16) + digit layout.",
        {"header": None},
    ),
    # ---- examples (E32) ---------------------------------------------------
    "conjura_mmm": Dataset(
        "conjura_mmm_data.csv",
        "https://ndownloader.figshare.com/files/46779652",
        "Anderson, A. (2024), Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce "
        "Brands, Conjura, figshare, doi:10.6084/m9.figshare.25314841 (CC BY 4.0).",
        "132,759 brand x territory x day rows (143 anonymised series from 93 e-commerce brands, "
        "2019-2024): first-time and all purchases (orders, units, original price, discount), and "
        "spend / clicks / impressions for Google (search, shopping, PMax, display, video), Meta "
        "(Facebook, Instagram, other) and TikTok, plus organic/direct/email/referral clicks. "
        "Missing spend means the brand did not use that channel.",
    ),
    # ---- challenges (C11) -------------------------------------------------
    "actg175": Dataset(
        "actg175.csv",
        "https://archive.ics.uci.edu/static/public/890/data.csv",
        "Hammer et al. (1996, New England Journal of Medicine 335:1081), AIDS Clinical Trials Group "
        "Study 175; UCI Machine Learning Repository dataset 890 (doi:10.24432/C5ZG8F), the same data "
        "as the R package `speff2trial` (Juraska et al.).",
        "2139 HIV-infected adults with 200-500 CD4 cells/mm3, randomised to zidovudine (ZDV, trt 0), "
        "ZDV + didanosine (trt 1), ZDV + zalcitabine (trt 2) or didanosine alone (trt 3). One row per "
        "patient: baseline covariates (age, wtkg, hemo, homo, drugs, karnof, race, gender, str2 = prior "
        "antiretroviral therapy, symptom, ...), CD4/CD8 counts at baseline (cd40, cd80) and at 20 +- 5 "
        "weeks (cd420, cd820), and time to a clinical event (time, cid).",
        {"index_col": "pidnum"},
    ),
    # ---- examples (E37) ---------------------------------------------------
    "ghcnd_seatac": Dataset(
        "ghcnd_USW00024233.csv.gz",
        "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00024233.csv.gz",
        "Menne et al. (2012), Global Historical Climatology Network - Daily (GHCN-Daily), NOAA National "
        "Centers for Environmental Information; station USW00024233 Seattle-Tacoma International Airport.",
        "Every daily observation at Sea-Tac since 1948 in long format, no header: station id, date "
        "(YYYYMMDD), element (TMAX/TMIN in tenths of a degree C, PRCP, SNOW, ...), value, measurement / "
        "quality / source flags and observation time. A blank quality flag means the value passed "
        "NOAA's checks. The file is updated daily, so the latest year is partial.",
        {"header": None, "names": ["station", "date", "element", "value", "mflag", "qflag", "sflag",
                                   "obs_time"],
         "dtype": {"date": str, "mflag": str, "qflag": str, "sflag": str}},
    ),
    "ghcnd_portland": Dataset(
        "ghcnd_USW00024229.csv.gz",
        "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00024229.csv.gz",
        "Menne et al. (2012), Global Historical Climatology Network - Daily (GHCN-Daily), NOAA National "
        "Centers for Environmental Information; station USW00024229 Portland International Airport, "
        "Oregon.",
        "Every daily observation at Portland airport since April 1938, same long format as ghcnd_seatac "
        "(TMAX in tenths of a degree C).",
        {"header": None, "names": ["station", "date", "element", "value", "mflag", "qflag", "sflag",
                                   "obs_time"],
         "dtype": {"date": str, "mflag": str, "qflag": str, "sflag": str}},
    ),
    "gistemp_global": Dataset(
        "gistemp_glb.csv",
        "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "GISTEMP Team, GISS Surface Temperature Analysis (GISTEMP v4), NASA Goddard Institute for Space "
        "Studies; Lenssen et al. (2019, Journal of Geophysical Research: Atmospheres 124:6307).",
        "Global mean land-ocean surface temperature anomaly (degrees C relative to 1951-1980) by month "
        "and season from 1880; column J-D is the calendar-year mean. '***' marks months not yet "
        "available. Updated monthly.",
        {"skiprows": 1, "na_values": "***"},
    ),
    # ---- examples (E36) ---------------------------------------------------
    "cces2018": Dataset(
        "cces18_common_vv.csv.gz",
        "https://raw.githubusercontent.com/JuanLopezMartin/MRPCaseStudy/master/data_public/chapter1/data/"
        "cces18_common_vv.csv.gz",
        "Ansolabehere, Schaffner & Luks, Cooperative Congressional Election Study 2018: Common Content "
        "(Harvard Dataverse, doi:10.7910/DVN/ZSBZ7K), as redistributed with Lopez-Martin, Phillips & "
        "Gelman, 'Multilevel Regression and Poststratification Case Studies' (GitHub "
        "JuanLopezMartin/MRPCaseStudy).",
        "60,000 respondents of the 2018 CCES online survey (YouGov). Only the columns the case study "
        "uses are read: CC18_321d (allow employers to decline coverage of abortions in insurance plans: "
        "1 support, 2 oppose), inputstate (state FIPS), gender (1 male, 2 female), race (1 White, "
        "2 Black, 3 Hispanic, 4-8 other groups), birthyr, educ (1 no HS ... 6 post-grad).",
        {"usecols": ["CC18_321d", "inputstate", "gender", "race", "birthyr", "educ"]},
    ),
    "mrp_poststrat": Dataset(
        "mrp_poststrat_df.csv",
        "https://raw.githubusercontent.com/JuanLopezMartin/MRPCaseStudy/master/data_public/chapter1/data/"
        "poststrat_df.csv",
        "Lopez-Martin, Phillips & Gelman, MRP Case Studies (chapter 1), built from the US Census "
        "Bureau's American Community Survey.",
        "Poststratification table: number of adults (n) in each of 12,000 cells of state (50, no DC) x "
        "ethnicity (White, Black, Hispanic, Other) x male (-0.5 female, +0.5 male) x age (6 groups) x "
        "education (5 groups).",
    ),
    "mrp_state_predictors": Dataset(
        "mrp_statelevel_predictors.csv",
        "https://raw.githubusercontent.com/JuanLopezMartin/MRPCaseStudy/master/data_public/chapter1/data/"
        "statelevel_predictors.csv",
        "Lopez-Martin, Phillips & Gelman, MRP Case Studies (chapter 1).",
        "For each of the 50 states: Republican share of the two-party vote in the 2016 presidential "
        "election (repvote) and Census region (Northeast, South, North Central, West).",
    ),
    # ---- examples (E38) ---------------------------------------------------
    "covid_hosp_triangle_de": Dataset(
        "covid_hosp_triangle_de.csv",
        "https://raw.githubusercontent.com/KITmetricslab/hospitalization-nowcast-hub/main/data-truth/"
        "COVID-19/COVID-19_hospitalizations_preprocessed.csv",
        "German COVID-19 Nowcast Hub (KIT / Wolffram et al. 2023, PLOS Computational Biology 19:e1011394), "
        "reformatted from the Robert Koch Institute's daily 'COVID-19-Hospitalisierungen in Deutschland' "
        "releases (github.com/robert-koch-institut).",
        "Reporting triangle of COVID-19 hospitalisations in Germany, 2021-04-06 to 2024: one row per "
        "Meldedatum (date the case was reported to the health authority) x location (DE and 16 states) x "
        "age group (00+ = all ages); value_kd = hospitalisations of those cases that were added to the "
        "data k days after the Meldedatum (k = 0..80, then value_>80d). Negative corrections have been "
        "redistributed to earlier delays by the hub.",
    ),
    # ---- examples (E39) ---------------------------------------------------
    "intcal20": Dataset(
        "intcal20.14c",
        "https://intcal.org/curves/intcal20.14c",
        "Reimer et al. (2020), The IntCal20 Northern Hemisphere radiocarbon age calibration curve "
        "(0-55 cal kBP), Radiocarbon 62:725-757, doi:10.1017/RDC.2020.41; distributed by intcal.org.",
        "Calibration curve on a calendar grid (5-yr steps in the Holocene, coarser further back): calendar "
        "age (cal BP, years before 1950), radiocarbon age (14C yr BP) and its 1-sd uncertainty, and "
        "Delta-14C (per mil) with its 1-sd uncertainty. Rows run from 55,000 cal BP down to 0.",
        {"comment": "#", "header": None,
         "names": ["cal_bp", "c14_bp", "c14_sd", "delta14c", "delta14c_sd"]},
    ),
    "shroud_damon1989": Dataset(
        "shroud_damon1989.csv",
        "https://www.shroud.com/nature.htm",
        "Damon et al. (1989), Radiocarbon dating of the Shroud of Turin, Nature 337:611-615, "
        "doi:10.1038/337611a0. HAND-TRANSCRIBED from Table 1 ('Basic data, individual measurements') "
        "of the paper; the URL is the full text of the paper (HTML) at shroud.com, not a CSV, so keep the "
        "cached file in data/. The 12 shroud rows were cross-checked against the `shroud` data of the "
        "CRAN package rice (Blaauw), and the per-lab weighted means reproduce Table 2.",
        "All 49 individual AMS measurements (14C yr BP, 1 sd) by the three laboratories (Arizona, Oxford, "
        "Zurich) on sample 1 (the Shroud of Turin) and three known-age controls: 2 = linen from a tomb at "
        "Qasr Ibrim, Nubia (11th-12th century AD), 3 = linen associated with an early 2nd-century AD "
        "mummy of Cleopatra from Thebes, 4 = threads from the cope of St Louis d'Anjou, Saint-Maximin "
        "(c. AD 1290-1310). meas_id is the paper's code (lab, sample, run, pretreatment, replicate).",
    ),
    "sluggan_moss": Dataset(
        "bchron_sluggan.rda",
        "https://raw.githubusercontent.com/cran/Bchron/master/data/Sluggan.rda",
        "Smith & Goddard (1991), A 12,500 year record of vegetational history at Sluggan Bog, Co. Antrim, "
        "N. Ireland, New Phytologist 118:167-187; as distributed in the R package Bchron (Haslett & "
        "Parnell 2008), from the European Pollen Database.",
        "R data file (bzip2-compressed XDR): a data frame of 31 radiocarbon dates down a peat core: id, "
        "ages (14C yr BP), ageSds, position (depth in cm), thickness (cm), calCurves. Several dates share "
        "a depth.",
    ),
    # ---- examples (E40) ---------------------------------------------------
    "potus2016_polls": Dataset(
        "potus2016_polls.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/all_polls.csv",
        "HuffPost Pollster 2016 presidential polls, as used for the 2016 backtest of the Economist's "
        "forecasting model (Heidemanns, Gelman & Morris 2020; GitHub TheEconomist/us-potus-model, MIT "
        "licence).",
        "3,168 rows of 2016 Clinton-Trump polls (May 2015 - 7 Nov 2016): state (-- = national), pollster, "
        "start/end date, number.of.observations, population (Likely/Registered Voters, Adults, and "
        "partisan sub-samples), mode (Internet, Live Phone, IVR/Online, ...), percentages trump, clinton, "
        "other, undecided, johnson, mcmullin, and question.iteration (several question versions of one poll).",
    ),
    "potus_results_1976_2016": Dataset(
        "potus_results_76_16.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/potus_results_76_16.csv",
        "Certified US presidential election results by state 1976-2016, as compiled in GitHub "
        "TheEconomist/us-potus-model.",
        "One row per year x state (50 states + DC): total_votes and the Democratic, Republican and other "
        "shares of the total vote.",
    ),
    "potus2012_states": Dataset(
        "potus2012_states.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/2012.csv",
        "2012 presidential results by state with electoral votes and adult population growth 2011-15, "
        "GitHub TheEconomist/us-potus-model.",
        "51 rows (50 states + DC): obama/romney percentages and vote counts, total_count, voting-age "
        "population, state_name, ev (electoral votes, 2012-2020 apportionment), adult_pop_growth_2011_15. "
        "The file has old-Mac CR line endings.",
        {"lineterminator": "\r"},
    ),
    "abramowitz_fundamentals": Dataset(
        "abramowitz_data.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/abramowitz_data.csv",
        "Inputs of Abramowitz's 'time for change' fundamentals model 1948-2016, as compiled in GitHub "
        "TheEconomist/us-potus-model.",
        "One row per election year: incumbent-party share of the two-party vote (incvote), June net "
        "presidential approval (juneapp), second-quarter annualised GDP growth (q2gdp) and other columns.",
    ),
    "acs2013_states": Dataset(
        "acs_2013_variables.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/acs_2013_variables.csv",
        "US Census Bureau American Community Survey (2013), state summaries compiled in GitHub "
        "TheEconomist/us-potus-model.",
        "51 rows: population, shares white, Black, Hispanic/other, college-educated, white working class "
        "(wwc_pct), median age, population density.",
    ),
    "state_urbanicity": Dataset(
        "urbanicity_index.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/urbanicity_index.csv",
        "State urbanicity index (average log population within 5 miles of a resident), compiled in GitHub "
        "TheEconomist/us-potus-model.",
        "51 rows: state_name, pop, average_log_pop_within_5_miles, state.",
    ),
    "state_white_evangelical": Dataset(
        "white_evangel_pct.csv",
        "https://raw.githubusercontent.com/TheEconomist/us-potus-model/master/data/white_evangel_pct.csv",
        "Share of white evangelical Christians by state, compiled in GitHub TheEconomist/us-potus-model.",
        "51 rows: state, pct_white_evangel.",
    ),
    # ---- examples (E41) ---------------------------------------------------
    "dutch_boys_bmi": Dataset(
        "gamlss_dbbmi.rda",
        "https://raw.githubusercontent.com/cran/gamlss.data/master/data/dbbmi.rda",
        "Fourth Dutch Growth Study 1996-7 (Fredriks et al. 2000, Pediatric Research 47:316-323; Archives "
        "of Disease in Childhood 82:107-112), data given by S. van Buuren; as distributed in the CRAN "
        "package gamlss.data (Stasinopoulos & Rigby), identical to dbbmi.rda in gamlss.data_6.0-7.tar.gz.",
        "R data file (xz-compressed XDR): a data frame `dbbmi` of 7,294 Dutch boys aged 0.03-21.7 years: "
        "age (years) and bmi (body-mass index, kg/m^2). Cross-sectional: one measurement per boy.",
    ),
    # ---- examples (E50) ---------------------------------------------------
    "scratch_assay_pc3": Dataset(
        "scratch_assay_pc3.csv",
        "https://raw.githubusercontent.com/ProfMJSimpson/NoiseModels/main/PDE_BinomialNoiseModel.jl",
        "Jin, Shah, Penington, McCue, Chopin & Simpson (2016), Reproducibility of scratch assays is "
        "affected by the initial degree of confluence: experiments, modelling and model selection, "
        "Journal of Theoretical Biology 390:136-145; counts as distributed in the code of Simpson, Murphy "
        "& Maclaren (2024), Modelling count data with partial differential equation models in biology, "
        "Journal of Theoretical Biology 580:111732. TRANSCRIBED from the arrays in that Julia file (the "
        "URL is code, not a CSV), so keep the cached file in data/.",
        "Scratch (wound-healing) assay with PC-3 prostate cancer cells: number of cells in each of 38 "
        "columns 50 um wide (centres x_um = 25..1875 um) across a 1,900 um field, counted at t_h = 0, 12, "
        "24, 36 and 48 hours after the scratch. A column fully packed holds about 122 cells (packing "
        "density 1.7e-3 cells/um^2 x 50 um x 1,430 um).",
    ),
    # ---- examples (E46) ---------------------------------------------------
    "blowfly": Dataset(
        "gamair_blowfly.rda",
        "https://raw.githubusercontent.com/cran/gamair/master/data/blowfly.rda",
        "Nicholson (1954), An outline of the dynamics of animal populations, Australian Journal of "
        "Zoology 2:9-65; as distributed in the CRAN package gamair (Wood, Generalized Additive Models).",
        "R data file (bzip2-compressed XDR): a data frame `blowfly` of 180 counts of adult sheep blowflies "
        "(Lucilia cuprina) in one of Nicholson's laboratory cultures, one count every two days: pop "
        "(adults; back-calculated from counts of dead flies) and day (an index, 0.5 to 90 in steps of 0.5).",
    ),
    # ---- examples (E54) ---------------------------------------------------
    "allen_sst_464212183": Dataset(
        "allen_464212183_long_square.npz",
        "https://api.brain-map.org/api/v2/well_known_file_download/491202878",
        "Allen Institute for Brain Science (2015), Allen Cell Types Database [dataset], available from "
        "celltypes.brain-map.org (cell page: celltypes.brain-map.org/experiment/electrophysiology/464212183); "
        "methods in Gouwens et al. (2019), Classification of electrophysiological and morphological neuron "
        "types in the mouse visual cortex, Nature Neuroscience 22:1182-1195. Used under the Allen Institute "
        "Terms of Use (alleninstitute.org/terms-of-use: research / noncommercial use with citation). "
        "EXTRACTED from the 62 MB NWB file at the URL (sweeps 19-36 and 44-46, the 'Long Square' "
        "current steps, every 10th sample), so keep the cached file in data/.",
        "NumPy .npz (open data.path(...) with np.load): whole-cell current-clamp recordings of one "
        "Sst-IRES-Cre;Ai14 aspiny interneuron in mouse primary visual cortex (specimen 464212183, male, "
        "P53). v_mv: 21 sweeps x 26,000 samples of membrane potential (mV, float32) at fs_hz = 20 kHz "
        "(downsampled from 200 kHz), a 1.3 s window with the 1 s square current step from t_on_s = 0.1 "
        "to t_off_s = 1.1 s; amp_pa: step amplitude (pA, -110 to +230, sorted); sweep: original sweep "
        "number; specimen_id.",
    ),
    "luria_delbruck_1943": Dataset(
        "luria_delbruck_1943.csv",
        "http://www.esp.org/foundations/genetics/classical/holdings/l/slmd-43.pdf",
        "Luria & Delbrueck (1943), Mutations of bacteria from virus sensitivity to virus resistance, "
        "Genetics 28:491-511 (PMC1209226), Tables 1-3. TRANSCRIBED from the ESP Foundations reprint at "
        "the URL (a PDF, not a CSV) and cross-checked against the dataset `luriadel` of the CRAN package "
        "flan (Mazoyer et al.), so keep the cached file in data/. Discrepancies: the reprint omits the 9th "
        "culture of experiment 1 (17, in flan; 9 values give the published mean 26.8); experiment 16's "
        "20th culture is 33 in the reprint and 35 in flan (35 gives the published mean 11.35, used here); "
        "experiments 10 and 21b average 24.0 and 44.2, not the published 23.8 and 48.2 (both sources "
        "agree on the values); experiment 23's frequency classes sum to 88, the paper says 87 cultures.",
        "E. coli B cultures tested for resistance to phage T1 (one row per culture, or per frequency "
        "class in table 3). table: 1 = ten samples from ONE culture (plating control), 2 = one sample from "
        "each of a series of parallel cultures, 3 = frequency distribution for 100 and 87(88) parallel "
        "cultures; experiment; medium (broth or synthetic); culture_ml and sample_ml (the fraction plated "
        "is sample_ml / culture_ml); cells_per_culture (bacteria per culture at the time of the test); "
        "culture (index); count_lo, count_hi (resistant colonies on the plate: equal for an exact count, "
        "a class such as 6-10 in table 3); n_cultures (cultures in that row).",
        {"comment": "#", "dtype": {"experiment": str}},
    ),
    # ---- examples (E55) ---------------------------------------------------
    "larsson_bursting": Dataset(
        "larsson2019_allelic_umis.csv",
        "https://raw.githubusercontent.com/sandberg-lab/txburst/master/data/SS3_c57_UMIs_concat.csv",
        "Larsson, Johnsson, Hagemann-Jensen, Hartmanis, Faridani, Reinius, Segerstolpe, Rivera, Ren & "
        "Sandberg (2019), Genomic encoding of transcriptional burst kinetics, Nature 565:251-254; data files "
        "of the paper's GitHub repository sandberg-lab/txburst (data/: SS3_c57_UMIs_concat.csv, "
        "SS3_cast_UMIs_concat.csv, cell_cycle_annotation.csv, slam_seq.csv, SS3_c57_UMIs_concat_ML.pkl). "
        "EXTRACTED: 24 genes from five files (the URL is the full 10,727-gene C57 table), so keep the "
        "cached file in data/.",
        "Allele-resolved single-cell RNA-seq UMI counts in 224 primary mouse fibroblasts from an F1 hybrid "
        "(C57BL/6J x CAST/EiJ), 24 genes, one row per cell x gene: cell; phase (G1, S, G2M, the "
        "repository's cell-cycle annotation); allelic_umis (all allele-assigned UMIs of the cell, both "
        "alleles, all 10,727 genes: a capture/size factor); gene; c57, cast (UMIs of that gene from each "
        "allele; empty = expressed but not assignable to an allele, i.e. missing); half_life_h (mRNA "
        "half-life in hours from the repository's SLAM-seq table); txburst_kon, txburst_koff, "
        "txburst_ksyn (the repository's published maximum-likelihood telegraph parameters for the C57 "
        "allele, in units of the mRNA degradation rate).",
    ),
    # ---- examples (E56) ---------------------------------------------------
    "tanouchi_mother_machine": Dataset(
        "tanouchi2017_mc4100_mother_machine.npz",
        "https://doi.org/10.6084/m9.figshare.c.3493548.v1",
        "Tanouchi, Pai, Park, Huang, Buchler & You (2017), Long-term growth data of Escherichia coli at a "
        "single-cell level, Scientific Data 4:170036 (doi:10.1038/sdata.2017.36); data on figshare "
        "(collection 3493548, files released under CC0): 'Zipped analysis data files of mother cells "
        "cultured at 25C / 27C / 37C' (ndownloader.figshare.com/files/6397359, 7235000, 6397368). "
        "PACKED: the 279 per-lineage text files of the three zips (columns frame, division flag, cell "
        "length) into one compressed NumPy file (fluorescence columns dropped), so keep the cached file "
        "in data/.",
        "NumPy .npz (open data.path(...) with np.load): mother-machine time-lapse of E. coli MC4100 "
        "(constitutive YFP) mother cells in LB at 25, 27 and 37 C, one frame per minute, about 70 "
        "generations per cell. lineage (279 names such as '37C_xy01_01'), temp_c; length_um "
        "(concatenated cell lengths, the major axis of the segmented mask, float32) with frame_offset "
        "(lineage k is length_um[frame_offset[k]:frame_offset[k+1]]); division_frame (frame indices, "
        "within the lineage, of the first frame after each detected division; the first entry is frame "
        "0, the start of the recording) with division_offset; minutes_per_frame = 1.",
    ),
    # ---- examples (E57) ---------------------------------------------------
    "heckert2022_spt": Dataset(
        "heckert2022_rara_nls_tracks.csv",
        "https://raw.githubusercontent.com/alecheckert/saspt/main/examples/u2os_rara_ht_7.48ms/region_4_7ms_trajs.csv",
        "Example data of the saSPT package (github.com/alecheckert/saspt, examples/, MIT licence), described "
        "with the method in Heckert, Dahal, Tjian & Darzacq (2022), Recovering mixtures of fast-diffusing "
        "states from short single-particle trajectories, eLife 11:e70169. EXTRACTED from 13 of the 22 "
        "example CSVs (u2os_rara_ht_7.48ms/region_4 and region_5; u2os_ht_nls_7.48ms/region_0..10; the URL "
        "is one of them), frames >= 1000, positions converted from pixels (0.16 um) to um, so keep the "
        "cached file in data/.",
        "Single-molecule tracks in nuclei of live human U2OS cells, 7.48 ms per frame (package settings: "
        "0.16 um pixels, 0.7 um focal depth), one row per localisation: condition (rara = RARA-HaloTag, the "
        "retinoic acid receptor alpha; nls = HaloTag-NLS, a free control), region (one nucleus / movie), "
        "track (trajectory id within the region), frame, x_um, y_um, loc_err_um (mean of the fitted x and y "
        "localisation errors). 34,401 RARA and 46,990 NLS localisations, tracks of 1 to 242 positions.",
    ),
    # ---- examples (E58) ---------------------------------------------------
    "dannhauser2022_dstorm": Dataset(
        "dannhauser2022_unc13_dstorm_roi.npz",
        "https://zenodo.org/api/records/7328805/files/dStorm647_Unc13_7.tif/content",
        "Mrestani (2022), Example raw dSTORM data of Brp (Alexa Fluor532) and Unc-13 (Alexa Fluor647), "
        "Zenodo doi:10.5281/zenodo.7328805 (CC BY 4.0); from Dannhaeuser, Mrestani, Gundelach, Pauli, Komma, "
        "Kollmannsberger, Sauer, Heckmann & Paul (2022), Endogenous tagging of Unc-13 reveals nanoscale "
        "reorganization at active zones during presynaptic homeostatic potentiation, Frontiers in Cellular "
        "Neuroscience 16:1074304 (Andor iXon Ultra 897 EMCCD, 127 nm pixels, 60x NA 1.49 objective). "
        "EXTRACTED from the 1.2 GB TIFF stack at the URL (15,000 frames of 200 x 200 px; frames 3000-4999 "
        "read with HTTP range requests, a 40 x 40 px region cropped), so keep the cached file in data/.",
        "NumPy .npz (open data.path(...) with np.load): raw camera frames (ADU, uint16) of a dSTORM "
        "recording of Unc-13-GFSTF (anti-GFP + Alexa Fluor647 F(ab')2) at a Drosophila larval "
        "neuromuscular junction, 10 ms exposures. frames: 2000 x 40 x 40 (frames first_frame = 3000 "
        "onwards; the crop starts at roi_row = 25, roi_col = 50 of the full field); pixel_nm = 127, "
        "exposure_s; and per-pixel statistics of the full 200 x 200 field over the same 2000 frames for "
        "camera calibration: field_median_adu, field_diff_var_adu2 (robust variance of frame-to-frame "
        "differences / 2), field_max_adu. No dark frames or gain settings are included.",
    ),
    # ---- examples (E59) ---------------------------------------------------
    "wolff2023_minflux_kinesin": Dataset(
        "wolff2023_minflux_kinesin.npz",
        "https://zenodo.org/records/7565676",
        "Wolff & Scheiderer (2023), data and MATLAB scripts for 'MINFLUX dissects the unimpeded walking of "
        "kinesin-1' (Wolff, Scheiderer et al., Science 379:1004, 2023), Zenodo doi:10.5281/zenodo.7565676 "
        "(CC BY 4.0). PACKED from the 38 MB zip: the authors' processed step tables "
        "(KinesinDataFiles/<construct>/<ATP>/allsteps_reeval.xls, read with xlrd) for 12 construct x ATP "
        "conditions (ATPgammaS left out), plus 30 raw 1-D MINFLUX traces of construct N356C (DOL1) from two "
        "files (10 uM: ...20220302sample2.txt, 1 mM: ...20220303sample2.txt) converted to positions with a "
        "Python port of the authors' calculateSCE.m (photon window 7-300, sign flipped so that walking is "
        "positive, shifted to start at 0), so keep the cached file in data/.",
        "NumPy .npz (open data.path(...) with np.load): kinesin-1 stepping on microtubules tracked with "
        "interferometric MINFLUX (a ~1 nm dye on a cysteine: N356C = coiled-coil stalk; T324C, K28C, E215C "
        "= motor head) at 10 uM, 100 uM or 1 mM ATP. Step table, one row per detected step of one trace: "
        "cond (index into cond_names / cond_construct / cond_atp_uM), step_nm (on-axis), offaxis_nm, "
        "dwell_s (time from this step to the next; 0 on the last two rows of a trace), plateau_sd_nm, "
        "photons, end_of_trace (1 on the last row of each trace), hmm_state (authors' step classes: "
        "2 bound-to-bound, 3 bound-to-unbound, 4 unbound-to-bound, 5/6 rare, 0 not classified; not valid "
        "for N356C). Raw traces: trace_t_s, trace_x_nm, trace_photons concatenated, trace k is "
        "[trace_offset[k]:trace_offset[k+1]], trace_cond its condition name.",
    ),
    # ---- examples (E60) ---------------------------------------------------
    "pytfm_colony_ko04": Dataset(
        "pytfm_colony_ko04.npz",
        "https://github.com/fabrylab/example_data_for_pyTFM",
        "Bauer, Prechová, Fischer, Thievessen, Gregor & Fabry (2021), pyTFM: A tool for traction force and "
        "monolayer stress microscopy, PLoS Computational Biology 17:e1008364; example data of the pyTFM "
        "tutorial (GitHub fabrylab/example_data_for_pyTFM, which states no licence). PACKED from "
        "clickpoints_tutorial/KO_analyzed/04u.npy, 04v.npy, 04tx.npy, 04ty.npy and the masks "
        "python_tutorial/force_measurement.png and cell_borders.png (holes filled, resampled to the PIV "
        "grid), so keep the cached file in data/.",
        "NumPy .npz (open data.path(...) with np.load): substrate displacement field under a small cell "
        "colony (7 cells; 'a critical cytoskeletal protein knocked out', per the tutorial) measured by PIV "
        "(20 um windows, 18 um overlap) between bead images before and after the cells were removed. ux_um, "
        "uy_um: displacements in um on a 189 x 193 grid (spacing grid_um = 2.117 um; rows run along +y, "
        "i.e. the images were flipped so that y points up and the y components changed sign). "
        "tx_pytfm_pa, ty_pytfm_pa: pyTFM's own tractions (finite-thickness FTTC followed by a Gaussian filter). "
        "mask_force: region pyTFM sums forces over; mask_colony: the colony footprint. Gel substrate: "
        "young_pa = 49 kPa, poisson = 0.49, thickness_um = 300. pyTFM's reported "
        "contractility (2.13e-6 N) and strain energy (2.46e-13 J) are included.",
    ),
    # ---- examples (E61) ---------------------------------------------------
    "pettmann2021_1g4_cd69": Dataset(
        "pettmann2021_1g4_cd69.csv",
        "https://elifesciences.org/articles/67092",
        "Pettmann, Huhn, Abu Shah, Kutuzov, Wilson, Dustin, Davis, van der Merwe & Dushek (2021), The "
        "discriminatory power of the T cell receptor, eLife 10:e67092 (CC BY 4.0). ASSEMBLED from two of the "
        "article's source-data files: Figure 2 source data 1 (elife-67092-fig2-data1-v3.zip, folder "
        "'Figure 2 - 1G4 blasts and U87', five CSVs) and Figure 1 source data 2 "
        "(elife-67092-fig1-data2-v3.csv, SPR affinities at 37 C), so keep the cached file in data/.",
        "Activation of primary human T cell blasts expressing the 1G4 TCR by U87 target cells pulsed "
        "with one of 8 variants of the NY-ESO-1 peptide (SLLMWITQV = 9V and single substitutions), one row "
        "per well: experiment (date code, 5 independent experiments), peptide, dose_uM (peptide pulsing "
        "concentration, 0 = unpulsed), cd69_pct (% CD69-positive T cells). Per peptide: sequence, "
        "kd_um, kd_sd_um, kd_n (SPR KD mean, SD and number of measurements; the Bmax-constrained estimate "
        "where it exceeds 20 uM, else the Bmax-fitted one, the rule stated in the paper) and kd_method.",
    ),
    # ---- examples (E64) ----
    "theoph": Dataset(
        "theoph.csv",
        f"{_RDATASETS}/datasets/Theoph.csv",
        "Theophylline pharmacokinetics from Boeckmann, Sheiner & Beal (1994), NONMEM Users Guide Part V; "
        "R package `datasets` (GPL-2 | GPL-3, part of R), via Rdatasets. Also in Pinheiro & Bates (2000).",
        "Serum theophylline concentration (conc, mg/L) in 12 subjects after a single oral dose (Dose, "
        "mg/kg, 3.1-5.9), 11 samples each over 25 h (Time, h since dosing; 132 rows), with body "
        "weight (Wt, kg). Some pre-dose (Time 0) samples are above zero.",
        {"index_col": 0},
    ),
    "warfarin_pkpd": Dataset(
        "warfarin_pkpd.csv",
        "https://github.com/nlmixr2/nlmixr2data/raw/main/data/warfarin.rda",
        "O'Reilly, Aggeler & Leong (1963) J Clin Invest 42:1542 and O'Reilly & Aggeler (1968) "
        "Circulation 38:169, as curated by Funaki, Holford & Fujita (2018) in R package `nlmixr2data` "
        "(GPL >= 3). CONVERTED from the package's `warfarin.rda` (R serialisation) to CSV, so keep "
        "the cached file in data/.",
        "Warfarin PK/PD in 32 subjects after one oral dose of 1.5 mg/kg (amt, mg, on the evid = 1 "
        "rows). 515 rows: id, time (h, 0-144), amt, dv (plasma warfarin mg/L when dvid = 'cp', 251 "
        "samples, four reported as 0 at 0.5 h; prothrombin complex activity, % of normal, when "
        "dvid = 'pca', 232 samples), evid, wt (kg, 40-102), age (y, 21-63), sex (27 male, 5 female).",
    ),
    # ---- examples (E63) ----
    "card_krueger_njmin": Dataset(
        "card_krueger_njmin.dat",
        "https://davidcard.berkeley.edu/data_sets/njmin.zip",
        "Card & Krueger (1994), Minimum wages and employment: a case study of the fast-food industry "
        "in New Jersey and Pennsylvania, AER 84(4):772-793; public-use file `public.dat` from "
        "njmin.zip on David Card's data page (posted by the author, no licence stated).",
        "Two telephone/personal surveys of 410 Burger King, KFC, Roy Rogers and Wendy's restaurants "
        "in New Jersey (331) and eastern Pennsylvania (79): Feb-Mar 1992, before NJ's minimum wage "
        "rose from $4.25 to $5.05 on 1 April 1992, and Nov-Dec 1992. Per store: chain, co_owned, "
        "state (1 = NJ), region dummies, full-time (empft), part-time (emppt) and manager (nmgrs) "
        "employees, starting wage (wage_st), prices and hours; wave-2 columns end in 2; status2 = 3 "
        "means closed permanently. Missing values are '.'.",
        {"sep": r"\s+", "header": None, "na_values": ".", "names": (
            "sheet chain co_owned state southj centralj northj pa1 pa2 shore ncalls empft emppt "
            "nmgrs wage_st inctime firstinc bonus pctaff meals open hrsopen psoda pfry pentree "
            "nregs nregs11 type2 status2 date2 ncalls2 empft2 emppt2 nmgrs2 wage_st2 inctime2 "
            "firstin2 special2 meals2 open2r hrsopen2 psoda2 pfry2 pentree2 nregs2 nregs112").split()},
        zip_member="public.dat",
    ),
    "banks_mississippi_1930": Dataset(
        "banks_mississippi_1930.csv",
        "https://raw.githubusercontent.com/pymc-labs/CausalPy/main/causalpy/data/banks.csv",
        "Richardson & Troost (2009), Monetary intervention mitigated banking panics during the Great "
        "Depression: quasi-experimental evidence from a Federal Reserve district border, 1929-1933, "
        "JPE 117(6):1031-1073; daily series as distributed with CausalPy (Apache-2.0).",
        "Daily counts, 1 Jul 1929 - 31 Aug 1934 (1,878 rows), of state banks in Mississippi in "
        "business (bib6, bib8) and in operation (bio6, bio8) in the Atlanta (6th) and St Louis "
        "(8th) Federal Reserve districts; the border splits the state. date (days since 1900), "
        "weekday, day, month, year.",
    ),
    "mlda_mortality": Dataset(
        "mlda_mortality.csv",
        "https://raw.githubusercontent.com/pymc-labs/CausalPy/main/causalpy/data/drinking.csv",
        "Carpenter & Dobkin (2009), The effect of alcohol consumption on mortality: regression "
        "discontinuity evidence from the minimum drinking age, AEJ Applied 1(1):164-182; cell means "
        "as used in Angrist & Pischke, Mastering 'Metrics (2015) ch. 4, distributed with CausalPy "
        "(Apache-2.0).",
        "US deaths per 100,000 person-years by age cell (agecell, 48 cells of ~30 days, ages 19.07-"
        "22.93; the two cells nearest 21 are absent) and cause: all, internal, external, alcohol, "
        "homicide, suicide, mva (motor vehicle accidents), drugs, externalother, each with a "
        "`...fitted` column from the published fit.",
        {"index_col": 0},
    ),
    "senate_rd": Dataset(
        "senate_rd.csv",
        "https://raw.githubusercontent.com/rdpackages/rdrobust/master/R/rdrobust_senate.csv",
        "Cattaneo, Frandsen & Titiunik (2015), Randomization inference in the regression "
        "discontinuity design: an application to party advantages in the U.S. Senate, J Causal "
        "Inference 3(1):1-24; example data of the `rdrobust` package (GPL-3).",
        "US Senate elections 1914-2010, one row per state x election (1,390 rows): margin "
        "(Democratic margin of victory at election t, % points; the running variable), vote "
        "(Democratic vote share in the next election for the same seat), state, year, class, "
        "termshouse, termssenate, population.",
    ),
    "card1995_schooling": Dataset(
        "card1995_schooling.csv",
        f"{_RDATASETS}/wooldridge/card.csv",
        "Card (1995), Using geographic variation in college proximity to estimate the return to "
        "schooling, in Aspects of Labour Market Behaviour; NLS Young Men cohort, via R package "
        "`wooldridge` (GPL-3), Rdatasets.",
        "3,010 men interviewed in 1976 (aged 24-34): lwage (log hourly wage, cents), educ (years), "
        "nearc4 / nearc2 (grew up near a 4-year / 2-year college in 1966), exper, expersq, black, "
        "south, smsa (1976), smsa66, reg661-reg669 (1966 region), parents' education, IQ, KWW.",
        {"index_col": 0},
    ),
    # ---- examples (E66) ----
    "upworthy_exploratory": Dataset(
        "upworthy_exploratory.csv",
        "https://osf.io/download/3vqmp/",
        "Matias, Munger, Aubin Le Quere & Ebersole (2021), The Upworthy Research Archive, a time "
        "series of 32,487 experiments in U.S. media, Scientific Data 8:195 "
        "(doi:10.1038/s41597-021-00934-7); OSF project jd64p (CC BY 4.0), file "
        "upworthy-archive-exploratory-packages-03.12.2020.csv (the URL). TRIMMED: kept the columns "
        "below, dropped excerpt/lede/share text/image URL/slug, added arm and content_id, so keep "
        "the cached file in data/.",
        "Exploratory sample of Upworthy's headline A/B tests, Jan 2013 - Apr 2015: 4,873 tests, "
        "22,666 packages (arms; 2-14 per test, mostly 4-6), one row per package: test_id "
        "(clickability_test_id), test_created (date of the test's first package), test_week "
        "(YYYYWW), arm (0.. within test, by creation time), content_id (arms of one test with the "
        "same id are identical in headline, image, excerpt, lede, share text and square image: "
        "natural A/A arms), headline, eyecatcher_id (image), impressions, clicks, significance "
        "(Upworthy's dashboard number), first_place, winner (editor's declared winner).",
    ),
    # ---- examples (E65) ----
    "ridgecrest_2019_comcat": Dataset(
        "ridgecrest_2019_comcat.csv",
        "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2016-01-01"
        "&endtime=2020-01-01&minlatitude=35.2&maxlatitude=36.4&minlongitude=-118.2"
        "&maxlongitude=-117.0&minmagnitude=2.0&orderby=time-asc&eventtype=earthquake",
        "USGS ANSS Comprehensive Earthquake Catalog (ComCat), FDSN event web service; Southern "
        "California Seismic Network solutions (network ci; 3 events from us; 1,877 events still "
        "status 'automatic'). US Government work, public domain. "
        "Downloaded 2026-09-28; ComCat revises events over time, so a fresh download can differ "
        "slightly - keep the cached file in data/.",
        "Every earthquake of magnitude 2.0 or more in the box 35.2-36.4 N, 118.2-117.0 W, "
        "2016-01-01 to 2019-12-31 (6,909 events), which contains the 2019 Ridgecrest, California "
        "sequence: the M6.4 of 2019-07-04 17:33:49 UTC and the M7.1 of 2019-07-06 03:19:53 UTC. "
        "Standard ComCat CSV columns: time (UTC, ISO 8601), latitude, longitude, depth (km), mag, "
        "magType (ml, mlr, mw...), nst, gap, dmin, rms, net, id, updated, place, type, "
        "horizontalError, depthError, magError, magNst, status, locationSource, magSource.",
        {"parse_dates": ["time"]},
    ),
    # ---- examples (E67) ----
    "atp_tour_matches": Dataset(
        "atp_tour_matches_2021_2026.csv",
        "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_2025.csv",
        "Jeff Sackmann, tennis_atp (github.com/JeffSackmann/tennis_atp), CC BY-NC-SA 4.0 "
        "(non-commercial, share-alike). The upstream repository was unreachable (404) on "
        "2026-09-28; downloaded from the archival mirror github.com/Aneeshers/tennis-sackmann-archive "
        "(atp/atp_matches_2021.csv ... atp_matches_2026.csv, June 2026 snapshot; the URL is one of "
        "the six files). ASSEMBLED: concatenated, kept tour-level events only and the columns below "
        "- keep the cached file in data/.",
        "ATP men's tour-level singles matches, 2021-01 to 2026-06-07 (Roland Garros 2026 is the "
        "last event): 14,844 matches, one row per match, winner first. Levels G (Grand Slam), M "
        "(Masters 1000), A (ATP 250/500 and other tour events), F (season finals); Davis Cup, "
        "Challengers and qualifying excluded. Columns: tourney_id, tourney_name, surface "
        "(Hard/Clay/Grass), tourney_level, tourney_date (YYYYMMDD, usually the Monday of the event "
        "week), round (R128...F, RR), best_of (3 or 5), match_num, winner_id, winner_name, loser_id, "
        "loser_name, winner_rank / loser_rank and *_rank_points (ATP ranking at the time), score "
        "(contains RET, W/O or DEF for retirements, walkovers, defaults).",
    ),
    "f1_race_results": Dataset(
        "f1_race_results_2021_2025.csv",
        "https://api.jolpi.ca/ergast/f1/2025/results.json?limit=100",
        "Jolpica-F1 API (successor of the Ergast motor-racing API; github.com/jolpica/jolpica-f1), "
        "data CC BY-NC-SA 4.0 per its Terms of Use. ASSEMBLED on 2026-09-28 from the paged "
        "/ergast/f1/<season>/results.json and /sprint.json endpoints for 2021-2025 (the URL is one "
        "page) into one table - keep the cached file in data/.",
        "Formula 1 results 2021-2025: 114 Grand Prix races and 24 sprint races, 2,758 rows, one "
        "per driver per session. Columns: season, round, session ('race' or 'sprint'; a sprint "
        "shares its weekend's round number), race, date, circuit, driver_id, driver_code, "
        "driver, constructor (team id), grid (0 = pit-lane start), position (FIA order including "
        "non-classified cars), position_text (the classification: a number if classified, R = "
        "retired / not classified, D = disqualified, W = withdrew before the start), status "
        "(Finished, +1 Lap, Collision, Engine...), laps, points.",
    ),
    # ---- examples (E68) ----
    "electricity_choice": Dataset(
        "electricity_choice.csv",
        "https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv",
        "Kenneth Train's stated-preference survey of electricity suppliers (Huber & Train 2001, "
        "Marketing Letters 12:259-269; Revelt & Train 2000, UC Berkeley working paper), distributed as `Electricity` in the R package "
        "mlogit (GPL >= 2) and, in long format, in the Python package xlogit (GPL-3; the URL). "
        "Verified identical, value for value, to mlogit 2.0-0's data/Electricity.rda.",
        "Stated-preference choice experiment: 361 residential customers each made up to 12 "
        "choices (4,308 choice tasks; 348 people did all 12) among 4 hypothetical electricity "
        "suppliers. Long format, one row per task x alternative (17,232 rows): choice (1 if "
        "chosen), id (person), alt (1-4, unlabelled), pf (fixed price, cents/kWh: 7 or 9; 0 when "
        "the contract is time-of-day or seasonal), cl (contract length in years: 0, 1 or 5; "
        "switching early costs a penalty), loc (local company), wk (well-known company), tod "
        "(time-of-day rates: 11c 8am-8pm, 5c otherwise), seas (seasonal rates: 10c summer, 8c "
        "winter, 6c spring/fall), chid (choice-task id).",
    ),
    # ---- examples (E69) ----
    "carcinoma_pathologists": Dataset(
        "carcinoma_pathologists.csv",
        "https://raw.githubusercontent.com/cran/poLCA/master/data/carcinoma.rda",
        "Holmquist, McMahan & Williams (1967, Archives of Pathology), as dichotomised in "
        "Landis & Koch (1977, Biometrics 33:363-374) and Agresti (2002, Categorical Data Analysis, "
        "Table 13.1); distributed as `carcinoma` in the CRAN package poLCA (GPL >= 2; the URL). "
        "CONVERTED from the .rda with R (values unchanged, a slide index added) - keep the CSV in data/.",
        "Seven pathologists (A-G) each classified the same 118 slides of the uterine cervix for "
        "carcinoma (in situ or invasive). One row per slide: slide (1-118), A..G (1 = no carcinoma, "
        "2 = carcinoma). There is no gold standard; 20 distinct rating patterns occur.",
    ),
    "anesthesia_dawid_skene": Dataset(
        "anesthesia_dawid_skene.csv",
        "https://raw.githubusercontent.com/cran/rater/master/data/anesthesia.rda",
        "Dawid & Skene (1979), Applied Statistics 28:20-28, Table 1; distributed as `anesthesia` in "
        "the CRAN package rater (Pullin, Gurrin & Vukcevic; GPL-2; the URL). CONVERTED from the .rda "
        "with R - keep the CSV in data/. An independent transcription (github.com/dallascard/"
        "dawid_skene) differs in one reading (patient 7, anaesthetist 1: 1,1,2 here vs 1,2,2).",
        "Pre-operative assessments of 45 patients by 5 anaesthetists on a 4-point ordinal scale of "
        "pre-operative health, made from a standard form; anaesthetist 1 assessed every form three "
        "times weeks apart, the others once (315 ratings). Long format: item (patient, 1-45), rater "
        "(1-5), rating (1-4).",
    ),
    # ---- examples (E70) ----
    "arctic_lake_sediment": Dataset(
        "arctic_lake_sediment.csv",
        "https://raw.githubusercontent.com/cran/compositions/master/data/ArcticLake.rda",
        "Aitchison (1986), The Statistical Analysis of Compositional Data, Data 5 (file ARCTIC.DAT "
        "of his CODA package); distributed as `ArcticLake` in the CRAN package compositions "
        "(van den Boogaart, Tolosana-Delgado & Bren; GPL >= 2; the URL). CONVERTED from the .rda with "
        "R (values unchanged, a sample index added) - keep the CSV in data/.",
        "Sand, silt and clay percentages of 39 sediment samples from an Arctic lake, with the water "
        "depth (m) at which each was taken. Columns: sample (1-39), sand, silt, clay (percent; rows "
        "sum to 100 up to rounding, 99.7-100.5), depth (10.4-103.7 m).",
    ),
    "uk_ge2024_england": Dataset(
        "uk_ge2024_england_candidates.csv",
        "https://electionresults.parliament.uk/general-elections/6/candidacies.csv",
        "UK Parliament election results service (House of Commons Library), general election of "
        "4 July 2024, candidacies file; Open Parliament Licence v3.0. DERIVED: English "
        "constituencies only and 11 of the 52 columns kept (renamed; values unchanged; independents' "
        "missing party set to 'Ind') - keep the CSV in data/ (the URL is the full UK file).",
        "One row per candidate standing in the 543 English constituencies (3,720 candidacies). "
        "Columns: region (9 English regions), constituency, ons_code, electorate, valid_votes (all "
        "valid votes in the seat), party (abbreviation: Lab, Con, RUK = Reform UK, LD, Green, WPB, "
        "SDP, ..., Ind), party_name, is_speaker (the Speaker, Chorley, whom the main parties did not "
        "oppose), is_independent, votes (candidate's votes), position (finishing place; 1 = winner).",
    ),
    # ---- examples (E72) ----
    "loc_posters_thumbs": Dataset(
        "loc_posters_thumbs.npz",
        "https://www.loc.gov/pictures/search/?co=pos&fo=json",
        "Library of Congress Prints & Photographs Division: Posters collection (search 'magazine', "
        "dated 1889-1905; search 'world war 1914-1918', dated 1914-1919) and Work Projects "
        "Administration Poster Collection (co=wpapos, dated 1935-1943); public domain / no known "
        "restrictions on publication per the LoC rights statements. BUILT by "
        "tools/fetch_loc_posters.py (seed 72): 250 randomly sampled items per era with a dated "
        "year and a reference image; each 640-px reference JPEG (the URL's `image.full`) resized "
        "to 64 x 48 with Lanczos. Keep the cached file in data/ (a rebuild draws from whatever "
        "the live search returns).",
        "NumPy .npz (open data.path(...) with np.load): thumbs (750 x 64 x 48 x 3 uint8 RGB, rows "
        "= height), era ('1890s magazine', 'WWI', 'WPA'; 250 each), year (first four-digit year "
        "in the LoC date), title, creator, loc_id, image_url. Many scans include the black or grey "
        "backing and a colour-checker strip around the poster (most often in the WWI group).",
    ),
    "msnbc_sessions": Dataset(
        "msnbc_sessions_sample.csv",
        "https://archive.ics.uci.edu/static/public/133/msnbc+com+anonymous+web+data.zip",
        "Heckerman, D. (1999), msnbc.com anonymous web data, UCI Machine Learning Repository "
        "(CC BY 4.0): IIS logs of msnbc.com and the news parts of msn.com for "
        "28 September 1999, 989,818 users. SUBSAMPLED by tools/build_e72_extras.py (seed 72): 60,000 "
        "random sessions - keep the cached file in data/.",
        "One row per user session (all page requests of one user that day): session (line number in "
        "the original file, from 0), n_views, pages (space-separated page-category codes in request "
        "order: 1 frontpage, 2 news, 3 tech, 4 local, 5 opinion, 6 on-air, 7 misc, 8 weather, "
        "9 msn-news, 10 health, 11 living, 12 business, 13 msn-sports, 14 sports, 15 summary, 16 bbs, "
        "17 travel). Cached pages were not logged.",
    ),
    "online_retail": Dataset(
        "online_retail_customer_products.csv",
        "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        "Chen, D. (2015), Online Retail, UCI Machine Learning Repository "
        "(CC BY 4.0); Chen, Sain & Guo (2012), Journal of Database Marketing & Customer Strategy "
        "Management 19:197-208. AGGREGATED by tools/build_e72_extras.py from the 541,909 "
        "transaction lines of the spreadsheet at the URL - keep the cached file in data/.",
        "A UK online retailer of giftware (many customers are wholesalers), orders 1 Dec 2010 - 9 Dec "
        "2011, one row per (customer, product): customer_id, country (the customer's most common), "
        "stock_code, n_invoices (orders containing the product), quantity (units). 4,335 customers, "
        "3,659 products, 266,226 rows; cancellations, returns, rows without a customer and "
        "non-product codes (postage, fees, adjustments) removed. Product names: online_retail_products.",
    ),
    "online_retail_products": Dataset(
        "online_retail_products.csv",
        "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        "As online_retail (UCI Online Retail, CC BY 4.0), built by tools/build_e72_extras.py.",
        "One row per product: stock_code, description (its most common description in the "
        "transactions), n_customers.",
    ),
    # ---- examples (E75) ----
    "bsds500_subset": Dataset(
        "bsds500_subset.npz",
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
        "Berkeley Segmentation Dataset BSDS500: Arbelaez, Maire, Fowlkes & Malik (2011), Contour "
        "detection and hierarchical image segmentation, IEEE TPAMI 33(5):898-916; human segmentations "
        "from Martin, Fowlkes, Tal & Malik (2001), ICCV. Images from the Corel collection. The Berkeley "
        "page allows downloading 'a portion of the dataset for non-commercial research and educational "
        "purposes' and asks users to cite the papers; no open licence. A SMALL SUBSET BUILT by "
        "tools/build_e75_bsds.py from the 70 MB archive (the URL) - keep the cached file.",
        "NumPy .npz (open data.path(...) with np.load): ten landscape photographs at half resolution "
        "(2 x 2 box average of 321 x 481): ids (BSDS ids), split ('train': 118035, 113044; 'test': "
        "100007, 8068, 3063, 228076, 97010, 29030, 16068, 108004), images (10 x 161 x 241 x 3 uint8 "
        "sRGB), human (10 x 7 x 161 x 241 int16: each annotator's segment labels 1, 2, ..., "
        "subsampled [::2, ::2] from the full-resolution maps; 0 = no such annotator), n_human (5-6 "
        "annotators per photograph).",
    ),
    # ---- examples (E80) ----
    "slacs_j1627": Dataset(
        "slacs_j1627_f814w.npz",
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/j9c701020_drc.fits",
        "NASA/ESA Hubble Space Telescope, ACS/WFC F814W, programme 10494 (PI L. Koopmans; SLACS "
        "follow-up), observed 2006-03-12, 4 exposures, 2,224 s; pipeline-drizzled, CTE-corrected product "
        "j9c701020_drc.fits from the Mikulski Archive for Space Telescopes (MAST, STScI). HST archival data "
        "are public; MAST asks for the acknowledgement 'Based on observations made with the NASA/ESA Hubble "
        "Space Telescope, obtained from the data archive at the Space Telescope Science Institute. STScI is "
        "operated by the Association of Universities for Research in Astronomy, Inc. under NASA contract "
        "NAS 5-26555.' The lens is SDSS J162746.44-005357.5 (Bolton et al. 2008, ApJ 682, 964). A SMALL "
        "EXTRACT BUILT by tools/build_e80_lens.py from the 215 MB FITS file (the URL) - keep the cached file.",
        "NumPy .npz (open data.path(...) with np.load), all arrays turned by a multiple of 90 degrees so "
        "that north is up and east left (north_residual_deg: the remaining angle of north from +y, towards "
        "-x): sci (121 x 121 cutout centred on the lens galaxy, electrons/s, sky subtracted by the pipeline), "
        "wht (the same, effective exposure time in s), blank_sci / blank_wht (128 x 128 empty sky nearby), "
        "star_stamps (10 x 41 x 41 isolated stars within 70\" of the lens, for the PSF) and star_xy (their "
        "positions in the original frame), pixel_scale (0.05 arcsec), exptime, photflam, photplam, ra_dec, "
        "lens_xy_full, date_obs, proposal, rootname, filter.",
    ),
    # ---- examples (E82) ----
    "edbo_arylation": Dataset(
        "edbo_arylation.csv",
        "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/data/direct_arylation/"
        "experiment_index.csv",
        "Shields, Stevens, Li, Parasram, Damani, Martinez Alvarado, Janey, Adams & Doyle (2021), Bayesian "
        "reaction optimization as a tool for chemical synthesis, Nature 590:89-96; data from the authors' "
        "repository github.com/b-shields/edbo (MIT licence, (c) 2020 Benjamin J. Shields). BUILT by "
        "tools/build_e82_arylation.py (SMILES replaced by the names in the repository's *-list.csv files) "
        "- keep the cached file.",
        "A fully enumerated high-throughput screen of a palladium-catalysed direct (C-H) arylation: every "
        "combination of 12 phosphine ligands x 4 carboxylate bases x 4 solvents x 3 concentrations x 3 "
        "temperatures, one yield each (1,728 rows): ligand, base, solvent, concentration (M: 0.057, 0.1, "
        "0.153), temperature (C: 90, 105, 120), yield (%, 0-100).",
    ),
    "edbo_arylation_game": Dataset(
        "edbo_arylation_game.csv",
        "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/arylation_game_summary.csv",
        "As edbo_arylation: the 'reaction optimisation game' of Shields et al. (2021), in which chemists "
        "chose experiments on the same grid and were shown the recorded yields. BUILT (reshaped to long "
        "format) by tools/build_e82_arylation.py.",
        "1,548 experiments by 50 participants (10-100 each): participant (0-49), area (Pharma, Academic, "
        "Other), expertise, experience (as self-reported), step (order played), ligand, base, solvent, "
        "concentration, temperature, yield.",
    ),
    "edbo_arylation_edbo_runs": Dataset(
        "edbo_arylation_edbo_runs.csv",
        "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/"
        "arylation_bo_results_GP-EI_bs=5.csv",
        "As edbo_arylation: the authors' own simulated EDBO campaigns (Gaussian-process surrogate, "
        "expected improvement, batches of 5, as the file name says). BUILT (long format) by tools/build_e82_arylation.py.",
        "50 simulated campaigns x 100 experiments: run, step (1-100), yield (%).",
    ),
    "edbo_arylation_ligand_dft": Dataset(
        "edbo_arylation_ligand_dft.csv",
        "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/data/direct_arylation/"
        "ligand-boltzmann_dft.csv",
        "As edbo_arylation: DFT descriptors of the ligands computed by the authors (conformer "
        "Boltzmann-weighted averages). BUILT by tools/build_e82_arylation.py (the '_Boltz' numeric "
        "columns only).",
        "12 rows (ligand) x 366 descriptors: energies (HOMO, LUMO, ...), dipole, volumes, atomic charges, "
        "NMR shifts and buried volumes of the phosphorus and its neighbours, and so on.",
    ),
    # ---- examples (E74) ----
    "woodblock_fading": Dataset(
        "woodblock_fading_spectra.csv",
        "https://zenodo.org/records/17014267",
        "Baines, Brokerhof, Christoforou, Jansen, van Leeuwen, Patin & Sauvage (2026), Balancing "
        "access and preservation: investigating light-induced fading of colourants used in Japanese "
        "woodblock prints, Journal of Paper Conservation 27:18-30; data on Zenodo 17014267 (CC BY "
        "4.0), Rijksmuseum / Cultural Heritage Agency of the Netherlands (RCE). BUILT by "
        "tools/build_e74_fading.py from the MFT and xenotest spreadsheets - keep the cached file.",
        "Reflectance spectra of reconstructions of Edo-period woodblock-print colourants printed on "
        "Japanese paper in 1999 by the Museum of Fine Arts, Boston, faded in 2024 by a microfading "
        "tester (instrument MFT: 2700 K LED spot, ~4.2 Mlx, no UV; three spots averaged) and in a "
        "xenon-arc chamber (XT: daylight-through-glass filter, 0.1 Mlx). One row per (instrument, "
        "colourant, dose, wavelength): dose_Mlxh (visible exposure, million lux hours), wavelength "
        "(380-730 nm, 10 nm; MFT averaged over +-4 nm), reflectance (0-1), refl_sd (MFT spot sd). "
        "Colourants: dayflower, safflower, turmeric, sappanwood, cochineal, yellowwood, orpiment, "
        "indigo, Prussian blue, vermilion, red lead, iron oxide, ochre, lead white, mica, brass "
        "powder, three mixtures (safflower+dayflower, indigo+orpiment, turmeric+orpiment), the bare "
        "paper, and (MFT only) Blue Wool references BW1-BW4.",
    ),
    "woodblock_fading_lab": Dataset(
        "woodblock_fading_lab.csv",
        "https://zenodo.org/records/17014267",
        "As woodblock_fading (Baines et al. 2026, Zenodo 17014267, CC BY 4.0).",
        "The CIELAB values the authors computed (D65, 10 degree observer): instrument, colourant, "
        "dose_Mlxh, L, a, b. MFT: every 10-second reading (0.1 MJ/m2 apart).",
    ),
    "cie1964_d65": Dataset(
        "cie1964_d65.csv",
        "http://www.cvrl.org/database/data/cmfs/ciexyz64_1.csv",
        "CIE 1964 10-degree colour-matching functions (CVRL, UCL) and CIE standard illuminant D65 "
        "(CIE 015; the 5 nm table as distributed with colour-science), assembled by "
        "tools/build_e74_fading.py.",
        "wavelength (360-830 nm, 1 nm), xbar, ybar, zbar, d65 (relative power, 100 at 560 nm; "
        "linearly interpolated from 5 nm).",
    ),
    "harunobu_impressions": Dataset(
        "harunobu_impressions.npz",
        "https://api.artic.edu/api/v1/artworks/20814",
        "Suzuki Harunobu (c. 1766), two impressions each of two designs from the series Eight Views of "
        "the Parlor (Zashiki hakkei): Descending Geese of the Koto Bridges (AIC 20814, 88966) and The "
        "Evening Glow of a Lamp (AIC 88968, 20817). Art Institute of Chicago, public domain, "
        "CC0 images via IIIF. Downloaded by tools/build_e74_fading.py.",
        "NumPy .npz (open data.path(...) with np.load): images (4 x 528 x 400 x 3 uint8 sRGB; the "
        "400-px-wide IIIF JPEGs trimmed top and bottom to a common height), aic_id, design "
        "('geese', 'lamp'), title.",
    ),
    # ---- examples (E76) ----
    "munsell_matt": Dataset(
        "munsell_matt_spectra.csv",
        "https://zenodo.org/records/3269912",
        "Munsell Colors Matt (spectrophotometer measured), University of Kuopio / University of "
        "Eastern Finland spectral color research database (Hauta-Kasari, n.d.; measured by J. "
        "Hiltunen), mirrored on Zenodo 3269912 by the colour-science project. Licence "
        "field on Zenodo: 'not specified'; cite the database. BUILT by tools/build_e76_metamers.py "
        "(10 nm means of the 1 nm data) - keep the cached file.",
        "Reflectance spectra of the 1,269 chips of the Munsell Book of Color, Matte Finish Collection "
        "(1976), measured on a Perkin-Elmer Lambda 9 at 1 nm. One row per chip: chip (e.g. '2.5R 9/2'), "
        "hue, hue_family (R, YR, Y, GY, G, BG, B, PB, P, RP), value, chroma, r380 ... r780 "
        "(reflectance 0-1 at 10 nm, the mean of the 1 nm readings within +-4 nm).",
    ),
    "cie_illuminants_e76": Dataset(
        "cie_illuminants_e76.csv",
        "https://raw.githubusercontent.com/colour-science/colour/develop/colour/colorimetry/datasets/"
        "illuminants/sds.py",
        "CIE illuminants (CIE 15:2004 tables, CIE 015:2018) as distributed with colour-science; A "
        "computed from Planck's law (CIE formula, checked against the table to 4e-6). Assembled by "
        "tools/build_e76_metamers.py.",
        "wavelength (380-780 nm, 5 nm) and relative spectral power of A (incandescent, 2856 K), FL2 "
        "(cool-white fluorescent), FL11 (narrow-band triphosphor fluorescent), LED_B2 and LED_B4 "
        "(phosphor-converted white LEDs).",
    ),
    # ---- examples (E77) ----
    "wcs_naming": Dataset(
        "wcs_naming.npz",
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/term.txt",
        "World Color Survey (Kay, Berlin, Maffi, Merrifield & Cook 2009, The World Color Survey, CSLI "
        "Publications); WCS Data Archives, https://linguistics.berkeley.edu/wcs/data.html (formerly "
        "www1.icsi.berkeley.edu/wcs), which asks that published work cite the archives and states no "
        "other licence. Raw files read from the GitHub mirror jvosten/wcs (data-raw/). BUILT by "
        "tools/build_e77_wcs.py - keep the cached file.",
        "NumPy .npz (open data.path(...) with np.load): naming (2,616 speakers x 330 chips, int16 term "
        "code - see wcs_terms - or -1 for no response), lang, speaker, age (-1 unknown), sex ('M', 'F', "
        "''); foci_lang, foci_speaker, foci_code, foci_chip (31k best-example choices; white/black bar "
        "choices mapped to chips A0/J0). 110 languages, ~24 speakers each; chip = column index + 1.",
    ),
    "wcs_terms": Dataset(
        "wcs_terms.csv",
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/dict.txt",
        "As wcs_naming (World Color Survey archives; built by tools/build_e77_wcs.py).",
        "One row per (language, term) used in the naming task: lang, code (0 = most used), abbrev (WCS "
        "abbreviation), term (transcription(s) from dict.txt), n_responses, n_speakers.",
    ),
    "wcs_chips": Dataset(
        "wcs_chips.csv",
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/cnum-vhcm-lab-new.txt",
        "As wcs_naming (World Color Survey archives: chip.txt and cnum-vhcm-lab-new.txt).",
        "The 330 WCS Munsell chips: chip (1-330), row (A-J, light to dark), col (0 achromatic, 1-40 "
        "hue), grid, munsell_hue, munsell_value, munsell_chroma, L, a, b (CIELAB as given by the "
        "archive).",
    ),
    "wcs_languages": Dataset(
        "wcs_languages.csv",
        "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/wcs_iso_codes.csv",
        "As wcs_naming (the archive's WCS_SIL_codes table, via jvosten/wcs).",
        "lang (1-110), name, iso639_3, family, country, n_speakers.",
    ),
    # ---- examples (E78) ----
    "kepler10_lc": Dataset(
        "kepler10_llc.csv.gz",
        "https://archive.stsci.edu/missions/kepler/lightcurves/0119/011904151/",
        "NASA Kepler mission long-cadence light curves of Kepler-10 (KIC 11904151), Mikulski Archive for "
        "Space Telescopes (MAST, STScI). NASA mission data, public (MAST asks users to acknowledge MAST and "
        "the Kepler mission, funded by NASA's Science Mission Directorate). BUILT by "
        "tools/build_e78_exoplanets.py from the 15 *_llc.fits files at the URL - keep the cached file.",
        "One row per 29.4-minute cadence, all 15 quarters on the archive (Q0-Q17 minus Q8, Q12, Q16, when the "
        "star fell on the failed CCD module 3): quarter, time (BKJD = BJD_TDB - 2454833, days), flux and "
        "flux_err (PDCSAP flux: systematics-corrected aperture photometry, e-/s; NaN where missing), quality "
        "(SAP_QUALITY bit mask, 0 = no flags). 53,029 rows.",
    ),
    "kepler10_archive": Dataset(
        "kepler10_archive.csv",
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps+where+hostname='Kepler-10'",
        "NASA Exoplanet Archive, Planetary Systems (ps) table, TAP query (NASA Exoplanet Science Institute / "
        "Caltech-IPAC; the Archive asks for the acknowledgement 'This research has made use of the NASA "
        "Exoplanet Archive, which is operated by the California Institute of Technology, under contract with "
        "NASA under the Exoplanet Exploration Program'). Queried 2026-09-30 by tools/build_e78_exoplanets.py.",
        "One row per published solution for Kepler-10 b, c and d (48 rows; default_flag = 1 marks the "
        "Archive's default): pl_orbper (days), pl_tranmid (BJD), pl_rade (Earth radii), pl_ratror (Rp/R*), "
        "pl_imppar, pl_trandur (hours), pl_trandep (per cent), pl_ratdor (a/R*), pl_orbeccen, st_rad, st_mass, "
        "st_dens (g/cm3) with errors, st_teff, st_logg, pl_refname (reference, HTML stripped), disc_year.",
    ),
    "borisov_astrometry": Dataset(
        "borisov_mpc_obs.csv",
        "https://data.minorplanetcenter.net/api/get-obs",
        "IAU Minor Planet Center observations API (ADES fields), object 2I/Borisov (C/2019 Q4). MPC data "
        "are public; the MPC asks for the acknowledgement 'This research has made use of data and/or "
        "services provided by the International Astronomical Union's Minor Planet Center.' BUILT by "
        "tools/build_e78_exoplanets.py (a POST-style JSON request) - keep the cached file.",
        "1,513 optical positions from discovery (2019-08-30) to 2019-10-15 UTC, deprecated rows removed, from "
        "88 observatories: obstime (ISO UTC), ra, dec (degrees, ICRF, as reported), stn (MPC observatory "
        "code), mag, band, astcat (reference star catalogue), rmsra, rmsdec (arcsec, where reported), mode, "
        "notes, and for the satellite NEOSSat (C53) its geocentric position: sys (ICRF_KM), pos1-pos3 (km).",
    ),
    "borisov_obscodes": Dataset(
        "borisov_obscodes.csv",
        "https://data.minorplanetcenter.net/api/obscodes",
        "IAU Minor Planet Center observatory codes API (as borisov_astrometry).",
        "The 88 observatory codes in borisov_astrometry: stn, longitude (degrees east), rhocosphi, "
        "rhosinphi (geocentric parallax constants in Earth radii; blank for the satellite C53), name.",
    ),
    "borisov_horizons": Dataset(
        "borisov_horizons.csv",
        "https://ssd.jpl.nasa.gov/api/horizons.api",
        "JPL Horizons ephemeris service (Solar System Dynamics group, JPL/Caltech; planetary ephemeris "
        "DE441; 2I solution JPL#54). Public service; cite Giorgini et al. (1996) / JPL Horizons. Queried "
        "2026-09-30 by tools/build_e78_exoplanets.py.",
        "Geometric heliocentric state vectors (ICRF, au and au/day, TDB), 2019-08-29 to 2019-10-17: body "
        "(earth hourly; jupiter and saturn system barycentres and 2I_jpl - JPL's own orbit of 2I, with "
        "non-gravitational terms - every 6 h), jd_tdb, x, y, z, vx, vy, vz.",
    ),
    "borisov_sbdb": Dataset(
        "borisov_sbdb.csv",
        "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=2I&full-prec=1",
        "JPL Small-Body Database API (Solar System Dynamics group, JPL/Caltech), queried 2026-09-30.",
        "JPL's published orbit of 2I/Borisov (solution 54, epoch JD 2458853.5 TDB): element, value, sigma, "
        "units - e, a, q, i, om, w, ma, tp, n, the non-gravitational parameters A1-A3 and DT, and rows for "
        "orbit_id, epoch, data_arc, first_obs, last_obs, n_obs_used, producer.",
    ),
    # ---- examples (E79) ----
    "planck_smica_patches": Dataset(
        "planck_smica_patches.npz",
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/",
        "Planck 2018 (PR3) SMICA CMB temperature maps (COM_CMB_IQU-smica_2048_R3.00_full, _hm1, _hm2; "
        "Planck Collaboration 2020, A&A 641, A4) and the common intensity confidence mask "
        "(COM_Mask_CMB-common-Mask-Int_2048_R3.00), from the IRSA mirror of the ESA Planck Legacy "
        "Archive. ESA asks users to acknowledge: 'Based on observations obtained with Planck "
        "(http://www.esa.int/Planck), an ESA science mission with instruments and contributions directly "
        "funded by ESA Member States, NASA, and Canada.' BUILT by tools/build_e79_cmb.py, which "
        "downloads only the needed HEALPix rows with HTTP range requests - keep the cached file.",
        "NumPy .npz (open data.path(...) with np.load): two 16 x 16 degree gnomonic patches, 256 x 256 "
        "pixels of 3.75 arcmin, each pixel the mean of 4 x 4 point samples of the Nside-2048 maps; rows "
        "run up in galactic latitude, columns towards decreasing galactic longitude. Per patch (prefix "
        "'lmc_' centred on (l, b) = (280, -35), around the Large Magellanic Cloud; 'north_' on (60, 55)): "
        "I (full-mission temperature, uK_CMB), hm1, hm2 (half-mission maps), inp (SMICA's inpainted "
        "map, released 'for PR purposes'), mask (fraction of samples kept by the common mask; 1 = "
        "usable), tmask (the same for SMICA's own confidence mask), centre. Global: reso_arcmin, n, sub, "
        "beam_fwhm_arcmin (5: SMICA's effective Gaussian beam), pixwin_2048 (HEALPix Nside-2048 "
        "temperature pixel window, ell = 0..4096, from healpy).",
    ),
    "planck_tt_theory": Dataset(
        "planck_theory_tt.txt",
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"
        "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt",
        "Planck 2018 best-fit LCDM theory spectra (base_plikHM_TTTEEE_lowl_lowE_lensing, the 'minimum' "
        "file; Planck Collaboration 2020, A&A 641, A6), Planck Legacy Archive via IRSA; acknowledgement "
        "as planck_smica_patches.",
        "ell (2-2508) and D_ell = ell (ell + 1) C_ell / 2 pi in uK^2 for TT, TE, EE, BB, and the lensing "
        "potential PP.",
        read_kwargs={"sep": r"\s+", "comment": "#", "header": None,
                     "names": ["ell", "TT", "TE", "EE", "BB", "PP"]},
    ),
    "planck_tt_binned": Dataset(
        "planck_tt_binned.txt",
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"
        "COM_PowerSpect_CMB-TT-binned_R3.01.txt",
        "Planck 2018 binned TT power spectrum (Planck Collaboration 2020, A&A 641, A5), Planck Legacy "
        "Archive via IRSA; acknowledgement as planck_smica_patches.",
        "83 bins from ell = 48 to 2499 (the full-sky, foreground-cleaned Plik spectrum): ell (bin "
        "centre), Dl, dDl_lo, dDl_hi (uK^2), bestfit.",
        read_kwargs={"sep": r"\s+", "comment": "#", "header": None,
                     "names": ["ell", "Dl", "dDl_lo", "dDl_hi", "bestfit"]},
    ),
    # ---- examples (E81) ----
    "ksba_wind": Dataset(
        "asos_ksba_2021_2023.csv.gz",
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=SBA&data=drct&data=sknt&data=tmpf"
        "&data=dwpf&year1=2021&month1=1&day1=1&year2=2024&month2=1&day2=1&tz=Etc/UTC&format=onlycomma"
        "&latlon=no&missing=empty&trace=empty&direct=no&report_type=3",
        "Iowa Environmental Mesonet (IEM, Iowa State University) ASOS/METAR archive, station SBA (Santa "
        "Barbara Municipal Airport, California; an FAA/NWS Automated Surface Observing System). Public "
        "US government observations redistributed by IEM, which asks to be acknowledged. BUILT by "
        "tools/build_e81_circular.py (the URL returns an uncompressed CSV) - keep the cached file.",
        "Routine hourly METAR reports, 2021-01-01 to 2023-12-31 (26,101 rows, at minute 53 of each hour): "
        "valid (UTC timestamp), drct (direction the wind blows FROM, degrees clockwise from true north, "
        "reported in steps of 10; 0 together with sknt = 0 means calm), sknt (speed, knots), tmpf and dwpf "
        "(air and dew-point temperature, Fahrenheit). Empty = missing (e.g. variable direction).",
    ),
    "bci_camera_trap": Dataset(
        "bci_camera_trap_times.csv",
        "https://ndownloader.figshare.com/files/3219290",
        "Rowcliffe, Kays, Kranstauber, Carbone & Jansen (2014), 'Activity level estimation data', figshare, "
        "doi:10.6084/m9.figshare.1160536 (CC BY 4.0), file BCItime.txt; data of Rowcliffe et al. (2014), "
        "'Quantifying levels of animal activity using camera trap data', Methods in Ecology and Evolution "
        "5:1170-1179. Also shipped as `BCItime` in the R package activity. CONVERTED from space- to "
        "comma-separated by tools/build_e81_circular.py - keep the cached file.",
        "17,820 camera-trap records from Barro Colorado Island, Panama, 2008: species (13: agouti, peccary, "
        "paca, rat, brocket, squirrel, coati, ocelot, tamandua, armadillo, opossum, mouse, tayra) and time "
        "(time of day as a fraction of 24 hours, to the minute; no dates).",
    ),
    # ---- examples (E83) ----
    "natality_births": Dataset(
        "natality2023_births_sample.csv.gz",
        "https://data.nber.org/nvss/natality/csv/2023/natality2023us.zip",
        "National Center for Health Statistics (NCHS), National Vital Statistics System, Natality public use "
        "file 2023 (all 3,605,081 US birth certificates of 2023; codes as in the NCHS 'User Guide to the 2023 "
        "Natality Public Use File'), read from NBER's CSV conversion (the URL; 188 MB). US government work, "
        "public domain; NCHS asks users to cite the source and not to attempt to identify anyone. BUILT by "
        "tools/build_e83_natality.py (a seeded random subsample) - keep the cached file.",
        "A simple random sample of 20,000 of the 3,491,735 SINGLETON births of 2023 (raw codes; 9 / 99 / "
        "99.9 / 999 / 9999 = unknown, empty = not reported): dob_mm, mager (mother's age), mrace6 (1 White, "
        "2 Black, 3 AIAN, 4 Asian, 5 NHOPI, 6 more than one race), dmar (1 married, 2 unmarried; not reported "
        "for California), meduc (1-8 education, 9 unknown), tbo_rec (total birth order), previs (prenatal "
        "visits), cig_0..cig_3 (cigarettes per day before pregnancy and in trimesters 1-3), m_ht_in (inches), "
        "bmi (pre-pregnancy), pwgt_r (pre-pregnancy weight, lb), wtgain (lb), rf_pdiab / rf_gdiab / rf_ghype "
        "(pre-pregnancy diabetes, gestational diabetes, gestational hypertension: Y/N/U), sex (M/F), combgest "
        "and oegest_comb (gestation in weeks, combined and obstetric estimate), dbwt (birth weight, grams).",
    ),
    # ---- examples (E84) ----
    "margarine": Dataset(
        "margarine_purchases.csv",
        "https://cran.r-project.org/src/contrib/Archive/bayesm/bayesm_3.1-6.tar.gz",
        "Allenby & Rossi (1991), 'Quality perceptions and asymmetric switching between brands', "
        "Marketing Science 10:185-205: an A.C. Nielsen scanner panel, distributed as `margarine` in the "
        "R package bayesm (Rossi; GPL >= 2; the URL is its CRAN source tarball). CONVERTED from .rda by "
        "tools/build_e84_margarine.py - keep the cached file.",
        "4,470 margarine purchases by 516 households, in time order within household (no dates): hhid, "
        "trip (1, 2, ... within household), choice (one of 10 products: Pk = Parkay, BB = Blue Bonnet, "
        "Fl = Fleischmann's, Hse = the store's house brand, Gen = generic, Imp = Imperial, SS = Shedd's "
        "Spread; _Stk = sticks, _Tub = tub), and price_<product>: the shelf price (US$ per pound) of every "
        "product on that shopping trip.",
    ),
    "margarine_demos": Dataset(
        "margarine_demos.csv",
        "https://cran.r-project.org/src/contrib/Archive/bayesm/bayesm_3.1-6.tar.gz",
        "As `margarine` (bayesm `margarine$demos`). CONVERTED by tools/build_e84_margarine.py - keep the "
        "cached file.",
        "Demographics of the 516 households: hhid, Income (US$ 1000s, bracket midpoints), Fs3_4 and Fs5 "
        "(family size 3-4, 5+), Fam_Size, college, whtcollar, retired (0/1).",
    ),
    # ---- examples (E85) ----
    "cj_cereal_incidence": Dataset(
        "cj_cereal_incidence.csv",
        "https://github.com/bradleyboehmke/completejourney/raw/master/data/transactions.rds",
        "84.51, 'The Complete Journey' (2017 grocery transactions of 2,469 households at one retailer), "
        "distributed as the R package completejourney (Boehmke, Davis & Delaney; CC0 1.0; the URL is its "
        "transaction file). DERIVED by tools/build_e85_cereal.py - keep the cached file.",
        "Half-year purchase incidence of cold cereals, the format panel providers deliver: household_id, half "
        "(1 = weeks 1-26 of 2017, 2 = weeks 27-53) for every household with a shopping trip in that half, and "
        "0/1 columns for 14 cereal products = manufacturer x segment (M194_family = manufacturer 194's "
        "all-family cereals; manufacturers are anonymised; Store = the retailer's own brand; segments kids, "
        "family, adult), 1 if bought at least once in the half.",
    ),
    "cj_activity": Dataset(
        "cj_activity.csv",
        "https://github.com/bradleyboehmke/completejourney/raw/master/data/transactions.rds",
        "As `cj_cereal_incidence`. DERIVED by tools/build_e85_cereal.py - keep the cached file.",
        "Shopping activity per household and half-year (all departments): household_id, half, trips "
        "(distinct baskets), spend (US$).",
    ),
    "cj_demographics": Dataset(
        "cj_demographics.csv",
        "https://github.com/bradleyboehmke/completejourney/raw/master/data/demographics.rda",
        "As `cj_cereal_incidence` (completejourney `demographics`, CC0). CONVERTED from .rda by "
        "tools/build_e85_cereal.py - keep the cached file.",
        "Demographics of 801 of the 2,469 households (bands as delivered): household_id, age, income, "
        "home_ownership, marital_status, household_size, household_comp, kids_count.",
    ),
    # ---- examples (E71) ----
    "speed_acc": Dataset(
        "wagenmakers2008_speed_acc.csv",
        "https://raw.githubusercontent.com/rtdists/rtdists/master/data/speed_acc.RData",
        "Wagenmakers, Ratcliff, Gomez & McKoon (2008), 'A diffusion model account of criterion shifts "
        "in the lexical decision task', Journal of Memory and Language 58:140-159, Experiment 1; "
        "distributed as `speed_acc` in the R package rtdists (Singmann et al.; GPL >= 3; the URL). "
        "CONVERTED from the xz-compressed .RData (R factors written as their labels; values "
        "unchanged) - keep the CSV in data/.",
        "Lexical decision (is this letter string a word?) with speed vs accuracy instructions "
        "alternating by block: 17 participants, 31,522 trials. Columns: id (1-17), block (1-20; "
        "participant 2 did 9), condition (speed / accuracy instruction), stim (item number), "
        "stim_cat (word / nonword), frequency (high, low, very_low for words; nw_high, nw_low, "
        "nw_very_low for nonwords made from words of that frequency), response (word, nonword, or "
        "error = invalid key), rt (seconds), censor (1 for the 171 trials the authors excluded: "
        "RT < 0.18 s or > 3 s, or an invalid response).",
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
    if p.suffix == ".geojson":
        raise ValueError(f"{name!r} is GeoJSON, not a table: open data.path({name!r}) with json.")
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

