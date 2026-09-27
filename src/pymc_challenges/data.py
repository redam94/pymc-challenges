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

