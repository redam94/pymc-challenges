"""Build the E83 dataset: a random subsample of US singleton births in 2023 from the NCHS natality
public-use file.

    .venv/bin/python tools/memguard.py --max-gb 3 -- .venv/bin/python tools/build_e83_natality.py

Source: National Center for Health Statistics (NCHS), National Vital Statistics System, "Natality
public use file, 2023" (3,605,081 birth records; codes as in the NCHS "User Guide to the 2023 Natality
Public Use File"). US government work, public domain; NCHS asks users not to attempt to identify any
individual and to cite the source. We read NBER's CSV conversion of the same file
(https://data.nber.org/nvss/natality/csv/2023/natality2023us.zip, 188 MB) because the NCHS FTP
server delivered about 30 KB/s here; NBER keeps the NCHS variable names and codes.

The zip is streamed (never unpacked to disk). Each singleton birth (dplural = 1) is kept with
probability 0.009 using numpy.random.default_rng(2023), then exactly 20,000 of the kept records are
drawn at random with the same generator. Columns are the raw codes of the user guide (unknown =
9, 99, 99.9, 999 or 9999 as documented; empty = item not reported):

    dob_mm (1-12), mager (mother's age, years), mrace6 (1 White, 2 Black, 3 AIAN, 4 Asian, 5 NHOPI,
    6 more than one race), dmar (1 married, 2 unmarried), meduc (1-8 education level, 9 unknown),
    tbo_rec (total birth order 1-8, 9 unknown), previs (prenatal visits), cig_0..cig_3 (cigarettes per
    day before pregnancy and in trimesters 1-3), m_ht_in (height, inches), bmi (pre-pregnancy BMI),
    pwgt_r (pre-pregnancy weight, lb), wtgain (weight gain, lb), rf_pdiab, rf_gdiab, rf_ghype
    (pre-pregnancy diabetes, gestational diabetes, gestational hypertension: Y / N / U), sex (M/F),
    combgest (combined gestation, weeks), oegest_comb (obstetric estimate of gestation, weeks), dbwt
    (birth weight, grams).
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TMP = ROOT / ".scratch" / "e83"
URL = "https://data.nber.org/nvss/natality/csv/2023/natality2023us.zip"
N_KEEP = 20_000
P_KEEP = 0.009
COLS = ["dob_mm", "mager", "mrace6", "dmar", "meduc", "tbo_rec", "previs", "cig_0", "cig_1", "cig_2",
        "cig_3", "m_ht_in", "bmi", "pwgt_r", "wtgain", "rf_pdiab", "rf_gdiab", "rf_ghype", "sex",
        "combgest", "oegest_comb", "dbwt"]
TEXT = {"rf_pdiab", "rf_gdiab", "rf_ghype", "sex"}


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    zpath = TMP / "natality2023us.zip"
    if not zpath.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, zpath)
    rng = np.random.default_rng(2023)
    rows, n_all, n_single = [], 0, 0
    with zipfile.ZipFile(zpath) as z, z.open("natality2023us.csv") as fh:
        reader = csv.reader(io.TextIOWrapper(fh, encoding="latin-1", newline=""))
        head = next(reader)
        ip = head.index("dplural")
        idx = [head.index(c) for c in COLS]
        for rec in reader:
            n_all += 1
            if rec[ip] != "1":
                continue
            n_single += 1
            if rng.random() >= P_KEEP:
                continue
            rows.append([rec[i] for i in idx])
    print(f"records {n_all:,}, singletons {n_single:,}, kept {len(rows):,}")
    df = pd.DataFrame(rows, columns=COLS)
    df = df.iloc[np.sort(rng.choice(len(df), N_KEEP, replace=False))].reset_index(drop=True)
    for c in COLS:
        if c not in TEXT:
            df[c] = pd.to_numeric(df[c].replace("", np.nan))
    df["bmi"] = df["bmi"].round(1)
    df.to_csv(DATA / "natality2023_births_sample.csv.gz", index=False)
    print(df.describe().T.to_string())
    for c in sorted(TEXT):
        print(c, df[c].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
