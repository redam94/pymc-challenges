"""Build the E123 datasets: a captive feeding trial of ten gray wolves, with what they were fed.

    python tools/build_e123_wolf_trial.py   # download (cached), derive, check, write data/e123_*.csv
    python tools/build_e123_wolf_trial.py --allow-new-source   # accept a workbook whose md5 has changed

Source: S1 Table of Derbridge JJ, Merkle JA, Bucci ME, Callahan P, Koprowski JL, Polfus JL and
Krausman PR (2015), "Experimentally derived d13C and d15N discrimination factors for gray wolves
and the impact of prior information in Bayesian mixing models", PLoS ONE 10(3): e0119940,
https://doi.org/10.1371/journal.pone.0119940. The table's caption: "d13C and d15N of hair from 10
gray wolves, and of tissue samples from white-tailed deer, beaver, and goose they consumed during a
captive feeding study, Forest Lake, Minnesota, USA, 2011-2012."
Licence, from the article: "This is an open access article distributed under the terms of the
Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction
in any medium, provided the original author and source are credited" (CC BY 4.0).

The workbook (181 kB, four sheets) is cached in ~/.cache/pymc-challenges/e123 (not committed) and
read with the standard library, because the project has no Excel reader among its dependencies.
Every value in the three tables is a measurement or a record; nothing is a summary statistic.

What the builder does, and nothing else:
  - turns Excel day numbers into dates and rounds the delta values to the 0.1 permil they were
    reported at (the workbook stores -19.4 as -19.399999999999999);
  - sheet "Wolf" -> e123_wolf_hair.csv, all 48 hair samples;
  - sheets "Muscle" and "Hair" -> e123_prey_tissue.csv; the three deer whose muscle value is NA are
    dropped from the muscle rows (their hair rows stay);
  - sheet "diet_summary" -> e123_wolf_feeding.csv: the rows of the ten wolves that have hair samples.
    The sheet also lists 25 other animals of the facility; they are used only to count how many
    animals shared each carcass. Rows without a wolf or without an item are dropped.
Which hair samples to use, which deer stand for the diet and how to weigh a carcass are choices of
the notebook.

e123_wolf_hair.csv, one row per hair sample:
  wolf        the animal's number at the Wildlife Science Center
  pair        number shared by the shoulder and the rump sample taken from one wolf on one day
  sample_id   the laboratory's label
  site        shoulder or rump
  date        YYYY-MM-DD the hair was collected
  d13C, d15N  permil (VPDB and air)

e123_prey_tissue.csv, one row per tissue sample of a food item:
  animal      the food item's label (D deer, B beaver, G goose), as in the feeding record
  species     deer, beaver or goose
  tissue      muscle or hair (hair for deer only)
  date        YYYY-MM-DD in the workbook's sample_date column
  d13C, d15N  permil

e123_wolf_feeding.csv, one row per wolf and food item:
  wolf        as above
  date        YYYY-MM-DD in the sheet's "Tissue Sample" column (or "Hair Sample" where that is the
              only one), which we read as the day the item was fed: with this reading the items up to
              31 October 2011 are the 106 deer, 14 beaver and 121 geese of the article. Empty for 10 rows.
  item        the food item's label
  species     deer, beaver, goose, or sheep (one item, labelled "S1 (SHEEP?)" in the source)
  shared_by   number of animals of the facility recorded with this item on this date
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e123"
URL = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0119940.s001"
MD5 = "cee9aec8eb187b143a98451a27e9e3b8"   # of the workbook as downloaded 2026-11-14 (181,369 bytes)
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
SPECIES = {"D": "deer", "B": "beaver", "G": "goose", "S": "sheep"}
STUDY_END = "2011-10-31"                    # the article's end of the hair-growth period


def fetch(allow_new: bool) -> Path:
    xlsx = CACHE / "journal.pone.0119940.s001.xlsx"
    if not xlsx.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            xlsx.write_bytes(resp.read())
    md5 = hashlib.md5(xlsx.read_bytes()).hexdigest()
    if md5 != MD5 and not allow_new:
        raise SystemExit(f"workbook md5 {md5} differs from the recorded {MD5}; pass --allow-new-source")
    return xlsx


def read_sheets(xlsx: Path) -> dict[str, pd.DataFrame]:
    """The workbook's sheets as tables of strings (first row = header), without an Excel library."""
    with zipfile.ZipFile(xlsx) as z:
        strings = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t"))
                   for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        out = {}
        for sheet in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS):
            rows = []
            for row in ET.fromstring(z.read("xl/" + rels[sheet.get(f"{{{NS['r']}}}id")])).findall(".//m:row", NS):
                cells = {}
                for c in row.findall("m:c", NS):
                    v = c.find("m:v", NS)
                    if v is not None:
                        col = re.match(r"[A-Z]+", c.get("r")).group(0)
                        cells[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
                rows.append(cells)
            cols = sorted({k for r in rows for k in r}, key=lambda s: (len(s), s))
            table = pd.DataFrame([[r.get(c) for c in cols] for r in rows[1:]],
                                 columns=[rows[0][c].strip() for c in cols])
            out[sheet.get("name")] = table
    return out


def day(serial: pd.Series) -> pd.Series:
    """Excel day numbers (1900 date system) as ISO dates; missing stays missing."""
    origin = datetime.date(1899, 12, 30)
    return serial.map(lambda s: None if pd.isna(s) else
                      (origin + datetime.timedelta(days=int(float(s)))).isoformat())


def delta(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.where(col != "NA"), errors="raise").round(1)


def build(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    w = sheets["Wolf"]
    hair = pd.DataFrame({"wolf": w["wolf_ID"].astype(int), "pair": w["unique"].astype(int),
                         "sample_id": w["sample_ID"], "site": w["location"], "date": day(w["sample_date"]),
                         "d13C": delta(w["C"]), "d15N": delta(w["N"])})
    hair = hair.sort_values(["wolf", "date", "site"], kind="stable").reset_index(drop=True)

    parts = []
    for name, tissue, id_col in (("Muscle", "muscle", "animal_ID"), ("Hair", "hair", "deer_ ID")):
        s = sheets[name]
        parts.append(pd.DataFrame({"animal": s[id_col], "species": s["species"], "tissue": tissue,
                                   "date": day(s["sample_date"]), "d13C": delta(s["C"]), "d15N": delta(s["N"])}))
    prey = pd.concat(parts, ignore_index=True).dropna(subset=["d13C", "d15N"]).reset_index(drop=True)

    f = sheets["diet_summary"].rename(columns={"Wolf ID": "wolf", "Lookup to WolfDiet_Summary": "item"})
    f = f[f["wolf"].notna() & f["item"].notna()].copy()
    f["date"] = day(f["Tissue Sample"]).fillna(day(f["Hair Sample"]))
    f["item"] = f["item"].str.replace(" (SHEEP?)", "", regex=False)
    f["wolf"] = f["wolf"].astype(int)
    f["shared_by"] = f.groupby(["item", f["date"].fillna("")])["wolf"].transform("nunique")
    f = f[f["wolf"].isin(hair["wolf"].unique())].copy()
    f["species"] = f["item"].str[0].map(SPECIES)
    feeding = (f[["wolf", "date", "item", "species", "shared_by"]]
               .sort_values(["wolf", "date", "item"], kind="stable", na_position="last").reset_index(drop=True))
    return {"e123_wolf_hair": hair, "e123_prey_tissue": prey, "e123_wolf_feeding": feeding}


def check(t: dict[str, pd.DataFrame]) -> None:
    hair, prey, feeding = t["e123_wolf_hair"], t["e123_prey_tissue"], t["e123_wolf_feeding"]
    assert len(hair) == 48 and hair["wolf"].nunique() == 10 and hair.notna().all().all()
    assert hair["site"].value_counts().to_dict() == {"shoulder": 24, "rump": 24}
    assert (hair.groupby("pair")["site"].nunique() == 2).all() and (hair.groupby("pair")["wolf"].nunique() == 1).all()

    # Table 1 of the article: n, mean and SD of d13C and d15N by species and tissue
    table1 = {("deer", "hair"): (23, -21.92, 3.03, 4.69, 1.13), ("deer", "muscle"): (20, -22.12, 2.35, 4.24, 0.99),
              ("beaver", "muscle"): (13, -24.80, 0.34, 2.32, 1.60), ("goose", "muscle"): (15, -25.57, 1.46, 4.00, 1.18)}
    for (species, tissue), (n, mc, sc, mn, sn) in table1.items():
        g = prey[(prey["species"] == species) & (prey["tissue"] == tissue)]
        got = (len(g), g["d13C"].mean(), g["d13C"].std(), g["d15N"].mean(), g["d15N"].std())
        assert got[0] == n and all(abs(a - b) < 0.006 for a, b in zip(got[1:], (mc, sc, mn, sn))), (species, tissue, got)

    assert feeding["species"].notna().all() and set(feeding["species"]) == {"deer", "beaver", "goose", "sheep"}
    assert feeding["wolf"].nunique() == 10 and feeding["date"].isna().sum() == 10
    window = feeding[feeding["date"] <= STUDY_END]             # the article: 106 deer, 14 beaver, 121 geese
    assert window.groupby("species")["item"].nunique().to_dict() == {"beaver": 14, "deer": 106, "goose": 121}
    assert set(prey["animal"]) <= set(feeding["item"])         # every sampled animal is in the feeding record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-new-source", action="store_true")
    args = ap.parse_args()
    tables = build(read_sheets(fetch(args.allow_new_source)))
    check(tables)
    for name, table in tables.items():
        out = DATA / f"{name}.csv"
        payload = table.to_csv(index=False, float_format="%.1f", lineterminator="\n").encode()
        out.write_bytes(payload)
        print(f"wrote {out.relative_to(ROOT)}: {len(table):,} rows, {len(payload) / 1e3:.1f} kB, "
              f"md5 {hashlib.md5(payload).hexdigest()}")
    f = tables["e123_wolf_feeding"]
    print("items fed to the ten wolves up to", STUDY_END, ":",
          f[f["date"] <= STUDY_END].groupby("species")["item"].nunique().to_dict())


if __name__ == "__main__":
    main()
