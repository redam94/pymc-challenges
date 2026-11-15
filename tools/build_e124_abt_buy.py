"""Build the E124 dataset: the same products in two shops' catalogues, with the true links.

    python tools/build_e124_abt_buy.py   # download (cached), derive, check, write data/e124_*.csv
    python tools/build_e124_abt_buy.py --allow-new-source   # accept an archive whose md5 has changed

Source: the Abt-Buy benchmark of the Database Group of the University of Leipzig (Prof. Erhard Rahm),
used in Köpcke, Thor and Rahm (2010), "Evaluation of entity resolution approaches on real-world match
problems", Proceedings of the VLDB Endowment 3(1). Product listings of two online shops, abt.com and
buy.com, and the "perfect mapping" between them.
https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip

Licence, as read on that page on 2026-11-14: "January 2019: These datasets are made available by the
database group of Prof. Erhard Rahm under the Creative Commons license. To give proper credit please
refer to this website and the VLDB2010 paper [1]". The words "Creative Commons license" link to
https://creativecommons.org/licenses/by/4.0/ (CC BY 4.0).

The archive holds three CSV files (Abt.csv, Buy.csv, abt_buy_perfectMapping.csv). It is cached in
~/.cache/pymc-challenges/e124 (not committed). The records are products, not people.

What the builder does, and nothing else:
  - reads Abt.csv as Latin-1 (it is not valid UTF-8) and writes UTF-8;
  - turns the price strings ("$1,299.00") into numbers; a missing price stays missing;
  - drops the long ``description`` columns, which the notebook does not use;
  - keeps the rows in the order of the source files.
Building the comparison of two records (what counts as agreement) is part of the notebook.

Derived tables:
  data/e124_abt.csv     id, name, price                 1,081 products of abt.com
  data/e124_buy.csv     id, name, manufacturer, price   1,092 products of buy.com
  data/e124_links.csv   id_abt, id_buy                  1,097 true links (the "perfect mapping")
Prices are in US dollars. 418 Abt and 590 Buy products have a price; 226 true links have both.
The mapping is almost one-to-one: 16 Abt ids and 5 Buy ids appear in two links.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e124"
URL = "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip"
MD5 = "99212bd8ce1f0f3c87deec9da41dff70"   # of the archive as downloaded 2026-11-14 (134,745 bytes)
N_ABT, N_BUY, N_LINKS = 1081, 1092, 1097


def fetch(allow_new: bool) -> zipfile.ZipFile:
    archive = CACHE / "zip" / "Abt-Buy.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            archive.write_bytes(resp.read())
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if md5 != MD5 and not allow_new:
        raise SystemExit(f"archive md5 {md5} differs from the recorded {MD5}; pass --allow-new-source")
    return zipfile.ZipFile(archive)


def price(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.str.replace(r"[$,]", "", regex=True), errors="raise")


def build(z: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    def read(member: str, encoding: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(z.read(member)), encoding=encoding, dtype=str)

    abt, buy, links = read("Abt.csv", "latin-1"), read("Buy.csv", "utf-8"), read("abt_buy_perfectMapping.csv", "utf-8")
    abt_out = pd.DataFrame({"id": abt["id"].astype(int), "name": abt["name"].str.strip(), "price": price(abt["price"])})
    buy_out = pd.DataFrame({"id": buy["id"].astype(int), "name": buy["name"].str.strip(),
                            "manufacturer": buy["manufacturer"].str.strip(), "price": price(buy["price"])})
    links_out = pd.DataFrame({"id_abt": links["idAbt"].astype(int), "id_buy": links["idBuy"].astype(int)})
    return {"e124_abt.csv": abt_out, "e124_buy.csv": buy_out, "e124_links.csv": links_out}


def check(tables: dict[str, pd.DataFrame]) -> None:
    abt, buy, links = tables["e124_abt.csv"], tables["e124_buy.csv"], tables["e124_links.csv"]
    assert (len(abt), len(buy), len(links)) == (N_ABT, N_BUY, N_LINKS)
    assert abt["id"].is_unique and buy["id"].is_unique and not links.duplicated().any()
    assert links["id_abt"].isin(abt["id"]).all() and links["id_buy"].isin(buy["id"]).all()
    assert abt["name"].notna().all() and buy["name"].notna().all()
    assert abt["price"].notna().sum() == 418 and buy["price"].notna().sum() == 590
    assert abt["price"].dropna().between(1, 5000).all() and buy["price"].dropna().between(1, 5000).all()
    assert links["id_abt"].duplicated().sum() == 16 and links["id_buy"].duplicated().sum() == 5
    both = links.merge(abt, left_on="id_abt", right_on="id").merge(buy, left_on="id_buy", right_on="id", suffixes=("_a", "_b"))
    assert both[["price_a", "price_b"]].notna().all(axis=1).sum() == 226
    assert "description" not in abt.columns and "description" not in buy.columns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-new-source", action="store_true")
    args = ap.parse_args()
    tables = build(fetch(args.allow_new_source))
    check(tables)
    for name, table in tables.items():
        payload = table.to_csv(index=False, lineterminator="\n").encode("utf-8")   # same bytes on every run
        (DATA / name).write_bytes(payload)
        print(f"wrote data/{name}: {len(table):,} rows, {len(payload) / 1e3:.0f} kB, md5 {hashlib.md5(payload).hexdigest()}")


if __name__ == "__main__":
    main()
