"""Build the E85 datasets: 6-month purchase incidence of cold cereals, shopping activity and
demographics, from 84.51's "The Complete Journey" grocery panel.

    uv run --no-project --with rdata --with pandas python tools/build_e85_cereal.py

(`rdata` reads R's .rda/.rds formats; it is not a project dependency, hence `--no-project --with`.)

Source: 84.51 (Kroger's data-science company), "The Complete Journey" study: one year (2017) of
grocery transactions of 2,469 households who are frequent shoppers at one retailer, with product
attributes and, for 801 households, demographics. Distributed as the R package completejourney
(Boehmke, Davis & Delaney; CC0 1.0), whose GitHub data directory holds the files read here.
Manufacturers are anonymised (numeric ids); `brand` is National or Private (the store brand).

What we write (all derived, nothing else changed):

* `cj_cereal_incidence.csv` - the format a panel provider typically delivers: one row per household
  and half-year (half 1 = weeks 1-26, half 2 = weeks 27-53) for every household with at least one
  shopping trip in that half, and a 0/1 column per cereal "product" = 1 if the household bought it
  at least once in that half. Products are manufacturer x segment cells of the COLD CEREAL category
  (segment = product_type KIDS / ALL FAMILY / ADULT CEREAL) bought by at least 150 households over
  the year: 14 cells, 99% of cereal purchases. Store brand = manufacturer 69 ("Private").
* `cj_activity.csv` - household, half, trips (distinct baskets, all departments), spend (US$).
* `cj_demographics.csv` - the package's demographics table as is (801 households).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TMP = ROOT / ".scratch" / "e85"
BASE = "https://github.com/bradleyboehmke/completejourney/raw/master/data"
SEG = {"KIDS CEREAL": "kids", "ALL FAMILY CEREAL": "family", "ADULT CEREAL": "adult"}


def fetch(name):
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / name
    if not path.exists():
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    obj = rdata.read_rds(path) if name.endswith(".rds") else rdata.read_rda(path)
    return obj if isinstance(obj, pd.DataFrame) else next(iter(obj.values()))


def main() -> None:
    tx, products, demos = fetch("transactions.rds"), fetch("products.rda"), fetch("demographics.rda")
    for df in (tx, products, demos):
        for c in df.columns:
            if c.endswith("_id"):
                df[c] = df[c].astype(str)
    tx["half"] = np.where(tx["week"] <= 26, 1, 2)
    activity = (tx.groupby(["household_id", "half"])
                .agg(trips=("basket_id", "nunique"), spend=("sales_value", "sum")).reset_index())
    activity["spend"] = activity["spend"].round(2)

    cer = tx.merge(products, on="product_id")
    cer = cer[(cer["product_category"] == "COLD CEREAL") & cer["product_type"].isin(SEG)].copy()
    maker = np.where(cer["brand"] == "Private", "Store", "M" + cer["manufacturer_id"])
    cer["product"] = maker + "_" + cer["product_type"].map(SEG)
    n_hh = cer.groupby("product")["household_id"].nunique()
    keep = n_hh[n_hh >= 150].sort_values(ascending=False).index
    print(f"kept {len(keep)} products, {cer['product'].isin(keep).mean():.1%} of cereal purchase lines")
    cer = cer[cer["product"].isin(keep)]

    inc = (cer.groupby(["household_id", "half", "product"]).size().unstack(fill_value=0) > 0).astype(int)
    inc = activity[["household_id", "half"]].merge(inc.reset_index(), how="left", on=["household_id", "half"])
    inc[list(keep)] = inc[list(keep)].fillna(0).astype(int)
    inc = inc[["household_id", "half", *keep]].sort_values(["half", "household_id"])
    inc.to_csv(DATA / "cj_cereal_incidence.csv", index=False)
    activity.sort_values(["half", "household_id"]).to_csv(DATA / "cj_activity.csv", index=False)
    demos.to_csv(DATA / "cj_demographics.csv", index=False)
    print(inc.groupby("half")[list(keep)].mean().T.round(3).to_string())
    print(inc.groupby("half").size(), len(demos))


if __name__ == "__main__":
    main()
