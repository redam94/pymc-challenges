"""Build the two extra E72 datasets (web sessions and retail baskets) from their UCI archives.

    uv run --no-project --with openpyxl --with pandas --with numpy python tools/build_e72_extras.py

openpyxl is only needed here, to read the retail spreadsheet once; the notebook reads the CSVs.

* data/msnbc_sessions_sample.csv - a seeded random sample of 60,000 of the 989,818 user sessions in
  the msnbc.com anonymous web data (UCI 133): session (line number in the original file, from 0),
  n_views, pages (space-separated category codes 1-17, in the order requested).
* data/online_retail_customer_products.csv - the UCI Online Retail transactions (UCI 352)
  aggregated to one row per (customer, product): customer_id, country, stock_code, n_invoices
  (number of orders containing the product), quantity (units); and data/online_retail_products.csv:
  stock_code, description (the product's most common description), n_customers. Cancellations,
  non-positive quantities, rows without a customer and non-product codes (postage, fees, manual
  adjustments, samples) are dropped.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
MSNBC = "https://archive.ics.uci.edu/static/public/133/msnbc+com+anonymous+web+data.zip"
RETAIL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
NON_PRODUCTS = {"POST", "D", "M", "DOT", "BANK CHARGES", "AMAZONFEE", "CRUK", "C2", "PADS", "S", "B", "m"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pymc-challenges"})) as r:
        return r.read()


def msnbc() -> None:
    with zipfile.ZipFile(io.BytesIO(fetch(MSNBC))) as zf:
        lines = gzip.decompress(zf.read("msnbc990928.seq.gz")).decode().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("% Sequences")) + 2
    seqs = [line.strip() for line in lines[start:]]
    seqs = [s for s in seqs if s]
    print(f"msnbc: {len(seqs):,} sessions")
    pick = np.sort(np.random.default_rng(72).choice(len(seqs), 60_000, replace=False))
    out = pd.DataFrame({"session": pick, "n_views": [len(seqs[i].split()) for i in pick],
                        "pages": [seqs[i] for i in pick]})
    out.to_csv(DATA / "msnbc_sessions_sample.csv", index=False)
    print(out.head(), out.n_views.describe(), sep="\n")


def retail() -> None:
    with zipfile.ZipFile(io.BytesIO(fetch(RETAIL))) as zf:
        df = pd.read_excel(io.BytesIO(zf.read("Online Retail.xlsx")), dtype={"StockCode": str, "InvoiceNo": str})
    print(f"retail: {len(df):,} rows")
    df = df[df["CustomerID"].notna() & (df["Quantity"] > 0) & ~df["InvoiceNo"].str.startswith("C")
            & ~df["StockCode"].isin(NON_PRODUCTS)]
    df = df.assign(customer_id=df["CustomerID"].astype(int), stock_code=df["StockCode"].str.upper(),
                   description=df["Description"].astype(str).str.strip())
    desc = df.groupby("stock_code")["description"].agg(lambda s: s.value_counts().index[0])
    country = df.groupby("customer_id")["Country"].agg(lambda s: s.value_counts().index[0])
    agg = df.groupby(["customer_id", "stock_code"]).agg(n_invoices=("InvoiceNo", "nunique"),
                                                         quantity=("Quantity", "sum")).reset_index()
    agg["country"] = agg["customer_id"].map(country)
    agg = agg[["customer_id", "country", "stock_code", "n_invoices", "quantity"]]
    agg.to_csv(DATA / "online_retail_customer_products.csv", index=False)
    products = desc.rename("description").reset_index()
    products["n_customers"] = products["stock_code"].map(agg.groupby("stock_code")["customer_id"].nunique())
    products.to_csv(DATA / "online_retail_products.csv", index=False)
    print(agg.head(), f"{agg.customer_id.nunique():,} customers, {agg.stock_code.nunique():,} products,"
          f" {len(agg):,} rows", sep="\n")


if __name__ == "__main__":
    import sys
    if "--retail-only" not in sys.argv:
        msnbc()
    retail()
