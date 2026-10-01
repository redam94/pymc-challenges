"""Build the E84 dataset: the margarine household purchase panel shipped with the R package bayesm.

    uv run --no-project --with rdata --with pandas python tools/build_e84_margarine.py

(`rdata` reads R's .rda format; it is not a project dependency, hence `--no-project --with`.)

Source: Allenby & Rossi (1991), "Quality perceptions and asymmetric switching between brands",
Marketing Science 10:185-205; distributed as `margarine` in the R package bayesm (Rossi, GPL >= 2),
whose source tarball on CRAN is the URL below. A.C. Nielsen scanner panel: 4,470 margarine
purchases by 516 households, each purchase with the shelf price of all 10 products that day.

bayesm keeps the purchases of a household in time order (rows of a household are contiguous), but
has no dates. We add `trip` (1, 2, ... within household, in the bayesm row order), write the chosen
product as its code instead of 1-10, and leave prices (US dollars per pound) untouched.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import rdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://cran.r-project.org/src/contrib/Archive/bayesm/bayesm_3.1-6.tar.gz"
PRODUCTS = ["Pk_Stk", "BB_Stk", "Fl_Stk", "Hse_Stk", "Gen_Stk", "Imp_Stk", "SS_Tub", "Pk_Tub",
            "Fl_Tub", "Hse_Tub"]  # bayesm's choice codes 1..10, in the order of its price columns


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        blob = tar.extractfile("bayesm/data/margarine.rda").read()
    tmp = ROOT / ".scratch" / "e84" / "margarine.rda"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(blob)
    m = rdata.read_rda(tmp)["margarine"]
    cp, demos = m["choicePrice"].reset_index(drop=True), m["demos"].reset_index(drop=True)
    assert list(cp.columns[2:]) == ["P" + p for p in PRODUCTS]
    assert (cp["hhid"].diff().fillna(0) != 0).sum() + 1 == cp["hhid"].nunique()  # contiguous
    cp["hhid"] = cp["hhid"].astype(int)
    cp.insert(1, "trip", cp.groupby("hhid").cumcount() + 1)
    cp["choice"] = [PRODUCTS[int(c) - 1] for c in cp["choice"]]
    cp = cp.rename(columns={"P" + p: "price_" + p for p in PRODUCTS})
    cp.to_csv(DATA / "margarine_purchases.csv", index=False)
    demos["hhid"] = demos["hhid"].astype(int)
    demos = demos.rename(columns={"Fs5.": "Fs5"})
    demos.to_csv(DATA / "margarine_demos.csv", index=False)
    print(cp.shape, demos.shape)
    print(cp["choice"].value_counts().to_string())


if __name__ == "__main__":
    main()
