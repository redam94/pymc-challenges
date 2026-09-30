"""Build the E82 datasets: the fully enumerated direct-arylation reaction screen of Shields et al.
(2021), the human "optimisation game" played on it, EDBO's own simulated campaigns, and DFT
descriptors of the 12 ligands.

    .venv/bin/python tools/build_e82_arylation.py

Source: github.com/b-shields/edbo (MIT licence, (c) 2020 Benjamin J. Shields), the code and data of
Shields, Stevens, Li, Parasram, Damani, Martinez Alvarado, Janey, Adams & Doyle (2021), "Bayesian
reaction optimization as a tool for chemical synthesis", Nature 590, 89-96.

* data/edbo_arylation.csv - one row per reaction of the full 12 x 4 x 4 x 3 x 3 grid (1,728):
  ligand, base, solvent (names from the repository's *-list.csv files), concentration (M),
  temperature (C), yield (%), from experiments/data/direct_arylation/experiment_index.csv.
* data/edbo_arylation_game.csv - experiments/arylation_game_summary.csv in long format: one row per
  experiment chosen by one of 50 chemists (participant 0-49, with area, expertise, experience as
  given), step (1, 2, ... in the order played), the conditions and the yield they were shown.
* data/edbo_arylation_edbo_runs.csv - experiments/arylation_bo_results_GP-EI_bs=5.csv in long
  format: run (0-49), step (1-100), yield - the authors' simulated EDBO campaigns (GP, expected
  improvement, batches of 5) on the same grid.
* data/edbo_arylation_ligand_dft.csv - the Boltzmann-weighted ('_Boltz') DFT descriptors of the 12
  ligands (experiments/data/direct_arylation/ligand-boltzmann_dft.csv), numeric columns only.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/"
DA = RAW + "data/direct_arylation/"


def get(url: str, tries: int = 4) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges"})
    for k in range(tries):  # raw.githubusercontent.com occasionally drops a TLS handshake
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return pd.read_csv(io.BytesIO(r.read()), encoding="utf-8-sig")
        except urllib.error.URLError:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)


def main() -> None:
    idx = get(DA + "experiment_index.csv")
    lig = get(DA + "ligand-list.csv")
    base = get(DA + "base-list.csv")
    sol = get(DA + "solvent-list.csv")
    out = pd.DataFrame({
        "ligand": idx.Ligand_SMILES.map(dict(zip(lig.Ligand_SMILES, lig.Ligand))),
        "base": idx.Base_SMILES.map(dict(zip(base.Base_SMILES, base.Base))),
        "solvent": idx.Solvent_SMILES.map(dict(zip(sol.Solvent_SMILES, sol.Solvent))),
        "concentration": idx.Concentration,
        "temperature": idx.Temp_C,
        "yield": idx["yield"],
    })
    assert out.notna().all().all() and len(out) == 1728
    assert out.groupby(["ligand", "base", "solvent", "concentration", "temperature"]).size().max() == 1
    out.to_csv(DATA / "edbo_arylation.csv", index=False)

    g = get(RAW + "arylation_game_summary.csv")
    rows = []
    for p, r in g.iterrows():
        k = 1
        while f"yield_{k}" in g.columns and pd.notna(r[f"yield_{k}"]):
            rows.append({"participant": p, "area": r.area, "expertise": r.expertise,
                         "experience": r.experience, "step": k, "ligand": r[f"ligand_{k}"],
                         "base": r[f"base_{k}"], "solvent": r[f"solvent_{k}"],
                         "concentration": r[f"conc_{k}"], "temperature": int(r[f"temp_{k}"]),
                         "yield": r[f"yield_{k}"]})
            k += 1
    game = pd.DataFrame(rows)
    # every human experiment must be a row of the grid, with the same yield
    chk = game.merge(out, on=["ligand", "base", "solvent", "concentration", "temperature"],
                     suffixes=("", "_grid"), how="left")
    assert chk.yield_grid.notna().all() and np.allclose(chk["yield"], chk.yield_grid)
    game.to_csv(DATA / "edbo_arylation_game.csv", index=False)

    b = get(RAW + "arylation_bo_results_GP-EI_bs=5.csv").drop(columns="Unnamed: 0")
    runs = b.stack().reset_index()
    runs.columns = ["run", "step", "yield"]
    runs["step"] = runs.step.astype(int) + 1
    runs.to_csv(DATA / "edbo_arylation_edbo_runs.csv", index=False)

    dft = get(DA + "ligand-boltzmann_dft.csv")
    dft = dft.assign(ligand=dft.ligand_SMILES.map(dict(zip(lig.Ligand_SMILES, lig.Ligand))))
    assert dft.ligand.notna().all()
    cols = [c for c in dft.columns if c.endswith("_Boltz") and pd.api.types.is_numeric_dtype(dft[c])]
    dft[["ligand"] + cols].to_csv(DATA / "edbo_arylation_ligand_dft.csv", index=False, float_format="%.6g")
    for f in ["edbo_arylation.csv", "edbo_arylation_game.csv", "edbo_arylation_edbo_runs.csv",
              "edbo_arylation_ligand_dft.csv"]:
        print(f, (DATA / f).stat().st_size // 1024, "KB")
    print(len(game), "human experiments by", game.participant.nunique(), "chemists;",
          len(cols), "ligand descriptors")


if __name__ == "__main__":
    main()
