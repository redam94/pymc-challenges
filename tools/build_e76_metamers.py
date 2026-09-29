"""Build the E76 datasets: reflectance spectra of 1,269 matt Munsell chips, and CIE illuminants.

    .venv/bin/python tools/memguard.py --max-gb 3 -- .venv/bin/python tools/build_e76_metamers.py

Only numpy and pandas are needed.

* data/munsell_matt_spectra.csv - reflectance spectra of the 1,269 chips of the Munsell Book of
  Color, Matte Finish Collection (Munsell Color, Baltimore, 1976), measured with a Perkin-Elmer
  Lambda 9 UV/VIS/NIR spectrophotometer at 1 nm from 380 to 800 nm (University of Kuopio, now
  University of Eastern Finland, spectral database; measurer Jouni Hiltunen; mirrored on Zenodo,
  record 3269912, by the colour-science project, licence "not specified"). One row per chip:
  chip (Munsell notation, e.g. "2.5R 9/2"), hue (e.g. "2.5R"), hue_family (R, YR, ..., RP),
  value, chroma, then r380 ... r780: reflectance (0-1) at 10 nm, each the mean of the 1 nm readings
  within +-4 nm.
* data/cie_illuminants_e76.csv - relative spectral power (5 nm, 380-780 nm) of CIE illuminants:
  A (incandescent, 2856 K; computed here from Planck's law with the CIE formula and checked
  against the CIE table), FL2 (cool-white halophosphate fluorescent), FL11 (narrow-band triphosphor
  fluorescent, 4000 K), LED-B2 and LED-B4 (phosphor-converted white LEDs, CIE 015:2018), from the
  tables distributed with colour-science (colour/colorimetry/datasets/illuminants/sds.py, CIE 15:2004
  tables and CIE 015:2018).
"""

from __future__ import annotations

import gzip
import io
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
ZENODO = "https://zenodo.org/api/records/3269912/files/{}/content"
SDS = ("https://raw.githubusercontent.com/colour-science/colour/develop/colour/colorimetry/"
       "datasets/illuminants/sds.py")
WL10 = np.arange(380, 781, 10)
WL5 = np.arange(380, 781, 5)
HUE_FAMILIES = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def munsell() -> None:
    raw = gzip.decompress(fetch(ZENODO.format("munsell380_800_1.asc.gz"))).decode()
    vals = np.array([float(v) for v in raw.split()])
    assert vals.size == 421 * 1269, vals.size
    spectra = vals.reshape(1269, 421)                     # chip x wavelength (380..800, 1 nm)
    readme = fetch(ZENODO.format("README.txt")).decode(errors="replace")
    index = readme[readme.index("The index of the ASCII spectra"):]
    pat = re.compile(r"^\s*(?:\d+\s+)?([\d.]+)\s+(RP|YR|GY|BG|PB|R|Y|G|B|P)\s+([\d.]+)\s*/\s*(\d+)\s*$")
    labels = [m.groups() for line in index.splitlines() if (m := pat.match(line))]
    assert len(labels) == 1269, len(labels)
    wl1 = np.arange(380, 801)
    cols = {}
    for w in WL10:
        m = np.abs(wl1 - w) <= 4
        cols[f"r{w}"] = spectra[:, m].mean(1)
    df = pd.DataFrame({
        "chip": [f"{h}{f} {v}/{c}" for h, f, v, c in labels],
        "hue": [f"{h}{f}" for h, f, _, _ in labels],
        "hue_family": [f for _, f, _, _ in labels],
        "value": [float(v) for _, _, v, _ in labels],
        "chroma": [int(c) for _, _, _, c in labels],
        **cols,
    })
    df.to_csv(DATA / "munsell_matt_spectra.csv", index=False, float_format="%.5g")
    print(f"Munsell: {len(df)} chips, reflectance {df[[f'r{w}' for w in WL10]].to_numpy().min():.4f}"
          f" - {df[[f'r{w}' for w in WL10]].to_numpy().max():.4f}; first {df.chip[0]}, last {df.chip.iloc[-1]}")
    print(df.hue_family.value_counts().reindex(HUE_FAMILIES).to_dict())


def table(src: str, name: str) -> dict[int, float]:
    block = src[src.index(f'"{name}": {{'):]
    block = block[len(name) + 5:block.index("}")]
    return {int(k): float(v) for k, v in re.findall(r"(\d+):\s*([-\d.]+)", block)}


def illuminants() -> None:
    src = fetch(SDS).decode()
    out = {"wavelength": WL5}
    # CIE standard illuminant A: Planck's law at 2848 K with c2 = 1.435e-2 m K, i.e. 2856 K on the
    # modern c2 = 1.4388e-2 m K scale, normalised to 100 at 560 nm (CIE 015).
    c2, T = 1.435e7, 2848.0                                   # nm K
    a = 100 * (560 / WL5) ** 5 * np.expm1(c2 / (T * 560)) / np.expm1(c2 / (T * WL5))
    a_tab = table(src, "A")
    err = np.max(np.abs(a / np.array([a_tab[w] for w in WL5]) - 1))
    print(f"A from Planck vs CIE table: max relative difference {err:.1e}")
    assert err < 1e-5
    out["A"] = a
    for name in ["FL2", "FL11", "LED-B2", "LED-B4"]:
        t = table(src, name)
        out[name.replace("-", "_")] = [t[w] for w in WL5]
    df = pd.DataFrame(out)
    df.to_csv(DATA / "cie_illuminants_e76.csv", index=False, float_format="%.6g")
    print(f"illuminants: {df.shape}")


if __name__ == "__main__":
    munsell()
    illuminants()
