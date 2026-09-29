"""Build the E74 datasets: light-fading of Japanese woodblock-print colourants, CIE tables, and
four Harunobu prints.

    uv run --no-project --with openpyxl --with pandas --with numpy --with pillow \
        python tools/build_e74_fading.py

openpyxl is only needed here, to read the spreadsheets once; the notebook reads the CSVs.

* data/woodblock_fading_spectra.csv - reflectance spectra of 19 colourants (plus three mixtures
  and the bare paper) printed in 1999 by the Museum of Fine Arts, Boston, faded by a microfading
  tester (MFT) and in a xenon-arc chamber (XT) at the Rijksmuseum / RCE in 2024 (Baines et al.
  2026, Zenodo 17014267, CC BY 4.0). Columns: instrument (MFT, XT), colourant, dose_Mlxh
  (visible exposure, million lux hours), wavelength (380-730 nm, 10 nm), reflectance, refl_sd
  (MFT: mean sd over the three spots; XT: blank). MFT spectra are averaged over +-4 nm around each
  10 nm point and thinned to every 4th of the 10-second readings. MFT also has the four Blue
  Wool references (BW1-BW4).
* data/woodblock_fading_lab.csv - the CIELAB values (D65, 10 degree observer) the authors
  computed: instrument, colourant, dose_Mlxh, L, a, b (MFT: every reading, 0.1 MJ/m2 apart).
* data/cie1964_d65.csv - wavelength (360-830 nm, 1 nm), xbar, ybar, zbar (CIE 1964 10 degree
  colour-matching functions, CVRL) and d65 (CIE standard illuminant D65 relative power,
  5 nm table linearly interpolated to 1 nm).
* data/harunobu_impressions.npz - four colour woodblock prints by Suzuki Harunobu (c. 1766),
  two impressions each of two designs, Art Institute of Chicago (public domain, CC0 images),
  resized to 400 px wide.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
ZENODO = "https://zenodo.org/records/17014267/files"
MFT_ZIP = f"{ZENODO}/MFT-data_BMFA-samples.zip?download=1"
XT_ZIP = f"{ZENODO}/RS-data_xenotest_BMFA-samples.zip?download=1"
CMF = "http://www.cvrl.org/database/data/cmfs/ciexyz64_1.csv"
D65 = ("https://raw.githubusercontent.com/colour-science/colour/develop/colour/colorimetry/"
       "datasets/illuminants/sds.py")
AIC = "https://www.artic.edu/iiif/2/{}/full/400,/0/default.jpg"
PRINTS = [  # (AIC id, image id, design, title)
    (20814, "0bb7c6f4-53d1-def8-28f2-c044348954ee", "geese",
     "Descending Geese of the Koto Bridges (Kotoji no rakugan)"),
    (88966, "35e7d372-8eb6-adc8-9ba8-9810ff64abae", "geese",
     "Descending Geese of the Koto Bridges (Kotoji no rakugan)"),
    (88968, "8af85036-2cf5-06f1-262b-2b7ed34b8856", "lamp",
     "The Evening Glow of a Lamp (Andon no sekisho)"),
    (20817, "fee65005-8776-5390-1f9e-ecb2c929fd1a", "lamp",
     "Evening Glow of a Lamp (Andon no sekisho)"),
]
NAMES = {"sappenwood": "sappanwood", "vermillon": "vermilion", "tumeric": "turmeric",
         "sfl-dfl": "safflower+dayflower", "ind-orp": "indigo+orpiment",
         "tum-orp": "turmeric+orpiment", "japanesePaper": "paper", "ironoxide": "iron oxide",
         "redlead": "red lead", "leadwhite": "lead white", "prussianblue": "Prussian blue",
         "brasspowder": "brass powder",
         "BWS0024": "BW1", "BWS0025": "BW2", "BWS0026": "BW3", "BWS0027": "BW4"}
WL = np.arange(380, 731, 10)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges",
                                               "AIC-User-Agent": "pymc-challenges"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def mft(zf: zipfile.ZipFile) -> tuple[list, list]:
    spec, lab = [], []
    for name in sorted(zf.namelist()):
        if not name.endswith(".xlsx"):
            continue
        code = re.search(r"MF\.(BWS\d+|[A-Za-z-]+)", name).group(1)
        col = NAMES.get(code, code)
        book = pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None, header=None)
        c = book["CIELAB"].iloc[2:, [0, 2, 3, 5, 7]].astype(float).to_numpy()
        he, hv = c[:, 0], c[:, 1]
        lab += [("MFT", col, h, *row[2:]) for h, row in zip(hv, c)]
        s = book["spectra"]
        wl = s.iloc[3:, 0].astype(float).to_numpy()
        means = s.iloc[3:, 1::2].astype(float).to_numpy()   # wavelength x dose
        sds = s.iloc[3:, 2::2].astype(float).to_numpy()
        he_s = s.iloc[0, 1::2].astype(float).to_numpy()
        hv_s = np.interp(he_s, he, hv)
        for j in range(0, len(he_s), 4):
            for w in WL:
                m = np.abs(wl - w) <= 4
                spec.append(("MFT", col, hv_s[j], w, means[m, j].mean(), sds[m, j].mean()))
        print(f"MFT {col:22s} {len(he_s)} readings to {hv_s[-1]:.2f} Mlxh")
    return spec, lab


def xt(zf: zipfile.ZipFile) -> tuple[list, list]:
    spec, lab = [], []
    for name in sorted(zf.namelist()):
        if not name.endswith(".xlsx"):
            continue
        code = re.search(r"2024-144_([A-Za-z-]+)_nb", name).group(1)
        col = NAMES.get(code, code)
        book = pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None, header=None)
        c = book["CIELAB"].iloc[1:, [0, 2, 3, 4, 5]].astype(float).to_numpy()
        t_hr, hv = c[:, 0], c[:, 1]
        lab += [("XT", col, row[1], *row[2:]) for row in c]
        s = book["spectra"]
        wl = s.iloc[1:, 0].astype(float).to_numpy()
        t_s = s.iloc[0, 1:].astype(float).to_numpy()
        hv_s = np.interp(t_s, t_hr, hv)
        vals = s.iloc[1:, 1:].astype(float).to_numpy()
        for j, h in enumerate(hv_s):
            for i, w in enumerate(wl):
                if int(w) in WL:  # cochineal was exported at 1 nm; the rest at 10 nm
                    spec.append(("XT", col, h, int(w), vals[i, j], np.nan))
        print(f"XT  {col:22s} {len(t_s)} readings to {hv_s[-1]:.2f} Mlxh")
    return spec, lab


def cie() -> None:
    cmf = pd.read_csv(io.BytesIO(fetch(CMF)), header=None, names=["wavelength", "xbar", "ybar", "zbar"])
    src = fetch(D65).decode()
    block = src[src.index('"D65": {'):]
    block = block[:block.index("}")]
    d65 = {int(k): float(v) for k, v in re.findall(r"(\d+):\s*([\d.]+)", block)}
    wl = np.array(sorted(d65))
    cmf = cmf[(cmf.wavelength >= 360) & (cmf.wavelength <= 830)].copy()
    cmf["d65"] = np.interp(cmf.wavelength, wl, [d65[w] for w in wl])
    cmf.to_csv(DATA / "cie1964_d65.csv", index=False, float_format="%.9g")
    print(f"CIE: {len(cmf)} rows, D65(560) = {d65[560]}")


def prints() -> None:
    from PIL import Image
    imgs = [np.asarray(Image.open(io.BytesIO(fetch(AIC.format(img)))).convert("RGB"))
            for _, img, _, _ in PRINTS]
    print("prints as downloaded:", [i.shape for i in imgs])
    h = min(i.shape[0] for i in imgs)
    imgs = [i[(i.shape[0] - h) // 2:(i.shape[0] - h) // 2 + h] for i in imgs]  # trim to a common height
    np.savez_compressed(DATA / "harunobu_impressions.npz", images=np.stack(imgs),
                        aic_id=np.array([p[0] for p in PRINTS]), design=np.array([p[2] for p in PRINTS]),
                        title=np.array([p[3] for p in PRINTS]))
    print("prints:", [i.shape for i in imgs])


def main() -> None:
    with zipfile.ZipFile(io.BytesIO(fetch(MFT_ZIP))) as zf:
        s1, l1 = mft(zf)
    with zipfile.ZipFile(io.BytesIO(fetch(XT_ZIP))) as zf:
        s2, l2 = xt(zf)
    spec = pd.DataFrame(s1 + s2, columns=["instrument", "colourant", "dose_Mlxh", "wavelength",
                                          "reflectance", "refl_sd"])
    lab = pd.DataFrame(l1 + l2, columns=["instrument", "colourant", "dose_Mlxh", "L", "a", "b"])
    spec.to_csv(DATA / "woodblock_fading_spectra.csv", index=False, float_format="%.5g")
    lab.to_csv(DATA / "woodblock_fading_lab.csv", index=False, float_format="%.5g")
    print(f"spectra {spec.shape}, lab {lab.shape}")
    cie()
    prints()


if __name__ == "__main__":
    main()
