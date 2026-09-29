"""Build the E77 datasets: the World Color Survey (WCS) naming and focus data in compact form.

    .venv/bin/python tools/build_e77_wcs.py

The WCS (Kay, Berlin, Maffi, Merrifield & Cook 2009, *The World Color Survey*, CSLI) asked on
average 24 speakers of each of 110 unwritten languages to name each of 330 Munsell chips and to
pick the best example(s) ("foci") of each of their colour terms. The archive
(https://linguistics.berkeley.edu/wcs/data.html, formerly www1.icsi.berkeley.edu/wcs) asks: "In
any published work based on these data, please cite these archives." It states no other licence.
The original text files are mirrored unchanged in the GitHub repository jvosten/wcs (folder
data-raw/, an R package whose own code is CC BY 4.0), which is where this script reads them.

Outputs (all small):

* data/wcs_chips.csv - the 330 chips: chip (WCS chip number 1-330), row (A-J, light to dark),
  col (0 = achromatic, 1-40 = hue columns), grid (e.g. 'E29'), munsell_hue, munsell_value,
  munsell_chroma, L, a, b (CIELAB given by the archive).
* data/wcs_languages.csv - lang (1-110), name, iso639_3, family, country, n_speakers.
* data/wcs_terms.csv - one row per (language, term abbreviation) seen in the naming data:
  lang, code (0, 1, ... in order of decreasing use within the language), abbrev (the WCS
  abbreviation), term (transcription(s) from dict.txt, ' | ' when the dictionary lists several),
  n_responses, n_speakers.
* data/wcs_naming.npz - naming (speakers x 330 int16: the code of the term speaker gave chip
  number c+1, -1 = no response '*' / '?'), lang, speaker (per row), age (-1 unknown), sex
  ('M', 'F', '' unknown); foci_lang, foci_speaker, foci_code, foci_chip (one row per chip a
  speaker marked as a best example of a term; A1-A40 / J1-J40 are mapped to A0 / J0; foci whose term is not in the naming data or whose
  grid code is invalid are dropped - the script prints how many).
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = "https://raw.githubusercontent.com/jvosten/wcs/master/data-raw/"


def fetch(name: str) -> str:
    req = urllib.request.Request(RAW + name, headers={"User-Agent": "pymc-challenges"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def main() -> None:
    chip = pd.read_csv(io.StringIO(fetch("chip.txt")), sep="\t", header=None,
                       names=["chip", "row", "col", "grid"])
    lab = pd.read_csv(io.StringIO(fetch("cnum-vhcm-lab-new.txt")), sep="\t")
    lab.columns = ["chip", "V", "H", "munsell_chroma", "munsell_hue", "munsell_value", "L", "a", "b"]
    chips = chip.merge(lab.drop(columns=["V", "H"]), on="chip").sort_values("chip")
    assert len(chips) == 330
    chips[["chip", "row", "col", "grid", "munsell_hue", "munsell_value", "munsell_chroma", "L", "a",
           "b"]].to_csv(DATA / "wcs_chips.csv", index=False)

    term = pd.read_csv(io.StringIO(fetch("term.txt")), sep="\t", header=None,
                       names=["lang", "speaker", "chip", "abbrev"], keep_default_na=False,
                       dtype={"abbrev": str})
    term["abbrev"] = term.abbrev.str.strip()
    missing = term.abbrev.isin(["*", "?", ""])
    print(f"term.txt: {len(term)} responses, {missing.sum()} non-responses ('*', '?', blank)")

    dct = pd.read_csv(io.StringIO(fetch("dict.txt")), sep="\t", keep_default_na=False, dtype=str)
    dct.columns = ["lang", "tnum", "term", "abbrev"]
    dct["lang"] = dct.lang.astype(int)
    gloss = dct.groupby(["lang", "abbrev"]).term.agg(lambda s: " | ".join(dict.fromkeys(s)))

    used = (term[~missing].groupby(["lang", "abbrev"])
            .agg(n_responses=("chip", "size"), n_speakers=("speaker", "nunique")).reset_index())
    used = used.sort_values(["lang", "n_responses", "abbrev"], ascending=[True, False, True])
    used["code"] = used.groupby("lang").cumcount()
    used["term"] = [gloss.get((l, a), "") for l, a in zip(used.lang, used.abbrev)]
    used[["lang", "code", "abbrev", "term", "n_responses", "n_speakers"]].to_csv(
        DATA / "wcs_terms.csv", index=False)
    code = {(l, a): c for l, a, c in zip(used.lang, used.abbrev, used.code)}

    spk = term[["lang", "speaker"]].drop_duplicates().sort_values(["lang", "speaker"]).reset_index(drop=True)
    row_of = {(l, s): i for i, (l, s) in enumerate(zip(spk.lang, spk.speaker))}
    naming = np.full((len(spk), 330), -1, np.int16)
    ri = np.array([row_of[(l, s)] for l, s in zip(term.lang, term.speaker)])
    ci = term.chip.to_numpy() - 1
    cv = np.array([-1 if m else code[(l, a)] for l, a, m in zip(term.lang, term.abbrev, missing)])
    naming[ri, ci] = cv

    sp = pd.read_csv(io.StringIO(fetch("spkr-lsas.txt")), sep="\t", header=None,
                     names=["lang", "speaker", "age", "sex"], dtype=str, keep_default_na=False)
    sp["lang"], sp["speaker"] = sp.lang.astype(int), sp.speaker.astype(int)
    sp = sp.drop_duplicates(["lang", "speaker"])        # one duplicated speaker number (lang 97)
    info = spk.merge(sp, on=["lang", "speaker"], how="left")
    age = pd.to_numeric(info.age, errors="coerce").fillna(-1).astype(int).to_numpy().copy()
    age[age <= 0] = -1
    sex = info.sex.fillna("").str.upper().str[:1].where(lambda s: s.isin(["M", "F"]), "").to_numpy()

    lines = [l for l in fetch("foci-exp.txt").replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
    foci = pd.DataFrame([l.split("\t")[:5] for l in lines], columns=["lang", "speaker", "resp", "abbrev", "grid"])
    foci["lang"], foci["speaker"] = foci.lang.astype(int), foci.speaker.astype(int)
    grid_chip = dict(zip(chips.grid, chips.chip))
    # The focus palette shows rows A (white) and J (black) as solid bars across all 40 hue columns,
    # so the expanded file lists A1 ... A40 / J1 ... J40 for one choice of white or black: map them
    # to the single chips A0 / J0 and drop the duplicates.
    g = foci.grid.str.strip()
    g = g.where(~g.str[0].isin(["A", "J"]), g.str[0] + "0")
    foci["chip"] = g.map(grid_chip)
    foci = foci.drop_duplicates(["lang", "speaker", "resp", "abbrev", "chip"])
    foci["code"] = [code.get((l, a.strip()), -9) for l, a in zip(foci.lang, foci.abbrev)]
    ok = foci.chip.notna() & (foci.code >= 0)
    print(f"foci-exp.txt: {len(foci)} focus choices; dropped {(~foci.chip.notna()).sum()} with an invalid "
          f"grid code and {(foci.chip.notna() & (foci.code < 0)).sum()} whose term is not in the naming data")
    foci = foci[ok]

    np.savez_compressed(
        DATA / "wcs_naming.npz", naming=naming, lang=spk.lang.to_numpy(np.int16),
        speaker=spk.speaker.to_numpy(np.int16), age=age.astype(np.int16), sex=sex.astype("U1"),
        foci_lang=foci.lang.to_numpy(np.int16), foci_speaker=foci.speaker.to_numpy(np.int16),
        foci_code=foci.code.to_numpy(np.int16), foci_chip=foci.chip.to_numpy(np.int16))

    iso = pd.read_csv(io.StringIO(fetch("wcs_iso_codes.csv")))
    iso = iso.rename(columns={"lang_nr": "lang", "lang_name": "name", "iso_693_3": "iso639_3",
                              "lang_country": "country"})
    iso["name"] = iso.name.str.replace(r"\s+", " ", regex=True).str.strip()
    iso["n_speakers"] = iso.lang.map(spk.groupby("lang").size())
    iso[["lang", "name", "iso639_3", "family", "country", "n_speakers"]].to_csv(
        DATA / "wcs_languages.csv", index=False)
    for f in ["wcs_chips.csv", "wcs_languages.csv", "wcs_terms.csv", "wcs_naming.npz"]:
        print(f"{f}: {(DATA / f).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
