"""Build the E116 dataset: every American English CDI Words & Sentences checklist aged 16-30 months.

    python tools/build_e116_wordbank.py   # download (cached), check licences, write data/e116_wordbank_ws.npz

Source: Wordbank (Frank, Braginsky, Yurovsky & Marchman 2017, J. Child Language 44:677-694), the
item-by-child "responses" slice for English (American) Words & Sentences (MacArthur-Bates CDI):
  https://langcog.github.io/wordbank-datapage/slices/responses/english_american_ws.parquet
  (2,981,861 bytes on 2026-11-06; 6,917,088 rows = 10,173 checklists x 680 words, ages 16-36)
The parquet is cached in ~/.cache/pymc-challenges/e116 (not committed). It is read by columns
with pyarrow: pandas on the whole file needs more than 2 GB.

Licence: "Unless otherwise stated, Wordbank and its datasets are licensed under a Creative Commons
Attribution 4.0 International License"; a few datasets are CC BY-NC 4.0. The slice has no dataset
column, so the build reads the datasets table embedded in the Wordbank contributors page
(https://langcog.github.io/wordbank-datapage/contributors.html: language, form, dataset_name,
contributor, license, n_admins) and ASSERTS that every English (American) WS dataset is CC-BY.
On 2026-11-06 all 18 are (10,604 administrations); the nine CC-BY-NC datasets are Korean, Ronga,
Tsonga, Tseltal and Peruvian Spanish. If a non-commercial dataset ever joins this slice, the build
stops rather than silently mixing licences.
The table counts 10,604 English (American) WS administrations but the item-level slice holds
10,173: Wordbank's item-level slices leave some administrations out, and without a dataset column
we cannot tell which. The licence inference does not depend on the counts matching: the slice is
English (American) WS by construction, so every row belongs to one of those 18 datasets (a subset
of their administrations), and all 18 are CC-BY.

Kept: the 9,537 checklists aged 16-30 months, the age range the CDI:WS was normed for (Fenson et
al. 2007). `is_norming` is False for every row of the slice, so the original norming sample cannot
be picked out; each `data_id` is one administration and the slice cannot tell repeat visits of one
child apart.

Output: data/e116_wordbank_ws.npz (NumPy, open with np.load(data.path("e116_wordbank_ws"))):
  produces_bits  uint8 (children, 85): np.packbits of the 0/1 "says this word" matrix along items
                 (np.unpackbits(..., axis=1, count=680) gives children x 680)
  answered_bits  uint8 (children, 85): packbits of 1 where the item was on the child's form
                 (440 checklists miss one or two items; their produces bit is 0)
  data_id        int32 (children,)
  age            int8  months, 16-30
  sex            int8  0 = boy, 1 = girl, -1 = not reported
  birth_order    int8  1 = first-born, 2.. = later, -1 = not reported
  education      int8  caregiver education 0 = primary ... 6 = graduate (EDUCATION), -1 = not reported
  ethnicity      int8  0 = non-Hispanic, 1 = Hispanic, -1 = not reported
  item_id        <U8   item_1 ... item_680 (instrument order)
  definition     <U40  the word as printed on the form ("ball", "choo choo", "dress (object)")
  category       <U20  the form's 22 semantic sections ("sounds", "animals", ... "connecting_words")
  uni_lemma      <U40  Wordbank's cross-language concept label
  EDUCATION      <U16  labels of the education codes
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e116_wordbank_ws.npz"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e116"
BASE = "https://langcog.github.io/wordbank-datapage/"
PARQUET = BASE + "slices/responses/english_american_ws.parquet"
CONTRIB = BASE + "contributors.html"
MD5 = "6cb17f2b920720ea96bef70db1a6135c"   # of the parquet as downloaded 2026-11-06
EDUCATION = ["Primary", "Some Secondary", "Secondary", "Some College", "College", "Some Graduate", "Graduate"]
BIRTH = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"]
AGES = (16, 30)


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "pymc-challenges"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def check_licences() -> None:
    """Every English (American) WS dataset in Wordbank's datasets table must be CC-BY."""
    html = fetch(CONTRIB, RAW / "contributors.html").read_text()
    start = html.index('{"contents":[{"name":"datasets"')
    table = json.JSONDecoder().raw_decode(html[start:])[0]["contents"][0]["value"]
    rows = [dict(zip(table, vals)) for vals in zip(*table.values())]
    ws = [r for r in rows if r["language"] == "English (American)" and r["form"] == "WS"]
    lic = sorted({r["license"] for r in ws})
    n = sum(int(r["n_admins"]) for r in ws)
    print(f"licences: {len(ws)} English (American) WS datasets, {n:,} administrations, licences {lic}")
    nc = sorted({(r["language"], r["dataset_name"]) for r in rows if r["license"] != "CC-BY"})
    print(f"  (non-CC-BY datasets elsewhere in Wordbank: {nc})")
    assert lic == ["CC-BY"], "a non-CC-BY dataset is in the English (American) WS slice: exclude it"


def main() -> None:
    check_licences()
    pq_path = fetch(PARQUET, RAW / "english_american_ws.parquet")
    md5 = hashlib.md5(pq_path.read_bytes()).hexdigest()
    if md5 != MD5:
        print(f"note: parquet md5 {md5} differs from the 2026-11-06 download ({MD5}); Wordbank was updated")
    t = pq.ParquetFile(pq_path).read(columns=["data_id", "item_id", "item_definition", "category", "uni_lemma",
                                              "age", "produces", "sex", "birth_order",
                                              "caregiver_education", "ethnicity"])
    assert t.num_rows == 6_917_088 or md5 != MD5
    # items in instrument order (item_1 ... item_680)
    items = pc.unique(t["item_id"]).to_pylist()
    items.sort(key=lambda s: int(s.split("_")[1]))
    J = len(items)
    assert J == 680
    item_idx = np.asarray(pc.index_in(t["item_id"], value_set=__import__("pyarrow").array(items)))
    did = t["data_id"].to_numpy()
    kids, child = np.unique(did, return_inverse=True)
    first = np.full(len(kids), -1)
    first[child[::-1]] = np.arange(len(did))[::-1]
    age = t["age"].to_numpy()[first]
    keep = (age >= AGES[0]) & (age <= AGES[1])
    print(f"checklists: {len(kids):,} in the slice, {keep.sum():,} aged {AGES[0]}-{AGES[1]} months")
    newpos = np.cumsum(keep) - 1
    rows_keep = keep[child]
    ci, ii = newpos[child[rows_keep]], item_idx[rows_keep]
    prod = t["produces"].fill_null(False).to_numpy(zero_copy_only=False)[rows_keep]
    n = int(keep.sum())
    Y = np.zeros((n, J), np.uint8)
    A = np.zeros((n, J), np.uint8)
    Y[ci, ii] = prod
    A[ci, ii] = 1

    def child_col(name, levels, start=0):
        v = t[name].take(__import__("pyarrow").array(first[keep])).to_pylist()
        return np.array([levels.index(x) + start if x in levels else -1 for x in v], np.int8)

    sex = child_col("sex", ["Male", "Female"])
    bo = child_col("birth_order", BIRTH, start=1)
    edu = child_col("caregiver_education", EDUCATION)
    eth = child_col("ethnicity", ["Non-Hispanic", "Hispanic"])
    # child-level attributes must be constant within a checklist
    for name in ["sex", "birth_order", "caregiver_education", "age"]:
        col = t[name]
        uniq = pc.count_distinct(pc.binary_join_element_wise(
            pc.cast(t["data_id"], "string"), pc.fill_null(pc.cast(col, "string"), "NA"), "|"))
        assert uniq.as_py() == len(kids), name
    meta = {}
    for c in ["item_definition", "category", "uni_lemma"]:
        pairs = pc.unique(pc.binary_join_element_wise(t["item_id"], t[c], "\t")).to_pylist()
        d = dict(p.split("\t", 1) for p in pairs)
        assert len(d) == J
        meta[c] = np.array([d[i] for i in items])
    np.savez_compressed(
        OUT, produces_bits=np.packbits(Y, axis=1), answered_bits=np.packbits(A, axis=1),
        data_id=kids[keep].astype(np.int32), age=age[keep].astype(np.int8), sex=sex, birth_order=bo,
        education=edu, ethnicity=eth, item_id=np.array(items), definition=meta["item_definition"],
        category=meta["category"], uni_lemma=meta["uni_lemma"], EDUCATION=np.array(EDUCATION))
    W = Y.sum(1)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): {n:,} checklists x {J} words, "
          f"{(A == 0).sum():,} unanswered cells in {(A.sum(1) < J).sum()} checklists")
    print(f"  girls {np.mean(sex == 1):.1%}, boys {np.mean(sex == 0):.1%}, sex missing {np.mean(sex < 0):.1%}; "
          f"birth order missing {np.mean(bo < 0):.1%}; education missing {np.mean(edu < 0):.1%}")
    for a in (16, 18, 24, 30):
        m = age[keep] == a
        print(f"  age {a}: {m.sum():4d} checklists, words 10/50/90%: {np.percentile(W[m], [10, 50, 90]).round()}")


if __name__ == "__main__":
    main()
