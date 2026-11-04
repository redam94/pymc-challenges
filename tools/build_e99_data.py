"""Build the E99 dataset: the CMU "Plaza" ultra-wideband range-only logs, as one small .npz.

Source: Djugash, Hamner & Roth (2009), "Navigating with ranging radios: Five data sets with ground
truth", Journal of Field Robotics 26(9):689-707. The original CMU host is gone; EPFL Infoscience
keeps a mirror, record 283435 (https://infoscience.epfl.ch/record/283435), file RangeOnlyDataset.zip
(3,616,837 bytes, md5 e8e8fb60c96f0c3ab6f9e47d87418df3). The record's licence field reads "n/a":
cite the paper.

In November 2026 the Infoscience bitstream URL answers a script with an AWS WAF captcha (HTTP 405),
so the zip must be fetched once in a browser and passed to this script:

    uv run python tools/build_e99_data.py /path/to/RangeOnlyDataset.zip

It keeps the two "Plaza" runs (the UWB ones; the Gesling runs use other radios) and writes
data/e99_plaza_uwb.npz with, for run k in (1, 2) and the original MATLAB names in brackets:

    p{k}_gt  (n, 4)  [Plaza{k}_GT]   time (s), x, y (m), heading (rad): GPS/INS ground truth
    p{k}_dr  (n, 3)  [Plaza{k}_DR]   time (s), distance travelled (m), heading change (rad) per step
    p{k}_drp (n, 4)  [Plaza{k}_DRp]  time, x, y, heading: the integrated dead-reckoned path
    p{k}_tl  (4, 3)  [Plaza{k}_TL]   beacon id, x, y (m): surveyed beacon positions
    p{k}_td  (m, 4)  [Plaza{k}_TD]   time (s), robot id (2), beacon id, range (m): UWB range log

Values are copied unchanged. Nothing here needs a network connection or a new dependency.
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

MD5 = "e8e8fb60c96f0c3ab6f9e47d87418df3"
OUT = Path(__file__).resolve().parents[1] / "data" / "e99_plaza_uwb.npz"


def main(zip_path: str) -> None:
    raw = Path(zip_path).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"md5 {md5} differs from the Infoscience record ({MD5}): wrong file?")
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for k in (1, 2):
            name = next(n for n in zf.namelist() if n.endswith(f"Plaza{k}.mat"))
            mat = sio.loadmat(io.BytesIO(zf.read(name)))
            for key in ("GT", "DR", "DRp", "TL", "TD"):
                arrays[f"p{k}_{key.lower()}"] = np.asarray(mat[f"Plaza{k}_{key}"], dtype=np.float64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    for k, v in arrays.items():
        print(f"  {k:7s} {v.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
