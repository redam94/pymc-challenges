"""Convert spatstat.data's `bei.rda` into data/bei_trees_terrain.npz for E86.

    uv run --no-project --with rdata --with numpy python tools/build_e86_bei.py path/to/bei.rda

The file is an R serialisation of a `ppp` point pattern and a list of two `im` pixel images. Keeps
the coordinates exactly as shipped. R's images are indexed [row = y, column = x].
"""

import sys
from pathlib import Path

import numpy as np
import rdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bei_trees_terrain.npz"

obj = rdata.conversion.convert(rdata.parser.parse_file(sys.argv[1]), default_encoding="utf8")
trees, extra = obj["bei"], obj["bei.extra"]
elev, grad = extra["elev"], extra["grad"]
assert np.allclose(elev["xcol"], grad["xcol"]) and np.allclose(elev["yrow"], grad["yrow"])
np.savez_compressed(
    OUT,
    x=np.asarray(trees["x"], float),
    y=np.asarray(trees["y"], float),
    xcol=np.asarray(elev["xcol"], float),
    yrow=np.asarray(elev["yrow"], float),
    elev=np.asarray(elev["v"], float),
    grad=np.asarray(grad["v"], float),
)
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
