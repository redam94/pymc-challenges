"""E92: pack what the notebook replays into data/e92_replay.npz. CPU only, NumPy only; reads results/E92/.

    E92_CORES=12,13 tools/e92_job.sh 3 2 -- .venv/bin/python tools/pack_e92.py

The stored run's arrays are not in git (only its JSON is), so the notebook's arrays ship as one derived
file in data/, as E88's and E90's do. It holds the replay chunk written by tools/score_e92.py, A''s mean
map, and the drift replicates of the 200 candidate sites (the same common random numbers the design used,
recomputed by run_e92_full.drift), so the notebook can draw tracks without the land-mask code in tools/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e92_full as r  # noqa: E402  (imports no JAX)

from pymc_challenges import data as reg  # noqa: E402
from pymc_challenges.bigrun import atomic_write, sha256_file  # noqa: E402

NOTICE = ("Argo float data and metadata from the Global Data Assembly Centre (Argo GDAC), SEANOE, "
          "https://doi.org/10.17882/42182, CC BY 4.0. These data were collected and made freely available by the "
          "International Argo Program and the national programs that contribute to it (https://argo.ucsd.edu, "
          "https://www.ocean-ops.org). The Argo Program is part of the Global Ocean Observing System. Changed: "
          "derived results of E92's stored run (tools/run_e92_full.py, tools/score_e92.py). Coastlines: Made with "
          "Natural Earth.")


def main():
    store = r.store
    arrays = store.load_chunk("replay_arrays", 0)
    arrays["map_mean_Afix"] = store.load_array("map_mean_Afix")
    inputs = store.load_json("design_inputs")
    sites = np.asarray(inputs["sites"])
    tr, _, _ = r.data()
    ocean = r.ocean_fn()
    starts, steps, gaps = r.tracks(tr)
    for rep in range(r.N_REP):
        la, lo, tt, valid, picks = r.drift(sites, starts, steps, gaps, ocean, rep)
        chunk = store.load_chunk("design", rep)  # draw 0 (the mode) of the primary model, replicate rep
        assert np.array_equal(picks, chunk["tracks"]) and np.array_equal(valid, chunk["valid"]), rep
        arrays[f"drift_lat_{rep}"] = la.astype(np.float32)
        arrays[f"drift_lon_{rep}"] = lo.astype(np.float32)
        arrays[f"drift_t_{rep}"] = tt.astype(np.float32)
        arrays[f"drift_valid_{rep}"] = valid
        arrays[f"design_picks_rep{rep}"] = chunk["picks"]  # greedy at the mode on this replicate
    # The declared operable-water rerun of the design (tools/score_e92_masked.py): same sites and tracks, its
    # own picks and validity; the first design's arrays above are kept as they were.
    masked = store.load_chunk("replay_masked", 0)
    op = r.operable_fn(ocean)
    for rep in range(r.N_REP):
        valid_m = r.drift(sites, starts, steps, gaps, op, rep)[3]
        assert np.array_equal(valid_m, masked[f"masked_valid_{rep}"]), rep
        arrays[f"masked_picks_rep{rep}"] = store.load_chunk("design_masked", rep)["picks"]
    arrays.update(masked)
    arrays["notice"] = np.array(NOTICE)
    out = reg.DATA_DIR / reg.REGISTRY["e92_replay"].filename  # data.path would try to download it
    atomic_write(out, lambda f: np.savez_compressed(f, **arrays))
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", sha256_file(out), len(arrays), "arrays")


if __name__ == "__main__":
    main()
