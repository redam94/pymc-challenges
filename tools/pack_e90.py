"""E90's derived summary for the notebook: one small file, built from the stored FULL run.

    python tools/pack_e90.py            # writes data/e90_nowcasts.npz

Reads the local unprocessed triangle (tools/build_e90_data.py) and the stored fits in results/E90/,
which are not in git. Writes only 7-day incidences per state and day, quantiles and alert
probabilities, and a few hundred draws on the eight hard dates: scores and derived summaries, no
reporting triangle (plans/large/E90.md, Q3). Plain NumPy, one core, seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e90_core as c  # noqa: E402
import run_e90_full as r  # noqa: E402
import score_e90 as s  # noqa: E402

from pymc_challenges.bigrun import atomic_write, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e90_nowcasts.npz"
DRAWS = 200
NOTICE = ("Derived from Robert Koch-Institut (2021-2022), COVID-19-Hospitalisierungen in Deutschland, CC BY 4.0, "
          "via the German COVID-19 Nowcast Hub (KIT; Wolffram et al. 2023). Changed: 7-day incidences per 100,000 "
          "(Destatis 31.12.2020 populations) and nowcast summaries from E90's own model. No endorsement is implied.")


def main():
    data = s.Data()
    store = s.store
    f32 = lambda x: np.asarray(x, np.float32)
    hard = [r.p.chunk_id(d) for d in r.HARD]
    thin = lambda keep: f32(keep[:: max(1, keep.shape[0] // DRAWS)][:DRAWS])
    honest = np.stack([thin(data.keep[i]) for i in hard])
    hindsight = np.stack([thin(store.load_chunk("hindsight", i)["keep"]) for i in hard])
    strata = np.stack([thin(store.load_chunk("strata", i)["keep"]) for i in hard])
    feb16 = r.p.chunk_id("2022-02-16")
    failed = store.load_chunk("failed_runs", r.FAILED_OFFSET["gpu_fits"] + feb16)["keep"]
    arrays = {
        "dates": np.array(data.all_dates), "states": np.array(c.STATES), "population": f32(data.pop),
        "frozen": f32(data.frozen), "final": f32(data.final), "factor": f32(data.factor),
        "ladder": f32(data.ladder), "stale": f32(data.stale),
        "q": f32(data.q), "p_cross": f32(data.pc), "q_recal": f32(data.qr), "p_recal": f32(data.pr),
        "k": f32(data.k), "quantiles": f32([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]),
        "hard_dates": np.array(r.HARD), "hard_honest": honest, "hard_hindsight": hindsight, "hard_strata": strata,
        "feb16_failed": f32(failed), "feb16_refit": f32(data.keep[feb16]), "notice": np.array(NOTICE),
    }
    assert np.isfinite(arrays["q"]).all() and arrays["frozen"].shape == (16, 227)
    atomic_write(OUT, lambda f: np.savez_compressed(f, **arrays))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), sha256 {sha256_file(OUT)[:12]}")


if __name__ == "__main__":
    main()
