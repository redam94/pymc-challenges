"""E92: score the operable-water rerun of the design (run_e92_full.py design_masked). CPU only.

    E92_CORES=12,13 tools/e92_job.sh 4 2 -- .venv/bin/python tools/score_e92_masked.py

A declared change made after the first design was seen: the first design is left as it is in
``scores.json``; this writes ``design_masked_summary`` (JSON) and the ``replay_masked`` chunk (arrays the
notebook draws), scored by the same function and the same advice rule as the first, then adds the new
files to the manifest without changing anything else in it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_e92 as s  # noqa: E402

store = s.store


def main():
    hs = s.jl("scores")["heldout"]
    ds, arrays = s.design_summary(hs, stage="design_masked", inputs_name="design_masked_inputs")
    inputs = s.jl("design_masked_inputs")
    first0 = store.load_chunk("design", 0)  # chunk 0: the primary model at its mode, drift replicate 1
    masked0 = store.load_chunk("design_masked", 0)
    valid = {"first": first0["valid"], "masked": masked0["valid"]}
    out = {"decision": ds["decision"], "models": ds["models"],
           "variants": {"gulf_weight_top20_changed": ds["gulf_weight_top20_changed"],
                        "fixed_sites_top20_changed": ds["fixed_sites_top20_changed"]},
           "excluded_sites": np.flatnonzero(~np.asarray(inputs["allowed"])).tolist(),
           "operable_km": inputs["operable_km"], "operable_min": inputs["operable_min"],
           "valid_share_mode_rep0": {"first_greedy": float(valid["first"][first0["picks"]].mean()),
                                     "first_gap": float(valid["first"][first0["gap_picks"]].mean()),
                                     "masked_greedy": float(valid["masked"][masked0["picks"]].mean()),
                                     "masked_gap": float(valid["masked"][masked0["gap_picks"]].mean())},
           "why": inputs["why"], "labels": "measured"}
    store.save_json("design_masked_summary", out)
    arrays = {f"masked_{k}": v for k, v in arrays.items() if k not in ("design_sites", "design_target_cells")}
    arrays["masked_allowed"] = np.asarray(inputs["allowed"])
    for rep in range(4):
        arrays[f"masked_valid_{rep}"] = store.load_chunk("design_masked", rep)["valid"]
    store.save_chunk("replay_masked", 0, arrays)
    m = store.manifest()
    keep = {k: v for k, v in m.items() if k not in ("files", "notebook")}
    store.write_manifest(**keep)
    print({k: out["decision"][k] for k in ("decision_model", "greedy_over_gap", "overlap_A_Afix", "region_tv",
                                           "recommendation")})
    for model, r in ds["models"].items():
        print(model, {k: r[k] for k in ("greedy_removed", "gap_removed", "random_removed", "greedy_over_gap",
                                        "greedy_over_random", "share_removed_greedy", "replicate_overlap_mean",
                                        "draw_overlap_with_mode_mean")})
    print(out["valid_share_mode_rep0"], "problems", store.verify()[:5])


if __name__ == "__main__":
    main()
