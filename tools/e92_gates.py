"""Write E92's pilot gates to results/E92/gates_2026-11-03.json BEFORE any pilot of that day runs.

    .venv/bin/python tools/e92_gates.py

Refuses to overwrite: once written, the lines are fixed, and a later change must be a new, dated file
that says what changed and why. The SHA-256 sidecar is written by the store.
"""

from __future__ import annotations

import sys

from pymc_challenges.bigrun import Store

NAME = "gates_2026-11-03"

GATES = {
    "written": "2026-11-03, before derive, climatology and every pilot below; no row data had been read",
    "plan": "plans/large/E92.md, pilot checklist, with lines made exact here where the plan left them loose",
    "P2_data": {
        "fit_profiles": {
            "what": "2023-2024 profiles in the masked box after reduction to one 100 dbar value, before holdout",
            "go": ">= 24000",
            "no_go": "widen to 15-58N (needs a new pull) or add 2022 to the fit years; replan",
        },
        "climatology": {
            "spec": "fixed before looking: base years 2015-2022 only; 1 degree cells (centres at .5); weighted "
                    "least squares on base profiles within 1000 km of the centre, tricube weight in distance; "
                    "local quadratic in position (6 terms) plus 3 annual harmonics (6 terms); cells with "
                    "fewer than 200 base profiles inside 1000 km or effective weight sum under 50 are left "
                    "empty; a profile's climatology is the bilinear blend of the four surrounding cells' "
                    "local fits evaluated at the profile's own place and day, renormalised over non-empty cells",
            "fallback": "if the Gulf Stream line below fails, ALSO fit a 500 km bandwidth and report both; "
                        "the 500 km fit may be used only with that said in the plan and the notebook",
        },
        "mode_offset": {
            "what": "R minus D+A mean anomaly, 2023-2024, within 5 degree x 5 degree x calendar-year cells "
                    "holding at least 20 of each, averaged with weight min(n_R, n_DA); raw difference and a "
                    "bootstrap se over cells are reported beside it",
            "go": "|difference| < 0.02 C",
            "no_go": "drop R rows and accept fewer profiles (plan); report the count left",
        },
        "anomaly_sd": {"what": "sd of 2023-2024 anomalies over the masked box", "go": "< 1.5 C"},
        "gulf_stream_mean": {
            "what": "base-year residual (in-sample) mean per 2 degree cell in 30-45N, 75-45W with at least 100 "
                    "base profiles; excess rms = sqrt(max(0, weighted mean(cell_mean^2 - se^2))), weights n",
            "go": "excess rms < 0.3 C",
            "no_go": "fallback bandwidth above, reported as a change",
        },
    },
    "P3_ladder": {
        "what": "n = 2000, 5000, 10000, 20000 real 2023-2024 locations (first n in time order of a fixed "
                "shuffle), Matern 3/2 two-scale kernel at the plan's illustrative scales; Cholesky, K^-1 "
                "from the factor, and the hand value-and-gradient; each GPU configuration in a fresh "
                "process with no previous result held; every output depends on the factor",
        "devices": "GPU f32, GPU f64, JAX CPU f32 and f64 (15 threads, cores 0-9,11-15), scipy LAPACK f32 "
                   "and f64 (15 OpenBLAS threads, same cores)",
        "gpu_f32_rate": {"what": "Cholesky flops n^3/3 over best wall time at n = 20000", "go": ">= 0.5 Tflop/s"},
        "gpu_peak_30k": {
            "what": "least-squares line peak_GiB = a x matrix_GiB + b over the measured GPU f32 gradient peaks "
                    "at n = 5000, 10000, 20000, evaluated at n = 30000; and the same for Cholesky alone",
            "go": "gradient projection <= 16.5 GiB",
            "no_go": "a blocked K^-1 (column blocks, only L held) or n = 20000 cap, whichever P3 times better",
        },
        "cpu_f64_rate": {"what": "scipy or JAX CPU f64 Cholesky at n = 20000, 15 threads, best of the two",
                         "go": ">= 200 Gflop/s"},
        "stop": "GPU fails both rate and memory AND CPU fails its rate: stop, n = 20000 cap (plan stop rule)",
    },
    "P4_gradient": {
        "what": "hand gradient (blocked trace of (a a^T - K^-1) dK, dK by autodiff inside each row block) "
                "against full autodiff through the Cholesky and against central finite differences "
                "(step 1e-5 in log space), n = 2000, float64, at the prior median and at 2 random prior "
                "draws, every hyperparameter; Matern 1/2 and 3/2",
        "go": "max relative difference < 1e-3 for every hyperparameter against both, where relative is "
              "|g_hand - g_ref| / max(|g_ref|, 1e-2 x max_j |g_ref_j|)",
        "no_go": "fix before any timing is trusted",
    },
    "P6_choices": {
        "what": "SMALL crop 25-45N, 75-40W, year 2024 at real density; MAP fit on 90% of floats (holdout "
                "floats chosen by a fixed hash of float_id); score = joint Gaussian predictive log density "
                "of each held-out float's profiles given the training data, at the MAP; paired across floats",
        "options": ["Matern 1/2 or 3/2", "equirectangular about 37.5N or Mercator"],
        "go": "an option wins only by more than 2 paired se of the summed per-float difference; otherwise "
              "take the simpler: Matern 1/2 (the exponential of Kuusela and Stein) and equirectangular",
        "prior_predictive": "95% prior interval of the total field sd sqrt(s1^2+s2^2+sn^2) inside 0.1-5 C",
        "edges": "every fitted log hyperparameter inside its prior's 1%-99% quantiles; else fix the mean first",
    },
    "P7_audit_price": {
        "what": "nutpie logp+dlogp of the PyMC model (pm.gp.Marginal-equivalent, same kernel and priors) on "
                "one thread at n = 400, 800, 2000, median of 50 jittered points after compile",
        "price": "21,000 gradients a chain (plan's assumption), 4 chains in parallel, one thread each",
        "line": "no pass line: it prices the audit, which the owner has accepted at n = 800 and 2,000",
        "n400_audit": {
            "what": "if time allows today: nutpie 4 chains, n = 400, against Laplace at the hand model's mode",
            "go": "log-hyperparameter means within 0.5 posterior sd; sd ratio 0.7-1.3; max r_hat 1.01",
            "no_go": "importance sampling from NUTS replaces Laplace at n = 2000 (plan)",
        },
    },
    "P9_precision": {
        "what": "20 iterates of a float64 L-BFGS path of the MAP fit on the n = 5000 subset (stored), then "
                "f32 and f64 log density and hand gradient at those iterates on the GPU at n = 5000, 10000, "
                "20000 (the 20000 f64 on GPU fits in memory)",
        "go": "max |f32 - f64| log density < 0.1 nat at n = 20000; min cosine(g32, g64) > 0.9999 at every n",
        "no_go": "the last L-BFGS steps and the Hessian in CPU f64 (or GPU f64 where it fits)",
    },
    "deferred_to_FULL_start": {
        "P8": "Vecchia m = 15, 30, 60 against exact at n = 3000; plan's line, unchanged",
        "P10": "design sensitivity at SMALL; plan's line, unchanged",
    },
    "safety": "before each job: /sys/devices/system/cpu/offline == '10,26' and the kernel log's "
              "machine-check count == 1; any change stops everything",
}


def main() -> None:
    store = Store("E92")
    if (store.path / f"{NAME}.json").exists():
        sys.exit(f"{NAME}.json exists: gates are written once. Write a new dated file instead.")
    digest = store.save_json(NAME, GATES)
    print(NAME, digest)


if __name__ == "__main__":
    main()
