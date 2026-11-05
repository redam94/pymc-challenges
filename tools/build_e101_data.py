"""Build the E101 dataset: a fixed random subsample of the Criteo Uplift Prediction Dataset v2.1,
plus the exact full-data counts per arm, as one small .npz.

Source: Diemert, Betlei, Renaudin & Amini (2018), "A Large Scale Benchmark for Uplift Modeling",
AdKDD workshop at KDD 2018. Licence CC BY-NC-SA 4.0 (non-commercial, share-alike). Criteo's
original link (http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz) answered 404 in November
2026; Criteo's own Hugging Face repository carries the same file:

    https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz
    (311,422,618 bytes, sha256 2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc)

Download it once, outside the repository, and pass its path:

    uv run python tools/build_e101_data.py /path/to/criteo-research-uplift-v2.1.csv.gz

The file has 13,979,592 rows: f0-f11 (anonymised dense features), treatment (1 = test, randomised,
about 85%), conversion, visit, exposure (the user was actually shown an ad; only possible in test).
The script streams it twice in chunks (peak memory about 2 GB) and writes
data/e101_criteo_uplift.npz with:

    X          (n, 12) float32   f0-f11 of the subsample (float64 in the source; rounded to float32)
    treatment, exposure, visit, conversion  (n,) uint8
    row        (n,) int64        row number in the source file (0-based, header excluded)
    cell_n, cell_visit, cell_conversion  (3,) int64   exact FULL-data counts for the three cells
               control, test-unexposed, test-exposed (control has no exposed users)
    cell_xsum, cell_xsq  (3, 12) float64   full-data sums and sums of squares of f0-f11 per cell
    basis_zz   (2, 91, 91)       full-data Z'Z per arm (control, test) for the basis
                                 Z = [1, r, r^2, r_i r_j (i < j)], r = the features' normal scores in the
                                 sample (tie-averaged ranks; other values interpolated)
    basis_zy   (2, 2, 91)        full-data Z'y per arm and outcome (visit, conversion)
    basis_cell_sum (3, 91)       full-data column sums of Z per cell
    n_total    ()  int64
    seed       ()  int64

The subsample is N_SAMPLE rows drawn without replacement, uniformly, with a fixed seed: a simple
random sample of the campaign audience, so both arms keep their randomised proportions.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
N_TOTAL = 13_979_592
N_SAMPLE = 300_000
SEED = 101
FEATS = [f"f{i}" for i in range(12)]
OUT = Path(__file__).resolve().parents[1] / "data" / "e101_criteo_uplift.npz"


def basis(R: np.ndarray) -> np.ndarray:
    """[1, R, R^2, R_i R_j (i < j)]: 91 columns from 12 normal scores."""
    pairs = [R[:, i] * R[:, j] for i in range(12) for j in range(i + 1, 12)]
    return np.column_stack([np.ones(len(R)), R, R ** 2, *pairs])


def main(gz_path: str) -> None:
    h = hashlib.sha256()
    with open(gz_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise SystemExit(f"sha256 {h.hexdigest()} differs from the published file: wrong file?")

    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(N_TOTAL, size=N_SAMPLE, replace=False))
    cell_n = np.zeros(3, np.int64)
    cell_visit = np.zeros(3, np.int64)
    cell_conv = np.zeros(3, np.int64)
    cell_xsum = np.zeros((3, 12))
    cell_xsq = np.zeros((3, 12))
    parts = []
    start = 0
    dtypes = {c: np.float64 for c in FEATS} | {c: np.int8 for c in ("treatment", "conversion", "visit", "exposure")}
    for chunk in pd.read_csv(gz_path, chunksize=1_000_000, dtype=dtypes):
        stop = start + len(chunk)
        t = chunk["treatment"].to_numpy()
        e = chunk["exposure"].to_numpy()
        if np.any((t == 0) & (e == 1)):
            raise SystemExit("exposed users in the control arm: unexpected")
        cell = np.where(t == 0, 0, np.where(e == 0, 1, 2))
        X = chunk[FEATS].to_numpy()
        v = chunk["visit"].to_numpy()
        c = chunk["conversion"].to_numpy()
        for k in range(3):
            m = cell == k
            cell_n[k] += m.sum()
            cell_visit[k] += v[m].sum()
            cell_conv[k] += c[m].sum()
            cell_xsum[k] += X[m].sum(axis=0)
            cell_xsq[k] += (X[m] ** 2).sum(axis=0)
        idx = keep[(keep >= start) & (keep < stop)] - start
        sub = chunk.iloc[idx].copy()
        sub["row"] = idx + start
        parts.append(sub)
        start = stop
        print(f"  {stop:>11,} rows", flush=True)
    if start != N_TOTAL:
        raise SystemExit(f"read {start:,} rows, expected {N_TOTAL:,}")
    s = pd.concat(parts, ignore_index=True)
    Xs = s[FEATS].to_numpy(np.float32).astype(np.float64)

    # Pass 2: sufficient statistics of a flexible basis, for an exact covariate-adjusted analysis of the
    # experiment on all users. Each feature is mapped to its normal score in the SAMPLE (tie-averaged ranks,
    # the same transform the notebook uses), interpolated for values the sample does not contain.
    maps = []
    for j in range(12):
        sc = stats.norm.ppf((stats.rankdata(Xs[:, j]) - 0.5) / len(Xs))
        v, i = np.unique(Xs[:, j], return_index=True)
        maps.append((v, sc[i]))
    P = 1 + 12 + 12 + 66
    zz = np.zeros((2, P, P))
    zy = np.zeros((2, 2, P))
    zsum = np.zeros((3, P))
    for chunk in pd.read_csv(gz_path, chunksize=500_000, dtype=dtypes):
        t = chunk["treatment"].to_numpy()
        e = chunk["exposure"].to_numpy()
        cell = np.where(t == 0, 0, np.where(e == 0, 1, 2))
        X = chunk[FEATS].to_numpy().astype(np.float32).astype(np.float64)
        Z = basis(np.column_stack([np.interp(X[:, j], *maps[j]) for j in range(12)]))
        for arm in (0, 1):
            m = t == arm
            zz[arm] += Z[m].T @ Z[m]
            for kk, lab in enumerate(("visit", "conversion")):
                zy[arm, kk] += Z[m].T @ chunk[lab].to_numpy()[m].astype(np.float64)
        for k in range(3):
            zsum[k] += Z[cell == k].sum(axis=0)
        print(f"  pass 2: {zsum[:, 0].sum():>13,.0f} rows", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X=s[FEATS].to_numpy(np.float32),
        treatment=s["treatment"].to_numpy(np.uint8),
        exposure=s["exposure"].to_numpy(np.uint8),
        visit=s["visit"].to_numpy(np.uint8),
        conversion=s["conversion"].to_numpy(np.uint8),
        row=s["row"].to_numpy(np.int64),
        cell_n=cell_n, cell_visit=cell_visit, cell_conversion=cell_conv,
        cell_xsum=cell_xsum, cell_xsq=cell_xsq,
        basis_zz=zz, basis_zy=zy, basis_cell_sum=zsum,
        n_total=np.int64(N_TOTAL), seed=np.int64(SEED),
    )
    md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, md5 {md5})")
    for k, name in enumerate(["control", "test, unexposed", "test, exposed"]):
        print(f"  {name:16s} n={cell_n[k]:>10,}  visit={cell_visit[k] / cell_n[k]:.4f}  "
              f"conversion={cell_conv[k] / cell_n[k]:.5f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
