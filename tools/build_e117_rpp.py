"""Build the E117 dataset: original and replication effects of the Reproducibility Project: Psychology.

    python tools/build_e117_rpp.py   # download (cached), derive, check, write data/e117_rpp_pairs.csv
    python tools/build_e117_rpp.py --allow-new-source   # accept a raw file whose md5 has changed

Source: the RPP master data file ``rpp_data.csv`` of the Open Science Collaboration (2015),
"Estimating the reproducibility of psychological science", Science 349:aac4716. OSF project
ezcuj (component ytpuq), https://osf.io/fgjvw/download. Licence CC0 1.0 Universal (checked
through https://api.osf.io/v2/nodes/ezcuj/?embed=license on 2026-11-06). The raw file is cached
in ~/.cache/pymc-challenges/e117 (not committed); it is latin-1 encoded.

Kept: the 97 studies with a correlation effect size and a sample size for BOTH the original and
the replication (the OSC's own effect-size subset: T_r..O., T_r..R., T_N..O., T_N..R.).

Derived columns (both arms, suffix _o for the original, _r for the replication):
  test, df1, df2, n, p, r      the OSC's recomputed test, degrees of freedom, sample size,
                               two-sided p (T_pval_USE) and effect as a correlation; r_r is negative
                               when the replication went the opposite way to the original
  n_eff                        the sample size the test's precision corresponds to: df2 + 2 for t and
                               F(1, df2) tests, the reported N for r, z and chi-square tests. A
                               test's df can differ from the head count either way (analyses on a
                               subset, after exclusions, over items, cells or trials); 30 of the 70
                               one-df t and F originals have df2 + 2 below N and 23 above it
  z, se                        Fisher z = atanh(r), se = 1 / sqrt(n_eff - 3)
  se_o_p                       the original's se implied by its own recomputed p-value,
                               z_o / Phi^-1(1 - p_o / 2); within 1% of se_o for most studies
  p_o_reported                 the p-value as the original paper reported it ("Reported P-value (O)")
  marginal_o                   True when the original was presented to the journal as .05 < p <= .10.
                               Four used originals have a recomputed p just above .05 (.0503-.0516);
                               three of them were reported as significant ("0.05", "<.05" and
                               "prep = .92", which is p of about .047), so only study 28 ("<.06") is
                               marginal. The publication filter acts on what the journal saw.
  single_df                    True when both arms are one-degree-of-freedom tests (t, F(1, .), r, z,
                               chi-square(1)), whose effect has a direction; 79 of 97
  claimed_effect               True when the original reported p <= 0.10; three originals reported a
                               null or a non-significant result (studies 26, 89, 135)
  use                          single_df and claimed_effect: the 76 pairs the model uses
  success                      replication p < .05 in the original's direction (from p_r and r_r)
  osc_replicate                the OSC's own yes/no coding of a significant replication
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e117_rpp_pairs.csv"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e117"
URL = "https://osf.io/fgjvw/download"
MD5 = "3befa6d4b217ca07b2366e81b0949087"   # of the file as downloaded 2026-11-06 (259,220 bytes)


def arm(d: pd.DataFrame, a: str) -> pd.DataFrame:
    """The derived columns of one arm ("O" original, "R" replication)."""
    test = d[f"T_Test.Statistic..{a}."].astype(str)
    df1, df2 = d[f"T_df1..{a}."], d[f"T_df2..{a}."]
    n = d[f"T_N..{a}."]
    n_eff = np.where(test.isin(["t", "F"]), df2 + 2, n)
    r = d[f"T_r..{a}."]
    out = pd.DataFrame({
        "test": test, "df1": df1, "df2": df2, "n": n, "p": d[f"T_pval_USE..{a}."], "r": r,
        "n_eff": n_eff,
    })
    out["z"] = np.arctanh(r)
    out["se"] = 1 / np.sqrt(out["n_eff"] - 3)
    out["one_df"] = test.isin(["t", "r", "z"]) | (test.isin(["F", "Chi2"]) & (df1 == 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-new-source", action="store_true",
                    help="continue when the raw file's md5 differs from the 2026-11-06 download")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / "rpp_data.csv"
    if not raw.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges"})
        raw.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    md5 = hashlib.md5(raw.read_bytes()).hexdigest()
    d = pd.read_csv(raw, encoding="latin1")
    if md5 != MD5:
        print(f"md5 {md5} of {raw} differs from the 2026-11-06 download {MD5}")
    d = d[d["Study Num"].notna()]
    assert len(d) == 167, len(d)
    d = d.dropna(subset=["T_r..O.", "T_r..R.", "T_N..O.", "T_N..R."]).reset_index(drop=True)
    assert len(d) == 97, len(d)

    o, r = arm(d, "O"), arm(d, "R")
    df = pd.DataFrame({
        "study": d["Study Num"].astype(int),
        "title": d["Study Title (O)"].str.strip(),
        "authors": d["Authors (O)"].str.strip(),
        "journal": d["Journal (O)"],
        "discipline": d["Discipline (O)"],
    })
    for name, a in (("o", o), ("r", r)):
        for c in ("test", "df1", "df2", "n", "n_eff", "p", "r", "z", "se"):
            df[f"{c}_{name}"] = a[c]
    df["se_o_p"] = df["z_o"] / stats.norm.isf(df["p_o"] / 2)
    df["p_o_reported"] = d["Reported P-value (O)"].astype(str).str.strip()
    # reported as significant although the recomputed p is a hair above .05 ("prep = .92" is the
    # Killeen replication probability, Phi(z / sqrt 2) = .92, i.e. z = 1.99, p = .047)
    reported_sig = df["p_o_reported"].isin(["0.05", "<.05", "prep = .92"])
    df["marginal_o"] = (df["p_o"] >= 0.05) & (df["p_o"] <= 0.10) & ~reported_sig
    df["single_df"] = o["one_df"] & r["one_df"]
    df["claimed_effect"] = df["p_o"] <= 0.10
    df["use"] = df["single_df"] & df["claimed_effect"]
    df["success"] = (df["p_r"] < 0.05) & (df["r_r"] > 0)
    df["osc_replicate"] = d["Replicate (R)"].str.lower().eq("yes")

    # checks
    assert (df["r_o"] > 0).all(), "originals are signed positive"
    assert df["single_df"].sum() == 79 and df["use"].sum() == 76, (df["single_df"].sum(), df["use"].sum())
    assert sorted(df.loc[~df["claimed_effect"], "study"]) == [26, 89, 135]
    u = df[df["use"]]
    zr = u["z_r"] / u["se_r"]
    assert ((zr > stats.norm.isf(0.025)) == u["success"]).all(), "success by p and by z agree"
    ratio = u["se_o_p"] / u["se_o"]
    assert ratio.between(0.85, 1.2).all(), ratio.describe()
    assert u["success"].sum() == 26, u["success"].sum()
    near = u[(u["p_o"] >= 0.05)]
    assert sorted(near["study"]) == [28, 37, 104, 129], sorted(near["study"])
    assert sorted(u.loc[u["marginal_o"], "study"]) == [28], "only study 28 was reported as marginal"

    for c in ("df1_o", "df2_o", "df1_r", "df2_r"):
        df[c] = df[c].round(3)
    for c in ("r_o", "r_r", "z_o", "z_r", "se_o", "se_r", "se_o_p"):
        df[c] = df[c].round(6)
    if md5 != MD5:
        if OUT.exists():  # the derived rows that changed, if the asserts above still held
            old = pd.read_csv(OUT).set_index("study")
            new = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False))).set_index("study")
            both = old.index.intersection(new.index)
            diff = (old.loc[both] != new.loc[both]) & ~(old.loc[both].isna() & new.loc[both].isna())
            print(f"studies in only one version: {sorted(set(old.index) ^ set(new.index))}")
            for st in both[diff.any(axis=1)]:
                cols = list(diff.columns[diff.loc[st]])
                print(f"study {st}: " + ", ".join(f"{c} {old.at[st, c]!r} -> {new.at[st, c]!r}"
                                                  for c in cols))
        if not args.allow_new_source:
            raise SystemExit("the OSF source file has changed: inspect the rows above, then rerun "
                             "with --allow-new-source")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} pairs, {df['use'].sum()} used, {OUT.stat().st_size / 1e3:.0f} KB")
    print(f"successes: all 97 {df['success'].sum()}, used 76 {u['success'].sum()}; "
          f"OSC 'yes' among the 97: {df['osc_replicate'].sum()}")
    print(u.groupby("discipline")["success"].agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
