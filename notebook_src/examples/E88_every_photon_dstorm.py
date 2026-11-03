# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # E88 · Every photon: a neural localiser tried on a full dSTORM movie, and how it failed
#
# | | |
# |---|---|
# | **Type** | Advanced worked example - read, run, modify |
# | **Data** | Real: two full dSTORM recordings of Unc-13 at *Drosophila* neuromuscular junctions (Mrestani 2022, Zenodo 7328805, CC BY 4.0), 15,000 frames each, as a 1.7 MB summary and small JSON tables of a stored run |
# | **Time** | About 1 minute on 2 cores (measured, the longer with a cold compile cache). This is a replay: the heavy run is not repeated |
# | **You will learn** | Reading a stored run honestly · gates written before the run, and printing the ones that failed · a paired-frame test of error bars that needs no ground truth · why a simulation pass says nothing about a microscope · a calibration certificate for a trained network · a NUTS audit of a fitter, live in PyMC · how a cross-correlation drift estimate breaks · why a random split flatters Fourier ring correlation · reporting "tried and failed" |
#
# > **Do not rerun the heavy parts on limited hardware.** Every network number here comes from one stored
# > run: 2.4 GB of raw movies, a GPU, 120,000 frames of inference and hours of wall time on a shared machine.
# > A normal run of this notebook touches none of that. It loads small JSON files from `results/E88/` and one
# > 1.7 MB summary, needs no GPU and no download, and refuses any other run level. The scripts
# > `tools/run_e88_classical.py`, `tools/run_e88_net.py` and `tools/run_e88_full.py` hold the heavy run. It ran on a
# > 16-core desktop with a 24 GB GPU and 64 GB of memory. Do not start it on a laptop.
#
# ## The brief
#
# > *From the lead of an imaging core, to the analyst. The lead is fictional, invented for this notebook; the
# > data and the numbers are real.* We are planning a repeat of the Unc-13 potentiation experiment. Last
# > time each field was 15,000 frames. On the camera that is only three minutes, but with 45 junctions per
# > channel it is 2.2 hours of scope time per channel and 54 GB per channel, and the dye bleaches throughout.
# > For these two recordings, how many frames would have given the same answer? How far apart are the two?
# > And I keep being told to replace our fitter with a neural network: would that change your answer, and
# > what would I have to check before trusting one?
#
# ## The answer, before the working
#
# **We tried the network on the real movies, and it failed. Keep your fitter.** The result on real data is
# the classical one.
#
# - **Frames: about 10,500-12,000, which is 20-30% fewer than 15,000, with a sampling range of 7,500-13,500.** That
#   rests on the classical pipeline on both full recordings, under a rule fixed before any full result was seen. It is
#   10,500 under the calibration the rule required and 12,000 under a calibration chosen afterwards. Those are two
#   points, not an interval: a block bootstrap of the count rule gives 7,500-13,500 under either calibration
#   [post-hoc]. The count and the band test set 10,500 jointly, by a narrow margin: in one of five fresh seed families
#   the band test fails there. If you cut, cut to 12,000. It holds for these two recordings and sets no lab policy.
# - **How far apart are the two recordings?** Their own answers differ by 3,000-7,000 frames (10,500 against
#   7,500, or 12,000 against 5,000), and recording 7 sets the number for both. That gap is not resolved: recording 32's
#   own answer runs from 2,000 to 12,000 across the bootstrap. The second recording is the only replication there is.
# - **The network failed the check that matters most.** On recording 32, which played no part in training,
#   its error bars are too small: the paired-frame z is 1.40 for network A and 1.63 for network B against a
#   failure line of 1.25. A fix (network B, trained on a simulator built from real data) halved the false
#   halo of faint detections around bright ones but made z worse, and the halo still fails on recording 32.
#   That is the written outcome B: the failure appears and the fix does not fix it.
# - **Your fitter's error bars are too small as well.** Classical z is 1.63 on the same brightness-matched set, so
#   both networks count as "no worse than classical" on the written line. That is no reason to switch. On the 4,337
#   pairs all four methods share, B is worse than classical (1.76 against 1.55), and A (1.44) has the same scatter
#   as single-emitter fits, 12.0 nm [post-hoc]. A also has the worse halo and replicate. Part of the excess is shared:
#   a floor of about 4-6 nm per frame in every method, which looks like the sample or stage and not the localiser. For
#   chosen K and B, most of it remains an unexplained scale.
# - **The network does not move the frames answer in any way we can trust.** The gated runs say 10,500 for A
#   and 12,000 for B, but they rest on a broken drift estimate, so that difference is not a finding. After
#   one post-hoc repair the networks give 12,000 (A) and 10,500 (B), or 15,000 and 13,500 under the other
#   calibration. Calibration and localiser each move N by 1,500-3,000 frames, about the same, and neither shift
#   stands clear of sampling noise.
# - **What the network does do it does on clean simulation:** recall 0.994 and 1.16 x the Cramér-Rao bound (CRLB)
#   after 40,000 steps on a flat background, and about 570-750 frames per second. Nothing in that passes to real data.
#   On the one simulated movie with a real background, half of A's rows match no emitter, and A also failed on a
#   harder simulator, which is the warning in miniature.
# - **The audit of the fitter failed largely through our own bug.** Check 4 stays failed as written. But every chain
#   started at the corner of its patch, because `with_data` in nutpie 0.16.11 does not move the start. Rerun with a
#   start on the spot, the ratio is 1.11-1.14 and all 100 single-emitter spots of recording 7 reach r_hat 1.01, with a few
#   divergences left, on the fitter's own positions and photons [post-hoc]. Laplace looks fine at K = 1 on that recording. K = 2 is the real open problem.
# - **Why it could not help much here.** The network's gain is at high density. The written overlap measure
#   gives 2.2% and 1.65%; a better post-hoc estimate gives about 5-9%, against 43% in E58's dense simulation.
#   The written stop before the full run fired, and the project owner chose to run it anyway, knowing it might
#   fail. It did, and that is useful: it is what this notebook reports.
# - **Before you trust any network, check four things:** paired frames, the halo, a population replicate and
#   a calibration certificate saying where it was never tested. All four are below.
#
# **What each number rests on.** Five labels are used throughout.
#
# | Label | Meaning |
# |---|---|
# | **Stored** | Measured once in the stored run of 2026-11-02 and read from `results/E88/`. Your run reloads it and cannot change it |
# | **Live** | Computed in this notebook from the summary file, so you can change it |
# | **Gated** | Judged against a line written to a file before the job ran. No line was moved afterwards |
# | **Post-hoc** | Added after seeing a result. It can explain a gate; it cannot overturn one |
# | **Extrapolated** | Arithmetic from a measured rate. Nothing was timed at that size |
#
# Every failed gate is printed with its line and value in section 9. Wall times from the full run are
# inflated: another notebook's job shared the machine and the GPU lock, so they are not a benchmark.
#
# ## What "the same answer" means here
#
# N is the smallest frame count at which, in **both** recordings, inside the authors' mask of terminal
# boutons, three things hold at N and at every larger N on the grid: resolution by Fourier ring correlation
# (FRC) is within 10% of its 15,000-frame value, the cluster count is within 10% of its 15,000-frame value,
# and the time-block bootstrap band at N overlaps the band at 15,000. "Frames" means the first N frames of a
# recording that ran to 15,000, so a truncated movie keeps the labelling and bleaching history of a full one.
# It is a proxy for a shorter acquisition. There is no ground truth here and there are no dark frames.
#
# ## Modelling decisions, stated up front
#
# | Decision | What this notebook does | Why, and what it costs |
# |---|---|---|
# | **Scope** | Two 647 nm recordings, inside the authors' masks. The record does not say how the two relate: same animal, junction or day is unknown | Between-sample variance cannot be estimated. **No lab-wide rule follows** |
# | **Rung** | Replay only. The plan's small rung, a base-16 network trained on a crop, was not built | Its pilot reached recall 0.80 and 1.86 x CRLB at E58's brightness (0.98 and 1.84 x at 2,000 photons), and a laptop's time is unmeasured. It would have taught the code without the result |
# | **Classical baseline** | E58's detection, Laplace fits and chosen K, with a 201-frame rolling background | The background falls during the movie, so a fixed one would be a poorly configured baseline. It also gives the fitter temporal context the network lacks: the network sees 3 frames, so long-lived bright objects stay in its input (section 10) |
# | **Network** | A DECODE-style U-Net written fresh from the paper in plain JAX: 3 frames in, per-pixel detection, position, photons and sds out, 471,784 weights | Per-pixel Gaussians are marginals. They are not a joint posterior over emitter sets |
# | **Network A** | 40,000 steps on E58's forward model: Gaussian PSF, flat background, independent frames | The expected failure, on purpose: it knows nothing of structured background or of blinks that last |
# | **Network B** | Warm start from A, 12,000 more steps on a simulator built from recording 7: stacked PSF, real background maps, persistence. The stack proved to be contaminated (section 7) | **Recording 32 played no part in the fit**, so it is the judged recording |
# | **Calibration** | Per block, per recording: the written action after P3 failed. A single line per recording is a labelled sensitivity run | The line is post-hoc, and no ruling has picked between them. Both are shown |
# | **Thresholds** | 10% on FRC and on count, fixed before any full result | Chosen as defaults by the analyst, not ruled on by the owner |
# | **Structure readout** | Connected components on a 10 nm render inside the mask, at least 7 events. Brp left out | **Our count is not the paper's.** HDBSCAN needs a dependency we do not have, and subclusters need the second channel |
# | **Training mode** | Deterministic GPU ops | The only setting under which a killed and resumed run matched exactly. It costs 3.4x in steps per second, like for like (64.3 s against 19.0 s for 3,000 steps) |
# | **Lines on the paired test** | Keep the absolute lines (pass 1.0-1.15, fail over 1.25) and add one relative to classical, set before any network result | The classical fitter fails the absolute line itself, so one alone would mislead |
#
# ## The plan
#
# 1. The movie, and what it costs
# 2. The first failure: a calibration that does not hold still
# 3. The classical answer: how many frames?
# 4. Is there anything for a network to find?
# 5. The instrument, and what it cost to build
# 6. A pass on simulation, and the certificate
# 7. Real frames: the four checks
# 8. A live audit in PyMC
# 9. The ledger: every gate, with its line
# 10. The second failure: drift
# 11. The third failure: a flattering split
# 12. The frames decision, all of it
# 13. What to tell the core

# %%
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")  # before NumPy: one BLAS thread per chain

import logging
import sys
import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import nutpie
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from IPython.display import HTML, Markdown, display
from matplotlib import animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter
from scipy import ndimage as ndi
from scipy import stats

from pymc_challenges import data
from pymc_challenges.bigrun import PreflightError, Store, banner, estimated, git_hash, measured

# ArviZ leaves a lazy stub under "jax", and Matplotlib's first plot tests `sys.modules.get("jax")`, which
# would import all of JAX for nothing. Nothing here uses JAX, so the stub goes.
sys.modules.pop("jax", None)

RANDOM_SEED = 88
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-variat")
logging.getLogger("pymc").setLevel(logging.WARNING)

LEVEL = os.environ.get("PYMC_CHALLENGES_RUN", "").strip().lower() or "replay"
if LEVEL != "replay":
    raise PreflightError(f"PYMC_CHALLENGES_RUN={LEVEL!r}: E88 has one level, replay. The heavy run is not a notebook "
                         "cell: it is tools/run_e88_full.py, and it is not for limited hardware.")

CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
PIXEL_NM, CYCLE_MS, FRAMES, SIDE, PATCH = 127.0, 11.742, 15000, 200, 11
RECS = {"r7": "recording 7", "r32": "recording 32"}
GRID = (2000, 3000, 4000, 5000, 6000, 7500, 9000, 10500, 12000, 13500, 15000)
CLASSICAL, NET_A, NET_B, SINGLE, MUTED, LINE_COL = "#0b6e8a", "#b2182b", "#d98200", "#5f6b73", "#9a9a9a", "#111111"
COLOUR = {"chosen": CLASSICAL, "K1": SINGLE, "a": NET_A, "b": NET_B}
NAME = {"chosen": "classical, chosen K", "K1": "classical, K = 1", "a": "network A", "b": "network B"}
DARK = plt.get_cmap("inferno").with_extremes(bad="black")



def markdown_table(frame):
    """A DataFrame as a Markdown table: pandas needs `tabulate` for this, and it is not a dependency here."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * frame.shape[1]]
    lines += ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in frame.to_numpy()]
    return Markdown("\n".join(lines))


store = Store("E88")
data.describe("dannhauser2022_e88_summary")
pack = np.load(data.path("dannhauser2022_e88_summary"), allow_pickle=False)
print(f"\nNotice carried in the file: {pack['attribution']}\n")
summary = store.load_json("summary")
certificate = store.load_json("certificate")
paired = store.load_json("paired_report")
halo = store.load_json("halo_report")
replicate = store.load_json("replicate_report")
audit = store.load_json("audit_nuts_report")
overlap_posthoc = store.load_json("p2b_overlap_posthoc")
posthoc = {name: store.load_json(f"posthoc_{name}") for name in ("sim", "real", "halo", "psf", "frames", "stuck", "nuts_cold", "nuts_warm")}
decisions = {"classical": store.load_json("p2b_decision"), "classical_line": store.load_json("p2b_decision_line")}
for net in ("a", "b"):
    for tail in ("", "_line", "_cdrift", "_line_cdrift"):
        decisions[net + tail] = store.load_json(f"decision_{net}{tail}")
started = time.perf_counter()
print(f"{CORES} cores here. Level: {LEVEL}. Written outcome of the stored run: {summary['outcome']}.")

# %%
display(Markdown(banner(
    level="replay",
    wall_time=measured("0.6-1.1", "min for this replay on 2 cores, warm and empty compile cache, builds of 2026-11-02"),
    peak_ram=measured("0.8-1.1", "GB, whole process tree, this replay, across builds"),
    peak_vram=measured(0, "GB in this run: every GPU number is stored"),
    disk=measured(1.7, "MB in data/, and about 0.5 MB of JSON in results/E88/"),
    download=measured(0, "MB for a reader. The stored run read 2.4 GB of TIFFs"),
    precision=measured("float32 in the stored networks, float64 here", ""),
    chains=measured(4, "chains on each of 4 live spots; 200 spots stored"),
    hardware=measured("Ryzen 5950X on 2 cores; stored GPU tables from an RTX 3090 Ti", ""),
    build_date=measured("2026-11-02", ""),
    git_hash=measured(git_hash() or "unknown", ""),
    smaller_machine=estimated("this replay is the small problem: no reader needs more", ""))))

# %% [markdown]
# ## 1. The movie, and what it costs
#
# **Source.** Mrestani (2022), *Example raw dSTORM data of Brp (Alexa Fluor532) and Unc-13 (Alexa Fluor647)*, Zenodo,
# doi:10.5281/zenodo.7328805, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper: Dannhäuser,
# Mrestani, Gundelach, Pauli, Komma, Kollmannsberger, Sauer, Heckmann & Paul (2022), *Endogenous tagging of Unc-13 reveals
# nanoscale reorganization at active zones during presynaptic homeostatic potentiation*, Frontiers in Cellular
# Neuroscience 16:1074304. **What we changed:** the two movies are reduced to a few raw frames and windows, statistics,
# masks on the 200 px grid and the output of our own analysis, and the lineups add simulated panels. No endorsement by
# the authors is implied. The record calls the files example data and numbers them. It does not say which animal, junction
# or condition each came from.
#
# dSTORM makes a few dye molecules blink on in each frame, finds each one to a few nanometres, and builds a
# picture from the positions. The left panels below are raw frames. The two large panels are built from the filtered
# localisations of each three-minute movie, 340,711 of them in recording 7, and the pale outlines are the authors' masks
# of the terminal boutons. The small box is the crop E58 used, which lies entirely outside the mask, so E58's companions
# and this notebook's numbers do not come from the same place.

# %%
seconds = FRAMES * CYCLE_MS / 1000
print(f"One movie: {FRAMES:,} frames x {CYCLE_MS} ms = {seconds:.1f} s on the camera (TIFF tags, measured).")
print(f"45 junctions per channel: {45 * seconds / 3600:.1f} h of scope time and {45 * 1.2098:.0f} GB "
      "(the 45 is the paper's 22 + 23 NMJs, one recording each is our assumption).")
for n in (10500, 12000):
    print(f"  At {n:,} frames: saves {45 * seconds * (1 - n / FRAMES) / 60:.0f} min and "
          f"{45 * 1.2098 * (1 - n / FRAMES):.0f} GB per channel [extrapolated, same assumption]")

frames_shown = pack["show_frames"]
fig = plt.figure(figsize=(13, 6.6), layout="constrained")
grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 2.05, 2.05])
vmin, vmax = np.percentile(pack["r7_frames"], [1, 99.9])
for row, f in enumerate(frames_shown):
    for col, rec in enumerate(RECS):
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(pack[f"{rec}_frames"][row], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{RECS[rec]}, frame {f:,}", fontsize=9)
        ax.axis("off")
for col, rec in enumerate(RECS):
    ax = fig.add_subplot(grid[:, 2 + col])
    render = np.ma.masked_equal(pack[f"{rec}_render_chosen"].astype(float), 0)
    ax.imshow(render, cmap=DARK, norm=LogNorm(1, 60), extent=(-0.5, 199.5, 199.5, -0.5), interpolation="nearest")
    ax.contour(pack[f"{rec}_mask"], levels=[0.5], colors="#9fe6ff", linewidths=0.8)
    if rec == "r7":
        ax.add_patch(Rectangle((49.5, 24.5), 40, 40, fill=False, ec="#55ff99", lw=1.3))
        ax.text(51, 22, "E58's crop", color="#55ff99", fontsize=9)
    ax.plot([155, 155 + 5000 / PIXEL_NM], [190, 190], color="w", lw=3)
    ax.text(155 + 2500 / PIXEL_NM, 186, "5 µm", color="w", ha="center", fontsize=9)
    ax.set_title(f"{RECS[rec]}: classical localisations and the mask", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black");

# %% [markdown]
# Two things to notice before any model runs. The two recordings are visibly different samples, with different
# coverage by their masks. And a mask is a small part of the frame: 12.8% and 7.2% of the pixels hold the
# structures the whole decision is about, so most of what the camera records never reaches the answer.
#
# The dye bleaches, and that is the reason a shorter movie might work at all. Here is how fast the signal runs
# out. The top axis gives camera seconds.

# %%
frames_at = (np.arange(30) + 0.5) * int(pack["block_frames"])
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), layout="constrained")
for rec, style in zip(RECS, ("-", "--")):
    per_frame = pack[f"{rec}_blobs_per_block"] / pack["block_frames"]
    in_mask = pack[f"{rec}_blobs_mask_per_block"] / pack["block_frames"]
    axes[0].plot(frames_at, per_frame, style, color=CLASSICAL, lw=2)
    axes[0].plot(frames_at, in_mask, style, color=NET_A, lw=2)
    axes[1].plot((np.arange(150) + 0.5) * 100, pack[f"{rec}_median_per_100"], style, color=SINGLE, lw=1.8)
    axes[1].text(450, pack[f"{rec}_median_per_100"][0] - 3, RECS[rec], fontsize=9, color=SINGLE, va="top")
    cum = np.cumsum(pack[f"{rec}_blobs_per_block"]) / pack[f"{rec}_blobs_per_block"].sum()
    print(f"{RECS[rec]}: {pack[f'{rec}_blobs_per_block'].sum():,} blobs, {pack[f'{rec}_blobs_mask_per_block'].sum():,} "
          f"in the mask ({pack[f'{rec}_blobs_mask_per_block'].sum() / pack[f'{rec}_blobs_per_block'].sum():.0%}); "
          f"first 5,000 frames hold {cum[9]:.0%}, first 10,500 hold {cum[20]:.0%}")
axes[0].text(8000, 26, "whole field", color=CLASSICAL, fontsize=10)
axes[0].text(8000, 7.5, "inside the mask", color=NET_A, fontsize=10)
axes[0].legend(handles=[Line2D([], [], color="k", ls="-", label="recording 7"),
                        Line2D([], [], color="k", ls="--", label="recording 32")], loc="upper right")
axes[0].set(xlabel="frame", ylabel="blobs per frame", ylim=(0, None))
axes[0].set_title("The blinks thin out to under half the starting rate", fontsize=11)
axes[1].set(xlabel="frame", ylabel="median pixel (ADU)")
axes[1].set_title("The background falls too, so a fixed one would be wrong", fontsize=11)
for ax in axes:
    top = ax.secondary_xaxis("top", functions=(lambda f: f * CYCLE_MS / 1000, lambda s: s * 1000 / CYCLE_MS))
    top.set_xlabel("seconds on the camera", fontsize=9);

# %% [markdown]
# The first third of each movie holds 43-44% of its blobs, and the first 10,500 frames about 77%. Nothing here says when
# structure stops gaining, because later blinks can fill in clusters that are still sparse. Only the decision curve can say that, and it needs
# calibrated, drift-corrected positions first.

# %% [markdown]
# ## 2. The first failure: a calibration that does not hold still
#
# A localiser needs photons, and the camera gives ADU. E58 fitted one offset and one gain from a photon-transfer
# curve on 2,000 frames. Gate P3 asked whether those hold along a full movie and between recordings: within 20
# ADU and 10% of the recording's median in every block, and between the two recordings.
#
# They do not. Stored, gated.

# %%
p3 = store.load_json("p3_calibration")
line = store.load_json("p3_line")
print(f"{'':14}{'offset moves':>14}{'slope moves':>14}{'line':>8}")
for rec in RECS:
    row = p3["recordings"][rec]
    print(f"{RECS[rec]:14}{row['offset_dev_max']:>10.0f} ADU{row['slope_rel_max']:>14.0%}{'20 ADU, 10%':>13}")
print(f"Between the two: {p3['between']['offset']:.0f} ADU, {p3['between']['slope_rel']:.0%}. Verdict: NO-GO, "
      "so the written action applies: calibrate per block and per recording.")
print("Post-hoc, one line per recording across blocks: " + "; ".join(
    f"{RECS[r]} slope {line[r]['slope']:.2f}, offset {line[r]['offset']:.0f}, r {line[r]['r']:.4f}" for r in RECS)
    + f". E58: slope {p3['e58']['slope']}, offset {p3['e58']['offset']:.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
for rec, marker in zip(RECS, ("o", "s")):
    row = p3["recordings"][rec]
    slopes, offsets = np.array(row["slopes"]), np.array(row["offsets"])
    dots = axes[0].scatter(slopes, offsets, c=frames_at, cmap="viridis", marker=marker, s=34, label=RECS[rec], lw=0)
    medians = pack[f"{rec}_median_per_100"].reshape(30, 5).mean(1)
    axes[1].plot(frames_at, (medians - offsets) / slopes, marker=marker, ms=3.5, color=CLASSICAL, lw=1.2,
                 label=f"{RECS[rec]}, per-block fit")
    axes[1].plot(frames_at, (medians - line[rec]["offset"]) / line[rec]["slope"], marker=marker, ms=3.5, color=NET_B,
                 lw=1.2, label=f"{RECS[rec]}, one line")
    axes[0].plot(line[rec]["slope"], line[rec]["offset"], marker=marker, ms=13, mfc="none", mec="k", mew=1.6)
    axes[0].annotate("post-hoc line", (line[rec]["slope"], line[rec]["offset"]), xytext=(8, -14),
                     textcoords="offset points", fontsize=8.5)
axes[0].plot(p3["e58"]["slope"], p3["e58"]["offset"], "*", ms=15, color=NET_A)
axes[0].annotate("E58", (p3["e58"]["slope"], p3["e58"]["offset"]), xytext=(9, 3), textcoords="offset points", color=NET_A)
fig.colorbar(dots, ax=axes[0], label="frame", shrink=0.8)
axes[0].legend(loc="upper left", fontsize=9)
axes[0].set(xlabel="fitted gain (ADU per count)", ylabel="fitted offset (ADU)")
axes[0].set_title("Block fits slide along one ridge as the movie runs", fontsize=11)
axes[1].legend(fontsize=8, loc="upper left", ncols=2)
axes[1].set(xlabel="frame", ylabel="background (counts per px)", ylim=(None, 86))
axes[1].set_title("Taken literally, the block fits make the background rise", fontsize=11);

# %% [markdown]
# Read the right panel. The raw background falls throughout the movie, yet the per-block calibration turns that
# into a background that climbs, which is not physically plausible. The left panel suggests why: each block's fit
# seems to trade offset against slope over a narrow range of pixel values. Across blocks, block mean against block
# variance falls on one straight line, with r near 0.9998, close to E58's numbers.
#
# **So there are two calibrations here, and neither is settled.** The gate failed and its written action was
# per-block fits, so per-block is the primary everywhere, because a gate cannot change. The one-line fit was found
# after the failure, so it is post-hoc and runs beside it as a sensitivity check. It is the more plausible of the
# two, and it matters later. No ruling has picked between them.

# %% [markdown]
# ## 3. The classical answer: how many frames?
#
# This is the result of the notebook, and it uses no network. E58's fitter ran over both movies: 367,071 and
# 245,335 blobs, about 3-4 minutes each on 13 CPUs. Rows with a posterior sd under 30 nm were kept, drift was
# corrected, gap-1 links were merged into events, and then the three tests were read at each frame count.
#
# The stored file has the verdict. We also recompute it here from the stored points, live, from the rule as
# written, so you can change the rule and see what moves.

# %%
def n_star(curve, allowance=0.10):
    """Smallest N on the grid where FRC, count and band agree with 15,000, there and at every larger N."""
    points, bands = curve["points"], curve["bands"]
    full, top = points["15000"], bands["15000"]

    def overlap(q, r):
        return q[0] <= r[1] and r[0] <= q[1]

    answer = FRAMES
    for N in reversed(GRID):
        p, q = points[str(N)], bands[str(N)]
        fine = (abs(p["frc_nm"] / full["frc_nm"] - 1) <= allowance and abs(p["count"] / full["count"] - 1) <= allowance
                and overlap(q["frc_q"], top["frc_q"]) and overlap(q["count_q"], top["count_q"]))
        if not fine:
            break
        answer = N
    return answer


def n_both(name, allowance=0.10):
    return max(n_star(decisions[name][rec], allowance) for rec in RECS)


for name in decisions:
    for rec in RECS:
        assert n_star(decisions[name][rec]) == decisions[name][rec]["N_star"], (name, rec)
print("Live rule matches every stored N*, across", len(decisions), "curve sets and both recordings.\n")
rows = []
for N in (2000, 5000, 7500, 9000, 10500, 12000, 15000):
    row = {"frames": N}
    for rec in RECS:
        p = decisions["classical"][rec]["points"][str(N)]
        c = decisions["classical"][rec]["checks"][str(N)]
        row[f"{rec} FRC nm"], row[f"{rec} clusters"], row[f"{rec} all three"] = round(p["frc_nm"], 1), int(p["count"]), c["all"]
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("\nN*, recording 7 / recording 32 / both, classical:")
print(f"  per block (written action, gated):  {decisions['classical']['r7']['N_star']:,} / "
      f"{decisions['classical']['r32']['N_star']:,} / {n_both('classical'):,}")
print(f"  one line (post-hoc sensitivity):    {decisions['classical_line']['r7']['N_star']:,} / "
      f"{decisions['classical_line']['r32']['N_star']:,} / {n_both('classical_line'):,}")
print("\nWhat the 10% allowance does to N for both, per block, then line:")
for allowance in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
    print(f"  {allowance:>5.1%}: {n_both('classical', allowance):>6,} {n_both('classical_line', allowance):>7,}")

frames_boot = posthoc["frames"]
print("\nPost-hoc, count rule only, classical, 100 block-bootstrap draws (200-frame blocks):")
for key, title in (("per_block", "per block"), ("line", "one line")):
    row = frames_boot["calibrations"][key]
    low, high = row["both_95"]
    print(f"  {title:10} N for both, 95% range {low:,.0f}-{high:,.0f}. Draws at each N: " + ", ".join(f"{int(n):,}: {k}" for n, k in row["both_counts"].items())
          + f". Recording 32 alone: {min(row['r32']['stars']):,}-{max(row['r32']['stars']):,}")
margins = {int(k): v["margin"] for k, v in frames_boot["band_seeds"].items()}
print("Band overlap at 10,500 of recording 7's count, in clusters, stored seed then five fresh families: "
      + ", ".join(f"{margins[k]:+.1f}" for k in (88, 1, 2, 3, 4, 5)) + ". A negative margin fails the test. FRC's band was not repeated.")
row = frames_boot["calibrations"]["per_block"]["r7"]
print(f"Recording 7, first N frames:           {row['points']}")
print(f"Same event count, thinned from all 15,000: {row['thinned']} (one thinning at each N)")
print(f"10,500 frames hold {row['events'][7] / row['events'][-1]:.0%} of recording 7's events")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.6), sharex=True, layout="constrained")
for col, rec in enumerate(RECS):
    for name, colour, label in (("classical", CLASSICAL, "per block, gated"), ("classical_line", NET_B, "one line, post-hoc")):
        curve = decisions[name][rec]
        frc = np.array([curve["points"][str(N)]["frc_nm"] for N in GRID])
        count = np.array([curve["points"][str(N)]["count"] for N in GRID])
        ok = np.array([curve["checks"][str(N)]["all"] for N in GRID])
        for row, y in enumerate((count, frc)):
            ax = axes[row, col]
            ax.plot(GRID, y, color=colour, lw=2, label=label)
            ax.scatter(np.array(GRID)[ok], y[ok], s=46, color=colour, zorder=3)
            ax.scatter(np.array(GRID)[~ok], y[~ok], s=46, facecolor="w", edgecolor=colour, zorder=3)
            ax.axhspan(y[-1] * 0.9, y[-1] * 1.1, color=colour, alpha=0.09, lw=0)
            ax.axvline(curve["N_star"], color=colour, lw=1.1, ls=":")
    axes[0, col].set_title(RECS[rec], fontsize=11)
    axes[1, col].set_xlabel("frames used (the first N of the movie)")
axes[0, 0].set_ylabel("clusters in the mask")
axes[1, 0].set_ylabel("FRC resolution (nm)")
axes[1, 0].set_ylim(25, 50)
axes[1, 1].set_ylim(25, 50)
axes[0, 0].legend(loc="lower right")
fig.suptitle("Count and its band decide, and FRC barely moves. Filled dot: all three tests pass there. Dotted line: N*. "
             "Shaded: 10% of the 15,000-frame value", fontsize=11);

# %% [markdown]
# **Reading it.** Under the written rule the answer is 10,500 frames, 30% fewer. Seven things stop that from being
# a clean number.
#
# - **FRC is no help.** At 2,000 frames it is already 32.8 nm in recording 7, against 35.3 nm at 15,000, so more frames
#   did not improve it. FRC cannot see this decision and should not be the headline here.
# - **The count never plateaus in recording 7.** It rises 9% from 10,500 to 15,000. N* does not come from the curve going
#   flat. A tighter allowance moves it up: 5% gives 12,000, or 13,500 under the line.
# - **Count and band set 10,500 jointly, narrowly.** At 9,000 both fail, and the count misses its 10% line by only 6
#   clusters. But 12.5-20% allowances leave N at 10,500 per block, because the band test still blocks it. At 10,500 the
#   two bands overlap by 11 clusters (900 against 889). In five fresh seed families the margin is +8 to +11 four times and
#   negative once, and a failure there would make N 12,000. So 12,000 is the safer end. This covers recording 7's count
#   band only.
# - **It moves with calibration.** Under the line, N for both is 12,000, and recording 32 drops from 7,500 to 5,000.
#   Hence "10,500-12,000". Those are two points, not an interval.
# - **Sampling noise is wider than that pair.** The bootstrap of the count rule gives 7,500-13,500 under either
#   calibration, with near-identical draws at each N, so this run does not show that calibration moves N at all. It is
#   a count-only, classical-only check with duplicated blocks, and it does not replace the gated value.
# - **The count measures events, not settling.** Thin all 15,000 frames at random to the same event count and you get
#   almost the same curve as the first N frames. Frame order carries almost nothing, so the rule says "about 77% of the
#   events". It does not show that structure has settled.
# - **The band and the points are different things.** Each bootstrap draw jitters events by their sd, adding error a
#   second time, so at 15,000 frames the band is 889-989 clusters and 45-56 nm against points of 801 and 35.3 nm. Test
#   3 compares bands with bands, so it stands, but the bands are not intervals around the points, and they are not
#   drawn here. They are not a posterior over resolution either: it is a time-block bootstrap with localisation
#   error propagated.

# %% [markdown]
# ## 4. Is there anything for a network to find?
#
# E58's headline was that overlapping emitters cost a classical pipeline recall: 0.758 even with Bayesian model
# selection at a density of 0.70 per µm². If real movies were that dense, a network might see what fitters merge.
# So gate P2b asked the question first, at no GPU cost: how often do blobs in the mask have a neighbour under 4 px?
# "Rare" was written as under 5% in both recordings, and rare meant stop before the full run.

# %%
over = {rec: decisions["classical"][rec]["overlap"] for rec in RECS}
rows = [("Written primary: 4-8 px annulus, scaled", [decisions["classical"]["overlap_primary"][r] for r in RECS], "gated", CLASSICAL),
        ("Detector, direct (merging hides most)", [over[r]["all"]["detector_under_4"] for r in RECS], "lower bound", SINGLE),
        ("Poisson at mask density", [over[r]["poisson_basis"]["share_under_4"] for r in RECS], "ignores clustering", SINGLE),
        ("Neighbour 1.5-4 px, frame 200 away", [overlap_posthoc[r]["gap200"]["px_1p5_to_4"] for r in RECS], "post-hoc", NET_B),
        ("Neighbour under 4 px, frame 200 away", [overlap_posthoc[r]["gap200"]["under_4"] for r in RECS], "post-hoc, upper side", NET_B),
        ("Two bright emitters in a fit", [over[r]["share_two_bright_1_4px"] for r in RECS], "inflated by misfit", MUTED),
        ("Fit chose K of 2 or more", [over[r]["share_K2plus"] for r in RECS], "inflated by misfit", MUTED)]
print(f"{'':42}{'rec 7':>8}{'rec 32':>8}  status")
for label, values, status, _ in rows:
    print(f"{label:42}{values[0]:>8.2%}{values[1]:>8.2%}  {status}")
dense = 1 - np.exp(-0.70 * np.pi * (4 * PIXEL_NM / 1000) ** 2)
print(f"\nE58's dense simulation, same Poisson sum at 0.70 per um^2: {dense:.0%} [extrapolated]")
assert decisions["classical"]["overlap_rare"], "the written verdict"
print("Written verdict: RARE, so the stop before FULL fired.")

fig, ax = plt.subplots(figsize=(11.5, 3.9), layout="constrained")
for i, (label, values, status, colour) in enumerate(rows[::-1]):
    ax.plot(values, [i, i], color=colour, lw=2.6, zorder=2)
    ax.scatter(values[0], i, s=60, color=colour, zorder=3)
    ax.scatter(values[1], i, s=60, facecolor="w", edgecolor=colour, lw=1.8, zorder=3)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=9.5)
ax.axvline(0.05, color=LINE_COL, lw=1.2)
ax.text(0.052, 0.45, "gate line, 5%", fontsize=9)
ax.axvline(dense, color=NET_A, lw=1.2, ls="--")
ax.text(dense * 0.97, 5.45, "E58's dense case", fontsize=9, color=NET_A, ha="right")
ax.set_xscale("log")
ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1], ["0.3%", "1%", "3%", "10%", "30%", "100%"])
ax.set(xlim=(0.003, 1))
ax.set_xlabel("share of in-mask blobs with a neighbour under 4 px (filled: recording 7, open: recording 32)", fontsize=10)
ax.set_title("The written measure says rare, a better one says just over the line, and either is far from dense", fontsize=11);

# %% [markdown]
# The gate says **rare** and stays rare. But the primary was always a lower bound, and it sat further below the truth
# than expected, because its annulus reaches outside narrow boutons where density falls. Post-hoc neighbours from
# another frame keep the spatial clustering and avoid detector merging, and they give about 5-9%. That is our best
# estimate, and it is an estimate and not a gate. The two fit-based rows read high because fits add emitters to
# absorb misfit: on a simulated control the same K measure read 18% with 11.5% of emitters truly near a neighbour, and
# 51% on the real crop. So they cannot be read as overlap.
#
# **What happened next.** The stop fired, along with P5, P6, P7 and the first P8'. The owner of the project looked
# at the pilots and ruled that the network should run in full anyway, because a tried-and-failed result is still
# useful. That ruling changed no gate. Every failed gate stayed failed, and the full run wrote its own gates first
# (`results/E88/full_gates.md`, in the repository as `full_gates.json`).

# %% [markdown]
# ## 5. The instrument, and what it cost to build
#
# The network is a DECODE-style U-Net (Speiser et al. 2021, *Nature Methods* 18:1082), written fresh from the paper,
# with no code copied. It takes 3 frames and predicts, per pixel, a detection probability, sub-pixel position,
# photons, background, and an sd for each. Training has no real data at all: a simulator inside the JIT draws
# emitters and camera noise, and the loss asks the network to recover them. So the network learns its simulator. The
# training code is in `tools/run_e88_net.py`; it needs a GPU and stays out of this notebook.
#
# Below is the bill. **Stored, wall times inflated** where marked: the job shared the machine with another
# notebook's run. Hatched bars are extrapolated.

# %%
train, infer = summary["train"], summary["infer"]
p4_gpu, p4_cpu = store.load_json("p4_gpu"), store.load_json("p4_cpu")
inference_s = sum(row["predict_seconds"] for net in infer.values() for row in net.values())
rates = sorted(row["frames_per_s"] for net in infer.values() for row in net.values())
classical_s = [store.load_json(f"classical_{r}_timing")["seconds"] for r in ("r7", "r32", "line_r7", "line_r32")]
ratio = p4_gpu["steps_per_s"] / p4_cpu["steps_per_s"]
print(f"Network A: {train['a']['steps']:,} steps in {train['a']['seconds']:.0f} s, {train['a']['steps_per_s']:.1f} steps/s, "
      f"{train['a']['vram']['peak_gb']:.2f} GB VRAM, {train['a']['weights']:,} weights, deterministic ops")
print(f"Network B: {train['b']['steps']:,} steps in {train['b']['seconds']:.0f} s, {train['b']['steps_per_s']:.1f} steps/s, "
      f"{train['b']['vram']['peak_gb']:.2f} GB VRAM, warm start from A")
print(f"Inference: 8 movies, 120,000 frames, {inference_s:.0f} s of prediction at {rates[0]:.0f}-{rates[-1]:.0f} frames/s, "
      f"{max(r['vram']['peak_gb'] for n in infer.values() for r in n.values()):.2f} GB VRAM")
print(f"Pilot P4, without deterministic ops: GPU {p4_gpu['steps_per_s']:.0f} steps/s on {p4_gpu['vram']['peak_gb']:.2f} GB, "
      f"CPU {p4_cpu['steps_per_s']:.2f} steps/s on 13 CPUs: {ratio:.0f}x. A pilot rate: the planned full CPU comparison was skipped.")
print(f"Classical: 4 runs in {sum(classical_s) / 60:.1f} min on 13 CPUs (one recording-7 run covers 28 of 30 blocks)")

bars = [("Train A, 40,000 steps, deterministic GPU", train["a"]["seconds"], False, NET_A),
        ("Train B, 12,000 steps, deterministic GPU", train["b"]["seconds"], False, NET_B),
        ("Train A at P4's GPU rate, not deterministic", 40000 / p4_gpu["steps_per_s"], True, NET_A),
        ("Train A at P4's CPU rate, 13 CPUs", 40000 / p4_cpu["steps_per_s"], True, NET_A),
        ("Use: both networks on both movies, both calibrations", inference_s, False, MUTED),
        ("Use: classical, recording 32 on 13 CPUs", classical_s[1], False, CLASSICAL),
        ("Camera time for one movie", seconds, False, SINGLE)]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), width_ratios=[1.5, 1], layout="constrained")
ax = axes[0]
for i, (label, value, extra, colour) in enumerate(bars[::-1]):
    ax.barh(i, value, color="none" if extra else colour, edgecolor=colour, hatch="///" if extra else None, lw=1.3)
    ax.text(value * 1.08, i, f"{value / 60:.1f} min" if value < 3600 else f"{value / 3600:.1f} h", va="center", fontsize=9)
ax.set_yticks(range(len(bars)), [b[0] for b in bars[::-1]], fontsize=9.3)
ax.set_xscale("log")
ax.set(xlim=(40, 60000))
ax.set_xlabel("seconds (log). Hatched: extrapolated from a measured rate", fontsize=10)
ax.set_title("Training is one bill, use is another", fontsize=11)
ax = axes[1]
ax.plot(train["a"]["log_steps"], train["a"]["log"], color=NET_A, lw=1.5, label="network A")
ax.plot(np.array(train["b"]["log_steps"]) + 40000, train["b"]["log"], color=NET_B, lw=1.5, label="network B, warm start")
ax.set(xlabel="training step", ylabel="loss at the logged step", ylim=(-45, 15))
ax.set_title("A's and B's losses, on different simulators", fontsize=11)
ax.legend(loc="upper right");

# %% [markdown]
# Three plain points from this bill.
#
# - **Training and use are different bills, paid by different people.** The GPU matters for training: 44x on the pilot's
#   rates without deterministic ops, and about 10.5x at the deterministic setting this run actually used, against the
#   same 13-CPU rate. The planned CPU run was skipped, so both are pilot rates. Use is cheap: 20-26 s to read one movie at batch 10. Whoever trains the network pays once,
#   and everyone who uses it inherits a certificate or the lack of one.
# - **The compute was never the hard part.** The training used under 0.3 GB of a 24 GB card, inference 2.5 GB, and the
#   GPU work adds up to about 21 minutes in these lines, and about 32 if all of validation counts [estimated]. The plan expected about 1 h 10 min. The series
#   was meant to show top-end compute, and E88 does not show it. Neither speed nor memory limited this result.
# - **Reproducibility cost more than memory did.** Two identical GPU runs had already differed by step 50, so the first
#   kill-and-resume test failed through non-determinism and not through the resume. Deterministic ops fixed that and
#   cost 3.4x in steps per second like for like (64.3 s against 19.0 s for the same 3,000 steps). The rest of the gap
#   between 49 steps/s and P4's 203 comes from different jobs on a shared machine. The owner has not ruled on this as the
#   published setting.

# %% [markdown]
# ## 6. A pass on simulation, and the certificate
#
# P5 asked for recall of at least 0.90 and RMSE within 1.3 x the CRLB at 5,000 steps. It missed narrowly at 1.38, and
# its written no-go said not to scale up. The full run did scale up, by the owner's ruling. Here is what more steps
# bought, on P5's own held-out movie. **Stored.** The 5,000-step no-go stays on the record whatever these show.

# %%
p5 = {"5,000 steps (P5, NO-GO)": store.load_json("p5_p5"), "A, 40,000 steps": store.load_json("p5_a"),
      "B, 12,000 more": store.load_json("p5_b"), "small, 1,500 steps": store.load_json("p5_small")}
rows = []
for name, r in p5.items():
    sparse, bright, dense_run = r["sparse_2000"], r["sparse_e58_brightness"], r["dense_e58_brightness"]
    rows.append({"network": name, "recall": sparse["tol2.0"]["recall"], "x CRLB": sparse["rmse_over_crlb"],
                 "median sd px": sparse["median_sd_px"], "recall, E58 brightness": bright["tol2.0"]["recall"],
                 "x CRLB, E58 brightness": bright["rmse_over_crlb"], "recall, dense": dense_run["tol2.0"]["recall"],
                 "x CRLB, dense": dense_run["rmse_over_crlb"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"\nLines at 2,000 photons, sparse: recall 0.90, 1.3 x CRLB. CRLB here {p5['A, 40,000 steps']['crlb_px']:.3f} px, "
      f"{p5['A, 40,000 steps']['crlb_nm']:.1f} nm.")


def cells(net, world, field):
    table = [r for r in certificate["simulation"][net] if r["world"] == world]
    photons, densities = sorted({r["photons"] for r in table}), sorted({r["density"] for r in table})
    out = np.full((len(photons), len(densities)), np.nan)
    for r in table:
        out[photons.index(r["photons"]), densities.index(r["density"])] = r[field]
    return out, photons, densities


def near_nominal(net, world):
    table = [r for r in certificate["simulation"][net] if r["world"] == world and r["in_prior"]]
    return sum(0.90 <= r["coverage95"] <= 0.98 for r in table), len(table)


fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, sharey=True, layout="constrained")
views = (("gauss", "coverage95", "95% coverage, simple world"), ("empirical", "coverage95", "95% coverage, empirical world"),
         ("empirical", "precision", "precision, empirical world"))
for row, net in enumerate(("a", "b")):
    for col, (world, field, title) in enumerate(views):
        ax = axes[row, col]
        values, photons, densities = cells(net, world, field)
        norm = TwoSlopeNorm(0.95, 0.40, 1.0)
        mesh = ax.pcolormesh(np.arange(6), np.arange(5), values, cmap="RdBu", norm=norm, edgecolor="w", lw=1.2)
        for i in range(4):
            for j in range(5):
                ax.text(j + 0.5, i + 0.5, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="w" if abs(values[i, j] - 0.95) > 0.28 else "k")
        ax.add_patch(Rectangle((4, 0), 1, 4, fill=False, hatch="xxx", ec="#555555", lw=0))
        ax.set_title(f"{NAME[net]}: {title}", fontsize=10)
        ax.set_xticks(np.arange(5) + 0.5, [f"{d:g}" for d in densities])
        ax.set_yticks(np.arange(4) + 0.5, [f"{p:,.0f}" for p in photons])
    axes[row, 0].set_ylabel("photons per emitter")
fig.supxlabel("emitters per µm² per frame. Hatched: outside the training prior", fontsize=10.5)
fig.colorbar(mesh, ax=axes, shrink=0.6, label="value (white is 0.95)")
fig.suptitle("Simulation only. Coverage is closest to nominal in the world nearest each network's training", fontsize=11)
for net in ("a", "b"):
    for world in ("gauss", "empirical"):
        print(f"{NAME[net]} in the {world} world: {near_nominal(net, world)[0]} of {near_nominal(net, world)[1]} "
              "cells in the prior have 95% coverage within 0.90-0.98")
emp_a = [r for r in certificate["simulation"]["a"] if r["world"] == "empirical" and r["in_prior"]]
emp_b = [r for r in certificate["simulation"]["b"] if r["world"] == "empirical" and r["in_prior"]]
print(f"Empirical world, in the prior: A precision under 0.9 in {sum(r['precision'] < 0.9 for r in emp_a)} of 16, lowest "
      f"{min(r['precision'] for r in emp_a):.2f}; B in {sum(r['precision'] < 0.9 for r in emp_b)} of 16, with recall under "
      f"0.9 in {sum(r['recall'] < 0.9 for r in emp_b)}")
own = [r for r in certificate["simulation"]["a"] if r["world"] == "gauss" and r["density"] == 0.7 and r["photons"] in (2000, 4000)]
print(f"A in its own world at 0.70 per um^2, 2,000-4,000 photons: {min(r['rmse_over_crlb'] for r in own):.1f}-"
      f"{max(r['rmse_over_crlb'] for r in own):.1f} x CRLB");

# %% [markdown]
# Taken alone, the table is a success: at 40,000 steps both networks are inside P5's lines. That is the trap.
#
# **A simulation pass says the network learned the simulator.** A was tested on the world that trained it, so
# its coverage looks fine. Move it to B's world, with a heavier stacked PSF and persistence, still a simulation, and its
# precision falls as far as 0.42, so most of its detections there match nothing. B repairs much of that in the
# empirical world, with precision under 0.9 in 4 cells against A's 11, at some cost in A's world and in recall. Neither
# result touches the microscope.
#
# P5's sd note matters too. The 5,000-step network predicted a median sd (0.050 px) below the CRLB (0.061 px), which an
# unbiased estimator cannot sustain. At 40,000 steps A's median sd sits near the CRLB and RMSE is 1.16 x, so its error
# bars look fine on this movie. The paired test in section 7 asks whether that lasts on real frames.

# %% [markdown]
# ## 7. Real frames: the four checks
#
# Here is the part no training loss can see. Each check is **stored**, all but check 3 are **gated**, and recording 32, which played no
# part in training, is the judged one. All are inside the mask.
#
# ### The stack B was built on
#
# B's fix starts here. Stacking 20,000 isolated bright spots from recording 7 gave the point spread function (PSF) B was
# trained on. As stored, it puts a quarter of the light beyond 3 px. A post-hoc control shows that most of that is not PSF:
# repeat the same stack where the emitter has gone, and subtract.

# %%
psf = store.load_json("psf_bg")
knots, profile = np.array(psf["knots_px"]), np.array(psf["profile"])
sigma = 1.05
gauss = np.exp(-knots**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
print(f"{psf['spots']:,} spots of {psf['pool']:,}: {psf['share_beyond_3px']:.1%} of energy beyond 3 px against "
      f"{psf['share_beyond_3px_gauss_105']:.1%} for a {sigma} px Gaussian; peak {psf['peak']:.3f} against "
      f"{psf['peak_gauss_105']:.3f}; sd of log width {psf['log_w_sd']:.3f}, which includes fit noise")
psf_control = posthoc["psf"]
print(f"Post-hoc control on the same {psf_control['spots']:,} spots. Stack: light over N {psf_control['stack_integral']:.2f}, "
      f"{psf_control['stack_share_signed']:.1%} beyond 3 px counted out to 9 px")
for name, title in (("after_100", "same sites, 100 frames away"), ("after_300", "same sites, 300 frames away"), ("random_sites", "random sites, same frames")):
    row = psf_control["controls"][name]
    print(f"  {title:28} leaves {row['integral_control']:.2f} of N, flat {row['pedestal_3_to_8px']:.4f} per px² at 3-8 px, core {row['core_under_1px']:.4f}. "
          f"Stack minus this: light over N {row['integral_difference']:.2f}, {row['share_difference']:.1%} beyond 3 px, peak {row['peak_difference']:.3f}")
width = psf_control["width"]
print(f"Width scale of a spot: median {width['stored_rule_median']:.2f}, log sd {width['stored_rule_log_sd']:.3f} with the stored fit; "
      f"{width['amplitude_free_median']:.2f}, {width['amplitude_free_log_sd']:.3f} with a free amplitude")
centres = (np.arange(len(psf_control["raw"])) + 0.5) * psf_control["knots_px"][1]
after = psf_control["controls"]["after_100"]
clean = np.array(after["difference"]) / after["integral_difference"]
fig, ax = plt.subplots(figsize=(8.6, 3.8), layout="constrained")
ax.fill_between(knots * PIXEL_NM, np.maximum(pack["psf_band"][0], 1e-6), pack["psf_band"][1], color=NET_B, alpha=0.35, lw=0)
ax.plot(knots * PIXEL_NM, np.maximum(profile, 1e-6), color=NET_B, lw=2, label="the stored stack, which B was trained on")
ax.plot(centres * PIXEL_NM, np.where(clean > 0, clean, np.nan), color=CLASSICAL, lw=2, label="stack minus the same sites 100 frames away, post-hoc")
ax.plot(knots * PIXEL_NM, gauss, color=NET_A, lw=2, ls="--", label="what A was trained on: Gaussian, 1.05 px")
ax.axvline(9 * PIXEL_NM, color=MUTED, lw=1, ls=":")
ax.text(9 * PIXEL_NM + 12, 0.002, "wings past here\nare not measured", fontsize=9, color=SINGLE)
ax.set_yscale("log")
ax.set(ylim=(1e-5, 0.3), xlim=(0, 1500), xlabel="distance from the spot centre (nm)", ylabel="share of photons per px²")
ax.set_title("Most of the stack's wings stay when the spot has gone, so they are not PSF", fontsize=11)
ax.legend(loc="upper right", fontsize=9);

# %% [markdown]
# **So the wing is mostly a contaminant.** The stack takes the mean of each pixel and subtracts a median, and a mean sits
# above a median here. Random sites leave a flat pedestal holding 0.23 of the photon count, and the sites themselves
# add about 0.09 more, because bright spots sit where other molecules and long-lived objects sit. Once that is subtracted
# the light sums to 1.02 of N, so classical photons are about right, and 5-6% lies beyond 3 px (10% against random
# sites), against 2.3% for the Gaussian. The real PSF has some wing, and far less than a quarter.
#
# B inherited the stored stack. Its simulator moves about 15-20% of each emitter's light into wings that are not PSF, and
# lowers the peak to 0.097 from about 0.11. **So B tests a contaminated stack, not "the real PSF".** Whether that explains
# B's z, or its recall cliff at 500 photons, was not measured. The stored fit's width of 0.84 is also an artefact of a fixed
# amplitude: freed, it is 0.96, which supports B's centre of 1.
#
# ### Check 1: the paired-frame test
#
# How do you test error bars with no ground truth? A molecule that stays on across frames gives two estimates of one
# point. If the sds are honest, the difference divided by its predicted sd is a standard normal. Links are made at a
# gap of 3 frames, because the network reads 3 frames at a time and a gap of 1 would share pixels. A mixture on the disc
# allows for false links, and every method is matched to classical chosen K on brightness, because z depends on it.
#
# Lines written beforehand: pass within 1.0-1.15, failure over 1.25, and between them undefined. A second line: "no
# worse than classical" if z is at or below classical z on the same brightness-matched set. That set is matched on
# brightness only: each method keeps its own pairs. **This test needs no ground truth,
# and it cannot see a bias common to both frames**, so a pass would be necessary, not sufficient.

# %%
verdicts = paired["verdicts"]
rows = []
for key, title in (("r32", "recording 32, judged"), ("r7", "recording 7"), ("r32_line", "recording 32, line calibration"),
                   ("r7_line", "recording 7, line calibration")):
    rows.append({"where": title, "A": verdicts[f"{key}_a"]["z"], "A 95% half": verdicts[f"{key}_a"]["halfwidth95"],
                 "A 1 se": verdicts[f"{key}_a"]["one_se"], "B": verdicts[f"{key}_b"]["z"],
                 "B 95% half": verdicts[f"{key}_b"]["halfwidth95"], "B 1 se": verdicts[f"{key}_b"]["one_se"],
                 "classical": verdicts[f"{key}_a"]["classical_z"], "K = 1": paired["rows"][key]["K1"]["gap3_R1.5"]["matched_chosen"]["z_sd"],
                 "K = 1, own basis": verdicts[f"{key}_a"]["k1_z"],
                 "absolute": f"{verdicts[f'{key}_a']['absolute']}, {verdicts[f'{key}_b']['absolute']}",
                 "relative": verdicts[f"{key}_a"]["relative"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
sim = paired["rows"]["simulated"]
print("\nSimulated baseline, gap 3, unweighted, whole field:",
      ", ".join(f"{NAME[m]} {sim[m]['gap3_R1.5']['unweighted']['z_sd']:.2f}" for m in ("a", "b", "chosen", "K1")))
print("It tests the method. It says nothing of the microscope.")
truth = posthoc["sim"]
print("Post-hoc, pairs with both ends on a true emitter: " + ", ".join(f"{NAME[m]} {truth[m]['z_both_true']:.2f}" for m in ("a", "b", "chosen", "K1")))
print("Share of rows within 2 px of a true emitter: " + ", ".join(f"{NAME[m]} {truth[m]['precision']:.2f}" for m in ("a", "b", "chosen", "K1"))
      + f". A's gap-3 pairs: {truth['a']['pairs']:,}, of which {truth['a']['pairs_neither']:,} join two false rows, yet the mixture calls {truth['a']['mixture_true_share']:.0%} true")
print(f"Network B at gap 1 on simulation: z {sim['b']['gap1_R1.5']['unweighted']['z_sd']:.2f}, where its 3-frame input shares pixels")

extra = posthoc["real"]
print("\nPost-hoc, the same links read other ways. Gated values above are unchanged.")
rows = []
for rec in ("r32", "r7"):
    common, floor = extra[rec]["common"], extra[rec]["floor"]
    for m in ("chosen", "K1", "a", "b"):
        row = paired["rows"][rec][m]["gap3_R1.5"]
        rows.append({"recording": RECS[rec], "method": NAME[m], "own pairs": row["pairs"], "matched z": row["matched_chosen"]["z_sd"], "matched to K = 1": row["matched_K1"]["z_sd"],
                     "unweighted z": row["unweighted"]["z_sd"], "unw. upper": row["unweighted"]["z_sd_hi"], "common z": common["z"][m], "common scatter nm": common["scatter_nm"][m],
                     "z tight / mid / wide sd": " / ".join(f"{v:.2f}" for v in extra[rec]["sd_tertiles"][m]), "residual scale": floor[m]["c"], "floor nm per frame": floor[m]["floor_nm"] / np.sqrt(2)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"Common pairs: {extra['r32']['common']['pairs']:,} on recording 32, {extra['r7']['common']['pairs']:,} on recording 7. Scatter by gap 1 / 3 / 10, K = 1, recording 32: "
      + " / ".join(f"{extra['r32']['by_gap_scatter_nm']['K1'][g]:.1f}" for g in ("1", "3", "10")) + " nm")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), layout="constrained")
ax = axes[0]
place = {"sim": 3, "r7": 2, "r32": 1}
offsets = {"a": 0.2, "b": 0.0, "chosen": -0.2}
for m, shift in offsets.items():
    judged, other = paired["rows"]["r32"][m]["gap3_R1.5"], paired["rows"]["r7"][m]["gap3_R1.5"]
    for y, row in ((place["r32"], judged), (place["r7"], other)):
        q = row["matched_chosen"]
        ax.errorbar(q["z_sd"], y + shift, xerr=[[q["z_sd"] - q["z_sd_lo"]], [q["z_sd_hi"] - q["z_sd"]]], fmt="o", ms=7,
                    color=COLOUR[m], capsize=3)
    ax.scatter(posthoc["sim"][m]["z_both_true"], place["sim"] + shift, s=48, marker="D", color=COLOUR[m])
ax.axvspan(1.0, 1.15, color="#2a7f62", alpha=0.15, lw=0)
ax.axvspan(1.25, 1.75, color=NET_A, alpha=0.07, lw=0)
ax.axvline(1.25, color=LINE_COL, lw=1.1)
ax.text(1.003, 3.42, "pass", fontsize=9, color="#2a7f62")
ax.text(1.262, 3.42, "failure", fontsize=9, color=NET_A)
ax.set_yticks([1, 2, 3], ["recording 32\n(judged)", "recording 7", "simulated,\ntrue pairs"])
ax.set(xlim=(0.98, 1.75), ylim=(0.6, 3.6))
ax.set_xlabel("z sd (1.0 means honest error bars)", fontsize=10)
ax.set_title("Simulation passes, real frames fail", fontsize=11)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COLOUR[m], label=NAME[m]) for m in offsets], loc="center right",
          bbox_to_anchor=(1.0, 0.72), fontsize=8.5)

ax = axes[1]
q = pack["z_quantiles"]
grid_z = np.linspace(-4.5, 4.5, 200)
for m in ("a", "b", "chosen"):
    ax.plot(pack[f"r32_z_{m}"], q, color=COLOUR[m], lw=2)
ax.plot(pack["sim_z_a"], q, color=NET_A, lw=1.2, ls=":", label="network A on simulation, false rows in")
ax.plot(grid_z, stats.norm.cdf(grid_z), color=LINE_COL, lw=1.3, label="honest: N(0, 1)")
ax.plot(grid_z, stats.norm.cdf(grid_z, scale=1.25), color=LINE_COL, lw=1.1, ls="--", label="the failure line: sd 1.25")
ax.set(xlim=(-4.5, 4.5), ylim=(0, 1), ylabel="cumulative share")
ax.set_xlabel("z on recording 32, gap 3 (raw, false links included)", fontsize=10)
ax.set_title("Real curves sit outside the failure line", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[2]
radii = (0.75, 1.0, 1.5, 2.0, 3.0)
for m in ("a", "b", "chosen"):
    z = [paired["rows"]["r32"][m][f"gap3_R{r}"]["matched_chosen"]["z_sd"] for r in radii]
    ax.plot(radii, z, "o-", color=COLOUR[m], lw=2)
    gap1 = paired["rows"]["r32"][m]["gap1_R1.5"]["matched_chosen"]["z_sd"]
    ax.scatter(1.5, gap1, marker="x", s=70, color=COLOUR[m], zorder=4)
ax.axhline(1.25, color=LINE_COL, lw=1.1)
ax.axvline(1.5, color=MUTED, lw=1, ls=":")
ax.text(1.53, 1.21, "written radius", fontsize=8.5, color=SINGLE)
ax.set(xlabel="link radius (px)", ylabel="z sd, recording 32")
ax.set_title("No radius rescues it. x marks gap 1", fontsize=11);

# %% [markdown]
# **Read this as a failure on every line that counts.** On the judged recording both networks are over 1.25, and the
# half-widths of about 0.03 leave no doubt. Recording 7, which B was built from, is also over the line for both, though A
# only just: its matched 1.267 has a lower bound of 1.253, and its unweighted z is 1.235 with an upper bound of 1.249, under
# the line. The matched value is the gated one, so the failure stands. Calibration does not rescue them: the line gives
# 1.39 and 1.61.
#
# **The simulated baseline needs care.** Taken raw it is 1.02-1.13, and A's 1.02 flatters. Half of A's rows on that movie
# match no emitter, and 34,028 of its 43,555 pairs join two false rows, stuck at fixed sites, that agree with themselves. The
# mixture still calls 97% of pairs true, because it cannot detect a false link at a fixed site. On pairs with both ends true,
# z is 1.06-1.13 for all four methods, so the method does work and what differs is the microscope. That is the value plotted.
# B's baseline also sits outside its own simulated world, and B's z of 0.59 at gap 1 shows what shared pixels can do.
#
# The relative line says "no worse than classical". True, and it needs care. Each method keeps its own pairs here:
# 8,187 for classical, 10,360 for A and 12,333 for B on recording 32. They are matched on brightness only. On that footing K
# = 1 is 1.43, which A (1.40) cannot be told from, and the 1.49 printed as "own basis" is K = 1 matched to itself. Matched to
# K = 1 instead, B is 1.69 against 1.49, which would read "worse" on the relative line. **B ties classical only on chosen K's
# basis.** On literally the same pairs, 4,337 on recording 32, z is chosen K 1.55, K = 1 1.51, A 1.44 and B 1.76, and A's
# scatter equals K = 1's at 12.0 nm. All of this is post-hoc, and the gated relative verdict stands.
#
# **A z of 1.4 does not mean the real scatter is 1.4 x the sd.** z falls as the sd widens, in every method: on recording 32,
# K = 1 goes 1.99, 1.41, 1.29 from the tightest third to the widest. That is the shape of a floor. A fit with a floor and a
# scale gives about 4-6 nm per frame on top of each sd in all four methods, so it looks like sample or stage and not any
# localiser, plus a residual scale of 1.1-1.2 for K = 1 and A and 1.3-1.5 for B and chosen K. Scatter also grows with the
# gap, so part of the floor is movement over time. **Advice: add a floor of a few nanometres before you scale any sd.**
#
# **B got worse than A,** for two different reasons. The fix took z from 1.40 to 1.63. On recording 32 B's scatter rose 20%
# over A's on common pairs. On recording 7 it rose 2% while B's sds shrank. The radius curve keeps rising for every method on
# recording 32, so real displacements have heavier tails than the mixture allows, and no fixed radius cures that there. On
# recording 7, A dips under 1.25 at radii of 0.75 and 1 px. Gap 1 looks better for the networks, 1.30 and 1.49, which is the
# trap the gap of 3 avoids, since at gap 1 they share frames. Classical also falls at gap 1, to 1.51, though its fits are
# independent, so shared pixels are not the whole story.
#
# **Recording 7 needs one more warning.** Long-lived bright objects, section 10, touch 45% of A's pairs and 42% of B's there.
# Without them z rises: unweighted A 1.23 to 1.34, B 1.38 to 1.55. Recording 32 moves less: A barely, B about 60% as much. So piles do not rescue any
# real z, and they mean recording 7's z mixes a pile population into the number.
#
# ### Check 2: the companion halo
#
# E58 found faint companions at 301-462 nm from bright spots: fitters explaining PSF wings with extra emitters. Do
# networks invent the same ring? Density of faint detections around bright ones, against distance. Failure:
# ring density over twice the far-field density.

# %%
print(f"{'halo ratio, line 2':22}{'rec 32':>9}{'rec 7':>9}{'rec 32 shifted':>16}{'rec 7 shifted':>15}")
for m in ("a", "b", "chosen"):
    print(f"{NAME[m]:22}{halo['r32'][m]['ratio']:>9.2f}{halo['r7'][m]['ratio']:>9.2f}{posthoc['halo']['r32'][m]['shifted']:>16.2f}{posthoc['halo']['r7'][m]['shifted']:>15.2f}")
print(f"{NAME['K1']:22}{posthoc['halo']['r32']['K1']['ratio']:>9.2f}{posthoc['halo']['r7']['K1']['ratio']:>9.2f}{posthoc['halo']['r32']['K1']['shifted']:>16.2f}{posthoc['halo']['r7']['K1']['shifted']:>15.2f}")
print("Shifted, post-hoc: faint rows taken from frames 200-800 away, so nothing in them can belong to the bright spot")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True, layout="constrained")
for ax, rec in zip(axes, ("r32", "r7")):
    for m in ("chosen", "a", "b"):
        row = halo[rec][m]
        edges = np.array(row["edges_nm"])
        ax.step(edges[:-1], np.maximum(row["density"], 1e-3), where="post", color=COLOUR[m], lw=2,
                label=f"{NAME[m]}: {row['ratio']:.1f} x far field")
        ax.hlines(row["density_far"], 1000, 1500, color=COLOUR[m], lw=1.2, ls="--")
    ax.axvspan(301, 462, color="#555555", alpha=0.13, lw=0)
    ax.set_yscale("log")
    ax.set(xlabel="distance from a bright detection (nm)", ylim=(1e-3, 30), title=RECS[rec] + (" (judged)" if rec == "r32" else ""))
    ax.legend(loc="upper right", fontsize=9)
axes[0].text(310, 0.0016, "E58's ring", fontsize=9)
axes[0].set_ylabel("faint detections per bright one, per µm²");

# %% [markdown]
# A fails on both recordings at about 5 x. B halves that: 2.69 on the judged recording, still a failure, and 1.95 on
# recording 7, a pass. Of all B's changes this is the one that did what it was built for, though the simulator changed
# several things at once, so the run cannot credit the PSF alone. Classical chosen K sits at 37-39, but that is not like
# for like: rows from K of 2 or more sit inside one blob by construction. K = 1 gives 0.01 and 0.00, which is no fairer:
# one row per blob is blindness inside a blob, not honesty. So this ratio has no fair classical comparator.
#
# Shifting the faint rows in time, post-hoc, gives 0.98-1.27 for the networks, so clustering of the sample does not
# inflate the gate.
#
# Look where the peaks are. Classical peaks inside E58's ring. A's excess peaks further out, at about 500 nm, so the
# written ring only catches its shoulder and 5.4 understates it. The ring was fixed in advance, so it stands.
#
# ### Check 3: the population replicate, and a lineup
#
# Simulate fresh frames from the fitted *population* of each method (density, brightness, PSF, background), never from
# the frames' own fitted emitters, which would look good by construction. Compare blobs. No line was written, so
# none is invented here. Before reading on, try the lineup: **one panel of nine is real.**

# %%
fig, axes = plt.subplots(3, 7, figsize=(12.5, 5.5), width_ratios=[1, 1, 1, 0.14, 1, 1, 1], layout="constrained")
for col in range(3):
    axes[col, 3].axis("off")
for offset, net in zip((0, 4), ("a", "b")):
    panels = pack[f"lineup_r32_{net}"]
    low, high = np.percentile(panels, [1, 99.7])
    for i in range(9):
        ax = axes[i // 3, offset + i % 3]
        ax.imshow(panels[i], cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(1.2, 5, str(i + 1), color="#ffd166", fontsize=10, weight="bold")
    axes[0, offset + 1].set_title(f"Lineup for {NAME[net]}, recording 32", fontsize=11);

# %% [markdown]
# The reveal is in the next cell. Decide first: which panel is real in each lineup, and how sure are you? Both lineups
# hold the same real frame, so comparing the two sides is a hint.

# %%
for net in ("a", "b"):
    print(f"{NAME[net]}: the real panel is number {int(pack[f'lineup_r32_{net}_real']) + 1}")
print()
rows = []
for rec in RECS:
    for net in ("a", "b"):
        r = replicate[rec][net]
        rows.append({"recording": RECS[rec], "network": NAME[net], "KS area": r["ks_area"], "KS peak": r["ks_peak"],
                     "blobs real": r["blobs_real"], "blobs replicate": r["blobs_replicate"],
                     "median area real, rep": tuple(r["median_area"]), "median peak real, rep": tuple(r["median_peak"])})
print(pd.DataFrame(rows).to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5), sharey=True, layout="constrained")
for ax, rec in zip(axes, ("r32", "r7")):
    real = np.array(replicate[rec]["a"]["profile_real"])
    px = np.arange(len(real))
    ax.plot(px, np.maximum(real, 1e-4), color="k", lw=2.4, label="real blobs")
    for net in ("a", "b"):
        ax.plot(px, np.maximum(replicate[rec][net]["profile_replicate"], 1e-4), color=COLOUR[net], lw=2, label=f"replicate from {NAME[net]}")
    ax.set_yscale("log")
    ax.set(xlabel="distance from the blob centre (px)", title=RECS[rec] + (" (judged)" if rec == "r32" else ""))
    ax.legend(fontsize=9)
axes[0].set_ylabel("mean height, peak = 1");

# %% [markdown]
# A's population gives far too many small blobs: more than twice the real count on recording 32, with a median area of 14
# px against 28. Its profile falls too fast at 3-7 px. B is closer on both KS distances, and on recording 32 its blobs
# are still too many, too small and too faint. **So the simple simulator fails a check you can run without any labels, and the empirical
# one improves on it without reaching the data.** These are descriptive: no line was written, and no verdict is given.
#
# ### Check 4: the NUTS audit
#
# Does the Laplace approximation behind the classical sds hold? NUTS on E58's spot model, 200 spots from the
# recording-7 mask, half with chosen K = 1 and half with K = 2. It shares the forward model with the fit, so it
# checks the approximation and not the model, and it cannot replace check 1. Line: median sd ratio within 0.8-1.25.

# %%
K, rhat, nuts_sd, laplace_sd = pack["audit_K"], pack["audit_rhat"], pack["audit_nuts_sd_px"], pack["audit_laplace_sd_px"]
good = rhat <= 1.01
print(f"Median NUTS / Laplace sd, line 0.8-1.25: {audit['median_sd_ratio']['all']:.2f} (K = 1 {audit['median_sd_ratio']['K1']:.2f}, "
      f"K = 2 {audit['median_sd_ratio']['K2']:.2f}). Converged only, descriptive: {audit['median_sd_ratio']['converged']:.2f} "
      f"(K = 1 {audit['median_sd_ratio']['K1_converged']:.2f}, K = 2 {audit['median_sd_ratio']['K2_converged']:.2f})")
print(f"P7's lines on 200 spots: max r_hat {audit['max_rhat']:.2f} (line 1.01), {audit['over_1_01']['K1'] + audit['over_1_01']['K2']} spots over "
      f"({audit['over_1_01']['K1']} K = 1, {audit['over_1_01']['K2']} K = 2), {audit['divergences']:,} divergences (line 0), minimum ESS {audit['min_ess']:.1f}")
print(f"Against the networks: matched {audit['nets']['a']['matched']} (A) and {audit['nets']['b']['matched']} (B) of "
      f"{audit['nets']['a']['of']} positions, median distance {audit['nets']['a']['median_distance_nm']:.1f} and "
      f"{audit['nets']['b']['median_distance_nm']:.1f} nm, median NUTS sd over network sd "
      f"{audit['nets']['a']['median_nuts_over_net_sd']:.2f} and {audit['nets']['b']['median_nuts_over_net_sd']:.2f}, all spots")
cold, warm, stuck_chains = posthoc["nuts_cold"], posthoc["nuts_warm"], posthoc["stuck"]
print("\nPost-hoc rerun, same posterior with the start on the spot, same seeds and spots, empty compile cache then warm:")
for tag, r in (("empty", cold), ("warm", warm)):
    print(f"  {tag:6} ratio {r['median_sd_ratio']['all']:.2f} (K = 1 {r['median_sd_ratio']['K1']:.2f}, K = 2 {r['median_sd_ratio']['K2']:.2f}; K = 2 converged "
          f"{r['median_sd_ratio']['K2_converged']:.2f}); over r_hat 1.01: K = 1 {r['over_1_01']['K1']}, K = 2 {r['over_1_01']['K2']}; "
          f"divergences {r['divergences']['all']:,} (K = 1 {r['divergences']['K1']}); chain gap on the unconverged, median {r['chain_gap_unconverged']['median']:.1f} nats")
print(f"  K = 1: the fitter's row sits within 0.5 px of the NUTS mean on {cold['K1_fit']['within_half_px']} of {cold['K1_fit']['of']} spots, photon ratio {cold['K1_fit']['median_N_ratio']:.2f}")
print(f"Stored form rerun on the 87 stuck spots: a gap of 20 nats or more on {stuck_chains['gap_20']['K1']} of {stuck_chains['of']['K1']} K = 1 and "
      f"{stuck_chains['gap_20']['K2']} of {stuck_chains['of']['K2']} K = 2; each stuck spot's best chain gives a ratio of {stuck_chains['best_chain_sd_ratio']['all']:.2f} "
      f"(K = 1 {stuck_chains['best_chain_sd_ratio']['K1']:.2f}, K = 2 {stuck_chains['best_chain_sd_ratio']['K2']:.2f})")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
ax = axes[0]
span = np.array([0.008, 12])
ax.fill_between(span, span * 0.8, span * 1.25, color="#2a7f62", alpha=0.15, lw=0)
ax.plot(span, span, color=LINE_COL, lw=0.9)
for k, marker in ((1, "o"), (2, "^")):
    keep = K == k
    ax.scatter(laplace_sd[keep & good], nuts_sd[keep & good], s=26, marker=marker, color=CLASSICAL, lw=0, label=f"K = {k}, r_hat at most 1.01")
    ax.scatter(laplace_sd[keep & ~good], nuts_sd[keep & ~good], s=30, marker=marker, facecolor="none", edgecolor=NET_A, label=f"K = {k}, r_hat over 1.01")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set(xlim=(0.012, 3), ylim=(0.012, 12), xlabel="Laplace sd (px)", ylabel="NUTS sd (px)")
ax.set_title("Stored run: at K = 1, converged spots agree. Many spots do not converge", fontsize=11)
ax.legend(fontsize=8.5, loc="lower right")
ax = axes[1]
ax.hist(np.log10(rhat[K == 1] - 1), bins=np.linspace(-4, 0.5, 34), color=CLASSICAL, alpha=0.8, label="K = 1")
ax.hist(np.log10(rhat[K == 2] - 1), bins=np.linspace(-4, 0.5, 34), histtype="step", color=NET_A, lw=2, label="K = 2")
ax.axvline(-2, color=LINE_COL, lw=1.2)
ax.text(-1.95, 13.5, "line: r_hat 1.01", fontsize=9)
ax.set_xticks([-3, -2, -1, 0], ["1.001", "1.01", "1.1", "2"])
ax.set(xlabel="r_hat of the spot (worst parameter)", ylabel="spots")
ax.set_title("Stored run: two populations of spot, clean or stuck", fontsize=11)
ax.legend();

# %% [markdown]
# It fails its line at 1.54, and 44% of spots (87 of 200) miss r_hat 1.01. Read the left panel. On K = 1 spots that converge, NUTS
# and Laplace agree (1.03). The excess in check 1 therefore does not look like a Laplace problem at K = 1: it points at
# the forward model, which both share. At K = 2 even converged spots sit above the band (1.48), so there the
# approximation does understate the sd. Stuck spots climb the plot because chains sit far apart, which inflates the
# pooled sd. The converged medians are descriptive only, since r_hat chose which spots got in.
#
# **Most of that failure is a bug in our audit, and the gate stays failed anyway.** It was written for this run, so the
# record does not move. But chains were never started on the spot. In nutpie 0.16.11 `with_data` swaps the data in the log
# density, and the start keeps the compile-time placeholders, here `cx = cy = 0`, the corner of the patch. `init_mean` is
# ignored for a PyMC model, which is why the pilot's Laplace start changed no K = 1 result. Rewrite the same posterior in
# non-centred form, so that the start follows the data, and the ratio is 1.11 with an empty compile cache and 1.14 with a
# warm one, inside 0.8-1.25. All 100 K = 1 spots reach r_hat 1.01, with 8-9 divergences left and on recording 7 only, and the fitter's row sits on the NUTS mean with a photon ratio of
# 1.00. **Laplace looks fine at K = 1 on this recording.**
#
# **K = 2 is the real problem.** There 35-45 spots stay over 1.01, with the ratio at 1.8-1.9 overall (1.7-1.8 with draws sorted on the
# wider axis) and about 1.5 on those that converge. Their chains sit a median 0.6 nats apart, so most are slow or competing
# states, though the largest gaps show some are still trapped. That is a claim about these rerun numbers, which are post-hoc, one form of the model and
# not a gate.

# %% [markdown]
# ## 8. A live audit in PyMC
#
# This one runs on your machine. The summary keeps the pixels of four audited spots, chosen by a rule and not by
# eye: for K = 1 and K = 2, the spot with the worst stored r_hat, and the first spot in audit order at or under
# 1.01. The model is E58's spot model with the fitter's mask of neighbouring blobs. `pm.Data` plus
# `nutpie.compile_pymc_model(...).with_data(...)` means each K compiles once. The prior predictive step belongs to E58;
# this section is the diagnostics. With two emitters, draws are sorted on x before r_hat, since label switching is not a
# sampling failure.
#
# We run it twice. First the stored form, centred, on the worst K = 1 spot, with the warmup kept so that you can see where
# the chains start. Then the same posterior written non-centred, `x = cx + 2 * rx`, whose start is the spot, on all four.
# **If you use `with_data`, check that your start does not depend on the data you swapped.**

# %%
PIX = np.arange(PATCH)
SQ2 = np.sqrt(2.0)


def spot_model(K, centred=False):
    """E58's spot model on one masked 11 x 11 patch: the same posterior as the Laplace fit it audits, in either form."""

    def pixel(u, s):
        return 0.5 * (pt.erf((PIX + 0.5 - u[..., None]) / (SQ2 * s)) - pt.erf((PIX - 0.5 - u[..., None]) / (SQ2 * s)))

    with pm.Model() as model:
        z = pm.Data("z", np.zeros(PATCH * PATCH))
        w = pm.Data("w", np.ones(PATCH * PATCH))
        cx, cy = pm.Data("cx", np.zeros(K)), pm.Data("cy", np.zeros(K))
        lb = pm.Data("lb", 5.0)
        # Centred, the stored form: its start keeps the placeholders above, whatever with_data supplies.
        # Non-centred: rb, rx and ry start at 0, so x and y start on the spot.
        logb = pm.Normal("logb", lb, 0.5) if centred else pm.Deterministic("logb", lb + 0.5 * pm.Normal("rb", 0.0, 1.0))
        logs = pm.Normal("logs", 0.0, 0.2)
        x = pm.Normal("x", cx, 2.0, shape=K) if centred else pm.Deterministic("x", cx + 2.0 * pm.Normal("rx", 0.0, 1.0, shape=K))
        y = pm.Normal("y", cy, 2.0, shape=K) if centred else pm.Deterministic("y", cy + 2.0 * pm.Normal("ry", 0.0, 1.0, shape=K))
        logN = pm.Normal("logN", np.log(1500.0), 1.0, shape=K)
        s = pt.exp(logs)
        image = pt.sum(pt.exp(logN)[:, None, None] * pixel(y, s)[:, :, None] * pixel(x, s)[:, None, :], axis=0)
        mu = (pt.exp(logb) + image).ravel() / 2
        pm.Potential("ll", pt.sum(w * (z * pt.log(mu) - mu - pt.gammaln(z + 1))))
    return model


def audit_spot(compiled_model, i, **options):
    """One spot, 4 chains, the stored seed and settings; the same summary for either form."""
    spot, cen = pack["spot_id"][i], pack["spot_cen"][i]
    k = int(K[spot])
    model = compiled_model.with_data(z=pack["spot_Z"][i].ravel().astype(float), w=pack["spot_W"][i].ravel().astype(float),
                                     cx=np.full(k, float(cen[0])), cy=np.full(k, float(cen[1])), lb=float(pack["spot_logb"][i]))
    t0 = time.perf_counter()
    trace = nutpie.sample(model, chains=4, tune=1000, draws=1000, cores=min(4, CORES), seed=int(pack["spot_seed"][i]), progress_bar=False, **options)
    post = trace.posterior
    x, y, logN = post["x"].values, post["y"].values, post["logN"].values
    if k == 2:
        order = np.argsort(x, axis=-1)
        x, y, logN = (np.take_along_axis(v, order, -1) for v in (x, y, logN))
    arrays = dict(x=x, y=y, logN=logN, logb=post["logb"].values[..., None], logs=post["logs"].values[..., None])
    r_hat = max(float(az.rhat(v[..., j])) for v in arrays.values() for j in range(v.shape[-1]))
    ess = min(float(az.ess(v[..., j])) for v in arrays.values() for j in range(v.shape[-1]))
    return trace, dict(K=k, stored_r_hat=float(rhat[spot]), live_r_hat=r_hat, live_ess=ess, seconds=time.perf_counter() - t0,
                       divergences=int(trace.sample_stats["diverging"].sum()), x=x, y=y, logN=logN, logb=arrays["logb"],
                       logp=trace.sample_stats["logp"].mean("draw").values)


t0 = time.perf_counter()
stored_form = nutpie.compile_pymc_model(spot_model(1, centred=True))
compiled = {k: nutpie.compile_pymc_model(spot_model(k)) for k in (1, 2)}
compile_s = time.perf_counter() - t0
worst_k1 = int(np.argmax(rhat[pack["spot_id"]] * (K[pack["spot_id"]] == 1)))
trace_stored, as_stored = audit_spot(stored_form, worst_k1, save_warmup=True)
first = trace_stored.warmup_posterior
print(f"Stored form, worst K = 1 spot, centre {pack['spot_cen'][worst_k1][0]:.1f}, {pack['spot_cen'][worst_k1][1]:.1f}. First warmup draw of each chain:")
print("  x " + " ".join(f"{v:5.1f}" for v in first["x"].values[:, 0, 0]) + "\n  y " + " ".join(f"{v:5.1f}" for v in first["y"].values[:, 0, 0]))
print(f"  r_hat {as_stored['live_r_hat']:.3f} (stored {as_stored['stored_r_hat']:.3f}), {as_stored['divergences']:,} divergences\n")
live = [audit_spot(compiled[int(K[spot])], i)[1] for i, spot in enumerate(pack["spot_id"])]
print(f"Compiled 3 models in {compile_s:.0f} s. The start on the spot, all four spots:")
print(pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} for row in live]).round(3).to_string(index=False))
print(f"Log density gap, best chain minus worst: " + ", ".join(f"{row['logp'].max() - row['logp'].min():,.0f}" for row in live))
print("\nEach chain, first emitter (sorted on x when K = 2):")
for row, name in zip([as_stored] + live, ("K = 1 worst, stored form", "K = 1 worst, start on the spot", "K = 1 clean", "K = 2 worst", "K = 2 clean")):
    print(f"  {name}")
    for c in range(4):
        print(f"    chain {c}: x {row['x'][c, :, 0].mean():5.1f}, sd of x {row['x'][c, :, 0].std():.2f}, photons "
              f"{np.exp(row['logN'][c, :, 0].mean()):7,.0f}, background {np.exp(row['logb'][c].mean()):4.0f}, mean logp {row['logp'][c]:8.1f}")

fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.2), layout="constrained")
palette = ("#b2182b", "#0b6e8a", "#d98200", "#2a7f62")
panels = [(worst_k1, as_stored, "K = 1, stored form")] + [(i, row, f"K = {row['K']}, start on the spot") for i, row in enumerate(live)]
for ax, (i, row, title) in zip(axes, panels):
    image = np.ma.masked_where(pack["spot_W"][i] == 0, pack["spot_Z"][i])
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    for c in range(4):
        ax.scatter(row["x"][c, ::10].ravel(), row["y"][c, ::10].ravel(), s=2.5, color=palette[c], alpha=0.5, lw=0)
    for c in range(4):
        ax.scatter(row["x"][c].mean(0), row["y"][c].mean(0), s=95, marker="X", color=palette[c], edgecolor="w", lw=1, zorder=4)
    ax.set(xlim=(-3.5, PATCH + 2.5), ylim=(PATCH + 2.5, -3.5), xticks=[], yticks=[])
    ax.set_title(f"{title}\nstored r_hat {row['stored_r_hat']:.2f}, live {row['live_r_hat']:.2f}", fontsize=9)
fig.suptitle("Dots: each chain's draws of position. X: each chain's mean. White pixels belong to neighbouring blobs", fontsize=11);

# %% [markdown]
# **The start explains the K = 1 trap.** The stored form starts every chain at the patch corner, nowhere near the spot, and it
# returns the stored r_hat of 3.012 in every compile-cache state we tried. Only chain 1 reaches the spot: about 6,000
# photons on a background of 112, with mean logp -435. The other three hold 160-830 photons on a background near 159, and
# they sit almost 2,000 nats lower. The light of the spot is being absorbed by a raised background, and in two of them the
# emitter roams at x sds of 2-4 px, the width of the prior, and leaves the patch. Start on the spot, with the same posterior
# and seed, and all four chains agree on about 6,000 photons, at r_hat 1.002 with no divergences.
#
# **This corrects how the stored run read its own audit.** The pilot and the plan called these "false modes" that a Laplace
# fit hides. A state 2,000 nats below another has a relative posterior weight of about exp(-2000), so it is no rival
# explanation of the pixels. The r_hat failure reports a bad start and a failed sampler, not a wider posterior, and the
# "NUTS sd" of such a spot is the distance between stuck chains.
#
# "Trap" does not cover all 87 stuck spots, though. Rerun in the stored form, 57 show a gap of at least 20 nats, 24 of 26 at K =
# 1 and 33 of 61 at K = 2. Of those 57, 23 reach the same level under plain L-BFGS, so nothing held them but a stalled
# chain. The other 30 have no such gap, and 28 of those are K = 2. That is why section 7's rerun, from the spot, matters more
# than any reading of the stuck chains.
#
# **The stored K = 2 numbers depend on the compile cache.** Same seed, inputs and machine, stored form, worst K = 2 spot: a warm
# PyTensor compile cache gives r_hat 3.113, 92 divergences and a gap of 2,723, and an empty cache gives 4.424, 2,515 and 3,147.
# One cache directory gave the second result on its first run and the first on its second. Core count made no difference, and
# K = 1 did not change. The cause is not established. Across the whole stored audit an empty cache gave a median ratio of
# 1.52, not 1.54, and 83 spots over the line, not 87, so every verdict holds. A reader's first run would see the same effect in
# the K = 2 clean row above, in its third decimal. The start on the spot brings this worst K = 2 spot to r_hat 1.003-1.004, and
# not every K = 2 spot follows: section 7 has 35-45 still over the line.
#
# **What that rests on, and what it does not change.** All of this is post-hoc. The gate was written for all 200 spots with
# the stored start, so it stays failed. The rerun is one reparameterisation, it is not a gate, and it says nothing of the
# K = 2 fits beyond the ratio. At K = 1 it is direct evidence for the fitter: its row sits on the NUTS mean on 99 of 100 spots.

# %% [markdown]
# ## 9. The ledger: every gate, with its line
#
# Everything judged so far, in one table, including the gates that made it awkward to continue. The pilot gates are
# `results/E88/pilot_gates.json`, a copy of a Markdown file written at 13:25 and made at 13:31, both before any pilot
# below ran. The full run's are `results/E88/full_gates.json`,
# written at 15:34 before any full job ran. Neither changed. **A later pass never replaces an earlier failure.**

# %%
p2 = store.load_json("p2_classical_r7")
audit_stage = next(row["seconds"] for row in store.stages() if row["stage"] == "audit_nuts")
p5p, p6, p7, p8 = p5["5,000 steps (P5, NO-GO)"], store.load_json("p6_p5"), store.load_json("p7_audit"), store.load_json("p8_resume")
p8_same = store.load_json("train_p8refd")["last_loss"] == store.load_json("train_p8kd")["last_loss"]
p8_slow, p8_fast = store.load_json("train_p8refd")["seconds"], store.load_json("train_p8ref")["seconds"]
v, grid_ratio, wh = verdicts, summary["grid"], p4_gpu["wilson_hilferty"]
p3r = p3["recordings"]
FAIL, PASS, NOTE = "**FAIL**", "pass", "no line"
ledger = [
    ("P3 calibration", "20 ADU, 10% along and between", f"along {p3r['r7']['offset_dev_max']:.0f} / {p3r['r32']['offset_dev_max']:.0f} ADU, "
     f"{p3r['r7']['slope_rel_max']:.0%} / {p3r['r32']['slope_rel_max']:.0%}; between {p3['between']['offset']:.0f} ADU, {p3['between']['slope_rel']:.0%}", FAIL),
    ("P2 time", "under 20 min on 13 CPUs", f"{p2['timing']['seconds']:.0f} s for {p2['timing']['chunks']} of 30 chunks, after a restart", PASS),
    ("P2 memory", "whole tree under 20 GB", "4.83 GB, read from the job log and not in the store", PASS),
    ("P2 drift", "under 10 nm", f"residual rms {p2['residual_rms_nm']:.1f} nm", PASS),
    ("P2b overlap", "not rare: 5% or more in a recording", f"{decisions['classical']['overlap_primary']['r7']:.1%} / {decisions['classical']['overlap_primary']['r32']:.2%}", FAIL + ": rare, stop fired"),
    ("FULL decision, written", "go needs every pilot and overlaps not rare", "go = false", "no-go; run on the owner's ruling"),
    ("P4 speed, VRAM", "30 steps/s, 8 GB", f"{p4_gpu['steps_per_s']:.0f} steps/s, {p4_gpu['vram']['peak_gb']:.2f} GB", PASS),
    ("P4 sampler TV", "0.030 at 20 photons, 0.015 at 88", f"{wh['20']['tv']:.3f}, {wh['88']['tv']:.3f}", PASS),
    ("P5, 5,000 steps", "recall 0.90, 1.3 x CRLB", f"{p5p['sparse_2000']['tol2.0']['recall']:.3f}, {p5p['sparse_2000']['rmse_over_crlb']:.2f} x", FAIL),
    ("P6 speed", "50 frames/s", f"{p6['batch8']['frames_per_s']:,.0f}-{p6['batch32']['frames_per_s']:,.0f} frames/s", PASS),
    ("P6 grid", "max / min 1.25", f"{p6['grid_max_over_min']:.2f}", FAIL),
    ("P7, 20 spots", "r_hat 1.01, 0 divergences", f"{p7['max_rhat']:.2f}, {p7['divergences']:,}", FAIL),
    ("P8' resume, first attempt", "1e-3 on loss and weights", f"loss {p8['max_rel_loss']:.1%} off, weights {p8['rel_l2']:.1e}", FAIL),
    ("P8' resume, deterministic ops", "same", f"final loss identical: {p8_same}", "pass, second attempt"),
    ("Check 1, A, recording 32", "z over 1.25 fails", f"{v['r32_a']['z']:.2f} (half-width {v['r32_a']['halfwidth95']:.3f}, 1 se {v['r32_a']['one_se']:.3f})", FAIL),
    ("Check 1, B, recording 32", "z over 1.25 fails", f"{v['r32_b']['z']:.2f} (half-width {v['r32_b']['halfwidth95']:.3f}, 1 se {v['r32_b']['one_se']:.3f})", FAIL),
    ("Check 1, A, B, recording 7", "z over 1.25 fails", f"{v['r7_a']['z']:.2f}, {v['r7_b']['z']:.2f}", FAIL),
    ("Check 1, relative, both", "at or under classical", f"{v['r32_a']['z']:.2f}, {v['r32_b']['z']:.2f} against {v['r32_a']['classical_z']:.2f}", "no worse than classical"),
    ("Check 2 halo, A", "ring over far field 2", f"{halo['r7']['a']['ratio']:.2f} / {halo['r32']['a']['ratio']:.2f}", FAIL),
    ("Check 2 halo, B", "ring over far field 2", f"{halo['r7']['b']['ratio']:.2f} / {halo['r32']['b']['ratio']:.2f}", FAIL + " on recording 32, pass on 7"),
    ("Check 3 replicate", "none written", f"KS area A {replicate['r32']['a']['ks_area']:.2f}, B {replicate['r32']['b']['ks_area']:.2f}", NOTE),
    ("Check 4 NUTS / Laplace sd", "0.8-1.25", f"{audit['median_sd_ratio']['all']:.2f}", FAIL),
    ("P7's lines, 200 spots", "r_hat 1.01, 0 divergences", f"{audit['max_rhat']:.2f}, {audit['divergences']:,}", FAIL),
    ("Audit time", "10 min", f"{audit_stage:.0f} s for 200 spots on 2 cores, inflated by sharing", PASS),
    ("P5's lines, A at 40,000", "recall 0.90, 1.3 x CRLB", f"{summary['p5_at_final']['a']['tol2.0']['recall']:.3f}, {summary['p5_at_final']['a']['rmse_over_crlb']:.2f} x", "inside; P5 stays failed"),
    ("P5's lines, B at 12,000 more", "same", f"{summary['p5_at_final']['b']['tol2.0']['recall']:.3f}, {summary['p5_at_final']['b']['rmse_over_crlb']:.2f} x", "inside; P5 stays failed"),
    ("P6's grid, all real frames, A", "1.25", f"{grid_ratio['a_r7']:.2f} / {grid_ratio['a_r32']:.2f}", FAIL),
    ("P6's grid, all real frames, B", "1.25", f"{grid_ratio['b_r7']:.2f} / {grid_ratio['b_r32']:.2f}", FAIL),
]
frame = pd.DataFrame(ledger, columns=["Gate", "Line", "Value (recording 7 / 32 where two)", "Verdict"])
display(markdown_table(frame))
print(f"{sum(FAIL in row[3] for row in ledger)} of {len(ledger)} rows failed. Written outcome: {summary['outcome']} "
      f"(A fails and B does not pass everything A failed).")

# %% [markdown]
# Some of those failures carry a diagnosis, and a diagnosis does not turn one into a pass:
#
# - **P6's grid looks like the sample, not a seam.** In the pilot the pattern moved with the image when the frames were
#   shifted (r 0.98 shifted back, 0.05 unshifted), and uniform simulated emitters gave 1.13. The gate was badly specified.
#   That diagnosis was not rerun on the full frames, so the failure stands and the cause there is unconfirmed.
# - **P5 passed at scale-up,** and the failure at 5,000 steps stands, because its written action was "do not scale up".
#   Whether the owner's ruling was right to override it is a judgement, not a measurement.
# - **Check 3 has no verdict,** because nobody wrote a line, and inventing one afterwards would be the very thing the
#   gates exist to prevent.

# %% [markdown]
# ## 10. The second failure: drift
#
# The written decision rule was applied to network rows exactly as it was to classical rows, including its drift method:
# for each 1,000-frame block, cross-correlate that block's render with the render of all other blocks. And on network
# rows it broke. The stored shifts are below. The smoke test had missed this, because its stand-in rows were classical,
# and it was found by reading the output after the run.

# %%
def shift_um(rec, tail, method):
    return pack[f"{rec}_drift{tail}_{method}"] * PIXEL_NM / 1000


print(f"{'':28}{'largest shift':>15}{'blocks over 1 um':>18}{'largest axis range':>20}")
for rec in RECS:
    for tail in ("", "_line"):
        for method in ("chosen", "a", "b"):
            d = shift_um(rec, tail, method)
            print(f"{RECS[rec]:13}{tail[1:] or 'per block':10}{method:6}{np.abs(d).max() * 1000:>12,.0f} nm"
                  f"{(np.abs(d).max(1) > 1).sum():>15} of 15{np.ptp(d, axis=0).max() * 1000:>15,.0f} nm")


def smooth(h):
    return ndi.gaussian_filter(h.astype(float), 1.0)


def cross(a, c):
    return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a - a.mean()) * np.conj(np.fft.rfft2(c - c.mean())), s=a.shape))


def peak_shift(cc, px_per_bin=0.25):
    """Sub-bin peak of a centred cross-correlation, by a quadratic through the peak on each axis."""
    y, x = np.unravel_index(cc.argmax(), cc.shape)

    def vertex(a, c, e):
        return 0.5 * (a - e) / (a - 2 * c + e)

    dy, dx = vertex(cc[y - 1, x], cc[y, x], cc[y + 1, x]), vertex(cc[y, x - 1], cc[y, x], cc[y, x + 1])
    return np.array([x + dx - cc.shape[1] // 2, y + dy - cc.shape[0] // 2]) * px_per_bin


block = int(pack["drift_demo_block"])
maps = {name: cross(smooth(pack[f"drift_demo_{name}_block"]), smooth(pack[f"drift_demo_{name}_rest"])) for name in ("a", "chosen")}
found = {name: -peak_shift(cc) for name, cc in maps.items()}
print(f"\nLive, block {block} of recording 7, from the stored renders:")
print(f"  network A: shift ({found['a'][0]:.1f}, {found['a'][1]:.1f}) px = {np.hypot(*found['a']) * PIXEL_NM / 1000:.1f} um. Stored: {pack['r7_drift_a'][block].round(1)}")
print(f"  classical: shift ({found['chosen'][0]:.2f}, {found['chosen'][1]:.2f}) px = {np.hypot(*found['chosen']) * PIXEL_NM:.0f} nm; "
      f"stored leave-block-out estimate of the same block {np.hypot(*p2['drift_leave_block_out_nm'][block]):.0f} nm")
assert np.allclose(found["a"], pack["r7_drift_a"][block], atol=1e-2), "the live estimate reproduces the stored fault"
pile_block = np.unravel_index(pack["drift_demo_a_block"].argmax(), (800, 800))
pile_rest = np.unravel_index(pack["drift_demo_a_rest"].argmax(), (800, 800))
to_px = lambda ij: np.array(ij[::-1]) * 0.25 - 0.375
print(f"  Fullest bin in the block: {pack['drift_demo_a_block'].max():,} rows at (x, y) {to_px(pile_block)}; "
      f"in the rest: {pack['drift_demo_a_rest'].max():,} rows at {to_px(pile_rest)}")
print(f"  Their separation: {to_px(pile_rest) - to_px(pile_block)} px. Typical occupied bin: "
      f"{int(np.median(pack['drift_demo_a_block'][pack['drift_demo_a_block'] > 0]))} row.")
print(f"  Full movie, fullest 0.25 px bin: A {pack['r7_render_a'].max():,}, B {pack['r7_render_b'].max():,}, classical {pack['r7_render_chosen'].max():,}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), width_ratios=[1, 1, 1.25], layout="constrained")
for ax, name, title in ((axes[0], "a", "network A"), (axes[1], "chosen", "classical")):
    cc = maps[name]
    ax.imshow(cc / cc.max(), cmap="magma", extent=(-200, 200, 200, -200), interpolation="nearest", vmin=-0.05, vmax=1)
    ax.set(xlim=(-120, 120), ylim=(120, -120), xlabel="shift (px)", ylabel="shift (px)")
    ax.set_title(f"{title}: block {block} against the rest", fontsize=10.5)
    peak = peak_shift(cc)
    ax.add_patch(plt.Circle(peak, 7, fill=False, ec="#55ff99", lw=1.6))
    ax.text(-108, -100, f"peak at ({peak[0] + 0:.1f}, {peak[1] + 0:.1f}) px", color="w", fontsize=9)
ax = axes[2]
blocks = np.arange(15)
for method, marker in (("chosen", "o"), ("a", "s"), ("b", "^")):
    for rec, fill in (("r7", True), ("r32", False)):
        d = np.hypot(*shift_um(rec, "", method).T) * 1000
        ax.scatter(blocks + {"chosen": 0, "a": 0.12, "b": -0.12}[method], d, s=38, marker=marker,
                   facecolor=COLOUR[method] if fill else "none", edgecolor=COLOUR[method], lw=1.3)
ax.axhline(10, color=LINE_COL, lw=1)
ax.text(5.6, 14, "P2's 10 nm line", fontsize=8.5)
ax.set_yscale("log")
ax.set(xlabel="1,000-frame block", ylabel="estimated shift (nm)")
ax.set_title("Real drift is nanometres. These faults are microns", fontsize=10.5)
ax.legend(handles=[Line2D([], [], marker=m, ls="", color=COLOUR[k], label=NAME[k]) for k, m in (("chosen", "o"), ("a", "s"), ("b", "^"))]
          + [Line2D([], [], marker="o", ls="", color="k", label="recording 7"),
             Line2D([], [], marker="o", ls="", mfc="none", color="k", label="recording 32")], fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.68), ncols=2);

# %% [markdown]
# **What broke.** The classical drift range on these recordings is 5-18 nm. The written method returned shifts of microns
# on network rows, up to 11.6 µm, in 1-4 of 15 blocks, for every network run except B per block on recording 32. A few
# fixed spots pile up hundreds or thousands of network rows in a single 0.25 px bin, where a typical bin holds one, and
# classical piles reach only 424. A cross-correlation peak is dominated by its strongest sources, so it lines up two
# *different* piles and calls the difference between them drift. The live result above is that fault on your machine:
# the peak falls at the separation of the two fullest bins, and a whole 1,000-frame block then gets moved by about 11 µm.
#
# **What the piles are, post-hoc.** The fullest spots are long-lived bright objects, not detector artefacts. Recording 7's at
# (x 36, y 99) reads about 2,500 ADU in block medians against 1,150 nearby, and it switches on and off for thousands of frames
# at a time. Recording 32's at (x 6, y 184) reads a median of about 1,550 ADU across its blocks. Only the fullest of each recording
# was read this way. The simulated movie points the same way: its background maps come from recording 7, and 93% of A's false
# rows and 91% of B's pile up there with no emitter present, and they lie within 1 px of the 20 fullest real pile bins about
# 13-18 times as often as true rows do. Only those 20 bins were tested.
#
# This is partly a fair-fight problem. The fitter subtracts a 201-frame rolling background, so these objects vanish for it.
# The network sees 3 frames and has no way to know one has been there for a minute. Classical's temporal context is part of
# this comparison.
#
# This matters because the gated decision curves used those shifts. Network B on recording 7 gave 86 clusters at 2,000
# frames, against 413 under the line. It was first misread as a calibration effect. It was this.
#
# **The repair, post-hoc and declared.** `results/E88/full_posthoc_drift.json` was written before the rerun, and only one
# repair was allowed: apply the stored classical drift of the same recording and calibration to the network rows, because
# drift belongs to the sample and not to the localiser.
# The gated curves stay on the record, and they are called unsound in section 12. A drift estimate robust to piles would be
# a new method, and it was not tried.
#
# **The general point.** A pipeline step validated on one kind of input is unvalidated on another. Classical rows and
# network rows have the same columns, and they are not the same distribution.

# %% [markdown]
# ## 11. The third failure: a flattering split
#
# This failure was predicted, and it has nothing to do with networks. FRC splits localisations in two and asks at what
# scale the two renders stop agreeing. A random split puts half a blink in each half. With 42% and 38% of filtered rows
# linking into the next frame, those halves share molecules, so FRC reports agreement that has nothing to do with resolution. The
# fix merges linked rows into events and splits by time blocks of 50 frames. Stored, and the curves are recomputed for the
# summary with assertions that they return the stored values.

# %%
split, split_post = store.load_json("frc_split"), store.load_json("frc_split_cdrift")
print(f"{'FRC at 15,000 frames, nm':32}{'random':>9}{'block':>9}{'gap':>8}")
for rec in RECS:
    r = split[rec]["chosen"]
    print(f"{RECS[rec] + ', classical':32}{r['random_split_nm']:>9.1f}{r['block_split_nm']:>9.1f}{r['gap_nm']:>8.1f}")
    for net in ("a", "b"):
        r = split_post[rec][net]
        print(f"{RECS[rec] + ', ' + NAME[net] + ', post-hoc':32}{r['random_split_nm']:>9.1f}{r['block_split_nm']:>9.1f}{r['gap_nm']:>8.1f}")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True, layout="constrained")
for ax, rec in zip(axes, RECS):
    for kind, ls, label in (("random", "--", "random split"), ("block", "-", "block split, merged events")):
        curve = pack[f"{rec}_frc_{kind}_chosen"]
        ax.plot(curve[:, 0], curve[:, 1], color=CLASSICAL, ls=ls, lw=2, label=label)
    ax.axhline(1 / 7, color=LINE_COL, lw=1.1)
    ax.text(300, 0.165, "threshold 1/7", fontsize=9)
    low, high = split[rec]["chosen"]["random_split_nm"], split[rec]["chosen"]["block_split_nm"]
    ax.axvspan(low, high, color=NET_A, alpha=0.18, lw=0)
    ax.annotate(f"{high - low:.1f} nm of flattery", (high, 0.55), xytext=(high + 12, 0.62), fontsize=9.5, color=NET_A)
    ax.set_xscale("log")
    ax.set(xlim=(600, 20), xlabel="spatial period (nm), fine detail to the right")
    ax.set_xticks([20, 30, 50, 100, 200, 500], ["20", "30", "50", "100", "200", "500"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"{RECS[rec]}, classical", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
axes[0].set_ylabel("Fourier ring correlation");

# %% [markdown]
# The split alone is worth up to 10 nm on classical rows and up to 12 nm on network rows. Even the block split is
# optimistic, since a molecule that returns after a long dark period can land in both halves. So always name the split
# when you quote FRC, and read a block-split number as a best case.

# %% [markdown]
# ## 12. The frames decision, all of it
#
# Now the brief's third question: would the network change the answer? The table has the gated runs and the post-hoc
# repair, each under both calibrations.

# %%
stored = summary["decision"]
post = store.load_json("decision_posthoc_cdrift")["decision"]
table = []
for label, key, source, status in (("Classical", "classical", stored, "reference"), ("Network A", "a", stored, "gated, unsound"),
                                   ("Network B", "b", stored, "gated, unsound"), ("Network A", "a", post, "post-hoc drift"),
                                   ("Network B", "b", post, "post-hoc drift")):
    a, b = source[key], source[f"{key}_line"]
    table.append({"method": label, "status": status, "per block": a["N_both"], "(rec 7, rec 32)": f"({a['r7']:,}, {a['r32']:,})",
                  "line": b["N_both"], "(rec 7, rec 32) ": f"({b['r7']:,}, {b['r32']:,})"})
print(pd.DataFrame(table).to_string(index=False))
print(f"\nClassical range quoted: {stored['classical']['N_both']:,}-{stored['classical_line']['N_both']:,}, or "
      f"{1 - stored['classical_line']['N_both'] / FRAMES:.0%}-{1 - stored['classical']['N_both'] / FRAMES:.0%} fewer frames")
for title, tail in (("per block", ""), ("one line", "_line")):
    values = [(stored if key == "classical" else post)[key + tail]["N_both"] for key in ("classical", "a", "b")]
    print(f"Spread of N for both over classical and post-hoc A and B, {title}: {max(values) - min(values):,}")
print(f"Moved by the line, post-hoc A: {post['a_line']['N_both'] - post['a']['N_both']:,}; post-hoc B: "
      f"{post['b_line']['N_both'] - post['b']['N_both']:,}; classical: {stored['classical_line']['N_both'] - stored['classical']['N_both']:,}")
print(f"Rows per frame in the first block of recording 7: B per block {pack['r7_rows_per_block_b'][0] / 500:.0f}, B under the line "
      f"{pack['r7_rows_per_block_b_line'][0] / 500:.0f}; A {pack['r7_rows_per_block_a'][0] / 500:.0f} and {pack['r7_rows_per_block_a_line'][0] / 500:.0f}; "
      f"classical {pack['r7_rows_per_block_chosen'][0] / 500:.0f} and {pack['r7_rows_per_block_chosen_line'][0] / 500:.0f}")

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), width_ratios=[1.15, 1.15, 1], layout="constrained")
for ax, rec in zip(axes, RECS):
    runs = (("classical", CLASSICAL, "-", 2.6, "classical"), ("a_cdrift", NET_A, "-", 2, "A, classical drift"),
            ("b_cdrift", NET_B, "-", 2, "B, classical drift"), ("a", NET_A, ":", 1.2, "A, gated drift (unsound)"),
            ("b", NET_B, ":", 1.2, "B, gated drift (unsound)"))
    for key, colour, ls, lw, label in runs:
        y = np.array([decisions[key][rec]["points"][str(N)]["count"] for N in GRID])
        ax.plot(GRID, y / y[-1], color=colour, ls=ls, lw=lw, label=label)
    ax.axhspan(0.9, 1.1, color="#555555", alpha=0.1, lw=0)
    ax.set(xlabel="frames used", ylim=(0, 1.2))
    ax.set_xticks([2000, 6000, 10000, 15000], ["2,000", "6,000", "10,000", "15,000"])
    ax.set_title(f"{RECS[rec].capitalize()}: clusters / value at 15,000", fontsize=10.5)
axes[0].legend(fontsize=8.5, loc="lower right")
ax = axes[2]
rows = [("Classical", stored["classical"], stored["classical_line"], CLASSICAL, True), ("A, post-hoc", post["a"], post["a_line"], NET_A, True),
        ("B, post-hoc", post["b"], post["b_line"], NET_B, True), ("A, gated", stored["a"], stored["a_line"], MUTED, False),
        ("B, gated", stored["b"], stored["b_line"], MUTED, False)]
ax.axvspan(10500, 12000, color=CLASSICAL, alpha=0.13, lw=0)
for i, (label, block_row, line_row, colour, sound) in enumerate(rows[::-1]):
    ax.plot([block_row["N_both"], line_row["N_both"]], [i, i], color=colour, lw=2.2, zorder=2)
    ax.scatter(line_row["N_both"], i, s=120, marker="D", facecolor="w", edgecolor=colour, lw=1.8, zorder=3)
    ax.scatter(block_row["N_both"], i, s=46, color=colour, zorder=4)
ax.set_yticks(range(5), [r[0] for r in rows[::-1]])
ax.set(xlim=(9500, 15800), ylim=(-0.6, 4.6), xlabel="N for both recordings")
ax.set_xticks([10500, 12000, 13500, 15000], ["10,500", "12,000", "13,500", "15,000"])
ax.set_title("Calibration and localiser move N by similar amounts", fontsize=10.5)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="per block"),
                   Line2D([], [], marker="D", ls="", mfc="w", color="k", label="one line"),
                   Patch(color=CLASSICAL, alpha=0.2, label="classical range")], fontsize=8, loc="lower right", bbox_to_anchor=(1.01, -0.01));

# %% [markdown]
# **Under the written rule the network changes the decision, and that is not a finding.** B's 12,000 against 10,500 rests
# on the drift fault. After the one declared repair, per block, B agrees with classical at 10,500 and A sits one grid step
# higher at 12,000, where recording 32 sets A's number, inside the range already quoted. Outcome D, "the network changes
# nothing", is false as written and about right per block. Under the line the repaired networks read 15,000 (A) and 13,500
# (B), above that range, but those are post-hoc curves from a network that failed its checks. The claim we stand behind is
# the classical one.
#
# **Calibration and localiser matter about equally here.** With the post-hoc repair, the line adds 3,000 frames to both
# networks and 1,500 to classical. Choosing a localiser spreads N by 1,500 per block and by 3,000 under the line. So the
# unsettled decision from section 2 is worth about as much to this answer as the network is, and the bootstrap in section 3
# says neither shift stands clear of sampling noise. In the first block of recording 7, B finds 55 rows per frame per block
# and 123 under the line, where classical finds 81 and 87, but that does not explain the shift: B's post-hoc count at 2,000
# frames barely moves, 402 against 419.
#
# Note what the post-hoc curves do not support: a claim that the network sees *more*. Look at the same 3 µm square through
# all three.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), layout="constrained")
for ax, m in zip(axes, ("chosen", "a", "b")):
    image = ndi.gaussian_filter(pack[f"zoom_{m}"].astype(float), 1.0)
    ax.imshow(np.ma.masked_less(image, 0.02), cmap=DARK, norm=LogNorm(0.02, max(image.max() * 0.4, 1)), interpolation="nearest")
    ax.plot([190, 290], [285, 285], color="w", lw=3)
    ax.text(240, 276, "1 µm", color="w", ha="center", fontsize=9)
    ax.set_title(f"{NAME[m]}: {int(pack[f'zoom_{m}'].sum()):,} rows", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
fig.suptitle("The densest 3 µm of recording 7's mask, by a rule, 10 nm bins, classical drift. A render is not evidence", fontsize=11);

# %% [markdown]
# They look alike, and the networks' pictures hold more rows and look a little crisper. **A render cannot tell you
# whether its error bars are honest, and these are the same rows that failed section 7.** A crisp picture is not a check.
#
# ### The whole recording in one movie
#
# Here is the decision as a process, on real data, classical rows. Left: a raw frame from a 40 px window. Middle: events so
# far in that window. Right: the whole mask's cluster count, against the 10% band. Watch the middle panel keep filling in
# after the count has entered the band. That is what a 10% allowance gives up.

# %%
ids = pack["animation_frames"]
raw = pack["animation_raw"]
events_xy, events_t = pack["animation_xy"], pack["animation_t"]
curve = np.array([decisions["classical"]["r7"]["points"][str(N)]["count"] for N in GRID])
counts = np.interp(ids, GRID, curve)
edges = np.arange(0, 40.01, 0.125)


def density_at(n):
    keep = events_t < n
    return np.histogram2d(events_xy[keep, 1], events_xy[keep, 0], bins=[edges, edges])[0]


plt.rcParams["animation.embed_limit"] = 12
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), width_ratios=[1, 1, 1.25], layout="constrained")
raw_image = axes[0].imshow(raw[0], cmap="gray", vmin=np.percentile(raw, 1), vmax=np.percentile(raw, 99.8), interpolation="nearest")
cloud = axes[1].imshow(np.ma.masked_equal(density_at(int(ids[0])), 0), cmap=DARK, norm=LogNorm(1, 12), interpolation="nearest")
axes[0].set_title("raw frame", fontsize=10)
axes[1].set_title("events so far", fontsize=10)
axes[1].set_facecolor("black")
for ax in axes[:2]:
    ax.set_xticks([])
    ax.set_yticks([])
ax = axes[2]
ax.plot(GRID, curve, color=CLASSICAL, lw=1.6)
ax.axhspan(curve[-1] * 0.9, curve[-1] * 1.1, color=CLASSICAL, alpha=0.12, lw=0)
ax.axvline(10500, color=LINE_COL, lw=1, ls=":")
ax.text(10600, 40, "N*", fontsize=9)
dot, = ax.plot([], [], "o", ms=9, color=NET_A)
label = ax.set_title("", fontsize=10)
ax.set(xlim=(0, 15500), ylim=(0, 900), xlabel="frames", ylabel="clusters, whole mask")


def update(i):
    raw_image.set_data(raw[i])
    cloud.set_data(np.ma.masked_equal(density_at(int(ids[i])), 0))
    dot.set_data([ids[i]], [counts[i]])
    label.set_text(f"frame {int(ids[i]):,} of 15,000, {ids[i] * CYCLE_MS / 1000:.0f} s")
    return raw_image, cloud, dot, label


movie = animation.FuncAnimation(fig, update, frames=len(ids), interval=220, blit=False)
html = movie.to_jshtml(fps=5, default_mode="once", embed_frames=True)
plt.close(fig)
print(f"{len(ids)} frames, {len(html) / 1e6:.1f} MB of HTML")
HTML(html)

# %% [markdown]
# ## 13. What to tell the core
#
# ### The certificate
#
# A trained network is an instrument, and it should ship with a calibration certificate, the way a balance does. Here is
# network B's, in one table, generated from the stored file. It is the part of this run we would hand over even if nothing
# else survived.

# %%
real = certificate["real"]
near = {(n, w): near_nominal(n, w)[0] for n in ("a", "b") for w in ("gauss", "empirical")}
ood = store.load_json("ood532")
page = pd.DataFrame([
    ("Instrument", certificate["instrument"]),
    ("Trained on, A", certificate["trained_on"]["a"]),
    ("Trained on, B", certificate["trained_on"]["b"] + ". The stacked PSF carries a pedestal and site structure: 5-6% of its light lies beyond 3 px after a control, not the stored 25%, post-hoc"),
    ("Range tested", f"density 0.02-1.5 per µm², of which the training prior covers {certificate['ranges']['density_per_um2'][0]}-{certificate['ranges']['density_per_um2'][1]}, 500-4,000 photons, background 110 photons in the coverage grid"),
    ("Coverage on simulation", f"Near-nominal in 16 cells: A own world {near[('a', 'gauss')]}, B in A's world {near[('b', 'gauss')]}; empirical world A {near[('a', 'empirical')]}, B {near[('b', 'empirical')]}"),
    ("Real data: error bars", f"Paired z, recording 32: A {real['paired']['r32_a']['z']:.2f}, B {real['paired']['r32_b']['z']:.2f}, classical {real['paired']['r32_a']['classical_z']:.2f}. **Failure line 1.25: fail**"),
    ("Real data: false detections", f"Halo, recording 32: A {real['halo']['r32']['a']:.2f}, B {real['halo']['r32']['b']:.2f}. **Line 2: fail**"),
    ("Real data: the fitter it replaces", f"NUTS over Laplace sd {real['nuts_median_sd_ratio']['all']:.2f} on all spots, **line 1.25: fail**. Our chains started at the patch corner: rerun post-hoc from the spot, {cold['median_sd_ratio']['all']:.2f}-{warm['median_sd_ratio']['all']:.2f}, K = 1 {cold['median_sd_ratio']['K1']:.2f}, K = 2 {warm['median_sd_ratio']['K2']:.2f}-{cold['median_sd_ratio']['K2']:.2f}, unresolved (sections 7-8)"),
    ("Grid pattern", f"Mean p by 8 px cell, max over min, recording 32: A {certificate['p6_grid_ratio_all_frames']['a_r32']:.2f}, B {certificate['p6_grid_ratio_all_frames']['b_r32']:.2f}. **Line 1.25: fail** as written, cause on full frames unconfirmed"),
    ("Never tested", "; ".join(certificate["never_tested"])),
    ("Output with no evidence", f"Four 532 nm frames, a different dye, gave A {ood['a']['found_per_frame']:.1f} and B {ood['b']['found_per_frame']:.1f} detections per frame with sds near {ood['b']['median_sd_nm']:.0f} nm. The certificate says: {ood['certificate']}"),
    ("Do not use for", "Position error bars at face value on these recordings; any count without the halo check; any other sample without repeating every check"),
], columns=["Item", "Finding"])
display(markdown_table(page))

# %% [markdown]
# Note the 532 row. A network given data it has never seen gives you a tidy table of detections with confident sds, and
# nothing in its output says that anything is wrong. Only the certificate does.
#
# ### What to tell the imaging core
#
# - **Frames.** For these two recordings, 10,500-12,000 frames would have given the same answer as 15,000 in both, by a
#   rule fixed in advance, with a sampling range of 7,500-13,500 on the count alone. Take 12,000 as the working number if
#   you cut at all: about 26 minutes and 11 GB per channel saved on our assumptions. 10,500 clears its tests by 11 clusters
#   and fails in one of five seed families. Run a pilot on your next sample before trusting either, because recording 7
#   alone sets this, its count never plateaus, and the rule is close to a statement about event number.
# - **The two recordings.** They agree on the method and differ on the number. Recording 32 settles 3,000-7,000 frames
#   earlier, a gap the bootstrap cannot resolve. The record does not say how the two relate, so we cannot tell how far
#   apart the next one will be.
# - **The network.** Do not replace your fitter on this evidence. It runs fast, it looks crisp and it passed simulation. It
#   also failed the paired test and the halo on data it had not seen, and its fix made one of those worse.
# - **Your fitter.** Its sds on these data are too small as well: z is 1.63 on recording 32. That is not a flat 1.6 x. Part of
#   the shortfall at the tight end is a floor of about 4-6 nm per frame, the same size in every method, and most of the
#   rest is an unexplained scale. Add a floor before you
#   scale, and test any classical sd before you treat it as a real error bar. The fitter's Laplace sds themselves hold at K =
#   1, where NUTS agrees once it starts on the spot. At K = 2 they are too small by a factor not yet settled.
# - **Before you trust any network, ask for:** the paired-frame z with a classical reference on literally common pairs; the halo;
#   a population replicate with a lineup; and a certificate that lists what it has never seen. All four can be run without
#   ground truth.
# - **Settle calibration.** Offset and gain moved this answer by about as much as the localiser did, 1,500-3,000 frames,
#   though neither shift stands clear of sampling noise. These recordings have no dark frames; recording some is the obvious next step [our judgement, not tested here].
#
# ### What this notebook cannot claim
#
# - **Nothing for a lab policy.** Two recordings, whose relation the record does not state, one dye pair. Between-sample
#   variance is not estimable.
# - **Nothing about structure.** Our count is connected components, not the paper's HDBSCAN subclusters, and it is not
#   compared to their Brp. It is a readout for a frames decision, and it is not a biological measurement.
# - **Nothing about a shorter acquisition.** First-N frames of a 15,000-frame movie is a proxy. A planned short movie
#   might use another laser power, and bleaching would differ.
# - **Nothing about networks in general.** This is one design, one training budget and one pair of simulators, and it was
#   not tuned, because tuning until something passed was ruled out. A better simulator may well do better. Density was
#   low here, and the network's advantage on simulation lies at high density.
# - **The error bar failure is only part explained.** A floor of about 4-6 nm per frame is common to all four methods, and
#   scatter grows with the gap, so sample or stage movement is the likely source, but nothing here tests that. The residual
#   scale of 1.3-1.5 for B and chosen K, and the heavy tails, are unexplained.
# - **Accuracy is untested.** Every real-data check is on precision or on consistency. A bias common to both frames of a
#   pair would pass all of them.
#
# ### Lapses in the stored run
#
# Reported because they bear on how far to trust it. The plan (`plans/large/E88.md`, sections 6b and 6c) has the full list.
#
# - The drift fault was caught by reading output after the run, not by the smoke test. The post-hoc table exists because of it.
# - The NUTS audit's chains started at the patch corner, and the pilot's Laplace start was silently ignored. This was found only
#   by a later audit. P7 and check 4 stay failed, and their numbers measure that start as much as the fitter.
# - The stored audit's K = 2 values, and any rerun's, depend on the state of the compile cache (section 8). The gate record stays.
# - The PSF stack went into B's simulator with no control, so B was trained on a contaminated PSF.
# - The attribution stored in the summary file was wrong until the audit, and the notebook did not show it. Fixed here.
# - The P8' pass came on a second attempt, under a flag its gate did not name.
# - The gates name `simulate_linked_movie` and `localise`. Both were replaced by patch-rendered, batched versions of the same
#   model, because the originals needed about 14 GB here.
# - The classical recording-7 timing covers 28 of 30 chunks, after an 8x slowdown from memory contention forced a restart
#   with a smaller fit batch.
# - The pilots' 14-worker plan could not be met with a core held offline, so all pools ran on 13 CPUs. The 44x is on 13
#   CPUs, and it is a pilot rate; the planned full CPU comparison was skipped by instruction.
# - Deterministic ops, the section 11 answers and the 10% thresholds were chosen by the analyst, as defaults. They were
#   not ruled on by the owner, who ruled on one thing: run it, and let it fail if it fails.

# %% [markdown]
# ### The main numbers, checked
#
# These are stored values, so a mismatch between prose and output would be our error. The assertions below test the numbers
# the argument rests on against the files they came from. They are not exhaustive. The pilot-P1 control figures in section 4
# (18%, 11.5%, 51%) are quoted from the plan and are in no stored file here.

# %%
near_value = lambda value, want, tol=0.006: abs(value - want) <= tol
assert summary["outcome"] == "B" and summary["fails_a"]["paired_z"] and summary["fails_b"]["paired_z"]
assert stored["classical"]["N_both"] == 10500 and stored["classical_line"]["N_both"] == 12000
assert (stored["classical"]["r7"], stored["classical"]["r32"], stored["classical_line"]["r7"], stored["classical_line"]["r32"]) == (10500, 7500, 12000, 5000)
assert stored["a"]["N_both"] == 10500 and stored["b"]["N_both"] == 12000 and post["a"]["N_both"] == 12000 and post["b"]["N_both"] == 10500
assert post["a_line"]["N_both"] == 15000 and post["b_line"]["N_both"] == 13500
assert near_value(v["r32_a"]["z"], 1.40) and near_value(v["r32_b"]["z"], 1.63) and near_value(v["r32_a"]["classical_z"], 1.63)
assert near_value(v["r32_a"]["k1_z"], 1.49) and near_value(v["r32_line_a"]["z"], 1.39) and near_value(v["r32_line_b"]["z"], 1.61)
assert all(v[f"{k}_{n}"]["absolute"] == "failure" for k in ("r32", "r7", "r32_line", "r7_line") for n in ("a", "b"))
assert all(v[f"{k}_{n}"]["relative"] == "no worse than classical" for k in ("r32", "r7") for n in ("a", "b"))
assert near_value(halo["r32"]["b"]["ratio"], 2.69) and near_value(halo["r7"]["b"]["ratio"], 1.95)
assert near_value(summary["p5_at_final"]["a"]["tol2.0"]["recall"], 0.994) and near_value(summary["p5_at_final"]["a"]["rmse_over_crlb"], 1.16)
assert near_value(audit["median_sd_ratio"]["all"], 1.54) and (audit["over_1_01"]["K1"] + audit["over_1_01"]["K2"]) / 200 == 0.435
assert decisions["classical"]["r7"]["blobs"] == 367071 and decisions["classical"]["r32"]["blobs"] == 245335
assert p2["rows_filtered"] == 340711
assert near_value(decisions["classical"]["r7"]["points"]["2000"]["frc_nm"], 32.8, 0.05) and near_value(decisions["classical"]["r7"]["points"]["15000"]["frc_nm"], 35.3, 0.05)
assert 1.08 < decisions["classical"]["r7"]["points"]["15000"]["count"] / decisions["classical"]["r7"]["points"]["10500"]["count"] < 1.10
assert near_value(dense, 0.43, 0.01) and near_value(ratio, 44, 0.5)
assert decisions["b"]["r7"]["points"]["2000"]["count"] == 86 and decisions["b_line"]["r7"]["points"]["2000"]["count"] == 413
band = decisions["classical"]["r7"]["bands"]["15000"]
assert (round(band["count_q"][0]), round(band["count_q"][1]), round(band["frc_q"][0]), round(band["frc_q"][1])) == (889, 989, 45, 56)
assert round(pack["r7_mask"].mean(), 3) == 0.128 and round(pack["r32_mask"].mean(), 3) == 0.072
assert pack["r7_render_chosen"].max() == 424 and min(split_post[r][n]["gap_nm"] for r in RECS for n in ("a", "b")) > 2.8
assert max(split["r32"]["chosen"]["gap_nm"], split["r7"]["chosen"]["gap_nm"]) < 10.0 and split_post["r32"]["b"]["gap_nm"] < 12
assert all(pack[f"{r}_blobs_per_block"][-1] < pack[f"{r}_blobs_per_block"][0] / 2 for r in RECS)
assert np.argmax(halo["r32"]["a"]["density"]) == 10 and 500 <= halo["r32"]["a"]["edges_nm"][10] < 550
assert np.abs(shift_um("r7", "_line", "b")).max() < 12 and max(np.abs(shift_um(r, t, "chosen")).max() for r in RECS for t in ("", "_line")) < 0.02
sim_rows, real_rows, boot = posthoc["sim"], posthoc["real"], frames_boot["calibrations"]
assert near_value(paired["rows"]["r32"]["K1"]["gap3_R1.5"]["matched_chosen"]["z_sd"], 1.427, 0.001) and near_value(paired["rows"]["r32"]["b"]["gap3_R1.5"]["matched_K1"]["z_sd"], 1.686, 0.001)
assert near_value(paired["rows"]["r7"]["a"]["gap3_R1.5"]["unweighted"]["z_sd"], 1.235, 0.001) and paired["rows"]["r7"]["a"]["gap3_R1.5"]["unweighted"]["z_sd_hi"] < 1.25
assert (near_value(sim_rows["a"]["precision"], 0.505, 0.001), near_value(sim_rows["b"]["precision"], 0.865, 0.001), sim_rows["a"]["pairs_neither"]) == (True, True, 34028)
assert all(1.06 <= sim_rows[m]["z_both_true"] <= 1.13 for m in ("a", "b", "chosen", "K1")) and near_value(sim["b"]["gap1_R1.5"]["unweighted"]["z_sd"], 0.59, 0.01)
assert real_rows["r32"]["common"]["pairs"] == 4337 and near_value(real_rows["r32"]["common"]["z"]["b"], 1.76, 0.01) and near_value(real_rows["r32"]["common"]["z"]["a"], 1.44, 0.01)
assert all(3.8 <= real_rows[r]["floor"][m]["floor_nm"] / np.sqrt(2) <= 6.0 for r in RECS for m in ("a", "b", "chosen", "K1"))
assert all(real_rows["r32"]["sd_tertiles"][m][0] > real_rows["r32"]["sd_tertiles"][m][2] for m in ("a", "b", "K1", "chosen"))
assert boot["per_block"]["both_95"] == [7500, 13500] == boot["line"]["both_95"] and min(boot["per_block"]["r32"]["stars"]) == 2000 and max(boot["per_block"]["r32"]["stars"]) == 12000
assert sum(v["margin"] < 0 for k, v in frames_boot["band_seeds"].items() if k != "88") == 1 and near_value(margins[88], 11, 0.1)
assert 0.05 <= psf_control["controls"]["after_100"]["share_difference"] <= 0.06 and near_value(psf_control["controls"]["random_sites"]["integral_control"], 0.23, 0.01)
assert (cold["over_1_01"]["K1"], warm["over_1_01"]["K1"], cold["K1_fit"]["within_half_px"]) == (0, 0, 99) and 0.8 <= cold["median_sd_ratio"]["all"] <= 1.25 and 0.8 <= warm["median_sd_ratio"]["all"] <= 1.25
assert near_value(stuck_chains["best_chain_sd_ratio"]["all"], 1.07, 0.01) and stuck_chains["K1_fit"]["on_best_chain"] == 25
assert near_value(real_rows["r7"]["persist_gap1_share_of_rows"], 0.42, 0.005) and near_value(real_rows["r32"]["persist_gap1_share_of_rows"], 0.38, 0.005)
assert near_value(min(r["precision"] for r in emp_a), 0.42, 0.01) and near_value(p8_slow / p8_fast, 3.4, 0.05) and near_value(p4_gpu["steps_per_s"] / p4_cpu["steps_per_s"], 44, 0.5)
assert decisions["b_cdrift"]["r7"]["points"]["2000"]["count"] == 402 and decisions["b_line_cdrift"]["r7"]["points"]["2000"]["count"] == 419
assert abs(as_stored["live_r_hat"] - 3.012) < 0.01 and all(abs(v) < 2.5 for v in np.r_[first["x"].values[:, 0, 0], first["y"].values[:, 0, 0]])
assert live[0]["live_r_hat"] <= 1.01 and live[1]["live_r_hat"] <= 1.01, "with the start on the spot both K = 1 spots converge, whatever the compile cache"
print("The stored numbers the argument rests on: checked. Section 8's K = 2 row is live and is not asserted, because it depends on the compile cache.")
print(f"Replay finished in {(time.perf_counter() - started) / 60:.1f} min on {CORES} cores.")

# %% [markdown]
# ## Try it yourself
#
# The first two run in seconds on the stored summary. Task 3 recompiles, which took 17-44 s here.
#
# 1. **Move the allowance, and the rule.** `n_both` takes the 10%. Replace "every larger N" with "this N only" in `n_star`,
#    and see which recording changes. What would an honest report say if the answer moved by two grid steps?
# 2. **Make drift robust.** Clip the renders passed to `cross` at a low count, say 3 rows per bin, so that no pile can
#    dominate, and rerun block 14 for network A and for classical. Does the classical answer survive? This is the new
#    method the run did not try.
# 3. **Move the start, then free K = 2.** In section 8, run the K = 2 spots with `centred=True`, and see how far the corner start
#    takes them. Then keep the non-centred form and tighten the position prior from 2.0 to 0.5 px. Do the chains now agree? Is
#    that a better model of the pixels, or only a better-behaved sampler? How would you tell the two apart?
#
# The plan's reader tasks asked you to change the simulator and retrain. They need a GPU, and a laptop's time for them is
# unmeasured, so they are left out here. The training code is `tools/run_e88_net.py`.
