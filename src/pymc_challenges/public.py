"""Graphics and wording for explaining a posterior to people who will never read the maths.

Shared by the "Bayes in public" notebooks (E108-E117). Everything here needs only numpy,
matplotlib and Pillow (which ships with matplotlib). Each plotting helper draws on an ``ax``
you pass in, so it works under ``az.style.use("arviz-variat")`` and inside any layout.

House conventions (from the public-communication toolkit):

- one saturated colour for "yes / the thing we count" (``HI``, blue) and one for "no / past the
  line" (``LO``, vermilion); everything else is grey. Blue against vermilion stays distinct
  under the common colour-vision deficiencies, and every helper also separates the two by
  position, hatching or a printed count, so colour is never the only cue;
- a sentence title that states the finding, text at 11-13 pt, labels on the marks rather
  than in a legend;
- probabilities in words *and* numbers (``calibrated_word`` plus the percentage in brackets),
  with the narrowest IPCC band that applies (a house rule; the IPCC bands overlap);
- whole percents, "<1%" and ">99%" at the ends, 89% equal-tailed ranges by default.

Example::

    from pymc_challenges import public as pub
    pub.how_sure(0.78, "floods are more frequent now", "medium", "the gauge moved in 2008")
"""

from __future__ import annotations

import io
import os
import textwrap
import warnings
from collections.abc import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

__all__ = [
    "HI", "LO", "INK", "MUTED", "LIGHT", "IPCC",
    "calibrated_word", "fmt_pct", "how_sure", "freq_range",
    "word_strip", "quantile_dotplot", "icon_array", "freq_tree", "gradient_intervals",
    "save_gif", "gif_contact_sheet",
]

INK, MUTED, LIGHT = "#222222", "#8a8a8a", "#d9d9d9"
HI, LO = "#2166ac", "#d55e00"   # blue and Okabe-Ito vermilion: safe for red-green colour blindness

# IPCC AR5 likelihood scale (Mastrandrea et al. 2010, Table 1), as disjoint bands: the narrowest
# term that applies is used, so "likely" means 66-90% here, "very likely" 90-99%.
IPCC = [
    (0.00, 0.01, "exceptionally unlikely"),
    (0.01, 0.10, "very unlikely"),
    (0.10, 0.33, "unlikely"),
    (0.33, 0.66, "about as likely as not"),
    (0.66, 0.90, "likely"),
    (0.90, 0.99, "very likely"),
    (0.99, 1.00, "virtually certain"),
]


# --------------------------------------------------------------------------- words

def calibrated_word(p: float) -> str:
    """The narrowest IPCC likelihood term for probability ``p`` (lower band wins on a tie).

    >>> calibrated_word(0.93)
    'very likely'
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    return next(w for lo, hi, w in IPCC if lo <= p <= hi)


def fmt_pct(p: float) -> str:
    """A probability as a whole percent, with "<1%" and ">99%" at the ends.

    >>> fmt_pct(0.004), fmt_pct(0.42), fmt_pct(0.996)
    ('<1%', '42%', '>99%')
    """
    p = float(p)
    if p < 0.005:
        return "<1%"
    if p >= 0.995:
        return ">99%"
    return f"{p:.0%}"


def how_sure(p: float, claim: str, confidence: str, why: str) -> str:
    """The "How sure are we" sentence pair: likelihood in words and numbers, then confidence.

    Likelihood words are for the model's probability; ``confidence`` ("low", "medium", "high")
    is for trust in the model and the data, with ``why`` as its reason (the IPCC split).

    >>> how_sure(0.78, "floods are more frequent now", "medium", "the gauge moved in 2008")
    'It is *likely* (78%) that floods are more frequent now. We have *medium confidence* in that figure: the gauge moved in 2008.'
    """
    return (f"It is *{calibrated_word(p)}* ({fmt_pct(p)}) that {claim}. "
            f"We have *{confidence} confidence* in that figure: {why}.")


def freq_range(p_draws, n: int = 100, prob: float = 0.89) -> tuple[int, int, int]:
    """Posterior draws of a rate as whole counts out of ``n``: (low, median, high), equal-tailed.

    >>> freq_range(np.random.default_rng(0).beta(15, 85, 4000))   # about (10, 15, 21)
    """
    a = (1 - prob) / 2
    lo, mid, hi = np.quantile(np.asarray(p_draws, float).ravel() * n, [a, 0.5, 1 - a])
    return int(np.floor(lo + 0.5)), int(np.floor(mid + 0.5)), int(np.floor(hi + 0.5))


# --------------------------------------------------------------------------- helpers

def _title(ax, text, fontsize=12.5, wrap=True):
    """Left-aligned sentence title; wrapped to the axes' width so neighbouring panels do not collide."""
    if not text:
        return
    if wrap:
        fig = ax.figure
        width_in = ax.get_position().width * fig.get_figwidth()
        chars = max(20, int(width_in * 72 / (fontsize * 0.56)))
        text = "\n".join(textwrap.fill(line, chars) for line in str(text).split("\n"))
    ax.set_title(text, loc="left", fontsize=fontsize, color=INK, pad=8)


def _shade(color, toward="white", t=0.5):
    c, w = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return tuple(c + (w - c) * t)


# --------------------------------------------------------------------------- word strip

def word_strip(ax, p: float, claim: str = "", *, band_colors: tuple[str, str] = (LO, HI)):
    """A 0-100% scale cut into the IPCC bands, with a marker at ``p`` labelled in words and numbers.

    >>> word_strip(plt.subplots(figsize=(9, 1.8))[1], 0.93, "The river floods at least once by 2050")
    """
    lo_c, hi_c = band_colors
    for lo, hi, w in IPCC:
        mid = (lo + hi) / 2
        t = abs(mid - 0.5) * 2                       # 0 in the middle band, 1 at the ends
        col = _shade(hi_c if mid > 0.5 else lo_c, "white", 0.85 - 0.6 * t)
        ax.add_patch(Rectangle((lo, 0), hi - lo, 1, fc=col, ec="white", lw=1.5))
        label = {"about as likely as not": "about as likely\nas not"}.get(w, w.replace("very ", "very\n"))
        if hi - lo >= 0.05:
            ax.text(mid, -0.12, label, ha="center", va="top", fontsize=9.5, color=INK, linespacing=1.05)
    for lo, hi, w in (IPCC[0], IPCC[-1]):            # 1%-wide end bands: label beside the scale
        ax.text(-0.01 if lo == 0 else 1.01, 0.5, w.replace(" ", "\n"), ha="right" if lo == 0 else "left",
                va="center", fontsize=9.5, color=INK, linespacing=1.05)
    ax.plot([p, p], [-0.02, 1.02], color=INK, lw=2.2, solid_capstyle="butt")
    ax.plot([p], [1.08], marker="v", ms=11, color=INK, clip_on=False)
    ax.text(min(max(p, 0.09), 0.91), 1.3, f"{calibrated_word(p)} ({fmt_pct(p)})", ha="center",
            va="bottom", fontsize=12, weight="bold", color=INK)
    for x in (0, 0.5, 1):
        if abs(x - p) < 0.06:                        # the marker sits there: its label says it
            continue
        ax.text(x, 1.02, f"{x:.0%}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.95, 1.85)
    ax.axis("off")
    _title(ax, claim)
    return ax


# --------------------------------------------------------------------------- quantile dotplot

def quantile_dotplot(ax, draws, n: int = 20, thresh: float | None = None, *, bins: int | None = None,
                     xlim: tuple[float, float] | None = None, color: str = HI, past_color: str = LO,
                     past: str = "above", unit: str = "possible outcomes", title: str | bool = True,
                     you_are_here: float | None = None, here_label: str = "you are here",
                     xlabel: str | None = None):
    """Quantile dotplot: ``n`` equal-probability quantiles of ``draws``, each a dot ("a 1-in-n chance").

    With ``thresh``, the bins are aligned so no column straddles the line, dots past it (``past`` =
    "above" or "below") take ``past_color``, and the default title counts them. ``you_are_here``
    draws a labelled marker (no mean line: Kale et al. 2021). Returns the quantiles.

    >>> quantile_dotplot(ax, y_draws, n=20, thresh=30, unit="possible arrival times", xlabel="minutes")
    """
    q = np.quantile(np.asarray(draws, float).ravel(), (np.arange(n) + 0.5) / n)
    lo, hi = xlim if xlim is not None else (q.min(), q.max())
    nb = bins or int(np.clip(round(1.6 * np.sqrt(n)) + 3, 8, 30))
    w = (hi - lo) / nb if hi > lo else 1.0
    anchor = thresh if thresh is not None else lo
    edge0 = anchor + np.floor((q.min() - anchor) / w) * w
    col = np.floor((q - edge0) / w + 1e-9).astype(int)
    centres = edge0 + (col + 0.5) * w
    if xlim is None:
        lo, hi = centres.min() - 0.75 * w, centres.max() + 0.75 * w
    if thresh is not None:
        is_past = q >= thresh if past == "above" else q < thresh
    else:
        is_past = np.zeros(n, bool)
    height = {}
    for x, hot in sorted(zip(centres, is_past), key=lambda t: t[0]):
        k = height.get(x, 0)
        ax.add_patch(Circle((x, (k + 0.5) * w), 0.46 * w, fc=past_color if hot else color, ec="none"))
        height[x] = k + 1
    top = max(height.values())
    head = top + (2.2 if you_are_here is not None else 0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.15 * w, head * w)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if thresh is not None:
        ax.plot([thresh, thresh], [-0.15 * w, (top + 0.4) * w], color=INK, lw=1.2)
    if you_are_here is not None:
        if you_are_here != thresh:
            ax.plot([you_are_here] * 2, [-0.15 * w, (top + 0.4) * w], color=INK, lw=1.2, ls=(0, (3, 2)))
        frac = (you_are_here - lo) / (hi - lo)
        ax.annotate(here_label, (you_are_here, (top + 0.4) * w), xytext=(you_are_here, (top + 1.5) * w),
                    ha="left" if frac < 0.6 else "right", va="center", fontsize=11, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=1, shrinkA=0, shrinkB=0))
    if title is True:
        if thresh is not None:
            k = int(is_past.sum())
            title = f"{k} of {n} {unit} are {'past' if past == 'above' else 'below'} the line"
        else:
            title = f"{n} {unit}, each a 1-in-{n} chance"
    _title(ax, title or None)
    return q


# --------------------------------------------------------------------------- icon array

def _person(ax, x, y, s, fc, ec, hatch):
    """A simple person icon in the unit cell at (x, y) (lower-left), size s."""
    head = Circle((x + 0.5 * s, y + 0.78 * s), 0.17 * s, fc=fc, ec=ec, hatch=hatch, lw=0.7)
    body = FancyBboxPatch((x + 0.24 * s, y + 0.04 * s), 0.52 * s, 0.5 * s,
                          boxstyle=f"round,pad=0,rounding_size={0.2 * s}", fc=fc, ec=ec, hatch=hatch, lw=0.7)
    ax.add_patch(head)
    ax.add_patch(body)


def icon_array(ax, lo: int, mid: int, hi: int, n: int = 100, ncol: int | None = None, *,
               label: str = "", what: str = "", color: str = HI, shape: str = "person",
               key: bool = True, title: str | bool = True):
    """An icon array of ``n`` with honest uncertainty: ``lo`` icons solid, ``lo``-``hi`` hatched, rest grey.

    Filled in reading order. ``what`` names the counted group in the key and title ("would
    be over the salt line"). Get (lo, mid, hi) from posterior draws with ``freq_range``.

    >>> icon_array(ax, *freq_range(p_draws, 100), what="are over the salt line", label="Adults")
    """
    lo, mid, hi = int(lo), int(mid), int(hi)
    if not 0 <= lo <= mid <= hi <= n:
        raise ValueError(f"need 0 <= lo <= mid <= hi <= n, got {lo}, {mid}, {hi}, {n}")
    ncol = ncol or (10 if n <= 100 else int(np.ceil(np.sqrt(n * 2))))
    nrow = int(np.ceil(n / ncol))
    light_bg = _shade(color, "white", 0.82)
    for i in range(n):
        r, c = divmod(i, ncol)
        x, y = c, nrow - 1 - r
        if i < lo:
            fc, ec, hatch = color, color, None
        elif i < hi:
            fc, ec, hatch = light_bg, color, "/////"
        else:
            fc, ec, hatch = LIGHT, LIGHT, None
        if shape == "person":
            _person(ax, x, y, 0.92, fc, ec, hatch)
        else:
            ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.8, 0.8, fc=fc, ec=ec, hatch=hatch, lw=0.7))
    bottom = 0
    if key:
        sw = 0.6 * max(1.0, ncol / 14)
        bottom = -(sw + 0.65 * max(1.0, ncol / 14))
        ax.add_patch(Rectangle((0, bottom + 0.2), sw, sw, fc=color, ec=color))
        ax.text(sw * 1.4, bottom + 0.2 + sw / 2, f"{lo} for sure", va="center", fontsize=11, color=INK)
        x2 = ncol * 0.42
        ax.add_patch(Rectangle((x2, bottom + 0.2), sw, sw, fc=light_bg, ec=color, hatch="/////", lw=0.7))
        ax.text(x2 + sw * 1.4, bottom + 0.2 + sw / 2, f"up to {hi - lo} more", va="center",
                fontsize=11, color=INK)
    ax.set_xlim(-0.1, ncol + 0.1)
    ax.set_ylim(bottom - 0.1, nrow + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is True:
        head = f"{label}: a" if label else "A"
        title = f"{head}bout {mid} in {n:,}{' ' + what if what else ''} (somewhere between {lo} and {hi})"
    _title(ax, title or None)
    return ax


# --------------------------------------------------------------------------- frequency tree

def _text_box(art):
    """xycoords callable: the window extent of a text's own glyphs (an Annotation's would include its link)."""
    return lambda renderer: mpl.text.Text.get_window_extent(art, renderer)


def freq_tree(ax, N: int, p1, p2_yes, p2_no, *, unit: str = "people",
              split1: tuple[str, str] = ("have it", "do not"),
              split2: tuple[str, str] = ("test positive", "test negative"),
              prob: float = 0.89, color: str = HI, title: str | bool = True) -> dict:
    """A natural-frequency tree: ``N`` split by ``p1``, then each branch by ``p2_yes`` / ``p2_no``.

    The leaves that ``split2[0]`` (e.g. "test positive") are highlighted, and the title reads the
    posterior off as a ratio of counts. Each probability may be a number or an array of posterior
    draws; with draws the counts come from the posterior means (integers that add up across every
    split) and the title gives the ``prob`` range of the ratio. Returns the counts (and range).

    >>> freq_tree(ax, 1000, prev_draws, sens_draws, fpr_draws, split1=("have it", "do not"))
    """
    d1, dy, dn = (np.atleast_1d(np.asarray(v, float)).ravel() for v in (p1, p2_yes, p2_no))
    a = int(round(N * d1.mean()))
    b = N - a
    ay = int(round(a * dy.mean()))
    bn = int(round(b * dn.mean()))
    out = {"N": N, "yes": a, "no": b, "yes_pos": ay, "yes_neg": a - ay, "no_pos": bn, "no_neg": b - bn}
    rng_txt = ""
    if max(d1.size, dy.size, dn.size) > 1:
        m = max(d1.size, dy.size, dn.size)
        r = [np.resize(v, m) for v in (d1, dy, dn)]
        ppv = r[0] * r[1] / (r[0] * r[1] + (1 - r[0]) * r[2])
        al = (1 - prob) / 2
        out["ratio_range"] = tuple(float(v) for v in np.quantile(ppv, [al, 1 - al]))
        rng_txt = f", range {out['ratio_range'][0]:.0%}-{out['ratio_range'][1]:.0%}"
    # (x, y, text, kind, parent); a child is an annotation tied to its parent's text, so each link
    # runs from box edge to box edge whatever the label lengths
    spec = [
        ("root", 0.0, 0.0, f"{N:,}\n{unit}", "root", None),
        ("yes", 1.0, 1.0, f"{a:,}\n{split1[0]}", "mid", "root"),
        ("no", 1.0, -1.0, f"{b:,}\n{split1[1]}", "mid", "root"),
        ("yes_pos", 2.0, 1.5, f"{ay:,} {split2[0]}", "hot", "yes"),
        ("yes_neg", 2.0, 0.5, f"{a - ay:,} {split2[1]}", "leaf", "yes"),
        ("no_pos", 2.0, -0.5, f"{bn:,} {split2[0]}", "hot", "no"),
        ("no_neg", 2.0, -1.5, f"{b - bn:,} {split2[1]}", "leaf", "no"),
    ]
    arts = {}
    for name, x, y, t, kind, parent in spec:
        hot = kind == "hot"
        style = dict(ha="left" if kind in ("hot", "leaf") else "center", va="center", fontsize=11.5,
                     color="white" if hot else INK, weight="bold" if hot else "normal", linespacing=1.1,
                     zorder={"root": 6, "mid": 5}.get(kind, 4),
                     bbox=dict(boxstyle="round,pad=0.4", fc=color if hot else "white",
                               ec=MUTED if kind == "leaf" else color, lw=1.2))
        if parent is None:
            arts[name] = ax.text(x, y, t, **style)
        else:
            arts[name] = ax.annotate(t, xy=(1, 0.5), xycoords=_text_box(arts[parent]), xytext=(x, y),
                                     textcoords="data",
                                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2, shrinkA=0,
                                                     shrinkB=0, relpos=(0, 0.5)), **style)
    ax.set_xlim(-0.45, 3.2)
    ax.set_ylim(-1.95, 1.95)
    ax.axis("off")
    if title is True:
        pos = ay + bn
        frac = ay / pos if pos else float("nan")
        title = (f"Of the {pos:,} who {split2[0]}, {ay:,} ({frac:.0%}{rng_txt}) {split1[0]}"
                 if pos else f"None of the {N:,} {split2[0]}")
    _title(ax, title or None)
    return out


# --------------------------------------------------------------------------- gradient intervals

def gradient_intervals(ax, draws: Sequence, names: Sequence[str], *, xlim: tuple[float, float] | None = None,
                       color: str = HI, median_tick: bool = True, ref: float | None = None,
                       ref_label: str | None = None, key: bool = True, xlabel: str | None = None,
                       title: str | None = None, smooth: float = 1.0, height: float = 0.7):
    """Gradient strips in place of error bars: each row's opacity follows its posterior density.

    ``draws`` is a list of 1-D arrays, one per name, drawn top to bottom. ``ref`` adds a labelled
    reference line; the key ("darker = more plausible") is written in the axes.

    >>> gradient_intervals(ax, [d_a, d_b, d_c], ["North", "Centre", "South"], ref=0, xlabel="change (%)")
    """
    draws = [np.asarray(d, float).ravel() for d in draws]
    if xlim is None:
        lo = min(np.quantile(d, 0.002) for d in draws)
        hi = max(np.quantile(d, 0.998) for d in draws)
        pad = 0.05 * (hi - lo)
        xlim = (lo - pad, hi + pad)
    grid = np.linspace(*xlim, 401)
    rgb = np.array(to_rgb(color))
    for i, d in enumerate(draws):
        dens = np.histogram(d, bins=grid)[0].astype(float)
        sd = np.std(d) / (grid[1] - grid[0])           # the draws' sd, in bins
        bw = max(1.0, smooth * 0.75 * sd * len(d) ** -0.2)   # a Silverman-like kernel width
        k = np.exp(-0.5 * (np.arange(-4 * bw, 4 * bw + 1) / bw) ** 2)
        dens = np.convolve(dens, k / k.sum(), mode="same")
        dens = dens / dens.max() if dens.max() > 0 else dens
        img = np.zeros((1, dens.size, 4))
        img[..., :3] = rgb
        img[..., 3] = dens ** 0.8                      # a touch of gamma so the tails stay visible
        y = len(draws) - 1 - i
        ax.imshow(img, extent=(xlim[0], xlim[1], y - height / 2, y + height / 2), aspect="auto",
                  interpolation="bilinear", origin="lower", zorder=2)
        if median_tick:
            m = np.median(d)
            ax.plot([m, m], [y - height / 2, y + height / 2], color="white", lw=1.6, zorder=3)
            ax.plot([m, m], [y - height / 2, y + height / 2], color=INK, lw=0.7, zorder=4)
    if ref is not None:
        ax.axvline(ref, color=INK, lw=1, ls=(0, (4, 2)), zorder=1)
        if ref_label:
            ax.text(ref, len(draws) - 0.5, " " + ref_label, ha="left", va="bottom", fontsize=10.5, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(draws) - 0.4 + (0.35 if (ref_label or key) else 0))
    ax.set_yticks(range(len(draws)), list(names)[::-1])
    ax.tick_params(axis="y", length=0, labelsize=11.5)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    if key:
        ax.text(1.0, 1.0, "darker = more plausible" + ("; tick = middle value" if median_tick else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel)
    _title(ax, title)
    return ax


# --------------------------------------------------------------------------- GIFs (the E95 pipeline)

def save_gif(fig, draw_frame: Callable[[int], object], n_frames: int, path: str, *, ms: int = 700,
             first_ms: int = 2000, last_ms: int = 4000, colors: int = 128, dpi: float | None = None,
             max_mb: float = 2.0):
    """Render ``draw_frame(k)`` for k = 0..n-1 on ``fig`` into an animated GIF, as E95 does.

    Each frame is saved as PNG, quantised to ``colors`` (median cut, no dither) and written with
    per-frame durations, holding the first and last frames. Build ``fig`` with
    ``plt.figure(..., layout="none")`` and fixed axes limits so frames do not jump, and name the
    file with the notebook's stem as prefix (``E108_usual_intake.gif``): the site copies every
    ``{stem}*.gif`` beside the page, and ``CARD_FIGURE`` may name it. Constrained layout is
    switched off while saving and restored after (not via ``rc_context``: see AUTHORING, E44).
    Keep ``ms`` >= 400 (WCAG 2.3.1: no more than three flashes a second). Returns the PIL frames
    for ``gif_contact_sheet``.

    >>> frames = save_gif(fig, show_frame, 40, "E108_usual_intake.gif")
    """
    from PIL import Image

    if n_frames < 1:
        raise ValueError("need at least one frame")
    if ms < 334:
        warnings.warn("frames shorter than 334 ms flash more than three times a second (WCAG 2.3.1)",
                      stacklevel=2)
    key = "figure.constrained_layout.use"
    old = mpl.rcParams[key]
    mpl.rcParams[key] = False
    frames = []
    try:
        for k in range(n_frames):
            draw_frame(k)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi or fig.dpi)
            rgb = Image.open(buf).convert("RGB")
            frames.append(rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                       dither=Image.Dither.NONE))
    finally:
        mpl.rcParams[key] = old
    if n_frames == 1:
        durations = [last_ms]
    else:
        durations = [first_ms] + [ms] * (n_frames - 2) + [last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, optimize=True, duration=durations)
    mb = os.path.getsize(path) / 1e6
    print(f"wrote {path}: {n_frames} frames, {frames[0].size[0]}x{frames[0].size[1]} px, {mb:.1f} MB")
    if mb > max_mb:
        warnings.warn(f"{path} is {mb:.1f} MB (budget {max_mb} MB): lower dpi, frames or colors",
                      stacklevel=2)
    return frames


def gif_contact_sheet(frames, picks: Sequence[int] | None = None, *, ncols: int = 2, width: float = 12.0,
                      captions: Sequence[str] | None = None, title: str | None = None):
    """The static companion of a GIF: ``picks`` frames (default six, evenly spaced) in a grid figure.

    For print and screen readers; pair it with alt text in the markdown link sentence. Two columns
    keep a 7-9 inch frame's own text readable; use three only for frames with large type.

    >>> gif_contact_sheet(frames, captions=["day 1", "day 2", ...], title="Six of the 40 frames")
    """
    n = len(frames)
    if picks is None:
        picks = sorted(set(np.linspace(0, n - 1, min(6, n)).round().astype(int)))
    picks = list(picks)
    nrows = int(np.ceil(len(picks) / ncols))
    w0, h0 = frames[0].size
    gap, cap, head = 0.15, (0.35 if captions else 0.0), (0.5 if title else 0.05)
    cell_w = (width - gap * (ncols + 1)) / ncols
    cell_h = cell_w * h0 / w0
    fig_h = head + nrows * (cap + cell_h + gap)
    fig = plt.figure(figsize=(width, fig_h), layout="none")
    for j, k in enumerate(picks):
        r, c = divmod(j, ncols)
        left = gap + c * (cell_w + gap)
        bottom = fig_h - head - (r + 1) * (cap + cell_h + gap) + gap
        ax = fig.add_axes([left / width, bottom / fig_h, cell_w / width, cell_h / fig_h])
        ax.imshow(np.asarray(frames[k].convert("RGB")))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(LIGHT)
        if captions:
            ax.set_title(captions[j], fontsize=11.5, color=INK, pad=4, loc="left")
    if title:
        fig.suptitle(title, fontsize=13, color=INK, x=gap / width, y=1 - 0.12 / fig_h, ha="left", va="top",
                     weight="bold")
    return fig
