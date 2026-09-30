"""Build the E80 dataset: an HST/ACS F814W image of the SLACS gravitational lens SDSS J1627-0053.

    uv run --no-project --with astropy --with numpy --with scipy --with truststore \
        python tools/build_e80_lens.py [path/to/j9c701020_drc.fits]

astropy is only needed here, to read the FITS file and its WCS once; the notebook reads the .npz.

Source: HST proposal 10494 (PI L. Koopmans, SLACS follow-up), visit 01, ACS/WFC F814W, four
exposures (2,224 s in total) on 2006-03-12, combined by the MAST pipeline (AstroDrizzle, CTE-corrected
"drc" product j9c701020_drc.fits, 0.05"/pixel, square kernel, pixfrac 1, units electrons/s). The 215 MB
file is downloaded from MAST (or read from the path given) and only small pieces are kept:

* ``sci``, ``wht``: a 121 x 121-pixel cutout (6") centred on the lens galaxy - the drizzled image
  (electrons/s, the sky already subtracted by the pipeline) and its weight map (the pipeline's
  "EXP" weighting: effective exposure time in seconds per output pixel);
* ``blank_sci``, ``blank_wht``: a 128 x 128-pixel patch of empty sky near the lens, for the noise
  level and the pixel-to-pixel noise correlation that drizzling introduces;
* ``star_stamps`` (k x 41 x 41), ``star_xy`` (their centres in the full frame, pixels): isolated,
  unsaturated stars within 1,500 pixels of the lens, for an empirical PSF;
* every array is turned by a multiple of 90 degrees (lossless) so that north is up and east left
  to within ``north_residual_deg`` (the angle of north from the +y axis, measured towards east =
  towards -x); ``pixel_scale`` (arcsec), ``exptime``, ``photflam``, ``photplam``, ``ra_dec``
  (J2000 of the cutout centre), ``lens_xy_full`` (the lens centre in the original frame).
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

try:  # use the operating system's certificate store (some networks re-sign TLS traffic)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

DATA = Path(__file__).resolve().parents[1] / "data"
URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/j9c701020_drc.fits"
RA, DEC = 246.943519, -0.899322          # SDSS J162746.44-005357.5 (the lens galaxy)
HALF = 60                                # cutout 121 x 121
STAMP = 20                               # star stamps 41 x 41


def main(path: str | None) -> None:
    from astropy.io import fits
    from astropy.wcs import WCS

    if path is None:
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "pymc-challenges"}),
                                    timeout=900) as r:
            raw = io.BytesIO(r.read())
    else:
        raw = path
    with fits.open(raw) as h:
        p, s = h[0].header, h[1].header
        sci = np.asarray(h[1].data, float)
        wht = np.asarray(h[2].data, float)
        wcs = WCS(s)
        cd = np.array([[s["CD1_1"], s["CD1_2"]], [s["CD2_1"], s["CD2_2"]]])
        meta = dict(exptime=float(p["EXPTIME"]), photflam=float(s["PHOTFLAM"]), photplam=float(s["PHOTPLAM"]),
                    date_obs=str(p["DATE-OBS"]), proposal=int(p["PROPOSID"]), rootname="j9c701020",
                    filt=str(p["FILTER2"]))
    sci = np.where(wht > 0, sci, 0.0)
    x0, y0 = wcs.all_world2pix([[RA, DEC]], 0)[0]
    # refine to the brightest pixel of the lens galaxy
    xi, yi = int(round(x0)), int(round(y0))
    box = sci[yi - 5:yi + 6, xi - 5:xi + 6]
    dy, dx = np.unravel_index(np.argmax(box), box.shape)
    xi, yi = xi - 5 + dx, yi - 5 + dy
    scale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600)

    # rotation by k * 90 degrees so that north is (nearly) up: find k from the CD matrix
    # d(RA, Dec)/d(x, y) = cd; north direction in pixel coordinates solves cd @ v = (0, 1)
    north = np.linalg.solve(cd, [0.0, 1.0])
    best = None
    for k in range(4):
        c, s_ = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        # np.rot90(a, k) turns the array counterclockwise (as displayed with origin="lower") by k*90 deg
        v = np.array([c * north[0] - s_ * north[1], s_ * north[0] + c * north[1]])
        ang = np.degrees(np.arctan2(-v[0], v[1]))        # angle of north from +y, towards -x (east)
        if best is None or abs(ang) < abs(best[1]):
            best = (k, ang)
    k_rot, north_res = best
    # np.rot90 works on (row, col) = (y, x) arrays displayed with origin="upper"; with origin="lower"
    # a counterclockwise turn by 90 degrees is np.rot90(a, -1)
    rot = lambda a: np.rot90(a, -k_rot, axes=(-2, -1)).copy()

    cut = (slice(yi - HALF, yi + HALF + 1), slice(xi - HALF, xi + HALF + 1))
    out = dict(sci=rot(sci[cut]).astype(np.float32), wht=rot(wht[cut]).astype(np.float32))

    # empty sky: the 128 x 128 patch within 600 pixels with the smallest 99.9th percentile
    best_patch = None
    for oy in range(-600, 601, 64):
        for ox in range(-600, 601, 64):
            if abs(ox) < 200 and abs(oy) < 200:
                continue
            sl = (slice(yi + oy, yi + oy + 128), slice(xi + ox, xi + ox + 128))
            if np.any(wht[sl] < 0.6 * np.median(wht[cut])):
                continue
            q = np.percentile(ndimage.uniform_filter(sci[sl], 3), 99.9)
            if best_patch is None or q < best_patch[0]:
                best_patch = (q, sl)
    out["blank_sci"] = rot(sci[best_patch[1]]).astype(np.float32)
    out["blank_wht"] = rot(wht[best_patch[1]]).astype(np.float32)

    # stars: compact, unsaturated, isolated local maxima
    mx = ndimage.maximum_filter(sci, 15)
    ys, xs = np.nonzero((sci == mx) & (sci > 15) & (sci < 150) & (wht > 0.8 * np.median(wht[cut])))
    cands = []
    for y, x in zip(ys, xs):
        if np.hypot(x - xi, y - yi) > 1500 or np.hypot(x - xi, y - yi) < 100:
            continue
        if y < 40 or x < 40 or y > sci.shape[0] - 40 or x > sci.shape[1] - 40:
            continue
        st = sci[y - STAMP:y + STAMP + 1, x - STAMP:x + STAMP + 1]
        tot = st[8:-8, 8:-8].sum()
        c3 = sci[y - 1:y + 2, x - 1:x + 2].sum() / tot
        yy, xx = np.indices(st.shape) - STAMP
        wgt = np.clip(st, 0, None) * (np.hypot(xx, yy) < 12)
        r2 = (wgt * (xx**2 + yy**2)).sum() / wgt.sum()
        others = st.copy()
        others[np.hypot(xx, yy) < 8] = 0
        if c3 > 0.56 and r2 < 10.5 and others.max() < 0.02 * sci[y, x] + 0.1:
            cands.append((np.hypot(x - xi, y - yi), x, y))
    cands.sort()
    cands = cands[:10]
    out["star_stamps"] = np.stack([rot(sci[y - STAMP:y + STAMP + 1, x - STAMP:x + STAMP + 1])
                                   for _, x, y in cands]).astype(np.float32)
    out["star_xy"] = np.array([(x, y) for _, x, y in cands], float)
    ra_c, dec_c = wcs.all_pix2world([[xi, yi]], 0)[0]
    out.update(pixel_scale=scale, north_residual_deg=north_res, rot90_k=k_rot, ra_dec=np.array([ra_c, dec_c]),
               lens_xy_full=np.array([xi, yi], float), exptime=meta["exptime"], photflam=meta["photflam"],
               photplam=meta["photplam"], date_obs=meta["date_obs"], proposal=meta["proposal"],
               rootname=meta["rootname"], filter=meta["filt"])
    np.savez_compressed(DATA / "slacs_j1627_f814w.npz", **out)
    print(f"lens at ({xi}, {yi}); scale {scale:.4f}\"; rot90 k={k_rot}, north residual {north_res:.2f} deg; "
          f"{len(cands)} stars at distances {[round(c[0]) for c in cands]}; blank patch q99.9 {best_patch[0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
