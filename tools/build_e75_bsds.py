"""Build the E75 dataset: a small subset of the Berkeley Segmentation Dataset (BSDS500).

    .venv/bin/python tools/build_e75_bsds.py

Downloads BSR_bsds500.tgz (70 MB) once into .scratch/e75/ and writes

* data/bsds500_subset.npz - ten landscape-format BSDS500 photographs (two from the TRAIN split,
  used in E75 only to choose the spatial coupling, and eight from the TEST split) at HALF
  resolution, with every human segmentation of each:

    ids      (10,)            BSDS image ids
    split    (10,)            'train' / 'test'
    images   (10, 161, 241, 3) uint8 sRGB, 2 x 2 box-averaged from 321 x 481
    human    (10, 7, 161, 241) int16 segment labels (1, 2, ...) of each annotator, subsampled
                               [::2, ::2] from the full-resolution label maps; 0 = no such
                               annotator (images have 5 to 7)
    n_human  (10,)            number of annotators per image

Source: Arbelaez, Maire, Fowlkes & Malik (2011), Contour detection and hierarchical image
segmentation, IEEE TPAMI 33(5):898-916; the images are from the Corel collection and the
segmentations from Martin, Fowlkes, Tal & Malik (2001), ICCV. The Berkeley page allows
downloading "a portion of the dataset for non-commercial research and educational purposes".
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
CACHE = ROOT / ".scratch" / "e75" / "BSR_bsds500.tgz"
OUT = ROOT / "data" / "bsds500_subset.npz"
PICK = [("train", "118035"), ("train", "113044"),
        ("test", "100007"), ("test", "8068"), ("test", "3063"), ("test", "228076"),
        ("test", "97010"), ("test", "29030"), ("test", "16068"), ("test", "108004")]
MAX_HUMAN = 7


def main() -> None:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, CACHE)
    images, human, n_human = [], [], []
    with tarfile.open(CACHE) as tf:
        for split, iid in PICK:
            base = "BSR/BSDS500/data"
            img = np.asarray(Image.open(tf.extractfile(f"{base}/images/{split}/{iid}.jpg")), float)
            assert img.shape == (321, 481, 3), (iid, img.shape)
            pad = np.pad(img, ((0, 1), (0, 1), (0, 0)), mode="edge")      # 322 x 482
            half = pad.reshape(161, 2, 241, 2, 3).mean((1, 3))
            images.append(np.round(half).astype(np.uint8))
            gt = sio.loadmat(tf.extractfile(f"{base}/groundTruth/{split}/{iid}.mat"))["groundTruth"]
            segs = np.zeros((MAX_HUMAN, 161, 241), np.int16)
            for a in range(gt.shape[1]):
                segs[a] = gt[0, a]["Segmentation"][0, 0][::2, ::2]
            human.append(segs)
            n_human.append(gt.shape[1])
            print(split, iid, "annotators", gt.shape[1],
                  "segments", [len(np.unique(s)) for s in segs[: gt.shape[1]]])
    np.savez_compressed(OUT, ids=np.array([p[1] for p in PICK]), split=np.array([p[0] for p in PICK]),
                        images=np.stack(images), human=np.stack(human), n_human=np.array(n_human))
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
