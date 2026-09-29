"""Build data/loc_posters_thumbs.npz: small thumbnails of Library of Congress posters (E72).

Three eras of printed "creatives" from the LoC Prints & Photographs poster collections (public
domain / no known restrictions): 1890s magazine and book posters, First World War posters, and
Works Progress Administration posters (1936-1943). For each era the script pages through the
LoC JSON API, keeps items with a dated year in the era and a reference image, draws a random
sample, downloads the 640-px reference JPEG and stores a 64 x 48 (height x width) RGB thumbnail.

    uv run python tools/fetch_loc_posters.py          # ~15 minutes, ~800 downloads
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "data" / "loc_posters_thumbs.npz"
UA = {"User-Agent": "pymc-challenges"}  # a browser-like UA triggers a Cloudflare challenge
PER_ERA = 250
H, W = 64, 48

# (era label, search URL template, allowed year range)
ERAS = [
    ("1890s magazine", "https://www.loc.gov/pictures/search/?q=magazine&co=pos&fo=json&c=100&sp={p}",
     (1889, 1905)),
    ("WWI", "https://www.loc.gov/pictures/search/?q=world%20war%201914-1918&co=pos&fo=json&c=100&sp={p}",
     (1914, 1919)),
    ("WPA", "https://www.loc.gov/pictures/search/?co=wpapos&fo=json&c=100&sp={p}", (1935, 1943)),
]


def get(url: str, tries: int = 5) -> bytes:
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network hiccups: back off and retry
            time.sleep(2 * (t + 1))
    raise RuntimeError(f"failed: {url}")


def year_of(s: str | None) -> int | None:
    m = re.search(r"(18|19)\d\d", s or "")
    return int(m.group(0)) if m else None


def listing(template: str, lo: int, hi: int) -> list[dict]:
    items, seen, p = [], set(), 1
    while True:
        d = json.loads(get(template.format(p=p)))
        for r in d["results"]:
            y = year_of(r.get("created_published_date"))
            full = (r.get("image") or {}).get("full")
            if y is None or not (lo <= y <= hi) or not full or r["pk"] in seen:
                continue
            seen.add(r["pk"])
            items.append({"pk": r["pk"], "title": r.get("title", ""), "year": y, "url": full,
                          "creator": r.get("creator") or ""})
        if not d.get("pages", {}).get("next"):
            break
        p += 1
        time.sleep(0.3)
    return items


def main() -> None:
    rng = np.random.default_rng(72)
    rows, thumbs = [], []
    for era, template, (lo, hi) in ERAS:
        items = listing(template, lo, hi)
        pick = rng.permutation(len(items))
        print(f"{era}: {len(items)} dated items with images", flush=True)
        n = 0
        for i in pick:
            if n == PER_ERA:
                break
            it = items[i]
            try:
                img = Image.open(io.BytesIO(get(it["url"]))).convert("RGB")
            except Exception:  # noqa: BLE001 - skip unreadable images
                continue
            if min(img.size) < 100:
                continue
            thumbs.append(np.asarray(img.resize((W, H), Image.Resampling.LANCZOS), dtype=np.uint8))
            rows.append({**it, "era": era})
            n += 1
            time.sleep(0.2)
        print(f"  kept {n}", flush=True)
    np.savez_compressed(
        OUT,
        thumbs=np.stack(thumbs),
        era=np.array([r["era"] for r in rows]),
        year=np.array([r["year"] for r in rows]),
        title=np.array([r["title"] for r in rows]),
        creator=np.array([r["creator"] for r in rows]),
        loc_id=np.array([str(r["pk"]) for r in rows]),
        image_url=np.array([r["url"] for r in rows]),
    )
    print(OUT, OUT.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
