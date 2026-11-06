"""Build data/e105_tour_pageviews.csv: daily English Wikipedia pageviews around three tour announcements.

Source: the Wikimedia REST API (per-article pageviews, agent=user, all-access), CC0. No key needed.
Each tour is listed with the article used, the announcement date and the date general sale opened;
the dates are public knowledge and the script checks that the series has a peak or bump near each.

    .venv/bin/python tools/build_e105_data.py
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "e105_tour_pageviews.csv"
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/daily/{s}/{e}"
UA = {"User-Agent": "pymc-challenges dataset builder (https://github.com/; contact m.reda94@gmail.com)"}

# tour, article, announcement, general on-sale (first public sale day), window start, window end
TOURS = [
    ("Coldplay: Music of the Spheres World Tour", "Coldplay", "2021-10-14", "2021-10-22", "2021-08-15", "2022-03-13"),
    ("Taylor Swift: The Eras Tour", "The_Eras_Tour", "2022-11-01", "2022-11-15", "2022-11-01", "2023-03-30"),
    ("Oasis: Live '25", "Oasis_(band)", "2024-08-27", "2024-08-31", "2024-06-28", "2025-01-24"),
]


def fetch(article: str, start: str, end: str) -> pd.Series:
    url = API.format(a=urllib.parse.quote(article, safe=""), s=start.replace("-", ""), e=end.replace("-", ""))
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    items = json.loads(r.text)["items"]
    return pd.Series({pd.Timestamp(i["timestamp"][:8]): int(i["views"]) for i in items}, name="views")


def main() -> None:
    rows = []
    for tour, art, ann, onsale, s, e in TOURS:
        v = fetch(art, s, e)
        v = v.reindex(pd.date_range(s, e), fill_value=0)
        df = pd.DataFrame({"tour": tour, "article": art, "date": v.index.strftime("%Y-%m-%d"), "views": v.to_numpy(),
                           "announced": ann, "on_sale": onsale})
        rows.append(df)
        near = v[pd.Timestamp(onsale) - pd.Timedelta(days=1): pd.Timestamp(onsale) + pd.Timedelta(days=3)]
        before = v[pd.Timestamp(onsale) - pd.Timedelta(days=4): pd.Timestamp(onsale) - pd.Timedelta(days=2)]
        print(f"{tour}: {len(v)} days, peak {v.max():,} on {v.idxmax():%Y-%m-%d}; on-sale bump {near.max():,} "
              f"against {before.mean():,.0f} just before")
    out = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
