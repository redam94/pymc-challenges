"""Build the E106 dataset: US film releases 2015-2019 from Wikidata, with daily English Wikipedia
pageviews of each film's article from 16 weeks before to 8 weeks after its US release.

    python tools/build_e106_data.py pull     # SPARQL + MediaWiki redirects + pageviews (resumable)
    python tools/build_e106_data.py derive   # write data/e106_film_attention.csv.gz

Sources (both public, no key, no login):
- Wikidata (https://www.wikidata.org), CC0 1.0. SPARQL endpoint https://query.wikidata.org/sparql:
  films (P31 = Q11424) with a US publication date (P577 qualified by P291 = Q30) between
  2015-11-01 and 2019-12-31, a box-office value (P2142), and an English Wikipedia article.
  Box office is taken in USD (Q4917) only: the worldwide value (qualifier P3005 = Q13780930) when
  present, else the US value (P3005 = Q30), else the largest unqualified USD value. Cost (P2130,
  USD) is the production budget where Wikidata has one. `series` = the film has "part of the
  series" (P179). Genres (P136) are kept as a few flags.
- Wikimedia pageviews REST API (https://wikimedia.org/api/rest_v1/), released CC0: per-article
  daily views, agent = user, all access methods, for the article AND for every redirect to it
  (MediaWiki API prop=redirects), summed by day. Articles are often created or moved before a
  release under another title ("Untitled ... film", a working title); the old title then becomes a
  redirect and keeps its own counts, so summing the redirects recovers that history. Daily
  pageviews start on 2015-07-01, which is why releases start in November 2015.

Release date: Wikidata lists premieres, festivals and the wide release side by side. We keep the
US-qualified dates (from 2014-06-01 on) and take the LAST one within 21 days of the first (a limited
opening followed by the wide release keeps the wide date); US dates more than 21 days after the first
are ignored (re-releases, home media mislabelled as releases), and a film whose first US release
falls before 2015-11-01 is dropped.

Facts this script depends on (checked on 2026-11-05):
- Wikidata box office is mostly worldwide totals and is filled for a minority of films; the set is
  a sample of releases tilted to bigger films, not the whole calendar.
- The pageviews API returns 404 for a title with no views in the range: those count as zero.
- US dates are read only up to D1, so a late-December 2019 limited release whose wide date fell in
  January 2020 keeps the limited date (1917: 2019-12-25 here, wide 2020-01-10). Reading dates to
  D1 + 21 days would fix it but changes the file; the notebook states the one affected film.
- Box office scope: of the 206 films the notebook keeps, 64 have an unqualified USD value
  (bo_scope 'unknown') and 1 a US-only value; the rest are worldwide. Foreign-language films with
  US releases (The Climbers, Shoplifters) pass the filters.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "e106_film_attention.csv.gz"
RAW = Path.home() / ".cache" / "pymc-challenges" / "e106"
UA = "pymc-challenges-e106/1.0 (teaching notebook; https://github.com/redam94/pymc-challenges)"
SPARQL = "https://query.wikidata.org/sparql"
MW = "https://en.wikipedia.org/w/api.php"
PV = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
D0, D1 = "2015-11-01", "2019-12-31"
Q0 = "2014-06-01"                   # US dates are read from here, so that an earlier first release is seen
PRE, POST = 112, 55                 # days before / after release kept (16 weeks, 8 weeks)
GENRES = {"animated": "Q202866", "horror": "Q200092", "comedy": "Q157443", "action": "Q188473",
          "superhero": "Q1535153", "drama": "Q130232", "scifi": "Q471839", "family": "Q1361932",
          "documentary": "Q93204", "thriller": "Q2484376", "romance": "Q1054574", "musical": "Q842256"}

QUERY = f"""
SELECT ?film ?filmLabel ?article ?usdate ?bo ?boUnit ?boPlace ?cost ?costUnit ?series ?genre WHERE {{
  ?film wdt:P31 wd:Q11424 .
  ?film p:P577 ?ds . ?ds ps:P577 ?usdate . ?ds pq:P291 wd:Q30 .
  FILTER(?usdate >= "{Q0}T00:00:00Z"^^xsd:dateTime && ?usdate <= "{D1}T00:00:00Z"^^xsd:dateTime)
  ?film p:P2142 ?bs . ?bs psv:P2142 ?bv . ?bv wikibase:quantityAmount ?bo ; wikibase:quantityUnit ?boUnit .
  OPTIONAL {{ ?bs pq:P3005 ?boPlace . }}
  OPTIONAL {{ ?film p:P2130 ?cs . ?cs psv:P2130 ?cv . ?cv wikibase:quantityAmount ?cost ;
              wikibase:quantityUnit ?costUnit . }}
  OPTIONAL {{ ?film wdt:P179 ?series . }}
  OPTIONAL {{ ?film wdt:P136 ?genre . }}
  ?article schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""


def fetch(url: str, attempts: int = 6, accept: str | None = None) -> bytes | None:
    hdr = {"User-Agent": UA}
    if accept:
        hdr["Accept"] = accept
    req = urllib.request.Request(url, headers=hdr)
    for a in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if a == attempts - 1:
                raise
            if e.code == 429:                       # rate limited: back off for longer
                time.sleep(15 * (a + 1))
                continue
        except (urllib.error.URLError, TimeoutError):
            if a == attempts - 1:
                raise
        time.sleep(2 ** a)
    return None


def pull() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    qf = RAW / "wikidata.csv"
    if not qf.exists():
        url = SPARQL + "?" + urllib.parse.urlencode({"query": QUERY})
        qf.write_bytes(fetch(url, accept="text/csv"))
    films = films_table(pd.read_csv(qf))
    rf = RAW / "redirects.json"
    red = json.loads(rf.read_text()) if rf.exists() else {}
    todo = [t for t in films.title if t not in red]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        cont: dict = {}
        while True:
            params = {"action": "query", "prop": "redirects", "titles": "|".join(batch), "rdlimit": "max",
                      "rdnamespace": "0", "format": "json", "formatversion": "2", **cont}
            js = json.loads(fetch(MW + "?" + urllib.parse.urlencode(params)))
            norm = {n["to"]: n["from"] for n in js["query"].get("normalized", [])}
            for pg in js["query"]["pages"]:
                key = norm.get(pg["title"], pg["title"]).replace(" ", "_")
                red.setdefault(key, [])
                red[key] += [r["title"].replace(" ", "_") for r in pg.get("redirects", [])]
            if "continue" not in js:
                break
            cont = js["continue"]
        for t in batch:
            red.setdefault(t, [])
        rf.write_text(json.dumps(red))
    pvdir = RAW / "pageviews"
    pvdir.mkdir(exist_ok=True)
    n = 0
    for _, f in films.iterrows():
        a = (f.release - pd.Timedelta(days=PRE)).strftime("%Y%m%d")
        b = (f.release + pd.Timedelta(days=POST)).strftime("%Y%m%d")
        for t in [f.title] + red.get(f.title, []):
            out = pvdir / (urllib.parse.quote(t, safe="") + f"_{a}.json")
            if out.exists():
                continue
            raw = fetch(f"{PV}/{urllib.parse.quote(t, safe='')}/daily/{a}00/{b}00")
            out.write_bytes(raw if raw is not None else b'{"items": []}')
            n += 1
            time.sleep(0.12)
    print(f"{len(films)} films, {sum(len(v) for v in red.values())} redirects, {n} new pageview requests")


def films_table(d: pd.DataFrame) -> pd.DataFrame:
    usd = d.boUnit.str.endswith("/Q4917")
    rows = []
    for qid, x in d.groupby("film"):
        dates = sorted(pd.to_datetime(x.usdate.str[:10]).unique())
        rel = max(t for t in dates if (t - dates[0]).days <= 21)
        xb = x[usd.loc[x.index]]
        if xb.empty:
            continue
        ww = xb[xb.boPlace.str.endswith("/Q13780930", na=False)].bo
        us = xb[xb.boPlace.str.endswith("/Q30", na=False)].bo
        bo = ww.max() if len(ww) else (us.max() if len(us) else xb[xb.boPlace.isna()].bo.max())
        if not np.isfinite(bo):
            continue
        cost = x[x.costUnit.str.endswith("/Q4917", na=False)].cost.max()
        title = urllib.parse.unquote(x.article.iloc[0].split("/wiki/")[1])
        label = x.filmLabel.iloc[0]
        if str(label).startswith("Q") and str(label)[1:].isdigit():   # no English label: use the article
            label = title.replace("_", " ")
        g = set(x.genre.dropna().str.rsplit("/", n=1).str[-1])
        rows.append({"qid": qid.rsplit("/", 1)[-1], "label": label,
                     "title": urllib.parse.unquote(x.article.iloc[0].split("/wiki/")[1]),
                     "release": pd.Timestamp(rel), "n_us_dates": len(dates),
                     "box_office_usd": float(bo), "bo_scope": "worldwide" if len(ww) else ("us" if len(us) else "unknown"),
                     "budget_usd": float(cost) if np.isfinite(cost) else np.nan,
                     "series": bool(x.series.notna().any()),
                     **{f"g_{k}": v in g for k, v in GENRES.items()}})
    f = pd.DataFrame(rows)
    return f[(f.release >= D0) & (f.release <= D1)].sort_values("release").reset_index(drop=True)


def derive() -> None:
    films = films_table(pd.read_csv(RAW / "wikidata.csv"))
    red = json.loads((RAW / "redirects.json").read_text())
    pvdir = RAW / "pageviews"
    out = []
    for _, f in films.iterrows():
        a = (f.release - pd.Timedelta(days=PRE)).strftime("%Y%m%d")
        days = pd.date_range(f.release - pd.Timedelta(days=PRE), f.release + pd.Timedelta(days=POST))
        tot, main = np.zeros(len(days)), np.zeros(len(days))
        titles = [f.title] + red.get(f.title, [])
        for k, t in enumerate(titles):
            items = json.loads((pvdir / (urllib.parse.quote(t, safe="") + f"_{a}.json")).read_text())["items"]
            for it in items:
                j = days.get_loc(pd.Timestamp(it["timestamp"][:8]))
                tot[j] += it["views"]
                if k == 0:
                    main[j] += it["views"]
        out.append(pd.DataFrame({"qid": f.qid, "date": days, "day": np.arange(-PRE, POST + 1),
                                 "views": tot.astype(np.int64), "views_main_title": main.astype(np.int64)}))
    pv = pd.concat(out)
    df = pv.merge(films.drop(columns=["title"]).assign(title=films.title), on="qid")
    cols = ["qid", "label", "title", "release", "n_us_dates", "box_office_usd", "bo_scope", "budget_usd", "series"] + \
        [c for c in df if c.startswith("g_")] + ["date", "day", "views", "views_main_title"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT, index=False, compression="gzip", date_format="%Y-%m-%d")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): {films.shape[0]} films, {len(df):,} rows")


if __name__ == "__main__":
    {"pull": pull, "derive": derive}[sys.argv[1]]()
