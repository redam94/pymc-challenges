"""Verification of the E118 data: registry, coverage, a seeded trace check against the raw sources, dating."""
import gzip, io, json, re, sys, unicodedata, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
from pymc_challenges import data

RAW = Path.home() / ".cache/pymc-challenges/e118/raw"
NAMES = ["e118_ideology", "e118_ideology_flags", "e118_results_2026", "e118_results_2026_margins", "e118_party_ie",
         "e118_party_direct", "e118_sources", "e118_build_log"]
print("== 2. describe / load")
D = {}
for n in NAMES:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data.describe(n)
    try:
        D[n] = data.load(n)
        print(f"  {n:<28} describe: {len(buf.getvalue()):>5} chars; load: {D[n].shape[0]:>6} rows x {D[n].shape[1]:>2} cols")
    except ValueError as e:
        js = json.load(open(data.path(n)))
        print(f"  {n:<28} describe: {len(buf.getvalue()):>5} chars; load raises by design ({str(e)[:38]}...); json keys: {len(js)}")
nom, ev, res, mg, ie, pdct, src = (D[n] for n in NAMES[:7])
FLAGS = [c for c in nom.columns[nom.columns.get_loc("nokken_poole_dim1_posthoc") + 2:]]

print("\n== 3. coverage of polled races (a race counts when its Democratic-side nominee has the measure)")
d = nom[(nom.side == "D") & nom.polled]
rows = []
for (c, o), g in d.groupby(["cycle", "office"]):
    races = set(g.race)
    row = {"cycle": c, "office": o, "polled": len(g), "D sat in Congress": int(g.served_before.sum()),
           "nominate (dated)": int(g.nominate_dim1.notna().sum()), "nominate (posthoc)": int(g.nominate_dim1_posthoc.notna().sum()),
           "any dated flag": int(g[FLAGS].fillna(False).any(axis=1).sum()),
           "flags available": int(sum(g[f].notna().any() for f in FLAGS)),
           "IE split": len(races & set(ie[(ie.cycle == c) & (ie.version == "eve") & ie.nominee].race)),
           "party direct": len(races & set(pdct[(pdct.cycle == c)].race)),
           "2026 counts": len(races & set(res.race)) if c == 2026 else "",
           "2026 counts, clean": len(races & set(res[res.two_party_consistent != False].race)) if c == 2026 else "",
           "2026 margin only": len((races & set(mg.race)) - set(res.race)) if c == 2026 else ""}
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== Democratic nominees in polled races carrying each measure (blank = no dated list that cycle)")
t = {}
for c, g in d.groupby("cycle"):
    t[c] = {f: (int(g[f].fillna(False).sum()) if g[f].notna().any() else "") for f in FLAGS}
    t[c]["nominate_dim1 (dated)"] = int(g.nominate_dim1.notna().sum())
    t[c]["nominate_dim1_posthoc"] = int(g.nominate_dim1_posthoc.notna().sum())
    t[c]["(polled D nominees)"] = len(g)
print(pd.DataFrame(t).to_string())
r = nom[(nom.side == "R") & nom.polled]
print("Republican nominees in polled races:", {c: {f: int(g[f].fillna(False).sum()) for f in ("hfc", "rsc", "rmsp", "rgg", "psc", "hfc_fund") if g[f].notna().any()} | {"nominate_posthoc": int(g.nominate_dim1_posthoc.notna().sum()), "n": len(g)} for c, g in r.groupby("cycle")})

print("\n== 4. trace check: 12 flags drawn with seed 118, each re-read from the raw file of its dated source")
def plain(x):
    return re.sub(r"[^a-z ]", " ", unicodedata.normalize("NFKD", re.sub("['’]", "", str(x))).encode("ascii", "ignore").decode().lower())
pick = ev.sample(12, random_state=118)
ok = 0
for r_ in pick.itertuples():
    sur = plain(r_.name.split(",")[0] if "," in r_.name else r_.name.split()[-1]).split()
    sur = [w for w in sur if len(w) > 2][-1]
    if str(r_.source_revid).isdigit() and len(str(r_.source_revid)) < 14:
        f = list((RAW / "wikipedia").glob(f"*__rev{r_.source_revid}.wikitext"))[0]
        lines = f.read_text(encoding="utf-8").splitlines()
        seg = lines[r_.line - 1: r_.line + 12]           # the bullet line, or the table row that starts there
        cut = next((i for i, x in enumerate(seg[1:], 1) if x.strip().startswith(("|-", "|}", "*", "=="))), len(seg))
        raw = " ".join(seg[:cut])
        meta = json.loads(f.with_suffix(".json").read_text())
        stamp = meta["timestamp"]
    else:
        f = list((RAW / "wayback").glob(f"*__{r_.source_revid}.html"))[0]
        b = f.read_bytes(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
        txt = plain(re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<(script|style).*?</\1>", " ", b.decode("utf-8", "replace"))))
        txt = " ".join(txt.split())
        given = [w for w in plain(r_.text.split(" | ")[0]).split() if len(w) > 2][:1]   # the name as the page prints it
        # every place the page prints the surname; keep the one that also carries the given name
        m = [x.start() for x in re.finditer(r"\b" + re.escape(sur) + r"\b", txt)]
        m = [i for i in m if not given or given[0] in txt[max(i - 40, 0): i + 10]]
        raw = txt[max(m[0] - 40, 0): m[0] + 60] if m else ""
        stamp = r_.source_time_utc
    given_ok = True
    if "," in r_.name or " " in r_.name:
        g = [w for w in plain(r_.name.split(",", 1)[1] if "," in r_.name else " ".join(r_.name.split()[:-1])).split() if len(w) > 2]
        given_ok = (not g) or any(w[:3] in plain(raw) for w in g)
    hit = (sur in plain(raw)) and given_ok
    sec = re.search(str(r_.section).split(" > ")[-1][:20], str(r_.section)) is not None
    ok += bool(hit)
    print(f"  {'OK ' if hit else 'FAIL'} {r_.cycle} {r_.race:<6} {r_.side} {r_.name[:24]:<24} {r_.flag:<14} {str(r_.source_revid):<14} {stamp[:10]} L{r_.line:<5} | {raw[:95]}")
print(f"  {ok}/12 source lines name the flagged nominee")

print("\n== 5. dating check")
ev["day"] = ev.source_time_utc.str[:10]
el = {2018: "2018-11-06", 2022: "2022-11-08", 2026: "2026-11-03"}
for c, g in ev.groupby("cycle"):
    print(f"  {c}: flags {len(g):>4}; latest source of any flag {g.source_time_utc.max()}  < election day {el[c]}: {g.source_time_utc.max()[:10] < el[c]}")
s2 = src[src.kind.isin(["wikipedia", "wayback"]) | ((src.kind == "voteview") & src.cycle.notna() & (src.version != "current file"))]
for c, g in s2.groupby("cycle"):
    print(f"  {int(c)}: sources {len(g):>2}; latest version {g.version_time_utc.max()} ({g.loc[g.version_time_utc.idxmax(), 'title'][:40]}); all before election day: {(g.version_time_utc.str[:10] < el[int(c)]).all()}")
print("  nominate_dim1 (dated) snapshots:", nom[nom.nominate_dim1.notna()].groupby("cycle").nominate_snapshot_utc.max().to_dict())
