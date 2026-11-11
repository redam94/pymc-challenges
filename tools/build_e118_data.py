"""Build E118's data: nominee ideology from sources dated before each election, provisional 2026
results, and independent expenditures split by who spent.

Raw downloads are cached in ~/.cache/pymc-challenges/e118/raw/ (or RAW_DIR). A file that is missing
from the cache is downloaded with a plain HTTP GET (public pages and public data files only, a
generic client name, a pause between requests); every file read is listed with its sha256 in
data/e118_build_log.json and, with the version that was read, in data/e118_sources.csv. The
downloads are untrusted data: parsed with the standard library and pandas, never executed. Run in
isolated mode, under the memory guard:

    .venv/bin/python tools/memguard.py --max-gb 3 -- .venv/bin/python -I tools/build_e118_data.py [RAW_DIR]

THE DATING RULE. Every ideology flag comes from a version of its source that existed before that
cycle's election day (2018-11-06, 2022-11-08, 2026-11-03): a Wikipedia revision selected as the
last one at or before the end of the eve (so the choice is reproducible: the last revision before
a past instant never changes), or an Internet Archive snapshot of an organisation's own page,
pinned by its capture time. A flag is true only when the source lists the person for that cycle;
nobody is classified by judgment and no gap is filled by inference. Measures that exist only in a
version made after the election are kept in columns whose names end in `_posthoc`.

Reads data/e94_races.csv and data/e94_polls.csv (the race frame, nominees and FEC IDs of E94).
Writes data/e118_ideology.csv, e118_ideology_flags.csv, e118_results_2026.csv,
e118_results_2026_margins.csv, e118_party_ie.csv.gz, e118_party_direct.csv, e118_sources.csv and e118_build_log.json.
`--offline` forbids any download (the build then needs a complete cache).
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OFFLINE = "--offline" in sys.argv
RAW = Path(ARGS[0]) if ARGS else Path.home() / ".cache/pymc-challenges/e118/raw"
OUT = Path(__file__).resolve().parents[1] / "data"

# Election days and the information cutoff ("eve"), as in tools/build_e94_data.py.
ELECTION = {2018: "2018-11-06", 2022: "2022-11-08", 2026: "2026-11-03"}
EVE = {2018: "2018-11-05", 2022: "2022-11-07", 2026: "2026-11-02"}
CYCLES = (2018, 2022, 2026)
# The last Congress that began before each election day.
CONGRESS = {2018: 115, 2022: 117, 2026: 119}
# The provisional 2026 results are the last Wikipedia revisions at or before this instant (UTC).
RESULTS_ASOF = "2026-11-11T22:20:00Z"
GZ = {"method": "gzip", "mtime": 0}
UA = "pymc-challenges-data-builder/1.0 (python-urllib)"
LOG: dict = {"cutoffs": EVE, "election_days": ELECTION, "results_2026_asof_utc": RESULTS_ASOF,
             "checks": {}, "counts": {}, "notes": {}}

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
ABBR = {v: k for k, v in STATES.items()}


# ---------------------------------------------------------------- raw cache and polite downloads
USED: dict[str, str] = {}
SOURCES: list[dict] = []
_LAST = {"t": 0.0}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def use(p) -> Path:
    p = Path(p)
    USED[str(p.relative_to(RAW))] = sha256(p)
    return p


def http_get(url: str, pause: float = 4.0, tries: int = 5) -> bytes:
    """One GET with a generic client name; waits `pause` seconds between requests and backs off on 429/5xx."""
    if OFFLINE:
        raise RuntimeError(f"--offline, and not in the cache: {url}")
    err = None
    for i in range(tries):
        wait = pause - (time.time() - _LAST["t"])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                _LAST["t"] = time.time()
                return r.read()
        except urllib.error.HTTPError as e:
            _LAST["t"], err = time.time(), e
            if e.code in (404, 410):
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            time.sleep(max(int(ra) if ra and ra.isdigit() else 0, 20 * (i + 1)))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            _LAST["t"], err = time.time(), e
            time.sleep(10 * (i + 1))
    raise err


def fetch_file(url: str, rel: str, pause: float = 2.0):
    """(path, {'url', 'retrieved_utc'}) of a public data file, cached at RAW/rel with its retrieval time."""
    p = RAW / rel
    meta = p.with_name(p.name + ".meta.json")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(http_get(url, pause))
        meta.write_text(json.dumps({"url": url, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    m = json.loads(meta.read_text()) if meta.exists() else {"url": url, "retrieved_utc": ""}
    use(p)
    return p, m


def safe(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", title.replace(" ", "_"))


def wiki_rev(title: str, cutoff: str, sub: str = "wikipedia"):
    """(meta, wikitext) of the last revision of a Wikipedia page at or before `cutoff` (ISO, UTC).

    meta: title, revid, timestamp, url (a permanent link to that revision). None if the page did not exist.
    """
    d = RAW / sub
    key = f"{safe(title)}__asof_{re.sub(r'[^0-9]', '', cutoff)[:12 if cutoff[11:19] != '23:59:59' else 8]}"
    hit = sorted(d.glob(key + "__rev*.json"))
    if not hit:
        if (d / (key + "__missing.json")).exists():
            return None, None
        d.mkdir(parents=True, exist_ok=True)
        q = {"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "rvstart": cutoff,
             "rvdir": "older", "rvprop": "ids|timestamp|content", "rvslots": "main", "redirects": 1,
             "format": "json", "formatversion": 2}
        js = json.loads(http_get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q)))
        page = js["query"]["pages"][0]
        if "revisions" not in page:
            (d / (key + "__missing.json")).write_text(json.dumps({"asked": title, "cutoff": cutoff}))
            return None, None
        rv = page["revisions"][0]
        meta = {"title": page["title"], "asked": title, "revid": rv["revid"], "timestamp": rv["timestamp"],
                "cutoff": cutoff, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "url": "https://en.wikipedia.org/w/index.php?title="
                       + urllib.parse.quote(page["title"].replace(" ", "_")) + f"&oldid={rv['revid']}"}
        f = d / f"{key}__rev{rv['revid']}.json"
        f.with_suffix(".wikitext").write_text(rv["slots"]["main"]["content"], encoding="utf-8")
        f.write_text(json.dumps(meta))
        hit = [f]
    meta = json.loads(hit[0].read_text())
    assert meta["timestamp"] <= cutoff, (title, meta["timestamp"], cutoff)
    return meta, use(hit[0].with_suffix(".wikitext")).read_text(encoding="utf-8")


def norm_name(s: str) -> str:
    s = re.sub("['\u2019\u2018\u02bb`]", "", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s)


STOP = {"jr", "sr", "ii", "iii", "iv", "mr", "mrs", "ms", "dr", "the", "de", "la", "van", "von", "phd", "del"}


def tokens(s: str, minlen: int = 2) -> set[str]:
    return {t for t in norm_name(s).split() if len(t) >= minlen and t not in STOP}


def pairs(s: pd.Series) -> dict:
    return {"_".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v) for k, v in s.items()}


# ---------------------------------------------------------------- wikitext: sections, list units, people
def heading_paths(text: str):
    """[(line number, 'Level 2 > Level 3 > ...', line)] for every line of a page's wikitext."""
    stack: dict[int, str] = {}
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        m = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*(?:<!--.*-->\s*)?$", ln)
        if m:
            lvl = len(m.group(1))
            h = re.sub(r"<ref.*?(/>|</ref>)", "", m.group(2))
            h = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", h)
            h = re.sub(r"\s*\(\d+\)\s*$", "", h).strip()   # 'Current (5)' -> 'Current'
            stack = {k: v for k, v in stack.items() if k < lvl}
            stack[lvl] = h
            continue
        out.append((i, " > ".join(stack[k] for k in sorted(stack)), ln))
    return out


def list_units(text: str, include: str, exclude: str | None = None):
    """List items and table rows under the headings whose path matches `include` (and not `exclude`).

    A unit is one bullet line, or one table row (the lines between two '|-'). Returns
    [(first line number, heading path, text of the unit)]. Prose lines are not units.
    """
    units, row = [], None
    for i, path, ln in heading_paths(text):
        ok = bool(re.search(include, path)) and not (exclude and re.search(exclude, path))
        s = ln.strip()
        if not ok or s.startswith("{|") or s.startswith("|}") or s.startswith("|+"):
            if row:
                units.append(tuple(row))
            row = None
            continue
        if s.startswith("|-"):
            if row:
                units.append(tuple(row))
            row = [i + 1, path, ""]
        elif s.startswith(("|", "!")) and row is not None:
            row[2] = (row[2] + " " + s).strip()
        elif re.match(r"^[*#]", s):
            if row:
                units.append(tuple(row))
                row = None
            units.append((i, path, s))
    if row:
        units.append(tuple(row))
    return [u for u in units if u[2] and not u[2].startswith("!")]


def strip_refs(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>", " ", s)
    s = re.sub(r"<ref.*?</ref>", " ", s, flags=re.S)
    return re.sub(r"<ref.*$", " ", s)


def person_refs(unit: str):
    """[(link target, shown name)] for the wikilinks and {{sortname}} templates of a unit."""
    u = strip_refs(unit)
    out = []
    for m in re.finditer(r"\{\{\s*sort\s*name\s*\|([^{}]*)\}\}", u, flags=re.I):
        a = [x.strip() for x in m.group(1).split("|") if "=" not in x]
        if len(a) >= 2:
            name = f"{a[0]} {a[1]}"
            out.append((a[2] if len(a) > 2 and a[2] else name, name))
    for m in re.finditer(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]", u):
        t = m.group(1).strip()
        if not re.match(r"(File|Image|Category):", t, flags=re.I):
            out.append((t, (m.group(2) or t).strip()))
    return out


_ORD = r"(\d+)(?:st|nd|rd|th)"


def place_of(unit: str):
    """(state, district) named in a unit: district is an int (0 at-large), 'S' for a Senate seat, or None."""
    u = strip_refs(unit)
    m = re.search(r"\[\[([A-Z][A-Za-z ]+?)'s " + _ORD + r" congressional district", u)
    if m and m.group(1) in ABBR:
        return ABBR[m.group(1)], int(m.group(2))
    m = re.search(r"\[\[([A-Z][A-Za-z ]+?)'s at-large congressional district", u)
    if m and m.group(1) in ABBR:
        return ABBR[m.group(1)], 0
    m = re.search(r"\{\{\s*ushr\s*\|\s*([A-Za-z ]+?)\s*\|\s*(\d+|AL|at-large)\s*[|}]", u, flags=re.I)
    if m:
        st = m.group(1) if m.group(1) in STATES else ABBR.get(m.group(1))
        if st:
            return st, int(m.group(2)) if m.group(2).isdigit() else 0
    m = re.search(r"\b([A-Z]{2})[-–‑ ](\d{1,2}|AL|at-large)\b", u)
    if m and m.group(1) in STATES:
        return m.group(1), int(m.group(2)) if m.group(2).isdigit() else 0
    sen = bool(re.search(r"United States Senate|U\.S\. Senat|Senator|\bSenate\b", u))
    for name, ab in sorted(ABBR.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", u) and not (
                name == "Washington" and "Washington, D.C." in u) and not (name == "Virginia" and "West Virginia" in u):
            return ab, ("S" if sen else None)
    return None, None


# ---------------------------------------------------------------- people: legislators, terms, Voteview
LEG = "https://unitedstates.github.io/congress-legislators"
VV_URL = "https://voteview.com/static/data/out/members/HSall_members.csv"
# Internet Archive captures of Voteview's member file made before an election day (none exists for 2018).
VV_SNAPSHOT = {2022: "20220505045700", 2026: "20261008235105"}


def load_people() -> pd.DataFrame:
    """Everyone who has served in Congress: bioguide, names, FEC IDs, Wikipedia title, terms (identity only)."""
    rows = []
    for f in ("legislators-current", "legislators-historical"):
        p, m = fetch_file(f"{LEG}/{f}.json", f"legislators/{f}.json")
        SOURCES.append({"kind": "crosswalk", "cycle": "", "key": f, "title": "unitedstates/congress-legislators",
                        "url": m["url"], "version": "", "version_time_utc": "", "retrieved_utc": m["retrieved_utc"],
                        "dated_before_election": "", "file": str(p.relative_to(RAW))})
        for d in json.loads(p.read_text(encoding="utf-8")):
            terms = [(t["start"], t["end"], t["type"], t["state"], t.get("district", -1)) for t in d["terms"]]
            if max(t[1] for t in terms) < "1990":
                continue
            n = d["name"]
            rows.append({"bioguide": d["id"]["bioguide"], "fec": d["id"].get("fec", []), "wikipedia": d["id"].get("wikipedia", ""),
                         "first": n.get("first", ""), "last": n.get("last", ""), "nickname": n.get("nickname", ""),
                         "middle": n.get("middle", ""), "official": n.get("official_full", ""), "terms": terms})
    return pd.DataFrame(rows).drop_duplicates("bioguide").reset_index(drop=True)


def served_before(terms, day: str) -> bool:
    return any(t[0] < day for t in terms)


def in_office_on(terms, day: str) -> bool:
    return any(t[0] <= day <= t[1] for t in terms)


def read_voteview(p: Path) -> pd.DataFrame:
    v = pd.read_csv(p, low_memory=False)
    v = v[v.chamber.isin(["House", "Senate"]) & v.bioguide_id.notna()]
    return v[["congress", "chamber", "bioguide_id", "icpsr", "state_abbrev", "party_code", "nominate_dim1",
              "nokken_poole_dim1", "nominate_number_of_votes"]]


def load_voteview() -> dict:
    """{'posthoc': today's file, 2022: capture of 2022-05-05, 2026: capture of 2026-10-08} (a missing capture is absent)."""
    out = {}
    p, m = fetch_file(VV_URL, "voteview/HSall_members__current.csv")
    SOURCES.append({"kind": "voteview", "cycle": "", "key": "HSall_members_current", "title": "Voteview member ideology (HSall_members.csv)",
                    "url": m["url"], "version": "current file", "version_time_utc": "", "retrieved_utc": m["retrieved_utc"],
                    "dated_before_election": False, "file": str(p.relative_to(RAW))})
    out["posthoc"] = read_voteview(p)
    for cycle, ts in VV_SNAPSHOT.items():
        url = f"https://web.archive.org/web/{ts}id_/{VV_URL}"
        try:
            p, m = fetch_file(url, f"voteview/HSall_members__wayback_{ts}.csv", pause=8.0)
        except Exception as e:  # the Internet Archive may refuse; the strict column then stays empty
            LOG["notes"][f"voteview_snapshot_{cycle}"] = f"not retrieved ({type(e).__name__}: {e}); nominate_dim1 is blank for {cycle}"
            continue
        if not p.read_bytes()[:20].startswith(b"congress,"):
            LOG["notes"][f"voteview_snapshot_{cycle}"] = "the capture is not the csv; ignored"
            continue
        when = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"
        assert when[:10] < ELECTION[cycle]
        SOURCES.append({"kind": "voteview", "cycle": cycle, "key": f"HSall_members_{ts}", "title": "Voteview member ideology (HSall_members.csv), Internet Archive capture",
                        "url": url, "version": ts, "version_time_utc": when, "retrieved_utc": m["retrieved_utc"],
                        "dated_before_election": True, "file": str(p.relative_to(RAW))})
        out[cycle] = read_voteview(p)
    return out


def nominate_for(vv: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """Per bioguide: first-dimension DW-NOMINATE in the last Congress, up to the one sitting at the election, served in."""
    v = vv[vv.congress <= CONGRESS[cycle]].sort_values(["congress", "nominate_number_of_votes"])
    v = v[v.nominate_dim1.notna()].groupby("bioguide_id").tail(1).set_index("bioguide_id")
    return v[["congress", "chamber", "nominate_dim1", "nokken_poole_dim1"]]


# ---------------------------------------------------------------- nominees and their identities
def build_nominees(races: pd.DataFrame, polls: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle, race and side: the nominee of e94_races, linked to a bioguide ID where one exists."""
    polled = set(map(tuple, polls[polls.kind.isin(["house", "senate"])][["cycle", "race"]].drop_duplicates().to_numpy()))
    fec2bio = {f: r.bioguide for r in people.itertuples() for f in r.fec}
    last_of = dict(zip(people.bioguide, people["last"]))
    wrong_ids: list[str] = []
    ptok = [(r.bioguide, tokens(r.last), tokens(" ".join([r.first, r.nickname, r.middle, r.official])),
             {t[3] for t in r.terms}, r.terms) for r in people.itertuples()]
    rows = []
    for r in races.itertuples():
        for side in ("D", "R"):
            cid = getattr(r, f"{side.lower()}_id")
            name = getattr(r, f"{side.lower()}_name")
            cid = "" if pd.isna(cid) else cid
            name = "" if pd.isna(name) else name
            if not name:
                continue
            multi = "|" in name
            # A row that stands for several candidates of one side (name 'A|B') is nobody in particular.
            bio, basis = ("" if multi else fec2bio.get(cid, "")), "fec_id"
            sur_row = "".join(norm_name(name.split(",")[0] if "," in name else name.split()[-1]).split())
            sur_leg = "".join(norm_name(last_of.get(bio, "")).split())
            if bio and not (tokens(last_of[bio]) & tokens(name)) and difflib.SequenceMatcher(None, sur_row, sur_leg).ratio() < 0.8:
                # The FEC ID on this row of e94_races belongs to someone with another surname: not this person.
                # (A misspelt surname in the FEC workbook - 'Buchson', 'Norcoss' - still passes.)
                wrong_ids.append(f"{r.cycle} {r.race} {side} {name}: {cid} is {last_of[bio]} ({bio})")
                bio = ""
            if not bio and not multi:
                # No FEC ID in the crosswalk: a unique person who served before, for this state, with the
                # nominee's surname and a given name in common.
                if "," in name:
                    sur, given = tokens(name.split(",")[0]), tokens(name.split(",", 1)[1])
                else:
                    tk = norm_name(name).split()
                    tk = [t for t in tk if t not in STOP and len(t) >= 2]
                    sur, given = set(tk[-1:]), set(tk[:-1])
                hits = [b for b, ls, gs, sts, terms in ptok if r.state in sts and sur & ls and given & gs
                        and served_before(terms, ELECTION[r.cycle])]
                bio, basis = (hits[0], "name+state") if len(hits) == 1 else ("", "")
            rows.append({"cycle": r.cycle, "office": r.office, "state": r.state, "race": r.race, "side": side,
                         "cand_id": cid, "name": name, "incumbent": bool(getattr(r, f"{side.lower()}_inc")),
                         "polled": (r.cycle, r.race) in polled, "multi_cands": multi,
                         "bioguide": bio, "bioguide_basis": basis if bio else ""})
    nom = pd.DataFrame(rows)
    LOG["checks"]["e94_fec_ids_that_belong_to_someone_else"] = wrong_ids
    dup = nom[nom.bioguide != ""].groupby(["cycle", "bioguide"]).size()
    LOG["checks"]["bioguides_on_two_nominee_rows"] = [f"{c} {b}: " + "; ".join(
        f"{x.race} {x.side} {x.name}" for x in nom[(nom.cycle == c) & (nom.bioguide == b)].itertuples()) for (c, b), n in dup.items() if n > 1]
    terms = people.set_index("bioguide").terms
    nom["served_before"] = [bool(b) and served_before(terms[b], ELECTION[c]) for b, c in zip(nom.bioguide, nom.cycle)]
    nom["in_office_on_eve"] = [bool(b) and in_office_on(terms[b], EVE[c]) for b, c in zip(nom.bioguide, nom.cycle)]
    return nom


# ---------------------------------------------------------------- dated lists -> flags
# kind 'members': a list of sitting members of Congress; every person named in a selected unit is matched to
#   a legislator in office on the eve (by the article the list links to, else by name within the state the
#   unit names), and through the legislator to a nominee.
# kind 'candidates': a table of endorsed candidates with a state and a district; the candidate is matched
#   to the nominee of that race by name.
# sel: {cycle: (heading-path pattern, excluded heading-path pattern)}. A cycle that is absent has no dated
#   list: the flag is blank (unknown) for that cycle, never False.
E18, E22, E26 = (EVE[c] + "T23:59:59Z" for c in CYCLES)
LISTS = [
    {"flag": "cpc", "label": "Congressional Progressive Caucus member", "kind": "members",
     "title": "Congressional Progressive Caucus",
     "sel": {2018: (r"^(House|Senate) members$", None), 2022: (r"(House|Senate) members$", r"Former"),
             2026: (r"(House|Senate) members$", r"Former")}},
    {"flag": "ndc", "label": "New Democrat Coalition member", "kind": "members", "title": "New Democrat Coalition",
     "sel": {2018: (r"^New Democrat Coalition members \(House\)", r"Former|Disaffiliated"), 2022: (r"^Membership$", None),
             2026: (r"Current members$", None)}},
    {"flag": "blue_dog", "label": "Blue Dog Coalition member", "kind": "members", "title": "Blue Dog Coalition",
     "sel": {2018: (r"^Members for the 115th Congress$", None, "List of members of the Blue Dog Coalition"),
             2022: (r"^Members of the Blue Dog Coalition$", None), 2026: (r"Membership > Current members$", None)}},
    {"flag": "hfc", "label": "House Freedom Caucus member (as compiled by Wikipedia; the caucus publishes no roster)",
     "kind": "members", "title": "Freedom Caucus",
     "sel": {2018: (r"^Membership$", None), 2022: (r"Current members$", None), 2026: (r"Current members$", None)}},
    {"flag": "rsc", "label": "Republican Study Committee member", "kind": "members", "title": "Republican Study Committee",
     "sel": {2018: (r"^Members$", None), 2022: (r"Current members", r"Former"), 2026: (r"Membership > Current members$", None)}},
    {"flag": "rmsp", "label": "Republican Main Street Partnership member", "kind": "members",
     "title": "Republican Main Street Partnership",
     "sel": {2018: (r"^Current members > (Senators|Representatives)$", None),
             2022: (r"^Current membership > U\.S\. (Senators|Representatives)$", None),
             # In 2026 the page lists its senators and says all members of the Main Street Caucus are affiliated.
             2026: [(r"^Affiliated members > U\.S\. senators$", None),
                    (r"^Membership > Current members$", None, "Main Street Caucus")]}},
    {"flag": "rgg", "label": "Republican Governance Group (Tuesday Group) member", "kind": "members",
     "title": "Republican Governance Group", "sel": {2022: (r"Current members$", None), 2026: (r"Current members$", None)}},
    {"flag": "psc", "label": "Problem Solvers Caucus member", "kind": "members", "title": "Problem Solvers Caucus",
     "sel": {2022: (r"^Membership > (Democrats|Republicans)$", None), 2026: (r"^Membership > (Democrats|Republicans)$", None)}},
    {"flag": "dsa_member", "label": "Listed as a current DSA member in the US House (Wikipedia's list of DSA officeholders)",
     "kind": "members", "title": "List of Democratic Socialists of America public officeholders",
     "sel": {2022: (r"United States House of Representatives > Current$", None),
             2026: (r"United States House of Representatives > Current$", None)}},
    {"flag": "jd", "label": "Justice Democrats endorsement", "kind": "candidates", "title": "Justice Democrats",
     "sel": {2018: (r"Candidates for U\.S\. (Senate|House)$", None), 2022: (r"2022 > U\.S\. (Senate|House)$", None),
             2026: (r"> 2026$", None)}},
    {"flag": "bnc", "label": "Brand New Congress endorsement", "kind": "candidates", "title": "Brand New Congress",
     "sel": {2018: (r"^Candidates$", None), 2022: (r"^Candidates > 2022 > U\.S\. (Senate|House)$", None)}},
    {"flag": "blue_dog_pac", "label": "Blue Dog PAC endorsement of a non-incumbent", "kind": "candidates",
     "title": "Blue Dog Coalition", "sel": {2026: (r"^Affiliate organizations > 2026$", None)}},
    {"flag": "hfc_fund", "label": "Freedom Caucus Fund endorsement of a non-incumbent", "kind": "candidates",
     "title": "Freedom Caucus", "sel": {2026: (r"^Affiliate organizations > 2026$", None)}},
]


# Internet Archive captures of organisations' own pages, pinned by capture time (each before election day).
# section: only entries under a heading line matching it; role 'members' goes through the legislators,
# role 'candidates' through the race the entry names.
WAYBACK = [
    {"flag": "blue_dog_site", "label": "Listed under 'Our Members' on the Blue Dog PAC's own site", "role": "members",
     "url": "https://bluedogdems.com/", "ts": {2018: "20181022062501", 2022: "20221107083124"}, "section": r"^Our Members$"},
    {"flag": "blue_dog_pac", "label": "Blue Dog PAC endorsement of a non-incumbent", "role": "candidates",
     "url": "https://bluedogdems.com/", "ts": {2018: "20181022062501", 2022: "20221107083124"}, "section": r"^Endorsed Candidates$"},
    {"flag": "newdem_fund", "label": "NewDem Action Fund endorsed candidate", "role": "candidates",
     "url": "https://newdemactionfund.com/candidates", "ts": {2022: "20221106185352", 2026: "20261101191604"},
     "section": r"Endorsed Candidates$"},
    {"flag": "newdem_watch", "label": "On the NewDem Action Fund's candidate watch list", "role": "candidates",
     "url": "https://newdemactionfund.com/candidates", "ts": {2022: "20221106185352"}, "section": r"Watch List$"},
    {"flag": "our_revolution", "label": "Our Revolution endorsement", "role": "candidates",
     "url": "https://ourrevolution.com/candidates/", "ts": {2018: "20181104071734"}, "section": None},
    {"flag": "red_to_blue", "label": "On the DCCC's Red to Blue list (a party programme, not an ideological label)",
     "role": "candidates", "url": "https://redtoblue.dccc.org/", "ts": {2018: "20181030021445", 2022: "20221106184002"},
     "section": None},
]
HEADINGS = r"^(Our Members|Endorsed Candidates|Members|\d{4} Endorsed Candidates|Endorsed Candidates|\d{4} NewDem Watch List|News|About)$"


class _Text(HTMLParser):
    """Visible text of a page, one line per block element."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        if tag in ("h1", "h2", "h3", "h4", "h5", "li", "p", "div", "a", "tr", "section"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.out.append(" ".join(d.split()) + " ")


def wayback_lines(url: str, ts: str):
    """(meta, visible text lines) of an Internet Archive capture, cached; gzip payloads are unpacked."""
    rel = "wayback/" + re.sub(r"[^A-Za-z0-9.-]", "_", re.sub(r"^https?://", "", url)) + f"__{ts}.html"
    p, m = fetch_file(f"https://web.archive.org/web/{ts}id_/{url}", rel, pause=8.0)
    raw = p.read_bytes()
    raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    t = _Text()
    t.feed(raw.decode("utf-8", errors="replace"))
    lines = [ln.strip() for ln in "".join(t.out).split("\n") if ln.strip()]
    when = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"
    return {"title": url, "revid": ts, "timestamp": when, "url": f"https://web.archive.org/web/{ts}/{url}",
            "retrieved_utc": m["retrieved_utc"], "file": str(p.relative_to(RAW))}, lines


_CODE = r"([A-Z]{2})-(\d{1,2}|AL)"
_ORDW = r"(\d+)(?:st|nd|rd|th)"


def page_entries(lines, section):
    """[(line number, section, text, name, state, district)] for the candidate or member entries of a page."""
    out, sec, seen = [], "", set()
    for i, ln in enumerate(lines):
        if re.match(HEADINGS, ln) or (len(ln) < 60 and re.search(r"(Endorsed Candidates|Watch List)$", ln)):
            sec = ln
            continue
        name = st = dist = None
        m = re.match(r"^(?:Rep\. )?(.+?)\s*\(" + _CODE + r"\)", ln)                      # 'Rep. Jim Cooper (TN-5)'
        if m:
            name, st, dist = m.group(1), m.group(2), m.group(3)
        if name is None:
            m = re.match(r"^(?:\w+ \| )?([A-Za-z]{2})-(\d{1,2}|AL)$", ln)                 # a name line, then 'CA-39' ('Nc-03' occurs)
            if m and i and not re.search(_CODE + r"$|^(Learn More|Visit|View|Donate|Get )", lines[i - 1]) and len(lines[i - 1].split()) <= 6:
                name, st, dist = lines[i - 1], m.group(1).upper(), m.group(2)
        if name is None:
            m = re.match(r"^((?:[A-Z][\w.'’-]*\.? ?){2,5}) " + _CODE + r"$", ln)             # 'Jevin Hodge AZ-01'
            if m and not re.match(r"(Get|Donate|Support|Involved)", ln):
                name, st, dist = m.group(1), m.group(2), m.group(3)
        if name is None and i >= 2 and re.match(r"^U\.?S\.? (House of Representatives|Senate)", ln):
            who = lines[i - 2]                                                              # 'Raul Grijalva Arizona'
            for full, ab in sorted(ABBR.items(), key=lambda kv: -len(kv[0])):
                if who.endswith(" " + full):
                    name, st = who[: -len(full) - 1], ab
                    break
            if name is not None:
                d = re.search(_ORDW + r" District|District (\d+)", ln)
                dist = "S" if "Senate" in ln else ((d.group(1) or d.group(2)) if d else None)
        if name is None or st not in STATES or name.isupper() or re.search(r"[A-Z]{4,}", name):
            continue
        dist = dist if dist in ("S", None) else (0 if dist == "AL" else int(dist))
        if section is not None and not re.search(section, sec):
            continue
        name = re.sub(r"^(Rep\.|Dr\.)\s+", "", name).strip()
        if (name, st, dist) in seen:
            continue
        seen.add((name, st, dist))
        out.append((i + 1, sec, f"{name} | {st}-{dist if dist is not None else '?'} | {ln}"[:240], name, st, dist))
    return out


def wiki_key(t: str) -> str:
    return " ".join(norm_name(t.replace("_", " ")).split())


def match_members(units, cycle, people):
    """Units of a members list -> [(bioguide, basis, line, path, text)], and the units that named nobody in office."""
    eve = EVE[cycle]
    pool = people[[in_office_on(t, eve) for t in people.terms]]
    by_wiki = {}
    for r in pool.itertuples():
        by_wiki.setdefault(wiki_key(r.wikipedia), []).append(r.bioguide)
        by_wiki.setdefault(wiki_key(re.sub(r"\s*\(.*\)$", "", r.wikipedia)), []).append(r.bioguide)
    ptok = [(r.bioguide, tokens(r.last), tokens(" ".join([r.first, r.nickname, r.middle, r.official])),
             {t[3] for t in r.terms if t[0] <= eve <= t[1]}) for r in pool.itertuples()]
    seat = {}
    for r in pool.itertuples():
        for t in r.terms:
            if t[0] <= eve <= t[1]:
                seat.setdefault((t[3], t[4] if t[2] == "rep" else "S"), []).append((r.bioguide, tokens(r.last)))
    hits, misses = [], []
    for line, path, text in units:
        st, dist = place_of(text)
        found = None
        for target, shown in person_refs(text):
            c = by_wiki.get(wiki_key(target), [])
            if len(set(c)) == 1:
                found = (c[0], "article")
                break
            # The link is to an older or ambiguous title: surname and a given name, within the unit's state.
            for nm in (shown, re.sub(r"\s*\(.*\)$", "", target)):
                tk = [t for t in norm_name(nm).split() if t not in STOP and len(t) >= 2]
                if len(tk) < 2:
                    continue
                c = [b for b, ls, gs, sts in ptok if set(tk[-1:]) & ls and set(tk[:-1]) & gs and (st is None or st in sts)]
                if len(set(c)) == 1:
                    found = (c[0], "name+state" if st else "name")
                    break
            if found:
                break
            # A nickname in the link ('Bill Keating', 'Dave Joyce'): the sitting member for the district the
            # unit names who has that surname. (A surname within a state is not enough: a departed member's
            # namesake would be flagged.)
            tk = [t for t in norm_name(shown).split() if t not in STOP and len(t) >= 2]
            if tk and st is not None:
                c = [b for b, ls in seat.get((st, dist), []) if tk[-1] in ls] if isinstance(dist, int) else []
                if len(set(c)) == 1:
                    found = (c[0], "surname+district")
                    break
        if found:
            hits.append((found[0], found[1], line, path, text))
        else:
            misses.append((line, path, text))
    return hits, misses


def first_cell_name(unit: str) -> str:
    """The name in the first cell of a table row (a {{sortname}}, a link or plain text), or of a bullet."""
    u = strip_refs(unit).lstrip("|*# ")
    depth, cut = 0, len(u)
    for i, ch in enumerate(u):
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "|" and depth == 0:
            if re.fullmatch(r"\s*(rowspan|colspan|style|align|class|scope|nowrap|data-sort-value)[^|]*", u[:i]):
                return first_cell_name(u[i + 1:])
            cut = i
            break
    cell = u[:cut]
    refs = person_refs(cell)
    if refs:
        name = refs[0][1]
    else:
        name = re.sub(r"\{\{[^{}]*\}\}", " ", cell)
        name = re.sub(r"'{2,}|<[^>]*>", " ", name)
    name = re.sub(r"\(.*?\)", " ", name)
    return " ".join(name.split())


def match_candidates(units, cycle, nom, office_hint=None):
    """Units of an endorsement table -> [(row index in nom, basis, line, path, text)]."""
    entries = []
    for line, path, text in units:
        st, dist = place_of(text)
        shown = first_cell_name(text)
        if not shown or st is None:
            continue
        if dist is None and re.search(r"Senate", path):
            dist = "S"
        entries.append((line, path, text, shown, st, dist))
    return match_entries(entries, cycle, nom)


def given_names(name: str):
    """(given-name words, initials) of 'Last, First M.' or 'First M. Last'."""
    if "," in name:
        g = norm_name(name.split(",", 1)[1]).split()
    else:
        g = [t for t in norm_name(re.sub(r"\(.*?\)", " ", name)).split() if t not in STOP][:-1]
    g = [t for t in g if t not in STOP]
    return [t for t in g if len(t) >= 2], "".join(t[0] for t in g)


def given_compatible(a: str, b: str) -> bool:
    """A short form of the same given name ('Matt'/'Matthew', 'Gabe'/'Gabriel'), or the same initials ('J.D.')."""
    wa, ia = given_names(a)
    wb, ib = given_names(b)
    if wa and wb:
        return any(x[:3] == y[:3] for x in wa for y in wb)
    return bool(ia) and bool(ib) and (ia == ib or ia.startswith(ib) or ib.startswith(ia)) and not (wa or wb)


def match_entries(entries, cycle, nom):
    """[(line, section, text, name, state, district)] -> hits [(row index in nom, basis, line, section, text)], misses."""
    nc = nom[(nom.cycle == cycle) & ~nom.multi_cands]
    hits, misses = [], []
    for line, path, text, shown, st, dist in entries:
        cand = tokens(shown)
        tk = [t for t in norm_name(shown).split() if t not in STOP and len(t) >= 2]
        if not tk:
            continue
        sur = tk[-1]
        if dist == "S":
            pool = nc[(nc.state == st) & (nc.office == "S")]
        elif dist is None:
            pool = nc[nc.state == st]
        else:
            pool = nc[nc.race == f"{st}-{dist:02d}"]
        found = None
        for i, r in pool.iterrows():
            nt = tokens(r["name"])
            nsur = tokens(r["name"].split(",")[0]) if "," in r["name"] else set(norm_name(r["name"]).split()[-1:])
            if len(cand & nt) >= 2 and sur in nt:
                found = (i, "race+name" if dist is not None else "state+name")
            elif dist is not None and sur in nsur and given_compatible(shown, r["name"]):
                found = found or (i, "race+surname")
        if found:
            hits.append((found[0], found[1], line, path, text))
        else:
            misses.append((line, path, text))
    return hits, misses


def excerpt(text: str, name: str, width: int = 240) -> str:
    """The source line, cut to `width` characters around the place where it names the person."""
    t = " ".join(strip_refs(text).split())
    if len(t) <= width:
        return t
    plain = unicodedata.normalize("NFKD", t).encode("ascii", "replace").decode().lower()
    sur = norm_name(name.split(",")[0] if "," in name else name).split()
    pos = max([plain.find(w) for w in (sur if "," in name else sur[-1:]) if len(w) > 2] + [-1])
    if pos < width - 40:
        return t[:width]
    a = max(pos - width // 2, 0)
    return "... " + t[a:a + width - 4]


def build_flags(nom: pd.DataFrame, people: pd.DataFrame):
    """The long evidence table (one row per nominee and flag) and {flag: cycles with a dated list}."""
    cut = dict(zip(CYCLES, (E18, E22, E26)))
    ev, avail, diag = [], {}, {}
    bio_rows = {}
    for i, r in nom[nom.bioguide != ""].iterrows():
        bio_rows.setdefault((r.cycle, r.bioguide), []).append(i)
    for L in LISTS:
        for cycle, spec in [(c, sp) for c, v in L["sel"].items() for sp in (v if isinstance(v, list) else [v])]:
            inc, exc = spec[0], spec[1]
            meta, txt = wiki_rev(spec[2] if len(spec) > 2 else L["title"], cut[cycle])
            if meta is None:
                continue
            assert meta["timestamp"][:10] < ELECTION[cycle]
            units = list_units(txt, inc, exc)
            if L["kind"] == "members":
                hits, misses = match_members(units, cycle, people)
                listed = len({h[0] for h in hits})
                out = [(i, h[1], h[2], h[3], h[4]) for h in hits for i in bio_rows.get((cycle, h[0]), [])]
            else:
                out, misses = match_candidates(units, cycle, nom)
                listed = len(units)
            if not units:
                continue
            if cycle not in avail.setdefault(L["flag"], []):
                avail[L["flag"]].append(cycle)
            src = {"kind": "wikipedia", "cycle": cycle, "key": L["flag"], "title": meta["title"], "url": meta["url"],
                   "version": meta["revid"], "version_time_utc": meta["timestamp"], "retrieved_utc": meta.get("retrieved_utc", ""),
                   "dated_before_election": True, "file": ""}
            if not any(s["kind"] == "wikipedia" and s["version"] == meta["revid"] and s["key"] == L["flag"] for s in SOURCES):
                SOURCES.append(src)
            seen = {(e["_i"], e["flag"]) for e in ev if e["cycle"] == cycle and e["flag"] == L["flag"]}
            n_before = len(seen)
            for i, basis, line, path, text in out:
                if (i, L["flag"]) in seen:
                    continue
                seen.add((i, L["flag"]))
                r = nom.loc[i]
                ev.append({"_i": i, "cycle": cycle, "race": r.race, "side": r.side, "cand_id": r.cand_id, "name": r["name"],
                           "flag": L["flag"], "match_basis": basis, "source_title": meta["title"], "source_revid": meta["revid"],
                           "source_time_utc": meta["timestamp"], "source_url": meta["url"], "section": path, "line": line,
                           "text": excerpt(text, r["name"])})
            diag[f"{L['flag']}_{cycle}" + (f"_{safe(meta['title'])}" if len(spec) > 2 else "")] = {
                "revid": meta["revid"], "revision_time": meta["timestamp"], "units": len(units), "listed": listed,
                "nominees_flagged": len(seen) - n_before, "units_matching_nobody": len(misses),
                "unmatched": [f"L{m[0]} " + " ".join(strip_refs(m[2]).split())[:150] for m in misses]}
    for Wb in WAYBACK:
        for cycle, ts in Wb["ts"].items():
            try:
                meta, lines = wayback_lines(Wb["url"], ts)
            except Exception as e:   # the Internet Archive refused: the flag stays blank for the cycle
                LOG["notes"][f"wayback_{Wb['flag']}_{cycle}"] = f"not retrieved ({type(e).__name__}); flag left blank"
                continue
            assert meta["timestamp"][:10] < ELECTION[cycle]
            entries = page_entries(lines, Wb["section"])
            if not entries:
                continue
            if Wb["role"] == "members":
                units = [(e[0], e[1], f"* [[{e[3]}]] ({e[4]}-{e[5]}) :: {e[2]}") for e in entries]
                hits, misses = match_members(units, cycle, people)
                out = [(i, h[1], h[2], h[3], h[4].split(" :: ", 1)[1]) for h in hits for i in bio_rows.get((cycle, h[0]), [])]
                misses = [(m_[0], m_[1], m_[2].split(" :: ", 1)[1]) for m_ in misses]
            else:
                out, misses = match_entries(entries, cycle, nom)
            avail.setdefault(Wb["flag"], [])
            if cycle not in avail[Wb["flag"]]:
                avail[Wb["flag"]].append(cycle)
            SOURCES.append({"kind": "wayback", "cycle": cycle, "key": Wb["flag"], "title": meta["title"], "url": meta["url"],
                            "version": meta["revid"], "version_time_utc": meta["timestamp"], "retrieved_utc": meta["retrieved_utc"],
                            "dated_before_election": True, "file": meta["file"]})
            seen = set()
            for i, basis, line, path, text in out:
                if (i, Wb["flag"]) in seen:
                    continue
                seen.add((i, Wb["flag"]))
                r = nom.loc[i]
                ev.append({"_i": i, "cycle": cycle, "race": r.race, "side": r.side, "cand_id": r.cand_id, "name": r["name"],
                           "flag": Wb["flag"], "match_basis": basis, "source_title": meta["title"], "source_revid": meta["revid"],
                           "source_time_utc": meta["timestamp"], "source_url": meta["url"], "section": path, "line": line,
                           "text": excerpt(text, r["name"])})
            diag[f"{Wb['flag']}_{cycle}"] = {
                "revid": meta["revid"], "revision_time": meta["timestamp"], "units": len(entries), "listed": len(entries),
                "nominees_flagged": len(seen), "units_matching_nobody": len(misses),
                "unmatched": [f"L{m_[0]} {m_[2]}"[:150] for m_ in misses]}
    LOG["checks"]["ideology_lists"] = diag
    LOG["ideology_flag_cycles"] = avail
    return pd.DataFrame(ev).drop(columns="_i"), avail


def build_ideology(races: pd.DataFrame):
    polls = pd.read_csv(OUT / "e94_polls.csv", low_memory=False)
    people = load_people()
    nom = build_nominees(races, polls, people)
    vv = load_voteview()
    for col in ("nominate_dim1", "nominate_congress", "nominate_dim1_posthoc", "nokken_poole_dim1_posthoc", "nominate_congress_posthoc"):
        nom[col] = np.nan
    nom["nominate_snapshot_utc"] = ""
    for cycle in CYCLES:
        sel = (nom.cycle == cycle) & nom.served_before
        post = nominate_for(vv["posthoc"], cycle)
        b = nom.loc[sel, "bioguide"]
        nom.loc[sel, "nominate_dim1_posthoc"] = b.map(post.nominate_dim1).to_numpy()
        nom.loc[sel, "nokken_poole_dim1_posthoc"] = b.map(post.nokken_poole_dim1).to_numpy()
        nom.loc[sel, "nominate_congress_posthoc"] = b.map(post.congress).to_numpy()
        if cycle in vv:
            pre = nominate_for(vv[cycle], cycle)
            nom.loc[sel, "nominate_dim1"] = b.map(pre.nominate_dim1).to_numpy()
            nom.loc[sel, "nominate_congress"] = b.map(pre.congress).to_numpy()
            ts = VV_SNAPSHOT[cycle]
            nom.loc[sel & nom.nominate_dim1.notna(), "nominate_snapshot_utc"] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"
            both = nom[sel & nom.nominate_dim1.notna() & nom.nominate_dim1_posthoc.notna()]
            d = (both.nominate_dim1 - both.nominate_dim1_posthoc).abs()
            LOG["checks"][f"nominate_{cycle}_snapshot_vs_today"] = {
                "n": int(len(both)), "corr": float(np.corrcoef(both.nominate_dim1, both.nominate_dim1_posthoc)[0, 1]),
                "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()), "moved_by_over_0.05": int((d > 0.05).sum())}
    ev, avail = build_flags(nom, people)
    flags = list(dict.fromkeys([L["flag"] for L in LISTS] + [w["flag"] for w in WAYBACK]))
    for f in flags:
        nom[f] = pd.array([pd.NA] * len(nom), dtype="boolean")
        for cycle in avail.get(f, []):
            sel = nom.cycle == cycle
            keys = set(map(tuple, ev[(ev.flag == f) & (ev.cycle == cycle)][["race", "side"]].to_numpy()))
            nom.loc[sel, f] = [(r, s) in keys for r, s in zip(nom.race[sel], nom.side[sel])]
    cols = ["cycle", "office", "state", "race", "side", "cand_id", "name", "incumbent", "polled", "multi_cands", "bioguide",
            "bioguide_basis", "served_before", "in_office_on_eve", "nominate_dim1", "nominate_congress",
            "nominate_snapshot_utc", "nominate_dim1_posthoc", "nokken_poole_dim1_posthoc", "nominate_congress_posthoc"]
    nom = nom[cols + flags]
    nom.to_csv(OUT / "e118_ideology.csv", index=False, float_format="%.4g")
    ev = ev.sort_values(["cycle", "race", "side", "flag"]).reset_index(drop=True)
    ev.to_csv(OUT / "e118_ideology_flags.csv", index=False)
    inc = nom[nom.incumbent]
    LOG["checks"]["incumbent_nominees_linked_to_bioguide"] = {
        str(c): f"{int((g.bioguide != '').sum())}/{len(g)}" for c, g in inc.groupby("cycle")}
    LOG["checks"]["incumbents_without_bioguide"] = [f"{r.cycle} {r.race} {r.side} {r.name}" for r in inc[inc.bioguide == ""].itertuples()]
    LOG["checks"]["non_incumbent_nominees_who_served_before"] = {
        str(c): [f"{r.race} {r.side} {r.name}" for r in g.itertuples()]
        for c, g in nom[~nom.incumbent & nom.served_before].groupby("cycle")}
    LOG["checks"]["e94_incumbent_but_not_in_office_on_the_eve"] = [
        f"{r.cycle} {r.race} {r.side} {r.name}" for r in nom[nom.incumbent & ~nom.in_office_on_eve & ~nom.multi_cands].itertuples()]
    LOG["counts"]["ideology_rows"] = pairs(nom.groupby(["cycle", "office", "side"]).size())
    LOG["counts"]["ideology_flags"] = pairs(ev.groupby(["cycle", "flag"]).size())
    return nom, ev


# ---------------------------------------------------------------- provisional 2026 results (Wikipedia)
def delink(t: str) -> str:
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    return re.sub(r"'{2,}", "", t)


def election_boxes(text: str):
    """[(line, heading path, title, [candidate dicts])] for every {{Election box}} table of a page."""
    paths = {i: p for i, p, _ in heading_paths(text)}
    lines = text.splitlines()
    out, cur = [], None
    for i, ln in enumerate(lines, 1):
        for piece in re.split(r"(?=\{\{\s*[Ee]lection box )", ln):
            m = re.match(r"\{\{\s*[Ee]lection box ([a-z ]+)", piece)
            if m:
                kind = m.group(1).strip()
                if kind.startswith("begin"):
                    cur = {"line": i, "path": paths.get(i, ""), "title": "", "cands": [], "buf": None, "total": None}
                    out.append(cur)
                    cur["buf"] = ["begin", piece]
                elif kind.startswith("end"):
                    cur = None
                elif cur is not None:
                    cur["buf"] = [kind, piece]
                    if "candidate" in kind:
                        cur["cands"].append(cur["buf"])
                    elif kind.startswith("total"):
                        cur["total"] = cur["buf"]
                    elif not kind.startswith("begin"):
                        cur["buf"] = [kind, piece]
            elif cur is not None and cur["buf"] is not None:
                cur["buf"][1] += "\n" + piece
            if cur is not None and cur["buf"] and cur["buf"][0] == "begin":
                t = re.search(r"\|\s*title\s*=\s*(.*)", cur["buf"][1])
                if t:
                    cur["title"] = " ".join(delink(strip_refs(t.group(1))).split())[:160]
    res = []
    for b in out:
        cands = []
        for kind, raw in b["cands"]:
            raw = delink(strip_refs(raw))
            g = lambda k: (re.search(r"\|\s*" + k + r"\s*=\s*([^|\n}]*)", raw) or [None, ""])[1].strip()
            votes = re.sub(r"[^\d]", "", g("votes"))
            pc = re.search(r"[\d.]+", g("percentage"))
            cands.append({"name": re.sub(r"\(.*?\)|\{\{.*", "", g("candidate")).strip(), "party": g("party"),
                          "votes": float(votes) if votes else np.nan, "pct": float(pc.group(0)) if pc else np.nan})
        tot = re.sub(r"[^\d]", "", (re.search(r"\|\s*votes\s*=\s*([^|\n}]*)", b["total"][1]) or [None, ""])[1]) if b["total"] else ""
        res.append((b["line"], b["path"], b["title"], cands, float(tot) if tot else np.nan))
    return res


def party_side_wiki(p: str) -> str:
    p = p.lower()
    if "democratic" in p:
        return "D"
    if "republican" in p:
        return "R"
    return "O"


def margins_2026(r26: pd.DataFrame, res: pd.DataFrame, polled: set) -> pd.DataFrame:
    """Margins WITHOUT counts: the 'Closest races' table of Wikipedia's national House article at RESULTS_ASOF.

    At that revision the table sits inside an HTML comment (an editor's draft, hidden from readers). It
    gives the winner's party and the margin to one decimal for races within 10 points. These are not
    counts, so they are written to their own file and never into e118_results_2026.csv; the builder
    reports how they compare with the counts where both exist.
    """
    meta, txt = wiki_rev("2026 United States House of Representatives elections", RESULTS_ASOF, sub="wikipedia_results_2026")
    cols = ["race", "office", "state", "winner", "flip", "margin_abs", "margin", "polled", "hidden_in_html_comment",
            "counts_margin", "status", "source_title", "source_revid", "source_time_utc", "source_url", "source_section", "source_line"]
    rows = []
    if meta is not None:
        SOURCES.append({"kind": "wikipedia_results", "cycle": 2026, "key": "US-H closest races", "title": meta["title"], "url": meta["url"],
                        "version": meta["revid"], "version_time_utc": meta["timestamp"], "retrieved_utc": meta.get("retrieved_utc", ""),
                        "dated_before_election": False, "file": ""})
        lines = txt.splitlines()
        paths = {i: p for i, p, _ in heading_paths(txt)}
        depth = 0   # inside <!-- ... -->
        hidden = {}
        for i, ln in enumerate(lines, 1):
            hidden[i] = depth > 0 or "<!--" in ln
            depth += ln.count("<!--") - ln.count("-->")
        counts = res.set_index("race").margin
        ok = r26[r26.contested & ~r26.multi_cands.fillna(False).astype(bool)].set_index("race")
        for i, ln in enumerate(lines, 1):
            m = re.match(r"^!\s*\{\{ushr\|([A-Za-z ]+)\|(\d+|AL)\|T\}\}", ln)
            if not m or not re.search(r"Closest races", paths.get(i, "")) or i + 1 >= len(lines):
                continue
            w = re.search(r"Party shading/(Democratic|Republican)", lines[i])
            pc = re.match(r"^\|\s*([\d.]+)%", lines[i + 1])
            st = ABBR.get(m.group(1), m.group(1) if m.group(1) in STATES else None)
            if not (w and pc and st):
                continue
            race = f"{st}-{0 if m.group(2) == 'AL' else int(m.group(2)):02d}"
            if race not in ok.index:
                continue        # uncontested or several candidates of one side: the margin is not D minus R
            side = w.group(1)[0]
            mg = float(pc.group(1))
            rows.append({"race": race, "office": "H", "state": st, "winner": side, "flip": "(flip)" in lines[i],
                         "margin_abs": mg, "margin": mg if side == "D" else -mg, "polled": race in polled,
                         "hidden_in_html_comment": bool(hidden[i]), "counts_margin": counts.get(race, np.nan),
                         "status": "uncertified, margin only (no counts), winner's party and margin as listed",
                         "source_title": meta["title"], "source_revid": meta["revid"], "source_time_utc": meta["timestamp"],
                         "source_url": meta["url"], "source_section": paths.get(i, ""), "source_line": i})
    mg = pd.DataFrame(rows, columns=cols)
    dup = mg[mg.race.duplicated(keep=False)]
    LOG["checks"]["results_2026_margins_listed_twice"] = [f"{r.race}: {r.winner} by {r.margin_abs} (line {r.source_line})" for r in dup.itertuples()]
    same = dup.groupby("race").margin.nunique()
    mg = mg[~mg.race.isin(same.index[same > 1])].drop_duplicates("race").reset_index(drop=True)   # a race listed twice with two margins is dropped
    mg.to_csv(OUT / "e118_results_2026_margins.csv", index=False)
    both = mg[mg.counts_margin.notna()]
    d = (both.margin - both.counts_margin).abs()
    LOG["checks"]["results_2026_margins_table"] = {
        "rows": int(len(mg)), "hidden_in_html_comment": int(mg.hidden_in_html_comment.sum()), "polled": int(mg.polled.sum()),
        "polled_without_counts": sorted(mg.race[mg.polled & mg.counts_margin.isna()]),
        "also_in_counts_file": int(len(both)), "agree_within_half_a_point": int((d <= 0.5).sum()),
        "disagree": [f"{r.race}: table {r.margin:+.1f}, counts {r.counts_margin:+.2f}" for r in both[d > 0.5].itertuples()]}
    return mg


def build_results_2026(races: pd.DataFrame) -> pd.DataFrame:
    """Uncertified counts for 2026 from Wikipedia's state and race articles as they stood at RESULTS_ASOF.

    A race enters only when one article's general-election results table gives a vote count for the
    Democratic-side and the Republican nominee of e94_races. Nothing is filled from race calls or percentages.
    """
    r26 = races[races.cycle == 2026]
    polls = pd.read_csv(OUT / "e94_polls.csv", low_memory=False)
    polled = set(polls[(polls.cycle == 2026) & polls.kind.isin(["house", "senate"])].race)
    pages = []
    for st in sorted(r26[r26.office == "H"].state.unique()):
        n = int((r26[(r26.office == "H")].state == st).sum())
        pages.append(("H", st, f"2026 United States House of Representatives election{'s' if n > 1 else ''} in {STATES[st]}"))
    for part in ("districts 1–26", "districts 27–52"):
        pages.append(("H", "CA", f"2026 United States House of Representatives elections in California ({part})"))
    for r in r26[r26.office == "S"].itertuples():
        pages.append(("S", r.state, f"2026 United States Senate {'special ' if r.special else ''}election in {STATES[r.state]}"))
    rows, pagelog = [], {}
    for office, st, title in pages:
        meta, txt = wiki_rev(title, RESULTS_ASOF, sub="wikipedia_results_2026")
        if meta is None:
            pagelog[title] = "no such page at the cutoff"
            continue
        SOURCES.append({"kind": "wikipedia_results", "cycle": 2026, "key": f"{st}-{office}", "title": meta["title"], "url": meta["url"],
                        "version": meta["revid"], "version_time_utc": meta["timestamp"], "retrieved_utc": meta.get("retrieved_utc", ""),
                        "dated_before_election": False, "file": ""})
        SOURCES[-1]["key"] += f" {meta['title'][-16:]}" if "districts" in meta["title"] else ""
        n_gen = n_filled = 0
        best = {}
        for line, path, btitle, cands, tot in election_boxes(txt):
            where = f"{path} | {btitle}"
            if re.search(r"primary|runoff|caucus|convention", where, flags=re.I) and not re.search(r"general", path, flags=re.I):
                continue
            if not re.search(r"general|results", path, flags=re.I) and not re.search(r"election", btitle, flags=re.I):
                continue
            if re.search(r"\b(20[01]\d|202[0-5])\b", btitle):
                continue                       # a table of an earlier election
            if office == "S":
                race = f"{st}-S"
            else:
                m = re.search(r"District (\d+)|(\d+)(?:st|nd|rd|th) district", where)
                one = int((r26[(r26.office == "H")].state == st).sum()) == 1
                if m:
                    race = f"{st}-{int(m.group(1) or m.group(2)):02d}"
                elif one:
                    race = f"{st}-00"
                else:
                    continue
            n_gen += 1
            if sum(c["votes"] > 0 for c in cands) >= 2:
                n_filled += 1
                best[race] = (line, path, btitle, cands, tot)   # the last such table under the race's heading
        pagelog[title] = f"rev {meta['revid']} {meta['timestamp']}: general tables {n_gen}, with counts {n_filled}"
        for race, (line, path, btitle, cands, tot) in best.items():
            rr = r26[r26.race == race]
            if not len(rr):
                continue
            rr = rr.iloc[0]
            if "|" in str(rr.d_name) or "|" in str(rr.r_name) or pd.isna(rr.d_name) or pd.isna(rr.r_name):
                continue
            listed = float(np.nansum([c["votes"] for c in cands]))
            # The table's own total row when it has one (some tables list only the leading candidates).
            total = tot if tot == tot and tot >= listed else listed

            def pick(name):
                nt = tokens(name)
                sur = norm_name(name).split()[-1]
                hit = [c for c in cands if sur in tokens(c["name"]) and len(nt & tokens(c["name"])) >= 1 and c["votes"] > 0]
                return hit[0] if len(hit) == 1 else None
            d, r = pick(rr.d_name), pick(rr.r_name)
            if d is None or r is None:
                continue
            gaps = [abs(100 * c["votes"] / total - c["pct"]) for c in (d, r) if c["pct"] == c["pct"]]
            consistent = (max(gaps) <= 0.5) if len(gaps) == 2 else np.nan
            # A table that lists only the leading candidates and has no total row prints percentages of a
            # larger total: its two-party share can still agree with its counts.
            two = (abs(d["pct"] / (d["pct"] + r["pct"]) - d["votes"] / (d["votes"] + r["votes"])) <= 0.005) if len(gaps) == 2 else np.nan
            rows.append({"race": race, "office": office, "state": st, "d_name": rr.d_name, "r_name": rr.r_name,
                         "d_votes": int(d["votes"]), "r_votes": int(r["votes"]), "total_votes": int(total),
                         "total_is_table_total_row": bool(tot == tot and tot >= listed),
                         "d_pct": round(100 * d["votes"] / total, 3), "r_pct": round(100 * r["votes"] / total, 3),
                         "margin": round(100 * (d["votes"] - r["votes"]) / total, 3),
                         "margin2": round(100 * (d["votes"] - r["votes"]) / (d["votes"] + r["votes"]), 3),
                         "table_consistent": consistent, "two_party_consistent": two,
                         "d_pct_source": d["pct"], "r_pct_source": r["pct"],
                         "dem2": round(d["votes"] / (d["votes"] + r["votes"]), 5),
                         "d_party_source": d["party"], "r_party_source": r["party"],
                         "pct_expected_in": np.nan, "polled": race in polled,
                         "status": "uncertified count, not final" + (
                             "; INCONSISTENT TABLE: its counts and percentages disagree" if two is False else
                             "; the table omits some candidates: use dem2 or margin2" if consistent is False else ""),
                         "source_title": meta["title"],
                         "source_revid": meta["revid"], "source_time_utc": meta["timestamp"], "source_url": meta["url"],
                         "source_section": path, "source_line": line, "retrieved_utc": meta.get("retrieved_utc", "")})
    res = pd.DataFrame(rows).sort_values(["office", "race"]).reset_index(drop=True)
    # Cross-check (not a source of votes): the latest election-night call of each race in e95_calls_2026.
    calls = pd.read_csv(OUT / "e95_calls_2026.csv")
    last = calls[calls.status == "called"].groupby("race").winner.last()
    res["call_winner_e95"] = res.race.map(last).fillna("")
    lead = np.where(res.margin > 0, "D", "R")
    both = res[res.call_winner_e95 != ""]
    LOG["checks"]["results_2026_vs_e95_calls"] = {
        "races_with_a_call": int(len(both)), "leader_agrees_with_call": int((lead[res.call_winner_e95 != ""] == both.call_winner_e95).sum()),
        "disagree": [f"{r.race}: margin {r.margin:+.2f}, called {r.call_winner_e95}" for r in both[lead[res.call_winner_e95 != ""] != both.call_winner_e95].itertuples()]}
    bad = res[res.two_party_consistent == False]  # noqa: E712
    part = res[(res.table_consistent == False) & (res.two_party_consistent == True)]  # noqa: E712
    LOG["checks"]["results_2026_tables_omitting_candidates"] = sorted(part.race)
    LOG["checks"]["results_2026_inconsistent_tables"] = [
        f"{r.race}: counts {r.d_votes:,} D / {r.r_votes:,} R of {r.total_votes:,} give {r.d_pct}/{r.r_pct}, the table prints {r.d_pct_source}/{r.r_pct_source}"
        for r in bad.itertuples()]
    LOG["counts"]["results_2026"] = {"rows": int(len(res)), "inconsistent_tables": int(len(bad)),
                                     "no_printed_percentages": int(res.table_consistent.isna().sum())}
    wrong = res[(res.d_party_source.map(party_side_wiki) == "R") | (res.r_party_source.map(party_side_wiki) == "D")]
    LOG["checks"]["results_2026_party_label_conflicts"] = [f"{r.race}: {r.d_name} listed as {r.d_party_source}; {r.r_name} as {r.r_party_source}" for r in wrong.itertuples()]
    res.to_csv(OUT / "e118_results_2026.csv", index=False)
    margins_2026(r26, res, polled)
    con = r26[r26.contested & ~r26.multi_cands.fillna(False).astype(bool)]
    have = set(res.race)
    LOG["checks"]["results_2026_pages"] = pagelog
    cov = {}
    for off in ("H", "S"):
        c = con[con.office == off]
        cp = c[c.race.isin(polled)]
        cov[off] = {"contested_races": int(len(c)), "with_counts": int(c.race.isin(have).sum()),
                    "polled_races": int(len(cp)), "polled_with_counts": int(cp.race.isin(have).sum()),
                    "polled_missing": sorted(cp.race[~cp.race.isin(have)])}
    LOG["checks"]["results_2026_coverage"] = cov
    st = con.assign(have=con.race.isin(have)).groupby("state").have.agg(["sum", "size"])
    LOG["checks"]["results_2026_states_with_no_counts"] = sorted(st.index[st["sum"] == 0])
    LOG["checks"]["results_2026_states_partly_counted"] = {s: f"{int(a)}/{int(b)}" for s, (a, b) in st[(st["sum"] > 0) & (st["sum"] < st["size"])].iterrows()}
    return res


# ---------------------------------------------------------------- independent expenditures, by spender
FEC = "https://www.fec.gov/files/bulk-downloads"
HILL = {"C00000935": "DCCC", "C00042366": "DSCC", "C00075820": "NRCC", "C00027466": "NRSC"}
SUPER = {"C00495028": "House Majority PAC", "C00484642": "Senate Majority PAC (SMP)",
         "C00504530": "Congressional Leadership Fund", "C00571703": "Senate Leadership Fund"}
CM_COLS = ["CMTE_ID", "CMTE_NM", "TRES_NM", "CMTE_ST1", "CMTE_ST2", "CMTE_CITY", "CMTE_ST", "CMTE_ZIP", "CMTE_DSGN",
           "CMTE_TP", "CMTE_PTY_AFFILIATION", "CMTE_FILING_FREQ", "ORG_TP", "CONNECTED_ORG_NM", "CAND_ID"]


def read_cm(year: int) -> pd.DataFrame:
    """FEC committee master file of a cycle."""
    p, m = fetch_file(f"{FEC}/{year}/cm{year % 100:02d}.zip", f"fec_cm/cm{year % 100:02d}.zip")
    SOURCES.append({"kind": "fec_cm", "cycle": year, "key": f"cm{year % 100:02d}", "title": "FEC committee master file",
                    "url": m["url"], "version": "", "version_time_utc": "", "retrieved_utc": m["retrieved_utc"],
                    "dated_before_election": "", "file": str(p.relative_to(RAW))})
    with zipfile.ZipFile(p) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = [ln.split("|")[:15] for ln in raw.splitlines() if ln.count("|") >= 14]
    return pd.DataFrame(rows, columns=CM_COLS)


def spender_groups(year: int, cm: pd.DataFrame) -> pd.DataFrame:
    """spe_id -> group and label, from fixed committee IDs and the committee master's own fields.

    hill_committee: DCCC, DSCC, NRCC, NRSC. aligned_superpac: House Majority PAC, Senate Majority PAC,
    Congressional Leadership Fund, Senate Leadership Fund. aligned_affiliate: a committee whose
    'connected organization' in the FEC committee master is one of those four, or which one of the four
    names as its own connected organization (WinSenate and Georgia Honor with SMP). party_other: any other
    party committee (FEC committee types X, Y, Z: the DNC, the RNC, state parties). Everything else: other.
    """
    cm = cm.drop_duplicates("CMTE_ID").set_index("CMTE_ID")
    names4 = {i: {cm.CMTE_NM.get(i, "").strip().upper()} for i in SUPER}
    out = {}
    for i, r in cm.iterrows():
        if i in HILL:
            out[i] = ("hill_committee", HILL[i], "")
        elif i in SUPER:
            out[i] = ("aligned_superpac", SUPER[i], "")
        elif r.CMTE_TP in ("X", "Y", "Z"):
            out[i] = ("party_other", "", "")
    for i, r in cm.iterrows():
        if i in out:
            continue
        conn = r.CONNECTED_ORG_NM.strip().upper()
        for s, nm in names4.items():
            their = cm.CONNECTED_ORG_NM.get(s, "").strip().upper()
            if (conn and conn in nm) or (their and their == r.CMTE_NM.strip().upper()):
                out[i] = ("aligned_affiliate", r.CMTE_NM.strip(), SUPER[s])
    g = pd.DataFrame.from_dict(out, orient="index", columns=["group", "spender", "affiliate_of"])
    g.index.name = "spe_id"
    return g


def read_ie(year: int, cands: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated general-election IEs, one row per item, in a final and an as-of-eve version.

    The filters, the prank-filer rule and the de-duplication are those of tools/build_e94_data.py
    (read_ie), kept line for line so the totals can be checked against data/e94_ie.csv.gz; this version
    keeps the spender and the support/oppose indicator, which E94 folds away.
    """
    p, m = fetch_file(f"{FEC}/{year}/independent_expenditure_{year}.csv", f"fec_ie/independent_expenditure_{year}.csv")
    SOURCES.append({"kind": "fec_ie", "cycle": year, "key": f"independent_expenditure_{year}",
                    "title": "FEC independent expenditures (bulk file)", "url": m["url"], "version": "",
                    "version_time_utc": "", "retrieved_utc": m["retrieved_utc"], "dated_before_election": "",
                    "file": str(p.relative_to(RAW))})
    df = pd.read_csv(p, dtype=str, low_memory=False)
    n0 = len(df)
    df["amt"] = pd.to_numeric(df.exp_amo, errors="coerce")
    df["receipt"] = pd.to_datetime(df.receipt_dat, format="%d-%b-%y", errors="coerce")
    df["dissem"] = pd.to_datetime(df.dissem_dt, format="%d-%b-%y", errors="coerce")
    df["expd"] = pd.to_datetime(df.exp_date, format="%d-%b-%y", errors="coerce")
    df["date"] = df.dissem.fillna(df.expd)
    LOG["counts"][f"ie_{year}_last_receipt_in_file"] = str(df.receipt.max().date())
    df = df[df.can_office.isin(["H", "S"]) & (df.ele_type == "G") & df.amt.notna() & df.receipt.notna()]
    # Prank filings: single items of $100M or more. Every filing of a spender that ever filed one is dropped.
    pranks = set(df.loc[df.amt >= 1e8, "spe_id"])
    bad = df.spe_id.isin(pranks)
    LOG["checks"][f"ie_{year}_prank_spenders"] = {"spenders": len(pranks), "rows": int(bad.sum()),
                                                 "dollars": float(df.amt[bad].sum()),
                                                 "names": sorted(set(df.spe_nam[bad].astype(str)))[:10]}
    df = df[~bad]
    df = df[(df.date >= pd.Timestamp(year - 1, 1, 1)) & (df.date <= pd.Timestamp(ELECTION[year]) + pd.Timedelta(days=1))]
    df = df.assign(amt=df.amt.round(2), cand_id=df.cand_id.fillna(""))
    eve = pd.Timestamp(EVE[year]) + pd.Timedelta(hours=23, minutes=59)

    def dedup(d):
        sup = set(d.loc[d.amndt_ind.fillna("N").str.startswith("A"), "prev_file_num"].dropna())
        d = d[~d.file_num.isin(sup)]
        # E94's call exactly (default sort): it decides which of two same-day duplicates is kept.
        return d.sort_values("receipt").drop_duplicates(["spe_id", "cand_id", "amt", "date"])

    final = dedup(df)
    pre = dedup(df[df.receipt <= eve])
    cols = ["cand_id", "can_office_state", "can_office_dis", "can_office", "cand_pty_aff", "sup_opp", "date", "spe_id",
            "spe_nam", "amt"]
    out = pd.concat([final[cols].assign(version="final"), pre[cols].assign(version="eve")], ignore_index=True)
    out = out.join(cands.set_index("cand_id")[["race", "side"]], on="cand_id")
    pty = out.cand_pty_aff.fillna("").str.upper()
    cand_side = np.where(pty.str.startswith("DEM"), "D", np.where(pty.str.startswith("REP"), "R", "O"))
    out["nominee"] = out.side.notna()
    out["cand_side"] = out.side.fillna(pd.Series(cand_side, index=out.index))
    s = out.cand_side
    # E94's fold: an item that does not say 'S' counts against the candidate it names.
    out["pro"] = np.where(out.sup_opp == "S", s, np.where(s == "D", "R", np.where(s == "R", "D", "O")))
    st = out.can_office_state.fillna("")
    dist = pd.to_numeric(out.can_office_dis, errors="coerce").fillna(0).astype(int)
    fallback = np.where(out.can_office == "S", st + "-S", st + "-" + dist.map("{:02d}".format))
    out["race"] = out.race.fillna(pd.Series(fallback, index=out.index))
    out["sup_opp"] = out.sup_opp.where(out.sup_opp.isin(["S", "O"]), "")
    LOG["counts"][f"ie_{year}"] = {"rows_raw": n0, "items_final": int(len(final)), "total_final": float(final.amt.sum()),
                                   "items_eve": int(len(pre)), "total_eve": float(pre.amt.sum())}
    return out


def build_party_ie(races: pd.DataFrame) -> pd.DataFrame:
    parts = []
    e94 = pd.read_csv(OUT / "e94_ie.csv.gz")
    for y in CYCLES:
        rr = races[races.cycle == y]
        cm_ = pd.concat([rr[["d_id", "race"]].rename(columns={"d_id": "cand_id"}).assign(side="D"),
                         rr[["r_id", "race"]].rename(columns={"r_id": "cand_id"}).assign(side="R")])
        cm_ = cm_[cm_.cand_id.fillna("") != ""].drop_duplicates("cand_id")
        it = read_ie(y, cm_)
        grp = spender_groups(y, read_cm(y))
        it = it.join(grp, on="spe_id")
        it["group"] = it.group.fillna("other")
        it["spender"] = it.spender.fillna("")
        it["affiliate_of"] = it.affiliate_of.fillna("")
        LOG["checks"][f"ie_{y}_aligned_affiliates"] = sorted(
            {f"{r.spe_id} {r.spender} (connected organisation: {r.affiliate_of})"
             for r in it[it.group == "aligned_affiliate"].itertuples()})
        oth = it[(it.group == "other") & (it.version == "eve")].groupby("spe_id").agg(
            name=("spe_nam", "first"), amt=("amt", "sum")).sort_values("amt", ascending=False).head(15)
        LOG["checks"][f"ie_{y}_largest_other_spenders_eve"] = [f"{i} | {r['name']} | {r.amt:.0f}" for i, r in oth.iterrows()]
        agg = it.groupby(["version", "race", "nominee", "cand_side", "sup_opp", "pro", "group", "spender", "affiliate_of"],
                         as_index=False).agg(amt=("amt", "sum"), items=("amt", "size"))
        agg.insert(0, "cycle", y)
        parts.append(agg)
        # Check against E94's folded file: the same dollars by version, race, side helped and nominee flag.
        # Items that name a nominee must agree cell by cell. Items with no candidate ID can differ by cell:
        # E94's de-duplication key (spender, candidate ID, amount, date) merges two such items for different
        # races, and which one survives depends on the row order of the download. The totals still agree.
        an = agg[agg.nominee].groupby(["version", "race", "pro"]).amt.sum()
        bn = e94[(e94.cycle == y) & e94.nominee].groupby(["version", "race", "pro"]).amt.sum()
        nb = pd.concat([an.rename("e118"), bn.rename("e94")], axis=1).fillna(0.0)
        nd = (nb.e118 - nb.e94).abs()
        LOG["checks"][f"ie_{y}_nominee_cells_vs_e94_ie"] = {
            v: {"cells": int(len(nd.xs(v))), "differing_by_over_1_dollar": int((nd.xs(v) > 1).sum()),
                "max_abs_difference": round(float(nd.xs(v).max()), 2)} for v in ("eve", "final")}
        a = agg.groupby(["version", "race", "pro", "nominee"]).amt.sum()
        b = e94[e94.cycle == y].groupby(["version", "race", "pro", "nominee"]).amt.sum()
        both = pd.concat([a.rename("e118"), b.rename("e94")], axis=1).fillna(0.0)
        diff = (both.e118 - both.e94)
        LOG["checks"][f"ie_{y}_vs_e94_ie"] = {
            v: {"e118_total": float(both.e118.xs(v).sum()), "e94_total": float(both.e94.xs(v).sum()),
                "cells": int(len(both.xs(v))), "cells_differing_by_over_1_dollar": int((diff.xs(v).abs() > 1).sum()),
                "max_abs_cell_difference": float(diff.xs(v).abs().max())} for v in ("eve", "final")}
    ie = pd.concat(parts, ignore_index=True)
    ie.to_csv(OUT / "e118_party_ie.csv.gz", index=False, compression=GZ, float_format="%.2f")
    LOG["counts"]["party_ie_rows"] = pairs(ie.groupby(["cycle", "version"]).size())
    LOG["counts"]["party_ie_dollars_eve_by_group"] = {
        str(y): {k: round(float(v)) for k, v in ie[(ie.cycle == y) & (ie.version == "eve")].groupby("group").amt.sum().items()}
        for y in CYCLES}
    return ie


# ---------------------------------------------------------------- party committees: coordinated and direct
PAS2_COLS = ["CMTE_ID", "AMNDT_IND", "RPT_TP", "TRANSACTION_PGI", "IMAGE_NUM", "TRANSACTION_TP", "ENTITY_TP", "NAME", "CITY",
             "STATE", "ZIP_CODE", "EMPLOYER", "OCCUPATION", "TRANSACTION_DT", "TRANSACTION_AMT", "OTHER_ID", "CAND_ID",
             "TRAN_ID", "FILE_NUM", "MEMO_CD", "MEMO_TEXT", "SUB_ID"]
PAS2_TYPES = {"24C": "coordinated", "24K": "contribution", "24Z": "in_kind"}


def build_party_direct(races: pd.DataFrame) -> pd.DataFrame:
    """What party committees spent with or gave to each nominee: coordinated party expenditures (FEC
    transaction type 24C), contributions (24K) and in-kind contributions (24Z), from the FEC's
    committee-to-candidate file (pas2). Independent expenditures are in e118_party_ie, not here.

    The file has transaction dates but no filing dates. 'pre_general' keeps transactions dated up to the
    close of books of the pre-general report (the 20th day before the election), which was due 12 days
    before the election and so was public by the eve; 'final' keeps everything dated up to election day
    that the file held when it was retrieved (for 2026 that is before the post-general reports).
    """
    parts = []
    for y in CYCLES:
        p, m = fetch_file(f"{FEC}/{y}/pas2{y % 100:02d}.zip", f"fec_pas2/pas2{y % 100:02d}.zip")
        SOURCES.append({"kind": "fec_pas2", "cycle": y, "key": f"pas2{y % 100:02d}", "title": "FEC contributions from committees to candidates (bulk file)",
                        "url": m["url"], "version": "", "version_time_utc": "", "retrieved_utc": m["retrieved_utc"],
                        "dated_before_election": "", "file": str(p.relative_to(RAW))})
        with zipfile.ZipFile(p) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), sep="|", names=PAS2_COLS, dtype=str, encoding="latin-1", quoting=3)
        cm = read_cm(y).drop_duplicates("CMTE_ID").set_index("CMTE_ID")
        party = set(cm.index[cm.CMTE_TP.isin(["X", "Y", "Z"])])
        df = df[df.CMTE_ID.isin(party) & df.TRANSACTION_TP.isin(PAS2_TYPES) & (df.MEMO_CD.fillna("") != "X")].copy()
        df["amt"] = pd.to_numeric(df.TRANSACTION_AMT, errors="coerce")
        df["dt"] = pd.to_datetime(df.TRANSACTION_DT, format="%m%d%Y", errors="coerce")
        day, close = pd.Timestamp(ELECTION[y]), pd.Timestamp(ELECTION[y]) - pd.Timedelta(days=20)
        n_bad = int((df.dt.isna() | (df.dt < pd.Timestamp(y - 1, 1, 1)) | (df.dt > pd.Timestamp(y, 12, 31))).sum())
        df = df[(df.dt >= pd.Timestamp(y - 1, 1, 1)) & (df.dt <= day) & df.amt.notna()]
        rr = races[races.cycle == y]
        cmap = pd.concat([rr[["d_id", "race"]].rename(columns={"d_id": "cand_id"}).assign(side="D"),
                          rr[["r_id", "race"]].rename(columns={"r_id": "cand_id"}).assign(side="R")])
        cmap = cmap[cmap.cand_id.fillna("") != ""].drop_duplicates("cand_id").set_index("cand_id")
        df = df.join(cmap, on="CAND_ID")
        tot_all = df.groupby("TRANSACTION_TP").amt.sum()
        df = df[df.race.notna()]
        df["group"] = np.where(df.CMTE_ID.isin(HILL), "hill_committee", "party_other")
        df["spender"] = df.CMTE_ID.map(HILL).fillna("")
        df["type"] = df.TRANSACTION_TP.map(PAS2_TYPES)
        for version, d in (("pre_general", df[df.dt <= close]), ("final", df)):
            a = d.groupby(["race", "side", "CAND_ID", "group", "spender", "type"], as_index=False).agg(amt=("amt", "sum"), items=("amt", "size"))
            a.insert(0, "version", version)
            a.insert(0, "cycle", y)
            parts.append(a)
        LOG["counts"][f"party_direct_{y}"] = {
            "pre_general_close_of_books": str(close.date()), "rows_with_dates_outside_the_cycle_dropped": n_bad,
            "dollars_all_party_committees_by_type": {PAS2_TYPES[k]: round(float(v)) for k, v in tot_all.items()},
            "dollars_to_nominees_by_type_final": {k: round(float(v)) for k, v in df.groupby("type").amt.sum().items()},
            "hill_committee_coordinated_final": {k: round(float(v)) for k, v in df[(df.type == "coordinated") & (df.group == "hill_committee")].groupby("spender").amt.sum().items()},
            "last_transaction_date": str(df.dt.max().date())}
    out = pd.concat(parts, ignore_index=True).rename(columns={"CAND_ID": "cand_id"})
    out.to_csv(OUT / "e118_party_direct.csv", index=False, float_format="%.2f")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    races = pd.read_csv(OUT / "e94_races.csv", dtype={"d_id": str, "r_id": str})
    races = races[races.cycle.isin(CYCLES)].copy()
    build_ideology(races)
    build_results_2026(races)
    build_party_ie(races)
    build_party_direct(races)
    src = pd.DataFrame(SOURCES).drop_duplicates(["kind", "cycle", "key", "version"]).reset_index(drop=True)
    src["sha256"] = [USED.get(f, "") for f in src.file]
    src.to_csv(OUT / "e118_sources.csv", index=False)
    # The dating rule, checked: every source of an ideology flag or of the pre-election DW-NOMINATE is a
    # version made before that cycle's election day.
    dated = src[src.kind.isin(["wikipedia", "wayback"]) | ((src.kind == "voteview") & (src.cycle != ""))]
    latest = {}
    for c in CYCLES:
        d = dated[dated.cycle == c]
        t = d.version_time_utc.max()
        assert t[:10] < ELECTION[c], (c, t)
        latest[str(c)] = {"sources": int(len(d)), "latest_version_time_utc": t, "election_day": ELECTION[c],
                          "latest_source": d.loc[d.version_time_utc.idxmax(), "title"]}
    LOG["checks"]["dating_rule"] = latest
    LOG["raw_files_sha256"] = USED
    (OUT / "e118_build_log.json").write_text(json.dumps(LOG, indent=1, default=str))
    for f in ("e118_ideology.csv", "e118_ideology_flags.csv", "e118_results_2026.csv", "e118_results_2026_margins.csv", "e118_party_ie.csv.gz",
              "e118_party_direct.csv", "e118_sources.csv", "e118_build_log.json"):
        print(f"{f:<26} {(OUT / f).stat().st_size / 1e3:8.0f} kB")


if __name__ == "__main__":
    main()
