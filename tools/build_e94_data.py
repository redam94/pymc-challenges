"""Build E94's processed data from the raw downloads in ~/.cache/pymc-challenges/e94/raw/.

The raw files are public downloads (FEC bulk files and results workbooks, MEDSL returns,
FiveThirtyEight poll and lean files, Wikipedia revisions frozen at 2026-11-02). They are treated as
untrusted data: parsed with the standard library and pandas, never executed. Run in isolated mode:

    JAX_PLATFORMS=cpu taskset -c 12,13 .venv/bin/python tools/memguard.py --max-gb 8 -- \
        .venv/bin/python -I tools/build_e94_data.py [RAW_DIR]

Writes data/e94_races.csv, e94_polls.csv, e94_reports.csv.gz, e94_ie.csv.gz, e94_rematch.csv,
e94_pollerr.csv and e94_build_log.json. Everything here is stdlib + numpy + pandas (no lxml,
openpyxl or bs4): a small xlsx reader and an HTML table extractor are included below.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/pymc-challenges/e94/raw"
OUT = Path(__file__).resolve().parents[1] / "data"

# Election days and the information cutoff ("eve") for each cycle. Polls created, reports received
# and IE items received after the eve's end (23:59) are dropped from the "as of eve" columns.
ELECTION = {2016: "2016-11-08", 2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2026: "2026-11-03"}
EVE = {2018: "2018-11-05", 2022: "2022-11-07", 2026: "2026-11-02"}
# gzip with a zero header timestamp, so a rebuild is byte-identical (the freeze records sha256 of the files).
GZ = {"method": "gzip", "mtime": 0}
LOG: dict = {"cutoffs": EVE, "election_days": ELECTION, "checks": {}, "counts": {}}

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
# Census regions, used for correlated race errors.
REGION = {}
for r, ss in {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}.items():
    for s in ss.split():
        REGION[s] = r
def party_side(p: str) -> str:
    """FEC workbook party labels ('D', 'DFL', 'D/WF', 'D*', 'W(R)/R', 'DEM/IP/WF') -> D, R or O."""
    toks = set(re.split(r"[/()*\s]+", str(p).upper())) - {""}
    if toks & {"D", "DEM", "DFL", "DNL"}:
        return "D"
    if toks & {"R", "REP"}:
        return "R"
    return "O"


def pairs(s: pd.Series) -> dict:
    return {"_".join(map(str, k)): int(v) for k, v in s.items()}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


USED: dict[str, str] = {}


def use(p) -> Path:
    p = Path(p)
    USED[str(p.relative_to(RAW))] = sha256(p)
    return p


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s)


def last_tokens(s: str, minlen: int = 3) -> set[str]:
    stop = {"jr", "sr", "ii", "iii", "iv", "mr", "mrs", "ms", "dr", "the", "de", "la", "van", "von", "phd"}
    return {t for t in norm_name(s).split() if len(t) >= minlen and t not in stop}


def read_cn(yy: int) -> pd.DataFrame:
    """FEC candidate master file (cn), the first 10 fields."""
    cols = ["CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_ELECTION_YR", "CAND_OFFICE_ST", "CAND_OFFICE",
            "CAND_OFFICE_DISTRICT", "CAND_ICI", "CAND_STATUS", "CAND_PCC"]
    with zipfile.ZipFile(use(RAW / f"fec/cn{yy:02d}.zip")) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = [ln.split("|")[:10] for ln in raw.splitlines() if ln.count("|") >= 9]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- stdlib readers
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def xlsx_read(path, want):
    """{sheet name: rows} for the sheets whose name contains `want` (stdlib zip + XML)."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rmap = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for s in wb.find("m:sheets", _NS):
        name = s.get("name")
        if want not in name:
            continue
        target = rmap[s.get(f"{{{_NS['r']}}}id")].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).iter(f"{{{_NS['m']}}}row"):
            vals = {}
            for c in row.findall("m:c", _NS):
                t, v = c.get("t"), c.find("m:v", _NS)
                if t == "s" and v is not None:
                    x = shared[int(v.text)]
                elif t == "inlineStr":
                    x = "".join(tt.text or "" for tt in c.iter(f"{{{_NS['m']}}}t"))
                else:
                    x = v.text if v is not None else None
                vals[_col(c.get("r"))] = x
            if vals:
                rows.append([vals.get(i) for i in range(max(vals) + 1)])
        out[name] = rows
    return out


class _TableParser(HTMLParser):
    """Tables with rowspan/colspan, each tagged with the h2/h3/h4 headings above it."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.stack, self.cell = [], [], None
        self.skipstack = []
        self.head = {"h2": "", "h3": "", "h4": ""}
        self.in_head, self.htext = None, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "script", "sup") or (tag == "span" and "display:none" in (a.get("style") or "")):
            self.skipstack.append(tag)
            return
        if tag in ("h2", "h3", "h4"):
            self.in_head, self.htext = tag, []
        if tag == "table":
            self.stack.append({"rows": [], "head": dict(self.head)})
        elif tag == "tr" and self.stack:
            self.stack[-1]["rows"].append([])
        elif tag in ("td", "th") and self.stack:
            def span(k):
                v = (a.get(k) or "1").strip(";")
                return int(v) if v.isdigit() else 1
            self.cell = {"t": [], "rs": span("rowspan"), "cs": span("colspan")}
        elif tag == "br" and self.cell is not None:
            self.cell["t"].append(" ")

    def handle_endtag(self, tag):
        if self.skipstack and tag == self.skipstack[-1]:
            self.skipstack.pop()
            return
        if tag in ("h2", "h3", "h4") and self.in_head == tag:
            txt = " ".join("".join(self.htext).split()).replace("[edit]", "")
            self.head[tag] = txt
            if tag == "h2":
                self.head["h3"] = self.head["h4"] = ""
            if tag == "h3":
                self.head["h4"] = ""
            self.in_head = None
        if tag in ("td", "th") and self.cell is not None and self.stack:
            if self.stack[-1]["rows"]:
                self.stack[-1]["rows"][-1].append(self.cell)
            self.cell = None
        elif tag == "table" and self.stack:
            self.tables.append(self.stack.pop())

    def handle_data(self, data):
        if self.skipstack:
            return
        if self.in_head:
            self.htext.append(data)
        if self.cell is not None:
            self.cell["t"].append(data)


def html_tables(path):
    p = _TableParser()
    p.feed(Path(path).read_text(encoding="utf-8"))
    out = []
    for t in p.tables:
        grid = {}
        for r, row in enumerate(t["rows"]):
            c = 0
            for cell in row:
                while (r, c) in grid:
                    c += 1
                txt = " ".join("".join(cell["t"]).split())
                for dr in range(min(cell["rs"], 500)):
                    for dc in range(min(cell["cs"], 50)):
                        grid[(r + dr, c + dc)] = txt
                c += cell["cs"]
        nr = max([k[0] for k in grid] + [-1]) + 1
        nc = max([k[1] for k in grid] + [-1]) + 1
        out.append({"head": t["head"], "rows": [[grid.get((r, c), "") for c in range(nc)] for r in range(nr)]})
    return out


def fnum(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- official results (FEC workbooks)
def fec_results(year: int, office: str) -> pd.DataFrame:
    """Candidate rows for the House ('H') or Senate ('S') general election of `year`."""
    path = use(RAW / f"fec_results/federalelections{year}.xlsx")
    want = "House Results" if office == "H" else "Senate Results"
    rows = list(xlsx_read(path, want).values())[0]
    hdr = [str(h or "").strip().upper() for h in rows[0]]

    def ix(*keys):
        for k in keys:
            for i, h in enumerate(hdr):
                if h.startswith(k):
                    return i
        return None

    i_st, i_d, i_id, i_inc = 1, 3, ix("FEC ID"), ix("(I)")
    i_last, i_first, i_party = ix("CANDIDATE NAME (LAST)"), ix("CANDIDATE NAME (FIRST)"), ix("PARTY")
    i_name = hdr.index("CANDIDATE NAME")
    i_gv, i_comb, i_win = ix("GENERAL VOTES"), ix("COMBINED GE PARTY TOTALS", "COMBINED GE"), ix("GE WINNER")
    rcv = [i for i, h in enumerate(hdr) if "RCV VOTES" in h]
    recs = []
    for r in rows[1:]:
        r = list(r) + [None] * (len(hdr) - len(r))
        fid = (r[i_id] or "").strip()
        if not fid or fid == "n/a" or r[i_st] not in STATES:
            continue
        gv = fnum(r[i_gv])
        unopp = str(r[i_gv] or "").strip().lower() == "unopposed"
        if unopp:
            gv = 0.0  # Florida does not print votes for unopposed candidates
        if np.isnan(gv):
            continue
        last_rcv = np.nan
        for i in rcv:
            if not np.isnan(fnum(r[i])):
                last_rcv = fnum(r[i])
        recs.append({
            "cycle": year, "office": office, "state": r[i_st], "dist": str(r[i_d] or "").strip(),
            "cand_id": fid, "inc": (r[i_inc] or "").strip() == "(I)", "last": (r[i_last] or "").strip(),
            "name": ((r[i_name] or "").strip() or f"{(r[i_last] or '').strip()}, {(r[i_first] or '').strip()}"),
            "party": (r[i_party] or "").strip().upper(),
            "votes": gv, "combined": fnum(r[i_comb]) if i_comb is not None else np.nan,
            "rcv_last": last_rcv, "unopposed": unopp,
            "winner": unopp or ((r[i_win] or "").strip().upper() == "W" if i_win is not None else False),
        })
    df = pd.DataFrame(recs)
    # Fusion candidates (CT, NY, SC) appear once per ballot line: sum their lines.
    agg = df.groupby(["cycle", "office", "state", "dist", "cand_id"], as_index=False).agg(
        inc=("inc", "max"), last=("last", "first"), name=("name", "first"), votes=("votes", "sum"),
        combined=("combined", "max"), rcv_last=("rcv_last", "max"), winner=("winner", "max"), unopposed=("unopposed", "max"),
        party=("party", lambda s: next((p for p in s if party_side(p) != "O"), s.iloc[0])))
    agg["votes"] = np.where(agg.combined > agg.votes, agg.combined, agg.votes)
    agg["side"] = agg.party.map(party_side)
    # Nonpartisan ballot labels ('N': Mississippi's 2018 special, Louisiana) take the FEC registration's party.
    nonp = agg.party.eq("N") & agg.cand_id.isin(FEC_PARTY)
    agg.loc[nonp, "side"] = agg.loc[nonp, "cand_id"].map(FEC_PARTY).map(party_side)
    return agg


def win_side(g, d, rr):
    w = g.loc[g.winner, "cand_id"] if g.winner.any() else g.cand_id.head(1)
    w = w.iloc[0]
    if len(d) and w == d.cand_id.iloc[0]:
        return "D"
    if len(rr) and w == rr.cand_id.iloc[0]:
        return "R"
    return g.set_index("cand_id").side.get(w, "O")


FEC_PARTY: dict = {}


def race_frame(cands: pd.DataFrame) -> pd.DataFrame:
    """One row per race: top D-side and top R candidate, two-party D share, incumbency."""
    out = []
    for (cycle, office, st, dist), g in cands.groupby(["cycle", "office", "state", "dist"]):
        if office == "S" and "special" not in dist.lower() and "unexpired" not in dist.lower() and dist.upper() not in ("S", "S-FULL TERM"):
            continue
        if office == "H":
            # Same-day specials for the rest of a term ('01 - UNEXPIRED TERM') are skipped.
            if "UNEXPIRED" in dist.upper() or not re.match(r"\d+", dist):
                continue
            dist = re.match(r"\d+", dist).group(0)
        use_rcv = g.rcv_last.notna().sum() >= 2
        v = g.rcv_last.fillna(0.0) if use_rcv else g.votes
        g = g.assign(v=v).sort_values("v", ascending=False)
        d = g[g.side == "D"].head(1)
        rr = g[g.side == "R"].head(1)
        # An independent who outpolls every Democrat is the D-side candidate (Sanders, King, McMullin).
        o = g[g.side == "O"].head(1)
        if len(o) and (not len(d) or o.v.iloc[0] > d.v.iloc[0]) and o.v.iloc[0] > 0.2 * g.v.sum():
            d = o
        special = office == "S" and ("unexpired" in dist.lower() or "special" in dist.lower())
        key = f"{st}-{int(dist):02d}" if office == "H" else f"{st}-S" + ("2" if special else "")
        rec = {"cycle": cycle, "office": office, "state": st, "race": key, "special": special,
               "d_id": d.cand_id.iloc[0] if len(d) else "", "r_id": rr.cand_id.iloc[0] if len(rr) else "",
               "d_name": d.name.iloc[0] if len(d) else "", "r_name": rr.name.iloc[0] if len(rr) else "",
               "d_votes": d.v.iloc[0] if len(d) else 0.0, "r_votes": rr.v.iloc[0] if len(rr) else 0.0,
               "d_inc": bool(d.inc.iloc[0]) if len(d) else False, "r_inc": bool(rr.inc.iloc[0]) if len(rr) else False,
               "winner": win_side(g, d, rr),
               "rcv": use_rcv,
               # Jungle-style ballots (nonpartisan labels, or a second candidate of the same side with over
               # 10% of the vote): a top-D vs top-R share is not a two-party vote there.
               "split": bool((g.party == "N").any()) or bool(
                   (g[g.side == "D"].v.iloc[1:2].sum() > 0.1 * g.v.sum()) or (g[g.side == "R"].v.iloc[1:2].sum() > 0.1 * g.v.sum()))}
        out.append(rec)
    df = pd.DataFrame(out)
    tot = df.d_votes + df.r_votes
    df["dem2"] = np.where((df.d_votes > 0) & (df.r_votes > 0), df.d_votes / tot.where(tot > 0, 1), np.nan)
    df["contested"] = df.dem2.notna()
    return df


# ---------------------------------------------------------------- partisan lean
def lean_tables():
    def parse(v):
        v = str(v).strip()
        if v.startswith(("R+", "D+")):
            return float(v[2:]) * (1 if v[0] == "D" else -1)
        return fnum(v)

    l18 = pd.read_csv(use(RAW / "fte_github/partisan-lean-2018/fivethirtyeight_partisan_lean_DISTRICTS.csv"),
                      encoding="utf-8-sig")
    l18 = dict(zip(l18.district, l18["2018"].map(parse)))
    s18 = pd.read_csv(use(RAW / "fte_github/partisan-lean-2018/fivethirtyeight_partisan_lean_STATES.csv"),
                      encoding="utf-8-sig")
    s18 = {ABBR[s]: parse(v) for s, v in zip(s18.state, s18["2018"]) if s in ABBR}
    l22 = pd.read_csv(use(RAW / "fte_github/partisan-lean/fivethirtyeight_partisan_lean_DISTRICTS.csv"))
    l22 = dict(zip(l22.district, l22["2022"].astype(float)))
    return l18, s18, l22


def dist538(race: str) -> str:
    st, d = race.split("-")
    return f"{st}-{max(int(d), 1)}"


# ---------------------------------------------------------------- FEC: candidates, reports
def read_weball(yy: str) -> pd.DataFrame:
    cols = ["CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS", "TRANS_FROM_AUTH",
            "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS",
            "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST",
            "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION",
            "GEN_ELECTION_PRECENT", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS",
            "CMTE_REFUNDS"]
    with zipfile.ZipFile(use(RAW / f"fec/weball{yy}.zip")) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = [ln.split("|") for ln in raw.splitlines() if ln.strip()]
    df = pd.DataFrame([r[:30] for r in rows if len(r) >= 30], columns=cols)
    for c in ("TTL_RECEIPTS", "TTL_DISB", "COH_COP", "TTL_INDIV_CONTRIB", "CAND_CONTRIB", "CAND_LOANS"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def read_reports(year: int) -> pd.DataFrame:
    p = use(RAW / f"fec_reports/{year}_HOUSE_SENATE_CAMPAIGNS_DOWNLOAD.csv")
    keep = ["CAND_ID", "CMTE_ID", "FEC_ELECTION_YR", "FORM_TP_CD", "RPT_TP", "RECEIPT_DT", "CVG_START_DT",
            "CVG_END_DT", "TTL_DISB", "TTL_RECEIPTS", "COH_COP"]
    df = pd.read_csv(p, encoding="latin-1", dtype=str)
    # A report's CAND_ID can be stale (a House member's committee that now runs a Senate campaign keeps
    # the House ID). The FEC's candidate-committee linkage file gives every candidate ID that each
    # committee serves this cycle (principal and authorised committees): a report row is listed under
    # each of them as well as under its own CAND_ID.
    df = df[[c for c in keep if c in df.columns]]
    df = df[(df.FORM_TP_CD == "F3") & (df.FEC_ELECTION_YR == str(year))].copy()
    with zipfile.ZipFile(use(RAW / f"fec_ccl/ccl{year % 100:02d}.zip")) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    ccl = pd.DataFrame([ln.split("|") for ln in raw.splitlines() if ln.strip()],
                       columns=["L_CAND_ID", "CAND_ELECTION_YR", "FEC_ELECTION_YR", "CMTE_ID", "CMTE_TP", "CMTE_DSGN", "LINKAGE_ID"])
    ccl = ccl[ccl.CMTE_DSGN.isin(["P", "A"])][["CMTE_ID", "L_CAND_ID"]].drop_duplicates()
    linked = df.merge(ccl, on="CMTE_ID").assign(CAND_ID=lambda d: d.L_CAND_ID).drop(columns="L_CAND_ID")
    df = pd.concat([df, linked], ignore_index=True)
    df = df.drop_duplicates(["CAND_ID", "CMTE_ID", "RPT_TP", "CVG_START_DT", "CVG_END_DT", "RECEIPT_DT"])
    df["receipt"] = pd.to_datetime(df.RECEIPT_DT, format="%d-%b-%y", errors="coerce")
    df["cvg_start"] = pd.to_datetime(df.CVG_START_DT, errors="coerce")
    df["cvg_end"] = pd.to_datetime(df.CVG_END_DT, errors="coerce")
    for c in ("TTL_DISB", "TTL_RECEIPTS", "COH_COP"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def money_at(reports: pd.DataFrame, cutoff: str | None, cvg_until: str | None = None) -> pd.DataFrame:
    r = reports
    if cutoff is not None:
        r = r[r.receipt <= pd.Timestamp(cutoff) + pd.Timedelta(hours=23, minutes=59)]
    if cvg_until is not None:
        r = r[r.cvg_end <= pd.Timestamp(cvg_until)]
    g = r.sort_values("cvg_end").groupby("CAND_ID")
    return pd.DataFrame({"receipts": g.TTL_RECEIPTS.sum(), "disb": g.TTL_DISB.sum(), "coh": g.COH_COP.last(),
                         "last_cvg": g.cvg_end.max()})


# ---------------------------------------------------------------- independent expenditures
def read_ie(year: int, cands: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated general-election IEs, one row per item, with first and as-of-eve information."""
    df = pd.read_csv(use(RAW / f"fec_ie/independent_expenditure_{year}.csv"), dtype=str, low_memory=False)
    n0 = len(df)
    df["amt"] = pd.to_numeric(df.exp_amo, errors="coerce")
    df["receipt"] = pd.to_datetime(df.receipt_dat, format="%d-%b-%y", errors="coerce")
    df["dissem"] = pd.to_datetime(df.dissem_dt, format="%d-%b-%y", errors="coerce")
    df["expd"] = pd.to_datetime(df.exp_date, format="%d-%b-%y", errors="coerce")
    df["date"] = df.dissem.fillna(df.expd)
    df = df[df.can_office.isin(["H", "S"]) & (df.ele_type == "G") & df.amt.notna() & df.receipt.notna()]
    # Prank filings: single items of $100M or more (the files hold $1e9-1e10 "expenditures" filed by
    # individuals). Every filing of a spender that ever filed one is dropped.
    pranks = set(df.loc[df.amt >= 1e8, "spe_id"])
    bad = df.spe_id.isin(pranks)
    LOG["checks"][f"ie_{year}_prank_spenders"] = {"spenders": len(pranks), "rows": int(bad.sum()),
                                                 "dollars": float(df.amt[bad].sum())}
    df = df[~bad]
    big = df.sort_values("amt", ascending=False).head(5)
    LOG["checks"][f"ie_{year}_largest_kept"] = [f"{r.spe_nam} | {r.cand_name} | {r.amt:.0f} | {r.exp_date}"
                                                for r in big.itertuples()]
    df = df[(df.date >= pd.Timestamp(year - 1, 1, 1)) & (df.date <= pd.Timestamp(ELECTION[year]) + pd.Timedelta(days=1))]
    df = df.assign(amt=df.amt.round(2), cand_id=df.cand_id.fillna(""))
    eve = pd.Timestamp(EVE[year]) + pd.Timedelta(hours=23, minutes=59)

    def dedup(d):
        # An amendment (A1, A2, ...) replaces the whole filing named in prev_file_num. The same item then
        # appears again on the spender's periodic report: one row per spender, candidate, amount and
        # date, keeping the earliest receipt (when the public first saw it). tran_id is unique only
        # within a filing, so it cannot be the key.
        sup = set(d.loc[d.amndt_ind.fillna("N").str.startswith("A"), "prev_file_num"].dropna())
        d = d[~d.file_num.isin(sup)]
        d = d.sort_values("receipt").drop_duplicates(["spe_id", "cand_id", "amt", "date"])
        return d

    final = dedup(df)
    pre = dedup(df[df.receipt <= eve])
    # When was each final item first seen? The earliest receipt of any row with its spender, candidate,
    # amount and date (so a 24-hour notice dates an item later restated on a periodic report).
    df["cand_id"] = df.cand_id.fillna("")
    first = df.groupby(["spe_id", "cand_id", "amt", "date"]).receipt.min().rename("first_receipt")
    final = final.join(first, on=["spe_id", "cand_id", "amt", "date"])
    LOG["counts"][f"ie_{year}"] = {"rows_raw": n0, "rows_general": int(len(df)), "items_final": int(len(final)),
                                   "total_final": float(final.amt.sum()), "items_eve": int(len(pre)),
                                   "total_eve": float(pre.amt.sum()),
                                   "total_final_first_seen_by_eve": float(final.amt[final.first_receipt <= eve].sum())}
    cols = ["cand_id", "can_office_state", "can_office_dis", "can_office", "cand_pty_aff", "sup_opp", "date", "spe_id", "amt"]
    out = pd.concat([final[cols + ["first_receipt"]].assign(version="final"),
                     pre[cols].assign(first_receipt=pre.receipt, version="eve")], ignore_index=True)
    # Map each item to a race through the candidate.
    cmap = cands.set_index("cand_id")[["race", "side"]]
    out = out.join(cmap, on="cand_id")
    pty = out.cand_pty_aff.fillna("").str.upper()
    cand_side = np.where(pty.str.startswith("DEM"), "D", np.where(pty.str.startswith("REP"), "R", "O"))
    side = out.side.fillna(pd.Series(cand_side, index=out.index))
    out["pro"] = np.where(out.sup_opp == "S", side, np.where(side == "D", "R", np.where(side == "R", "D", "O")))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- polls: FiveThirtyEight 2018/2022
def poll_pick(df: pd.DataFrame) -> pd.DataFrame:
    """One question per poll: likely voters first, then registered, then the largest sample."""
    pr = {"lv": 0, "rv": 1, "v": 2, "a": 3}
    df = df.assign(_p=df.population.map(pr).fillna(4), _n=-df.sample_size.fillna(0))
    return df.sort_values(["poll_id", "_p", "_n"]).groupby("poll_id").head(1).drop(columns=["_p", "_n"])


def fte_polls(races: pd.DataFrame) -> pd.DataFrame:
    out = []
    g = pd.read_csv(use(RAW / "fte_wayback/generic_ballot_polls_historical__20250118200335.csv"), low_memory=False)
    s = pd.read_csv(use(RAW / "fte_wayback/senate_polls_historical__20250118200335.csv"), low_memory=False)
    h = pd.read_csv(use(RAW / "fte_wayback/house_polls_historical__20250118200335.csv"), low_memory=False)
    for df in (g, s, h):
        df["created"] = pd.to_datetime(df.created_at, format="%m/%d/%y %H:%M", errors="coerce")
        df["end"] = pd.to_datetime(df.end_date, format="%m/%d/%y", errors="coerce")
        df["start"] = pd.to_datetime(df.start_date, format="%m/%d/%y", errors="coerce")
    for cycle in (2018, 2022):
        eve = pd.Timestamp(EVE[cycle]) + pd.Timedelta(hours=23, minutes=59)
        x = g[(g.cycle == cycle) & (g.created <= eve) & (g.end <= eve)]
        LOG["counts"][f"generic_{cycle}_dropped_created_after_eve"] = int(((g.cycle == cycle) & (g.created > eve)).sum())
        x = poll_pick(x)
        for _, r in x.iterrows():
            out.append({"cycle": cycle, "kind": "generic", "race": "US", "pollster": r.pollster,
                        "partisan": {"DEM": "D", "REP": "R"}.get(r.partisan, ""), "internal": False,
                        "start": r.start, "end": r.end, "created": r.created, "n": r.sample_size,
                        "population": r.population, "dem": r.dem, "rep": r.rep, "source": "538"})
        for kind, df in (("senate", s), ("house", h)):
            x = df[(df.cycle == cycle) & (df.stage == "general") & (~df.hypothetical.fillna(False).astype(bool))
                   & (df.created <= eve) & (df.end <= eve)]
            rr = races[(races.cycle == cycle) & (races.office == ("S" if kind == "senate" else "H"))]
            nmatch = nmiss = 0
            for qid, q in x.groupby("question_id"):
                st = ABBR.get(q.state.iloc[0])
                if st is None:
                    continue
                if kind == "house":
                    cand = rr[rr.race == f"{st}-{int(q.seat_number.iloc[0]):02d}"]
                else:
                    cand = rr[rr.state == st]
                # Assign the question to the race whose two nominees it names.
                hit = None
                for _, race in cand.iterrows():
                    dl, rl = last_tokens(race.d_name), last_tokens(race.r_name)
                    dq = q[q.candidate_name.map(lambda n: bool(last_tokens(n) & dl))]
                    rq = q[q.candidate_name.map(lambda n: bool(last_tokens(n) & rl))]
                    if len(dq) == 1 and len(rq) == 1:
                        hit = (race, dq.pct.iloc[0], rq.pct.iloc[0])
                if hit is None:
                    nmiss += 1
                    continue
                nmatch += 1
                race, dp, rp = hit
                r = q.iloc[0]
                out.append({"cycle": cycle, "kind": kind, "race": race.race, "pollster": r.pollster,
                            "partisan": {"DEM": "D", "REP": "R"}.get(r.partisan, ""),
                            "internal": bool(r.internal) if pd.notna(r.internal) else False,
                            "start": r.start, "end": r.end, "created": r.created, "n": r.sample_size,
                            "population": r.population, "dem": dp, "rep": rp, "source": "538",
                            "poll_id": r.poll_id})
            LOG["counts"][f"{kind}_{cycle}_questions_matched"] = nmatch
            LOG["counts"][f"{kind}_{cycle}_questions_unmatched"] = nmiss
    df = pd.DataFrame(out)
    # One question per poll and race (538 lists LV and RV versions as separate questions).
    pr = {"lv": 0, "rv": 1, "v": 2, "a": 3}
    df["_p"] = df.population.map(pr).fillna(4)
    df["poll_id"] = df.poll_id.fillna(-1)
    a = df[df.kind != "generic"].sort_values("_p").groupby(["poll_id", "race"]).head(1)
    df = pd.concat([df[df.kind == "generic"], a]).drop(columns="_p")
    return df


# ---------------------------------------------------------------- 2026 from Wikipedia (frozen 2026-11-02)
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}


def parse_end_date(txt: str, default_year=2026):
    t = norm_name(txt.replace("–", " ").replace("-", " "))
    toks = t.split()
    months = [(i, MONTHS[w]) for i, w in enumerate(toks) if w in MONTHS]
    nums = [int(x) for x in re.findall(r"\d+", txt.replace("–", " "))]
    year = next((n for n in reversed(nums) if n > 1900), default_year)
    days = [n for n in nums if n <= 31]
    if not months or not days:
        return pd.NaT
    m = months[-1][1]
    try:
        return pd.Timestamp(year, m, days[-1])
    except ValueError:
        return pd.NaT


def parse_sample(txt: str):
    m = re.search(r"([\d,]+)\s*\((\w+)\)", txt)
    n = fnum(m.group(1)) if m else fnum(re.sub(r"[^\d]", "", txt.split("(")[0]) or "nan")
    pop = re.search(r"\((LV|RV|A|V)\)", txt)
    return n, (pop.group(1).lower() if pop else "")


def pct(txt):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*%", txt or "")
    return float(m.group(1)) if m else np.nan


def wiki_candidates(txt: str):
    """'▌Name (Party) ▌Name (Party)' -> [(name, party)]."""
    out = []
    for part in re.split(r"▌+", txt):
        m = re.match(r"\s*(.+?)\s*\(([^)]+)\)\s*$", part.strip())
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def side_of(party: str) -> str:
    p = party.lower()
    if p.startswith("democratic") or p == "dfl" or "democratic–farmer" in p:
        return "D"
    if p.startswith("republican"):
        return "R"
    return "O"


RUNNING = r"nominated|advanced|running for re|re-election|reelection|seeking"
NOT_RUNNING = r"lost|retir|defeat|resign|vacant"


def frame_2026():
    house_page = glob.glob(str(RAW / "wikipedia/asof_20261102/2026_United_States_House*.html"))[0]
    senate_page = glob.glob(str(RAW / "wikipedia/asof_20261102/2026_United_States_Senate*.html"))[0]
    recs = []
    for t in html_tables(use(house_page)):
        r = t["rows"]
        if len(r) < 2 or r[1][:2] != ["Location", "2026 PVI"]:
            continue
        for row in r[2:]:
            m = re.match(r"(.+?) (\d+|at-large)$", row[0])
            if not m or m.group(1) not in ABBR:
                continue
            st = ABBR[m.group(1)]
            d = 0 if m.group(2) == "at-large" else int(m.group(2))
            pvi = row[1]
            pv = float(pvi[2:]) * (1 if pvi.startswith("D") else -1) if pvi[:2] in ("D+", "R+") else (0.0 if "EVEN" in pvi.upper() else np.nan)
            cands = wiki_candidates(row[6])
            status = row[5]
            running = bool(re.search(RUNNING, status, re.I)) and not re.search(NOT_RUNNING, status, re.I)
            recs.append({"cycle": 2026, "office": "H", "state": st, "race": f"{st}-{d:02d}", "pvi": pv,
                         "member": row[2].split(" Redistricted from")[0], "member_party": side_of(row[3]),
                         "status": status, "inc_running": running, "cands": cands})
    hs = pd.DataFrame(recs)
    LOG["counts"]["house_2026_member_rows"] = len(hs)
    # Two members drawn into one district appear on two rows: one row per district, with the party of
    # the incumbent(s) still running. Incumbents of both parties running against each other: no incumbent.
    def collapse(g):
        run = g[g.inc_running]
        parties = set(run.member_party) - {"O"}
        first = g.iloc[0].copy()
        first["member"] = "; ".join(g.member)
        first["status"] = "; ".join(g.status)
        first["inc_running"] = len(parties) == 1
        first["member_party"] = parties.pop() if len(parties) == 1 else (first.member_party if not len(run) else "")
        first["n_members"] = len(g)
        return first
    hs = pd.DataFrame([collapse(g) for _, g in hs.groupby("race", sort=False)])
    LOG["counts"]["house_2026_districts"] = len(hs)
    tabs = html_tables(use(senate_page))
    srecs = []
    for t in tabs:
        r = t["rows"]
        if len(r) > 2 and r[1][:3] == ["State", "PVI", "Senator"] and ("Results" in r[1] or "Status" in r[1]):
            si = r[1].index("Results") if "Results" in r[1] else r[1].index("Status")
            ci = r[1].index("Candidates")
            for row in r[2:]:
                name = re.sub(r"\s*\((special|Class \d)\)", "", row[0])
                if name not in ABBR:
                    continue
                pvi = row[1]
                pv = float(pvi[2:]) * (1 if pvi.startswith("D") else -1) if pvi[:2] in ("D+", "R+") else 0.0
                running = bool(re.search(RUNNING, row[si], re.I)) and not re.search(NOT_RUNNING, row[si], re.I)
                srecs.append({"cycle": 2026, "office": "S", "state": ABBR[name], "race": f"{ABBR[name]}-S",
                              "pvi": pv, "member": row[2], "member_party": side_of(row[3]), "status": row[si],
                              "inc_running": running, "cands": wiki_candidates(row[ci]),
                              "special": "Class 3" in row[0] or "special" in row[0]})
        if len(r) > 2 and r[0][:2] == ["Parties", "Parties"]:
            hdr = r[1]
            for row in r:
                if row[:2] == ["Not up", "Not up"]:
                    LOG["senate_2026_not_up"] = {hdr[i]: row[i] for i in range(2, len(hdr))}
    ss = pd.DataFrame(srecs)
    LOG["counts"]["senate_2026_races"] = len(ss)
    df = pd.concat([hs, ss], ignore_index=True)
    # Nominees: one D-side (Democrat, or the leading independent where no Democrat runs) and one R.
    dn, rn, dside_ind = [], [], []
    for cands in df.cands:
        d = [n for n, p in cands if side_of(p) == "D"]
        r = [n for n, p in cands if side_of(p) == "R"]
        ind = [n for n, p in cands if p.lower() == "independent"]
        if not d and ind:
            d, flag = ind, True
        else:
            flag = False
        dn.append(d[0] if len(d) == 1 else ("|".join(d) if d else ""))
        rn.append(r[0] if len(r) == 1 else ("|".join(r) if r else ""))
        dside_ind.append(flag)
    df["d_name"], df["r_name"], df["d_is_ind"] = dn, rn, dside_ind
    df["n_cands"] = df.cands.map(len)
    return df.drop(columns="cands")


def wiki_polls_2026(fr: pd.DataFrame) -> pd.DataFrame:
    out = []
    files = sorted(glob.glob(str(RAW / "wikipedia_races_asof_20261102/*.html")))
    for f in files:
        title = Path(f).name.split("__rev")[0].replace("_", " ")
        sen = "Senate" in title
        m = re.search(r"in (.+)$", title)
        st = ABBR.get(m.group(1)) if m else None
        if st is None:
            continue
        for t in html_tables(use(f)):
            r = t["rows"]
            if len(r) < 2 or "Poll source" not in r[0][0]:
                continue
            hdr = r[0]
            # Which race: Senate page -> the state's seat; House page -> the district heading above.
            if sen:
                race = f"{st}-S"
            else:
                hd = " ".join([t["head"]["h2"], t["head"]["h3"]])
                dm = re.search(r"District (\d+)", hd)
                if dm:
                    race = f"{st}-{int(dm.group(1)):02d}"
                elif re.search(r"at-large", hd, re.I) or "election in" in title:
                    race = f"{st}-00"
                else:
                    race = f"{st}-statewide"
            row = fr[fr.race == race]
            if race.endswith("statewide") and "Democratic" in hdr and "Republican" in hdr:
                kind, di, ri = "state_generic", hdr.index("Democratic"), hdr.index("Republican")
            elif len(row):
                row = row.iloc[0]
                dl, rl = last_tokens(row.d_name), last_tokens(row.r_name)
                di = next((i for i, h in enumerate(hdr) if i >= 4 and dl and last_tokens(h) & dl), None)
                ri = next((i for i, h in enumerate(hdr) if i >= 4 and rl and last_tokens(h) & rl), None)
                if di is None or ri is None:
                    continue
                # A table that also offers a third named candidate (a primary or a three-way) is kept only
                # when it is the head-to-head: exactly two named candidate columns.
                named = [h for h in hdr[4:] if h and not re.match(r"(Other|Undecided|Margin|Others)", h)]
                if len(named) > 2:
                    continue
                kind = "senate" if sen else "house"
            else:
                continue
            for rr in r[1:]:
                if len(rr) <= max(di, ri) or rr[0] == hdr[0]:
                    continue
                dp, rp = pct(rr[di]), pct(rr[ri])
                if np.isnan(dp) or np.isnan(rp):
                    continue
                n, pop = parse_sample(rr[2])
                ps = rr[0]
                partisan = "R" if re.search(r"\(R\)", ps) else ("D" if re.search(r"\(D\)", ps) else "")
                out.append({"cycle": 2026, "kind": kind, "race": race, "pollster": re.sub(r"\s*\((R|D)\)", "", ps).strip(),
                            "partisan": partisan, "internal": False, "start": pd.NaT, "end": parse_end_date(rr[1]),
                            "created": pd.NaT, "n": n, "population": pop, "dem": dp, "rep": rp, "source": "wikipedia"})
    df = pd.DataFrame(out).drop_duplicates(["race", "pollster", "end", "dem", "rep", "n"])
    cutoff = pd.Timestamp(EVE[2026])
    LOG["counts"]["wiki_polls_2026_end_after_eve_dropped"] = int((df.end > cutoff).sum())
    return df[df.end <= cutoff]


def generic_aggregates_2026() -> pd.DataFrame:
    out = []
    for f in sorted(glob.glob(str(RAW / "wikipedia_generic_agg_weekly/asof_*.wikitext"))):
        txt = use(f).read_text(encoding="utf-8")
        snap = re.search(r"asof_(\d{4}-\d\d-\d\d)", f).group(1)
        for blk in txt.split("|-")[1:]:
            lines = [ln for ln in blk.splitlines() if ln.startswith("|")]
            if len(lines) < 6:
                continue
            src = re.sub(r"<ref.*", "", lines[0]).strip("|[] ")
            src = src.split("|")[-1].strip("[] ") if "[[" in lines[0] else src
            vals = [re.sub(r"\{\{[^}]*\}\}\s*\|", "", ln).strip("|' ") for ln in lines]
            rep, dem = pct(vals[3]), pct(vals[4])
            if np.isnan(rep) or np.isnan(dem):
                continue
            out.append({"cycle": 2026, "kind": "generic_avg", "race": "US", "pollster": src, "partisan": "",
                        "internal": False, "start": pd.NaT, "end": parse_end_date(vals[2]),
                        "created": pd.Timestamp(snap), "n": np.nan, "population": "", "dem": dem, "rep": rep,
                        "source": "wikipedia-aggregate", "admin": vals[1]})
    df = pd.DataFrame(out)
    df = df[df.end <= pd.Timestamp(EVE[2026])]
    return df.drop_duplicates(["pollster", "end", "dem", "rep"])


# ---------------------------------------------------------------- historical polling error (538 raw_polls)
def polling_errors() -> pd.DataFrame:
    """Per cycle and race type: mean signed error of the final-three-weeks polls (D margin, points)."""
    p = pd.read_csv(use(RAW / "fte_github/pollster-ratings/raw_polls.csv"), low_memory=False)
    keep = p.type_simple.isin(["House-G", "Sen-G", "House-G-US"])
    p = p[keep].copy()
    p["poll_margin"] = p.cand1_pct - p.cand2_pct
    p["true_margin"] = p.cand1_actual - p.cand2_actual
    d1 = p.cand1_party.eq("DEM")
    sign = np.where(d1, 1.0, np.where(p.cand2_party.eq("DEM"), -1.0, np.nan))
    p["err_d"] = (p.poll_margin - p.true_margin) * sign
    p = p[np.isfinite(p.err_d) & (p.time_to_election <= 21)]
    p["generic"] = p.location.eq("US")
    race = p.groupby(["cycle", "type_simple", "generic", "race_id", "location"]).agg(err=("err_d", "mean"), n=("err_d", "size")).reset_index()
    return race


# ---------------------------------------------------------------- rematches (Levitt 1994)
COURT_REDRAWS = {(2006, "TX"), (2006, "GA"), (2016, "FL"), (2016, "VA"), (2016, "NC"), (2018, "PA"), (2020, "NC")}


def rematches(res_by_year: dict, weball: dict) -> pd.DataFrame:
    """Consecutive-cycle House rematches between the same two FEC candidate IDs in the same district."""
    rows = []
    years = sorted(res_by_year)
    for y in years:
        if y - 2 not in res_by_year or y in (2012, 2022):
            continue
        a, b = res_by_year[y - 2], res_by_year[y]
        m = b.merge(a, on=["race", "d_id", "r_id"], suffixes=("", "_prev"))
        m = m[m.contested & m.contested_prev & (m.d_id != "") & (m.r_id != "")]
        m = m[[(y, s) not in COURT_REDRAWS for s in m.state]]
        for c, yy in (("", y), ("_prev", y - 2)):
            w = weball[yy].set_index("CAND_ID")
            m[f"d_disb{c}"] = m.d_id.map(w.TTL_DISB)
            m[f"r_disb{c}"] = m.r_id.map(w.TTL_DISB)
        rows.append(m[["cycle", "state", "race", "d_id", "r_id", "d_name", "r_name", "dem2", "dem2_prev", "d_inc",
                       "r_inc", "d_inc_prev", "r_inc_prev", "d_disb", "r_disb", "d_disb_prev", "r_disb_prev"]])
    return pd.concat(rows, ignore_index=True)


def medsl_house_year(year: int, wb: pd.DataFrame) -> pd.DataFrame:
    """2004-2014 House races from MEDSL votes, candidates matched to FEC IDs by name within district+party."""
    m = MEDSL_HOUSE[(MEDSL_HOUSE.year == year) & (MEDSL_HOUSE.stage == "gen") & (~MEDSL_HOUSE.special.astype(bool))
                    & (~MEDSL_HOUSE.writein.astype(bool))]
    m = m[m.party.isin(["democrat", "republican", "democratic-farmer-labor", "democratic-farm-labor"])]
    m = m.groupby(["state_po", "district", "party", "candidate"], as_index=False).candidatevotes.sum()
    wbh = wb[wb.CAND_ID.str.startswith("H")].copy()
    wbh["dist"] = pd.to_numeric(wbh.CAND_OFFICE_DISTRICT, errors="coerce")
    out = []
    for (st, d), g in m.groupby(["state_po", "district"]):
        rec = {"cycle": year, "state": st, "race": f"{st}-{int(d):02d}", "office": "H"}
        for side, parties, fpty in (("d", ("democrat", "democratic-farmer-labor", "democratic-farm-labor"), "DEM"),
                                    ("r", ("republican",), "REP")):
            gg = g[g.party.isin(parties)].sort_values("candidatevotes", ascending=False)
            rec[f"{side}_votes"] = gg.candidatevotes.iloc[0] if len(gg) else 0.0
            rec[f"{side}_name"] = gg.candidate.iloc[0] if len(gg) else ""
            rec[f"{side}_id"], rec[f"{side}_inc"] = "", False
            if len(gg):
                pool = wbh[(wbh.CAND_OFFICE_ST == st) & (wbh.dist == max(int(d), 0) if int(d) > 0 else wbh.dist.isin([0, 1]))
                           & wbh.CAND_PTY_AFFILIATION.str.startswith(fpty[:3])]
                toks = last_tokens(gg.candidate.iloc[0])
                hit = pool[pool.CAND_NAME.map(lambda n: bool(last_tokens(n.split(",")[0]) & toks))]
                if len(hit) >= 1:
                    hit = hit.sort_values("TTL_DISB", ascending=False)
                    rec[f"{side}_id"] = hit.CAND_ID.iloc[0]
                    rec[f"{side}_inc"] = hit.CAND_ICI.iloc[0] == "I"
        out.append(rec)
    df = pd.DataFrame(out)
    tot = df.d_votes + df.r_votes
    df["dem2"] = np.where((df.d_votes > 0) & (df.r_votes > 0), df.d_votes / tot.where(tot > 0, 1), np.nan)
    df["contested"] = df.dem2.notna()
    return df


MEDSL_HOUSE = None


def main():
    global MEDSL_HOUSE
    OUT.mkdir(exist_ok=True)
    l18, s18, l22 = lean_tables()

    weball = {y: read_weball(f"{y % 100:02d}") for y in range(2004, 2027, 2)}
    for y in (2016, 2018, 2020, 2022):
        FEC_PARTY.update(dict(zip(weball[y].CAND_ID, weball[y].CAND_PTY_AFFILIATION.str[:1])))

    # ---------- results 2016-2022 (FEC workbooks, with FEC IDs)
    cand_rows, races = [], []
    for y in (2016, 2018, 2020, 2022):
        for off in ("H", "S"):
            c = fec_results(y, off)
            cand_rows.append(c)
            races.append(race_frame(c))
    cands = pd.concat(cand_rows, ignore_index=True)
    races = pd.concat(races, ignore_index=True)
    # Lean on the matching lines: 538's 2018 lean for 2016-2020, the 2022 lean for 2022. PA (2018 map),
    # NC (2020 map) and the 2016 court maps differ from the 2018 lines, so lean is blanked where the
    # district's lines changed between the result and the lean.
    def lean_for(row):
        if row.office == "S":
            if row.cycle == 2022:
                return S22.get(row.state, np.nan)
            return s18.get(row.state, np.nan)
        key = dist538(row.race)
        if row.cycle == 2022:
            return l22.get(key, np.nan)
        if (row.cycle, row.state) in {(2016, "PA"), (2020, "NC"), (2016, "FL"), (2016, "VA"), (2016, "NC")}:
            return np.nan
        return l18.get(key, np.nan)

    # 2022 state lean: mean of the state's 2022 district leans (538 published no 2022 state file).
    l22s = pd.Series(l22)
    S22 = l22s.groupby(l22s.index.str[:2]).mean().to_dict()
    l18s = pd.Series(l18)
    prox18 = l18s.groupby(l18s.index.str[:2]).mean()
    both = pd.DataFrame({"proxy": prox18, "state": pd.Series(s18)}).dropna()
    LOG["checks"]["state_lean_proxy_vs_538_2018"] = {
        "corr": float(both.corr().iloc[0, 1]), "mean_abs_diff": float((both.proxy - both.state).abs().mean()),
        "max_abs_diff": float((both.proxy - both.state).abs().max())}
    races["lean"] = races.apply(lean_for, axis=1)
    races["region"] = races.state.map(REGION)
    races["inc_party"] = np.where(races.d_inc, "D", np.where(races.r_inc, "R", ""))
    races["open"] = races.inc_party == ""
    # California's 2022 special for the last weeks of Padilla's term is not a seat in the next Congress.
    races["seat"] = ~((races.cycle == 2022) & (races.race == "CA-S2"))

    # ---------- money known by each eve (2018, 2022), and final cycle totals (weball)
    rep = {y: read_reports(y) for y in (2018, 2022, 2026)}
    for side in ("d", "r"):
        for col in ("rec_eve", "disb_eve", "coh_eve", "disb_final", "rec_final"):
            races[f"{side}_{col}"] = np.nan
    for y in (2016, 2018, 2020, 2022):
        w = weball[y].set_index("CAND_ID")
        sel = races.cycle == y
        for side in ("d", "r"):
            ids = races.loc[sel, f"{side}_id"]
            races.loc[sel, f"{side}_disb_final"] = ids.map(w.TTL_DISB).to_numpy()
            races.loc[sel, f"{side}_rec_final"] = ids.map(w.TTL_RECEIPTS).to_numpy()
        if y in EVE:
            m = money_at(rep[y], EVE[y])
            for side in ("d", "r"):
                ids = races.loc[sel, f"{side}_id"]
                races.loc[sel, f"{side}_rec_eve"] = ids.map(m.receipts).fillna(0.0).to_numpy()
                races.loc[sel, f"{side}_disb_eve"] = ids.map(m.disb).fillna(0.0).to_numpy()
                races.loc[sel, f"{side}_coh_eve"] = ids.map(m.coh).fillna(0.0).to_numpy()

    # ---------- 2026 frame (Wikipedia, frozen) + money from FEC filings received by 2026-11-02
    fr = frame_2026()
    w26 = weball[2026]
    r26 = rep[2026]
    g12 = set(r26.loc[r26.RPT_TP == "12G", "CAND_ID"])
    m26 = money_at(r26, EVE[2026])
    w26 = w26[w26.CAND_ID.isin(g12)].copy()
    w26["dist"] = pd.to_numeric(w26.CAND_OFFICE_DISTRICT, errors="coerce").fillna(0).astype(int)
    w26["race"] = np.where(w26.CAND_ID.str.startswith("S"), w26.CAND_OFFICE_ST + "-S",
                           w26.CAND_OFFICE_ST + "-" + w26.dist.map("{:02d}".format))
    w26["side"] = np.where(w26.CAND_PTY_AFFILIATION.str.startswith("DEM") | w26.CAND_PTY_AFFILIATION.eq("DFL"), "D",
                           np.where(w26.CAND_PTY_AFFILIATION.str.startswith("REP"), "R", "O"))
    w26["receipts_eve"] = w26.CAND_ID.map(m26.receipts).fillna(0.0)
    # Candidate pool per race: everyone in the FEC candidate master (cn26) registered for that state, office
    # and district, whether or not a pre-general report has arrived. (Matching only against candidates with
    # a 12G found 86% of R nominees; a nominee who filed no 12G, or whose FEC party is UNK, was lost.)
    cn = read_cn(26)
    cn["dist"] = pd.to_numeric(cn.CAND_OFFICE_DISTRICT, errors="coerce").fillna(0).astype(int)
    cn = cn[cn.CAND_OFFICE.isin(["H", "S"])].copy()
    cn["race"] = np.where(cn.CAND_OFFICE == "S", cn.CAND_OFFICE_ST + "-S",
                          cn.CAND_OFFICE_ST + "-" + cn.dist.map("{:02d}".format))
    pty = cn.CAND_PTY_AFFILIATION.fillna("")
    cn["side"] = np.where(pty.str.startswith("DEM") | pty.eq("DFL"), "D", np.where(pty.str.startswith("REP"), "R", "O"))
    cn["receipts_eve"] = cn.CAND_ID.map(m26.receipts).fillna(0.0)
    cn["has12"] = cn.CAND_ID.isin(g12)
    cn["yr26"] = cn.CAND_ELECTION_YR.eq("2026")
    ids, ok, how = {"d": [], "r": []}, {"d": [], "r": []}, {"d": [], "r": []}
    for _, row in fr.iterrows():
        pool = cn[cn.race == row.race]
        for side, nm in (("d", row.d_name), ("r", row.r_name)):
            want = "D" if side == "d" and not row.d_is_ind else ("O" if side == "d" else "R")
            other = "R" if want == "D" else "D"
            p = pool[pool.side != other]       # the nominee's party, or an FEC party of UNK/other
            names = nm.split("|") if nm else []
            sur = set().union(*[last_tokens(x, 2) for x in names]) if names else set()
            hit = []
            for n in p.CAND_NAME:
                fec_sur, _, fec_first = n.partition(",")
                full = last_tokens(n, 2)
                # surname tokens shared, or every name token of a Wikipedia name in the full FEC name
                # (FEC files Ashley Hinson as "ARENHOLZ, ASHLEY HINSON")
                hit.append(bool(last_tokens(fec_sur, 2) & sur) or any(
                    len(last_tokens(x, 2)) >= 2 and last_tokens(x, 2) <= full for x in names))
            named = p[np.array(hit, dtype=bool)] if len(p) else p
            # Prefer the nominee's own party, a 12G on file, money reported, then a 2026 registration.
            pick = named.assign(same=named.side.eq(want)).sort_values(
                ["same", "has12", "receipts_eve", "yr26"], ascending=False)
            ids[side].append(pick.CAND_ID.iloc[0] if len(pick) else "")
            ok[side].append(len(named) > 0)
            how[side].append("" if not len(pick) else ("12G" if pick.has12.iloc[0] else "cn26"))
    fr["d_id"], fr["r_id"] = ids["d"], ids["r"]
    fr["d_id_name_match"], fr["r_id_name_match"] = ok["d"], ok["r"]
    con = (fr.d_name != "") & (fr.r_name != "")
    LOG["checks"]["fec_id_2026_name_match_rate"] = {
        s: float(np.mean(np.array(ok[s])[con.to_numpy()])) for s in ("d", "r")}
    LOG["checks"]["fec_id_2026_unmatched"] = {
        s: [f"{a}: {b}" for a, b in fr.loc[con & ~fr[f"{s}_id_name_match"], ["race", f"{s}_name"]].itertuples(index=False)]
        for s in ("d", "r")}
    LOG["checks"]["fec_id_2026_matched_without_12G"] = {
        s: int(sum(h == "cn26" for h, c in zip(how[s], con) if c)) for s in ("d", "r")}
    for side in ("d", "r"):
        # A nominee with no FEC ID is missing, not $0: the model treats the money signal as unknown.
        # A matched candidate with no report received by the eve reported nothing: $0.
        has = fr[f"{side}_id"] != ""
        for col, src in (("rec_eve", m26.receipts), ("disb_eve", m26.disb), ("coh_eve", m26.coh)):
            fr[f"{side}_{col}"] = np.where(has, fr[f"{side}_id"].map(src).fillna(0.0), np.nan)
    fr["lean"] = 2.0 * fr.pvi  # Cook PVI is on the two-party share scale; 538 lean on the margin scale.
    fr["region"] = fr.state.map(REGION)
    fr["inc_party"] = np.where(fr.inc_running, fr.member_party, "")
    fr["inc_party"] = np.where(fr.inc_party == "O", "", fr.inc_party)
    fr["open"] = fr.inc_party == ""
    fr["d_inc"] = fr.inc_party == "D"
    fr["r_inc"] = fr.inc_party == "R"
    fr["contested"] = (fr.d_name != "") & (fr.r_name != "")
    fr["special"] = fr.special.fillna(False).astype(bool)
    fr["seat"] = True
    fr["split"] = False  # 2026: no results yet. Louisiana's November House ballot is a jungle: see multi_cands.
    fr["multi_cands"] = fr.d_name.str.contains("|", regex=False) | fr.r_name.str.contains("|", regex=False)

    # Validate the 12G nominee proxy on 2022: share of general-election major-party candidates with a 12G.
    g12_22 = set(rep[2022].loc[rep[2022].RPT_TP == "12G", "CAND_ID"])
    r22 = races[(races.cycle == 2022) & races.contested]
    LOG["checks"]["nominees_2022_with_12G"] = float(np.mean([i in g12_22 for i in pd.concat([r22.d_id, r22.r_id])]))

    allr = pd.concat([races, fr], ignore_index=True)
    allr["seat_key"] = allr.race
    cols = ["cycle", "office", "state", "race", "region", "special", "seat", "lean", "pvi", "inc_party", "open", "d_inc", "r_inc",
            "d_id", "r_id", "d_name", "r_name", "d_is_ind", "contested", "split", "dem2", "winner", "rcv", "d_votes", "r_votes",
            "d_rec_eve", "r_rec_eve", "d_disb_eve", "r_disb_eve", "d_coh_eve", "r_coh_eve", "d_rec_final", "r_rec_final",
            "d_disb_final", "r_disb_final", "member", "status", "multi_cands", "d_id_name_match", "r_id_name_match"]
    allr = allr[cols]
    allr.to_csv(OUT / "e94_races.csv", index=False, float_format="%.6g")
    LOG["counts"]["races_rows"] = pairs(allr.groupby(["cycle", "office"]).size())

    # National House two-party vote (sum of the district votes, contested and not).
    nat = allr[(allr.office == "H") & allr.cycle.isin([2016, 2018, 2020, 2022])].groupby("cycle")[["d_votes", "r_votes"]].sum()
    LOG["national_house_dem2"] = (nat.d_votes / (nat.d_votes + nat.r_votes)).round(4).to_dict()

    # ---------- polls
    polls = pd.concat([fte_polls(races), wiki_polls_2026(fr), generic_aggregates_2026()], ignore_index=True)
    # 538's generic-ballot average (an aggregate for the backtest, like 2026's): the eve value per cycle.
    top = pd.read_csv(use(RAW / "fte_github/congress-generic-ballot/generic_topline_historical.csv"))
    # 538's generic-ballot topline history ends in 2016 (timestamps of September 2020), so it cannot stand in
    # for 2018/2022 aggregates; the notebook builds its own rolling average for the aggregate-only backtest.
    LOG["fte_generic_topline_last_date"] = str(top.modeldate.iloc[-1])
    polls.to_csv(OUT / "e94_polls.csv", index=False, float_format="%.4g", date_format="%Y-%m-%d")
    LOG["counts"]["polls"] = pairs(polls.groupby(["cycle", "kind"]).size())

    # ---------- reports for the nowcast (general-election D and R candidates only)
    keep = []
    for y in (2018, 2022, 2026):
        rr = allr[allr.cycle == y]
        idmap = pd.concat([rr[["d_id", "race", "office"]].rename(columns={"d_id": "cand_id"}).assign(side="D"),
                           rr[["r_id", "race", "office"]].rename(columns={"r_id": "cand_id"}).assign(side="R")])
        idmap = idmap[idmap.cand_id != ""].drop_duplicates("cand_id").set_index("cand_id")
        x = rep[y][rep[y].CAND_ID.isin(idmap.index)]
        x = x.groupby(["CAND_ID", "RPT_TP", "cvg_start", "cvg_end"], as_index=False).agg(
            receipt=("receipt", "max"), disb=("TTL_DISB", "sum"), receipts=("TTL_RECEIPTS", "sum"),
            coh=("COH_COP", "sum"))
        x = x.join(idmap, on="CAND_ID")
        x.insert(0, "cycle", y)
        keep.append(x)
    reports = pd.concat(keep, ignore_index=True).rename(columns={"CAND_ID": "cand_id", "RPT_TP": "rpt"})
    reports.to_csv(OUT / "e94_reports.csv.gz", index=False, compression=GZ, float_format="%.2f", date_format="%Y-%m-%d")
    LOG["counts"]["reports_rows"] = reports.groupby("cycle").size().astype(int).to_dict()

    # ---------- IE, aggregated by race, side, date and first receipt date
    ies = []
    for y in (2018, 2022, 2026):
        cm = pd.concat([allr[allr.cycle == y][["d_id", "race"]].rename(columns={"d_id": "cand_id"}).assign(side="D"),
                        allr[allr.cycle == y][["r_id", "race"]].rename(columns={"r_id": "cand_id"}).assign(side="R")])
        cm = cm[cm.cand_id != ""].drop_duplicates("cand_id")
        it = read_ie(y, cm)
        st = it.can_office_state.fillna("")
        dist = pd.to_numeric(it.can_office_dis, errors="coerce").fillna(0).astype(int)
        fallback = np.where(it.can_office == "S", st + "-S", st + "-" + dist.map("{:02d}".format))
        it["race"] = it.race.fillna(pd.Series(fallback, index=it.index))
        it["nominee"] = it.side.notna()
        agg = it.groupby(["version", "race", "pro", "nominee", "date", "first_receipt"], as_index=False).amt.sum()
        agg.insert(0, "cycle", y)
        ies.append(agg)
    ie = pd.concat(ies, ignore_index=True)
    ie.to_csv(OUT / "e94_ie.csv.gz", index=False, compression=GZ, float_format="%.2f", date_format="%Y-%m-%d")

    # ---------- rematches: 2006-2010 and 2014 (MEDSL + weball names), 2016-2022 (FEC workbooks)
    MEDSL_HOUSE = pd.read_csv(use(RAW / "medsl_github/1976-2018-house.csv"), encoding="latin-1", low_memory=False)
    # National House vote 1976-2018 (MEDSL, general elections incl. specials on the day) for the midterm prior.
    g = MEDSL_HOUSE[(MEDSL_HOUSE.stage == "gen") & (~MEDSL_HOUSE.writein.astype(bool))]
    dv = g[g.party.isin(["democrat", "democratic-farmer-labor", "democratic-farm-labor"])].groupby("year").candidatevotes.sum()
    rv = g[g.party.eq("republican")].groupby("year").candidatevotes.sum()
    LOG["national_house_dem2_medsl"] = (dv / (dv + rv)).round(4).to_dict()
    # 2024 national House vote (the 2026 midterm prior's starting point): Wikipedia's Federal results table.
    f24 = glob.glob(str(RAW / "wikipedia_2024_house_results/2024_United_States_House*.html"))[0]
    for t in html_tables(use(f24)):
        if t["head"]["h3"] == "Federal" and len(t["rows"]) > 3:
            v = {row[1]: fnum(row[2]) for row in t["rows"][2:] if len(row) > 2}
            d24, r24 = v.get("Democratic Party"), v.get("Republican Party")
            LOG["national_house_2024_wikipedia"] = {"dem": d24, "rep": r24, "dem2": round(d24 / (d24 + r24), 4)}
            break
    # Scale check for 2026's lean: Cook's 2022 PVI (on the 2024 page) against 538's 2022 lean, same lines.
    pv = {}
    for t in html_tables(f24):
        r = t["rows"]
        if len(r) > 2 and len(r[1]) > 1 and r[1][:2] == ["Location", "2022 PVI"] and t["head"]["h2"] != "Crossover seats":
            for row in r[2:]:
                m = re.match(r"(.+?) (\d+|at-large)$", row[0])
                if m and m.group(1) in ABBR and row[1][:2] in ("D+", "R+", "EV"):
                    v = 0.0 if row[1].upper().startswith("EVEN") else float(row[1][2:]) * (1 if row[1][0] == "D" else -1)
                    pv[f"{ABBR[m.group(1)]}-{1 if m.group(2) == 'at-large' else int(m.group(2))}"] = v
    redrawn24 = {"AL", "GA", "LA", "NC", "NY"}
    both = pd.DataFrame({"pvi": pd.Series(pv), "lean": pd.Series(l22)}).dropna()
    both = both[~both.index.str[:2].isin(redrawn24)]
    if len(both) > 50:
        slope, icpt = np.polyfit(2 * both.pvi, both.lean, 1)
        LOG["checks"]["lean538_2022_vs_2xcook_pvi"] = {
            "n": int(len(both)), "slope": float(slope), "intercept": float(icpt),
            "corr": float(np.corrcoef(both.pvi, both.lean)[0, 1]),
            "resid_sd": float(np.std(both.lean - (icpt + slope * 2 * both.pvi)))}
    res_by_year = {}
    for y in (2004, 2006, 2008, 2010, 2012, 2014):
        res_by_year[y] = medsl_house_year(y, weball[y])
    for y in (2016, 2018, 2020, 2022):
        res_by_year[y] = races[(races.cycle == y) & (races.office == "H")].copy()
    rm = rematches(res_by_year, weball)
    rm.to_csv(OUT / "e94_rematch.csv", index=False, float_format="%.6g")
    LOG["counts"]["rematches"] = rm.groupby("cycle").size().astype(int).to_dict()
    LOG["checks"]["medsl_fec_id_match"] = {y: {"d": float((res_by_year[y].d_id != "").mean()),
                                               "r": float((res_by_year[y].r_id != "").mean())} for y in (2004, 2006, 2008, 2010, 2012, 2014)}

    # ---------- historical polling error (538 pollster ratings raw polls)
    pe = polling_errors()
    pe.to_csv(OUT / "e94_pollerr.csv", index=False, float_format="%.4g")

    LOG["raw_files_sha256"] = USED
    (OUT / "e94_build_log.json").write_text(json.dumps(LOG, indent=1, default=str))
    for f in ("e94_races.csv", "e94_polls.csv", "e94_reports.csv.gz", "e94_ie.csv.gz", "e94_rematch.csv",
              "e94_pollerr.csv", "e94_build_log.json"):
        print(f"{f:<22} {(OUT / f).stat().st_size / 1e3:8.0f} kB")


if __name__ == "__main__":
    main()
