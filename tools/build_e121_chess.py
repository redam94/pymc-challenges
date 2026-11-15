"""Build the E121 datasets: rated blitz games on Lichess, July to December 2014, with engine evaluations.

    python tools/build_e121_chess.py     # download (cached), stream, sample, check, write data/e121_chess_*.csv.gz

Source: the Lichess open database, https://database.lichess.org/ ("Database exports are released under
the Creative Commons CC0 license. Use them for research, commercial purpose, publication, anything you
like. You can download, modify and redistribute them, without asking for permission.", read
2026-11-14). Six monthly files of rated standard games, lichess_db_standard_rated_2014-07 ... 2014-12
(.pgn.zst, 1.27 GB together, 6.7 million games), each checked against the sha256 Lichess publishes.
About one game in six carries Stockfish evaluations as ``[%eval ...]`` comments, which is what makes
move quality measurable here without running an engine.

A second source gives each sampled account one bit: whether the public Lichess API reports it today as
marked for a violation of the terms of service (``tosViolation`` in POST https://lichess.org/api/users,
up to 300 ids per request; this builder sends them 25 seconds apart, with the
User-Agent "pymc-challenges"). The Lichess source code sets that field for accounts marked for engine
assistance or for boosting. Accounts that are closed today show no mark either way and are left out.
The answers are cached with the raw files and are as of the day they were fetched (2026-11-14).

Raw files and intermediate tables stay in ~/.cache/pymc-challenges/e121 (never committed): the account
names exist only there.

What the builder does:
  1. streams the six files through the ``zstd`` command and keeps the rated blitz games (ratings on
     Lichess are separate for each speed, so one speed gives one coherent rating), leaving out games
     with a titled player: 2,842,294 games, 437,076 of them with evaluations;
  2. calls an analysed game usable if it has at least 30 plies and an evaluation for every ply kept
     (the first 100; a final mating move has none), and an account eligible if it played at least 60
     rated blitz games in the six months, at least 16 of them usable analysed games, and is not closed
     (8,886 accounts before the last condition, of which 1,246 are closed, 541 marked, 7,099 neither);
  3. keeps EVERY eligible marked account and a random 2,000 of the others (a case-control sample:
     the column ``weight`` says how many eligible accounts each row stands for), and for each kept
     account at most 24 usable analysed games, drawn at random (an account can appear in more games
     than that, as the opponent in another sampled account's games);
  4. replaces account names by codes (A0001, ...) and games by codes (G00001, ...), both in random
     order, and drops everything else that identifies a game: its address, the clock time, the opening,
     the moves themselves. A day becomes one of 13 two-week periods.

Three tables are written.

data/e121_chess_accounts.csv.gz, one row per sampled account:
  account      code
  marked       1 if the Lichess API reports the account as marked for a terms-of-service violation
  weight       eligible accounts of the same kind (marked or not) per sampled account
  n_blitz      rated blitz games in the six months
  n_analysed   usable analysed games among them (at most 24 are in the games table)
  elo_first    published blitz rating before the account's first game of the window
  elo_last     published blitz rating after its last game of the window

data/e121_chess_results.csv.gz, the results of ALL rated blitz games of the sampled accounts, counted:
  account      code
  period       0..12, two-week periods from 1 July 2014 (the last one runs to 31 December)
  opp_elo      the opponent's published rating before the game, rounded to 50 points
  own_elo      the account's own published rating before the game, averaged over the cell
  n            games in the cell
  wins, draws  how many of them the account won and drew

data/e121_chess_games.csv.gz, one row per sampled analysed game:
  game         code
  period       0..12
  white, black account codes (empty when that player is not a sampled account)
  white_elo, black_elo   published blitz ratings before the game
  result       1-0, 0-1 or 1/2-1/2
  plies        half-moves played (evaluations are kept for the first 100)
  evals        the engine evaluation after each half-move, in centipawns from White's side, separated by
               spaces; M and -M for a forced mate for White and for Black; empty for a final mating move
  flags        one character per half-move: 0 quiet, 1 capture, 2 check, 3 capture with check
"""

from __future__ import annotations

import array
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path.home() / ".cache" / "pymc-challenges" / "e121"
RAW = CACHE / "raw"
BASE = "https://database.lichess.org/standard/"
UA = {"User-Agent": "pymc-challenges"}
SHA256 = {  # from https://database.lichess.org/standard/sha256sums.txt, read 2026-11-14
    "2014-07": "d73fa278979e3dd36d9af98c7b09c5eb03f8644ec5beccddc191af7b4959cad8",
    "2014-08": "95d5cd0f00577f40fb3bbf31de5ec9ff25108b9d9f0f16d74b7ddd627206d8d2",
    "2014-09": "238085acf009cff4d3c31a2e3bfd4f5a544edd8b1b5dd247a974943c0168176b",
    "2014-10": "093366fd90fe6c2c75a76e80d7a52c9fcf98101a9be1ae2241cf1a7ee87dcb3b",
    "2014-11": "7c6266941459c130547bbbb8df227ea8a40a21602b55ead854da44d013d45a22",
    "2014-12": "4589a1af622a893d196bc8eaede657652ce65dc79d2f289ff65fadd6a7076af4",
}
DAY0 = date(2014, 7, 1).toordinal()
N_PERIODS, PERIOD_DAYS = 13, 14
MIN_BLITZ, MIN_ANALYSED, MAX_GAMES, MIN_PLIES, MAX_PLIES = 60, 16, 24, 30, 100
N_UNMARKED = 2000                 # sampled accounts that carry no mark
SEED = 121
RESULT = {b"1-0": 2, b"1/2-1/2": 1, b"0-1": 0}
MOVE = re.compile(rb"\d+\.+ (\S+)(?: \{ \[%eval (\S+)\] \})?")


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for month, sha in SHA256.items():
        f = RAW / f"lichess_db_standard_rated_{month}.pgn.zst"
        if not f.exists():
            req = urllib.request.Request(BASE + f.name, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as resp, open(f.with_suffix(".part"), "wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            f.with_suffix(".part").rename(f)
            time.sleep(2)
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        assert h.hexdigest() == sha, f"{f.name}: sha256 differs from the one Lichess publishes"


def scan() -> None:
    """Stream the monthly files once; cache the rated blitz games (headers) and their evaluations."""
    if (CACHE / "blitz_games.npz").exists() and (CACHE / "blitz_moves.tsv").exists():
        return
    if shutil.which("zstd") is None:
        raise SystemExit("the zstd command is needed to read the .pgn.zst files")
    names: dict[bytes, int] = {}
    cols = {k: array.array("i") for k in ("w", "b", "welo", "belo", "wdiff", "bdiff", "res", "day", "sec", "ev", "plies")}
    n = n_all = 0
    with open(CACHE / "blitz_moves.tsv.part", "wb") as moves_out:
        for month in SHA256:
            proc = subprocess.Popen(["zstd", "-dc", str(RAW / f"lichess_db_standard_rated_{month}.pgn.zst")],
                                    stdout=subprocess.PIPE, bufsize=1 << 22)
            h: dict[bytes, bytes] = {}
            for line in proc.stdout:
                if line[:1] == b"[":
                    sp = line.find(b" ")
                    h[line[1:sp]] = line[sp + 2:line.rfind(b'"')]
                    continue
                if len(line) < 2:
                    continue
                n_all += 1
                event = h.get(b"Event", b"").split()
                try:
                    if len(event) < 2 or event[1] != b"Blitz" or b"WhiteTitle" in h or b"BlackTitle" in h:
                        raise KeyError
                    res = RESULT[h[b"Result"]]
                    y, m_, d_ = h[b"UTCDate"].split(b".")
                    hh, mm, ss = h[b"UTCTime"].split(b":")
                    row = (int(h[b"WhiteElo"]), int(h[b"BlackElo"]), int(h[b"WhiteRatingDiff"]), int(h[b"BlackRatingDiff"]))
                except (KeyError, ValueError):       # another speed, a titled account, or a header without ratings
                    h = {}
                    continue
                toks = MOVE.findall(line)
                has_ev = b"%eval" in line
                for key in (b"White", b"Black"):
                    if h[key] not in names:
                        names[h[key]] = len(names)
                for k, v in zip(cols, (names[h[b"White"]], names[h[b"Black"]], *row, res,
                                       max(date(int(y), int(m_), int(d_)).toordinal() - DAY0, 0),
                                       int(hh) * 3600 + int(mm) * 60 + int(ss), int(has_ev), len(toks))):
                    cols[k].append(v)
                if has_ev:
                    evs = ["" if not e else ("M" if e[1:2] != b"-" else "-M") if e[:1] == b"#"
                           else str(int(round(float(e) * 100))) for _, e in toks]
                    flags = "".join(str(int(b"x" in san) + 2 * int(b"+" in san or b"#" in san)) for san, _ in toks)
                    moves_out.write(f"{n}\t{' '.join(evs)}\t{flags}\n".encode())
                n += 1
                h = {}
            proc.wait()
            assert proc.returncode == 0, month
            print(f"  {month}: {n_all:,} games read, {n:,} rated blitz games kept", flush=True)
    inv = [b""] * len(names)
    for k, v in names.items():
        inv[v] = k
    (CACHE / "blitz_names.txt").write_bytes(b"\n".join(inv))
    np.savez(CACHE / "blitz_games.npz", n_all=n_all, **{k: np.frombuffer(v, dtype=np.int32) for k, v in cols.items()})
    (CACHE / "blitz_moves.tsv.part").rename(CACHE / "blitz_moves.tsv")


def usable(evs: list[str]) -> bool:
    """At least MIN_PLIES plies, and an evaluation for each of the first MAX_PLIES (a last mating move has none)."""
    n = len(evs)
    if n < MIN_PLIES:
        return False
    kept = evs[:MAX_PLIES]
    return "" not in kept[:-1] and (kept[-1] != "" or n <= MAX_PLIES)


def account_status(wanted: list[str]) -> dict[str, list[int]]:
    """[marked, closed] for each account name, from the public Lichess API; cached."""
    store = CACHE / "account_status.json"
    status = json.loads(store.read_text()) if store.exists() else {}
    todo = [a for a in wanted if a.lower() not in status]
    for k in range(0, len(todo), 300):
        batch = todo[k:k + 300]
        req = urllib.request.Request("https://lichess.org/api/users?profile=false", data=",".join(batch).encode(),
                                     headers={**UA, "Content-Type": "text/plain"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            got = {u["id"]: [int(bool(u.get("tosViolation"))), int(bool(u.get("disabled")))] for u in json.load(resp)}
        for a in batch:
            status[a.lower()] = got.get(a.lower(), [0, 1])       # an account the API no longer knows counts as closed
        store.write_text(json.dumps(status))
        time.sleep(25)                       # the endpoint allows 8,000 accounts per ten minutes
    return status


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    z = np.load(CACHE / "blitz_games.npz")
    g = pd.DataFrame({k: z[k] for k in z.files if k != "n_all"})
    names = (CACHE / "blitz_names.txt").read_text().split("\n")
    assert len(names) == max(g["w"].max(), g["b"].max()) + 1

    moves: dict[int, tuple[str, str]] = {}
    with open(CACHE / "blitz_moves.tsv") as f:
        for line in f:
            gi, evs, flags = line.rstrip("\n").split("\t")
            if usable(evs.split(" ")):
                moves[int(gi)] = (evs, flags)
    g["usable"] = g.index.isin(moves.keys())

    long = pd.concat([g[["w", "usable"]].rename(columns={"w": "a"}), g[["b", "usable"]].rename(columns={"b": "a"})])
    per = long.groupby("a").agg(n_blitz=("usable", "size"), n_analysed=("usable", "sum"))
    elig = per[(per["n_blitz"] >= MIN_BLITZ) & (per["n_analysed"] >= MIN_ANALYSED)].copy()
    status = account_status([names[a] for a in elig.index])
    elig["marked"] = [status[names[a].lower()][0] for a in elig.index]
    elig["closed"] = [status[names[a].lower()][1] for a in elig.index]
    counts = {"games_read": int(z["n_all"]), "blitz": len(g), "blitz_analysed": int(g["ev"].sum()),
              "blitz_usable": int(g["usable"].sum()), "accounts": len(per), "eligible": len(elig),
              "eligible_closed": int(elig["closed"].sum()), "eligible_marked": int(elig["marked"].sum())}
    elig = elig[elig["closed"] == 0]
    rng = np.random.default_rng(SEED)
    unmarked = np.sort(elig.index[elig["marked"] == 0].to_numpy())
    counts["eligible_unmarked"] = len(unmarked)
    keep = np.concatenate([np.sort(elig.index[elig["marked"] == 1].to_numpy()),
                           rng.choice(unmarked, size=N_UNMARKED, replace=False)])
    code = {a: f"A{i + 1:04d}" for i, a in enumerate(rng.permutation(keep))}
    acc = elig.loc[keep, ["marked", "n_blitz", "n_analysed"]].copy()
    acc["weight"] = np.where(acc["marked"] == 1, 1.0, len(unmarked) / N_UNMARKED).round(4)

    # every rated blitz game of a sampled account, one row per (account, game)
    g = g.sort_values(["day", "sec"], kind="stable")
    side = []
    for me, opp, me_elo, opp_elo, me_diff, sign in (("w", "b", "welo", "belo", "wdiff", 1), ("b", "w", "belo", "welo", "bdiff", -1)):
        s = g[g[me].isin(code.keys())]
        side.append(pd.DataFrame({"a": s[me].to_numpy(), "day": s["day"].to_numpy(), "sec": s["sec"].to_numpy(),
                                  "elo": s[me_elo].to_numpy(), "after": (s[me_elo] + s[me_diff]).to_numpy(),
                                  "opp_elo": s[opp_elo].to_numpy(), "score2": 1 + sign * (s["res"].to_numpy() - 1),
                                  "gi": s.index.to_numpy(), "usable": s["usable"].to_numpy()}))
    pg = pd.concat(side, ignore_index=True).sort_values(["a", "day", "sec"], kind="stable")
    acc["elo_first"] = pg.groupby("a")["elo"].first()
    acc["elo_last"] = pg.groupby("a")["after"].last()
    pg["period"] = np.minimum(pg["day"] // PERIOD_DAYS, N_PERIODS - 1)
    pg["opp_bin"] = (np.round(pg["opp_elo"] / 50) * 50).astype(int)
    res = (pg.assign(win=pg["score2"] == 2, draw=pg["score2"] == 1).groupby(["a", "period", "opp_bin"])
           .agg(n=("win", "size"), wins=("win", "sum"), draws=("draw", "sum"), own_elo=("elo", "mean")).reset_index())
    res["own_elo"] = res["own_elo"].round().astype(int)

    # at most MAX_GAMES usable analysed games per account, at random
    cand = pg[pg["usable"]].copy()
    cand["u"] = rng.random(len(cand))
    chosen = np.unique(cand.sort_values(["a", "u"]).groupby("a").head(MAX_GAMES)["gi"].to_numpy())
    gcode = {gi: f"G{i + 1:05d}" for i, gi in enumerate(rng.permutation(chosen))}
    gs = g.loc[chosen]
    games = pd.DataFrame({
        "game": [gcode[gi] for gi in chosen],
        "period": np.minimum(gs["day"].to_numpy() // PERIOD_DAYS, N_PERIODS - 1),
        "white": [code.get(a, "") for a in gs["w"]], "black": [code.get(a, "") for a in gs["b"]],
        "white_elo": gs["welo"].to_numpy(), "black_elo": gs["belo"].to_numpy(),
        "result": gs["res"].map({2: "1-0", 1: "1/2-1/2", 0: "0-1"}).to_numpy(),
        "plies": gs["plies"].to_numpy(),
        "evals": [" ".join(moves[gi][0].split(" ")[:MAX_PLIES]) for gi in chosen],
        "flags": [moves[gi][1][:MAX_PLIES] for gi in chosen],
    }).sort_values("game").reset_index(drop=True)

    accounts = (acc.assign(account=[code[a] for a in acc.index])
                [["account", "marked", "weight", "n_blitz", "n_analysed", "elo_first", "elo_last"]]
                .sort_values("account").reset_index(drop=True))
    results = (res.assign(account=res["a"].map(code)).rename(columns={"opp_bin": "opp_elo"})
               [["account", "period", "opp_elo", "own_elo", "n", "wins", "draws"]]
               .sort_values(["account", "period", "opp_elo"]).reset_index(drop=True))
    return accounts, results, games, counts


def check(accounts: pd.DataFrame, results: pd.DataFrame, games: pd.DataFrame, counts: dict) -> None:
    assert accounts["account"].is_unique and games["game"].is_unique
    assert accounts["account"].str.fullmatch(r"A\d{4}").all() and games["game"].str.fullmatch(r"G\d{5}").all()
    assert (accounts["marked"] == 1).sum() == counts["eligible_marked"]     # a closed account shows no mark
    assert (accounts["marked"] == 0).sum() == N_UNMARKED
    assert set(results["account"]) == set(accounts["account"])
    assert (results.groupby("account")["n"].sum().reindex(accounts["account"]).to_numpy() == accounts["n_blitz"].to_numpy()).all()
    assert ((results["wins"] + results["draws"]) <= results["n"]).all() and results["period"].between(0, N_PERIODS - 1).all()
    sampled = set(accounts["account"])
    assert ((games["white"].isin(sampled)) | (games["black"].isin(sampled))).all()
    assert set(games["white"]) | set(games["black"]) <= sampled | {""}
    n_ev = games["evals"].str.count(" ") + 1
    assert (n_ev == np.minimum(games["plies"], MAX_PLIES)).all() and (games["flags"].str.len() == n_ev).all()
    assert games["plies"].min() >= MIN_PLIES
    per_acc = pd.concat([games["white"], games["black"]]).value_counts().drop("", errors="ignore")
    assert per_acc.reindex(accounts["account"]).min() >= MIN_ANALYSED
    for table in (accounts, results, games):          # nothing that names an account or a game
        assert not {"site", "name", "username", "id", "date", "time", "opening", "eco"} & {c.lower() for c in table.columns}


def write(table: pd.DataFrame, name: str) -> str:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:        # mtime=0: byte-identical rebuilds
        gz.write(table.to_csv(index=False).encode())
    out = DATA / name
    out.write_bytes(buf.getvalue())
    md5 = hashlib.md5(buf.getvalue()).hexdigest()
    print(f"wrote data/{name}: {len(table):,} rows, {out.stat().st_size / 1e6:.2f} MB, md5 {md5}")
    return md5


def main() -> None:
    fetch()
    scan()
    accounts, results, games, counts = build()
    check(accounts, results, games, counts)
    print(counts)
    for table, name in ((accounts, "e121_chess_accounts.csv.gz"), (results, "e121_chess_results.csv.gz"),
                        (games, "e121_chess_games.csv.gz")):
        write(table, name)
    print(f"sampled accounts: {len(accounts):,} ({int(accounts['marked'].sum())} marked); "
          f"each unmarked one stands for {accounts['weight'].max():.2f} eligible accounts")


if __name__ == "__main__":
    main()
