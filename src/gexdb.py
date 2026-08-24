"""Persist dealer gamma snapshots to SQLite, so history accumulates.

Free option chains are a live snapshot only. There is no free archive: the
DoltHub options dataset carries greeks but no open interest, and open interest
is the whole input to GEX; the Wayback copies of the CBOE endpoint are a
handful of stray captures and web.archive.org is unreachable from this network
anyway. So the only way to own GEX history is to start keeping it.

Two things are stored for every reading:

  * the aggregate metrics, which is what a strategy trades on;
  * the per-strike profile, so a question asked in a year's time is not limited
    to the metrics that seemed interesting today.

Persistence is text-first and the database is derived from it. This session
runs in a container that is reclaimed when it ends, so the only durable store
is the repository -- and a SQLite file that grows every day would rewrite a
large binary blob in each commit. Instead every day writes one small immutable
CSV plus one appended JSON line, which git stores cleanly, and `rebuild()`
reconstructs the database from them on any machine.
"""

from __future__ import annotations

import csv
import gzip
import glob
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gex as G  # noqa: E402

DB = "data/gex/gex.db"
RAW = "data/gex/raw"
DAILY = "data/gex/daily"
METRICS = "data/gex/metrics.jsonl"

# Only these two buckets get a per-strike file. The wider buckets are supersets
# whose profiles can be recomputed from the raw payload if they are ever
# wanted, and writing all four would quadruple the daily file for no new
# information.
STRIKE_BUCKETS = ("all", "0dte")

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
  id            INTEGER PRIMARY KEY,
  symbol        TEXT NOT NULL,
  trade_date    TEXT NOT NULL,     -- the chain's own quote date
  captured_utc  TEXT NOT NULL,     -- when we fetched it
  bucket        TEXT NOT NULL,     -- 'all', '0dte', '5dte', ...
  spot          REAL,
  total_gex     REAL,
  gamma_flip    REAL,
  flip_dist_pct REAL,
  call_wall     REAL,
  put_wall      REAL,
  n_contracts   INTEGER,
  total_oi      REAL,
  raw_path      TEXT,
  UNIQUE (symbol, trade_date, bucket)
);
CREATE TABLE IF NOT EXISTS strike_gex (
  snapshot_id   INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
  strike        REAL NOT NULL,
  gex           REAL,
  call_oi       REAL,
  put_oi        REAL,
  PRIMARY KEY (snapshot_id, strike)
);
CREATE INDEX IF NOT EXISTS ix_snap_sym_date ON snapshots(symbol, trade_date);
"""

BUCKETS = {"all": None, "0dte": 0, "5dte": 5, "30dte": 30}


def day_file(symbol, trade_date, root=DAILY):
    return os.path.join(root, f"{symbol.strip('_')}_{trade_date}.csv.gz")


def write_day(symbol, trade_date, rows, root=DAILY):
    """One immutable per-strike file per symbol per day.

    A new small file each day is what git is good at; rewriting one growing
    file is what it is bad at.
    """
    os.makedirs(root, exist_ok=True)
    path = day_file(symbol, trade_date, root)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "strike", "gex", "call_oi", "put_oi"])
        w.writerows(rows)
    return path


def append_metrics(row, path=METRICS):
    """Append one day's aggregate metrics, skipping an exact repeat."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = (row["symbol"], row["trade_date"], row["bucket"])
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    old = json.loads(line)
                except Exception:
                    continue
                if (old.get("symbol"), old.get("trade_date"),
                        old.get("bucket")) == key:
                    return False
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return True


def connect(path=DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def save_raw(symbol, trade_date, payload, root=RAW):
    """Keep the original bytes; re-deriving beats re-fetching what is gone."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{symbol.strip('_')}_{trade_date}.json.gz")
    if not os.path.exists(path):
        with gzip.open(path, "wt") as fh:
            json.dump(payload, fh)
    return path


def quote_date(payload) -> str:
    """The date the chain describes, which is not always today.

    A snapshot pulled at the weekend still describes Friday's close, so keying
    rows on the wall clock would write a Saturday row that never existed.
    """
    ts = (payload.get("timestamp") or "").split(" GMT")[0].strip()
    data = payload.get("data", {})
    last = data.get("last_trade_time") or ""
    if last:
        return last[:10]
    return ts[:10] if ts else ""


def record(symbols=("_SPX", "_NDX"), path=DB, buckets=BUCKETS, chain_path=None):
    """Fetch each symbol once and write every DTE bucket from that one chain."""
    con = connect(path)
    out = []
    for sym in symbols:
        try:
            payload = G.fetch_chain(sym, chain_path)
        except Exception as e:
            out.append({"symbol": sym, "error": str(e)})
            continue

        spot, ch = G.parse_chain(payload)
        tdate = quote_date(payload)
        strike_rows = []
        raw = save_raw(sym, tdate, payload)
        captured = payload.get("timestamp", "")

        for name, dte in buckets.items():
            grid, tot = G.profile(ch, spot, max_dte=dte)
            total = float(G.gex_at(spot, ch, max_dte=dte)[0].sum())
            flip = G.gamma_flip(grid, tot)
            ks, agg = G.by_strike(ch, spot, max_dte=dte)
            cw, pw = G.walls(ks, agg, spot)
            used = np.ones(len(ch["strike"]), bool) if dte is None else ch["dte"] <= dte

            cur = con.execute(
                """INSERT INTO snapshots
                   (symbol, trade_date, captured_utc, bucket, spot, total_gex,
                    gamma_flip, flip_dist_pct, call_wall, put_wall,
                    n_contracts, total_oi, raw_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, trade_date, bucket) DO UPDATE SET
                     captured_utc=excluded.captured_utc, spot=excluded.spot,
                     total_gex=excluded.total_gex, gamma_flip=excluded.gamma_flip,
                     flip_dist_pct=excluded.flip_dist_pct,
                     call_wall=excluded.call_wall, put_wall=excluded.put_wall,
                     n_contracts=excluded.n_contracts, total_oi=excluded.total_oi,
                     raw_path=excluded.raw_path
                   WHERE substr(excluded.captured_utc, 1, 10) <= snapshots.trade_date
                      OR substr(snapshots.captured_utc, 1, 10) > snapshots.trade_date
                   RETURNING id""",
                (payload.get("symbol", sym), tdate, captured, name, spot, total,
                 flip, None if flip is None else (spot / flip - 1) * 100.0,
                 cw, pw, int(used.sum()), float(ch["oi"][used].sum()), raw))
            got = cur.fetchone()
            if got is None:
                # The stored row was captured on the session it describes and
                # this one was not -- a pre-open fetch the following Monday
                # carries live quotes but still reports Friday's date, and
                # letting it overwrite would replace a settled close with a
                # partial one. Keep what is there.
                sid = con.execute(
                    "SELECT id FROM snapshots WHERE symbol=? AND trade_date=?"
                    " AND bucket=?",
                    (payload.get("symbol", sym), tdate, name)).fetchone()[0]
                continue
            sid = got[0]

            con.execute("DELETE FROM strike_gex WHERE snapshot_id=?", (sid,))
            rows = []
            m = used
            for k in np.unique(ch["strike"][m]):
                at = m & (ch["strike"] == k)
                rows.append((sid, float(k), float(agg[ks == k][0]) if (ks == k).any() else 0.0,
                             float(ch["oi"][at & (ch["right"] > 0)].sum()),
                             float(ch["oi"][at & (ch["right"] < 0)].sum())))
            con.executemany(
                "INSERT INTO strike_gex (snapshot_id,strike,gex,call_oi,put_oi)"
                " VALUES (?,?,?,?,?)", rows)

            metrics_row = {
                "symbol": payload.get("symbol", sym), "trade_date": tdate,
                "captured_utc": captured, "bucket": name, "spot": spot,
                "total_gex": total, "gamma_flip": flip,
                "flip_dist_pct": None if flip is None else (spot / flip - 1) * 100.0,
                "call_wall": cw, "put_wall": pw,
                "n_contracts": int(used.sum()),
                "total_oi": float(ch["oi"][used].sum()),
            }
            append_metrics(metrics_row)
            if name in STRIKE_BUCKETS:
                strike_rows.extend([(name,) + r[1:] for r in rows])
            if name == "all":
                out.append({"symbol": sym, "trade_date": tdate, "spot": spot,
                            "total_gex": total, "gamma_flip": flip,
                            "strikes": len(rows)})

        if strike_rows:
            write_day(sym, tdate, strike_rows)
        con.commit()
    con.close()
    return out


def rebuild(path=DB, daily=DAILY, metrics=METRICS):
    """Reconstruct the database from the committed text files.

    This is what makes the container being ephemeral survivable: a fresh clone
    plus this call is a complete database.
    """
    if os.path.exists(path):
        os.remove(path)
    con = connect(path)
    ids = {}
    if os.path.exists(metrics):
        with open(metrics) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cur = con.execute(
                    """INSERT OR REPLACE INTO snapshots
                       (symbol, trade_date, captured_utc, bucket, spot, total_gex,
                        gamma_flip, flip_dist_pct, call_wall, put_wall,
                        n_contracts, total_oi, raw_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) RETURNING id""",
                    (r["symbol"], r["trade_date"], r.get("captured_utc", ""),
                     r["bucket"], r.get("spot"), r.get("total_gex"),
                     r.get("gamma_flip"), r.get("flip_dist_pct"),
                     r.get("call_wall"), r.get("put_wall"),
                     r.get("n_contracts"), r.get("total_oi"), None))
                ids[(r["symbol"].strip("^_"), r["trade_date"], r["bucket"])] = cur.fetchone()[0]

    n_strikes = 0
    for f in sorted(glob.glob(os.path.join(daily, "*.csv.gz"))):
        base = os.path.basename(f)[:-len(".csv.gz")]
        sym, tdate = base.rsplit("_", 1)
        with gzip.open(f, "rt") as fh:
            for row in csv.DictReader(fh):
                sid = ids.get((sym, tdate, row["bucket"]))
                if sid is None:
                    continue
                con.execute(
                    "INSERT OR REPLACE INTO strike_gex "
                    "(snapshot_id,strike,gex,call_oi,put_oi) VALUES (?,?,?,?,?)",
                    (sid, float(row["strike"]), float(row["gex"]),
                     float(row["call_oi"]), float(row["put_oi"])))
                n_strikes += 1
    con.commit()
    con.close()
    return len(ids), n_strikes


def history(symbol="^SPX", bucket="all", path=DB):
    """Every stored reading for a symbol, oldest first."""
    import pandas as pd
    con = connect(path)
    df = pd.read_sql_query(
        "SELECT * FROM snapshots WHERE symbol=? AND bucket=? ORDER BY trade_date",
        con, params=(symbol, bucket))
    con.close()
    return df


def summary(path=DB):
    con = connect(path)
    rows = con.execute(
        """SELECT symbol, bucket, COUNT(*) n, MIN(trade_date), MAX(trade_date)
           FROM snapshots GROUP BY symbol, bucket ORDER BY symbol, bucket""").fetchall()
    con.close()
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        n, k = rebuild()
        print(f"rebuilt {DB} from committed files: {n} snapshots, {k:,} strike rows")
        sys.exit()
    got = record(chain_path=sys.argv[1] if len(sys.argv) > 1 else None)
    for r in got:
        print(json.dumps(r))
    print("\nstored so far:")
    for sym, bucket, n, lo, hi in summary():
        print(f"  {sym:<8s} {bucket:<6s} {n:>4d} days   {lo} .. {hi}")
