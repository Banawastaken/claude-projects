"""Databento market-by-order: price the request first, then download.

MBO is every individual order's lifecycle -- add, modify, cancel, fill -- and it
is the largest data product there is. A single session of one liquid future runs
to gigabytes, and Databento bills by the byte, so the way to lose a $150 trial
credit is to type a plausible-looking date range and press go.

Nothing here downloads before `metadata.get_cost` has priced it and the price
has cleared a budget you set. `plan()` is the command to run first; it costs
nothing and tells you what a request would spend.

Set DATABENTO_API_KEY in the environment. The key is read from there and never
written to disk or into any file this repo commits.
"""

from __future__ import annotations

import os
import sys

BUDGET_USD = float(os.environ.get("DATABENTO_BUDGET", "25.0"))

# Cheap-to-expensive. MBO is the one the research actually wants; the others
# exist so a budget can buy a longer window when MBO cannot.
SCHEMAS = {
    "mbo": "every order event (what the research needs, largest)",
    "mbp-1": "top of book, one level, on every change",
    "mbp-10": "ten levels on every change",
    "tbbo": "trades with the quote at the time of the trade",
    "trades": "trades only (smallest)",
    "ohlcv-1s": "one-second bars",
}


def client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit(
            "DATABENTO_API_KEY is not set.\n"
            "Get the key from https://databento.com (Portal -> API keys), then:\n"
            "  export DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxxxxxx\n"
            "It is read from the environment only and never written to disk.")
    import databento as db
    return db.Historical(key)


def datasets():
    c = client()
    return list(c.metadata.list_datasets())


def plan(dataset="GLBX.MDP3", symbols=("ES.c.0",), schema="mbo",
         start=None, end=None, stype_in="continuous"):
    """What a request would cost and how big it is. Downloads nothing."""
    c = client()
    kw = dict(dataset=dataset, symbols=list(symbols), schema=schema,
              start=start, end=end, stype_in=stype_in)
    cost = c.metadata.get_cost(mode="historical", **kw)
    size = c.metadata.get_billable_size(**kw)
    return {"dataset": dataset, "symbols": list(symbols), "schema": schema,
            "start": start, "end": end,
            "cost_usd": float(cost), "bytes": int(size),
            "gib": int(size) / 1024 ** 3}


def fetch(dataset="GLBX.MDP3", symbols=("ES.c.0",), schema="mbo",
          start=None, end=None, stype_in="continuous",
          out="data/mbo", budget=None, force=False):
    """Download only after the priced cost clears the budget."""
    budget = BUDGET_USD if budget is None else budget
    p = plan(dataset, symbols, schema, start, end, stype_in)
    print(f"request: {schema} {','.join(symbols)} {start} -> {end}")
    print(f"  billable {p['gib']:.3f} GiB, cost ${p['cost_usd']:.2f}, "
          f"budget ${budget:.2f}")
    if p["cost_usd"] > budget and not force:
        raise SystemExit(
            f"refusing: ${p['cost_usd']:.2f} exceeds the ${budget:.2f} budget.\n"
            "Shorten the window, use a cheaper schema (tbbo or trades), or "
            "raise DATABENTO_BUDGET deliberately.")

    os.makedirs(out, exist_ok=True)
    tag = f"{dataset}_{'-'.join(symbols)}_{schema}_{start}_{end}".replace("/", "-")
    path = os.path.join(out, tag + ".dbn.zst")
    if os.path.exists(path):
        print(f"  already downloaded: {path}")
        return path
    c = client()
    data = c.timeseries.get_range(
        dataset=dataset, symbols=list(symbols), schema=schema,
        start=start, end=end, stype_in=stype_in)
    data.to_file(path)
    print(f"  wrote {path} ({os.path.getsize(path)/1024**2:.1f} MiB)")
    return path


def load(path):
    """Read a stored DBN file into a DataFrame."""
    import databento as db
    return db.DBNStore.from_file(path).to_df()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "datasets":
        for d in datasets():
            print(" ", d)
    elif cmd == "plan":
        # plan <schema> <start> <end> [symbol]
        schema, start, end = sys.argv[2], sys.argv[3], sys.argv[4]
        sym = sys.argv[5] if len(sys.argv) > 5 else "ES.c.0"
        p = plan(symbols=(sym,), schema=schema, start=start, end=end)
        print(f"{p['schema']:<9s} {p['start']} -> {p['end']}  "
              f"{p['gib']:.3f} GiB  ${p['cost_usd']:.2f}")
    elif cmd == "compare":
        # price every schema over the same window, so the trade-off is visible
        start, end = sys.argv[2], sys.argv[3]
        sym = sys.argv[4] if len(sys.argv) > 4 else "ES.c.0"
        print(f"{'schema':<10s}{'GiB':>9s}{'cost':>10s}   what it is")
        for s, note in SCHEMAS.items():
            try:
                p = plan(symbols=(sym,), schema=s, start=start, end=end)
                print(f"{s:<10s}{p['gib']:>9.3f}{p['cost_usd']:>10.2f}   {note}")
            except Exception as e:
                print(f"{s:<10s}{'-':>9s}{'-':>10s}   {str(e)[:60]}")
    else:
        print(__doc__)
        print("commands:\n  datasets\n  plan <schema> <start> <end> [symbol]"
              "\n  compare <start> <end> [symbol]")
