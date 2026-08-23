"""Fetch split-adjusted share counts for the PEAD universe."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edgar import shares_outstanding  # noqa: E402

DATA = "data/pead"


def main():
    with open(os.path.join(DATA, "events.json")) as fh:
        ev = json.load(fh)
    with open(os.path.join(DATA, "universe.json")) as fh:
        uni = json.load(fh)["ok"]

    out, missing = {}, []
    for i, tk in enumerate(sorted(uni)):
        if tk not in ev:
            continue
        s = shares_outstanding(ev[tk]["cik"])
        if len(s) >= 8:
            out[tk] = s
        else:
            missing.append(tk)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(uni)}  have {len(out)}", flush=True)

    with open(os.path.join(DATA, "shares.json"), "w") as fh:
        json.dump(out, fh)
    print(f"\n{len(out)} names with >= 8 share observations; "
          f"{len(missing)} without")


if __name__ == "__main__":
    main()
