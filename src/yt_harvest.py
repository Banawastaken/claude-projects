"""Harvest titles, descriptions and transcripts for a channel's long-form videos."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yt_fetch import channel_videos, session, transcript  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "matfinog")


# The channel listing costs a request against a hard-throttled IP, so the
# long-form ids read off it earlier are reused when they are still valid.
KNOWN = ["bMs9PzQzrqM", "IeaFAj83Ufg", "ueVIjmC9glg", "mUeDgNf84o0",
         "3qMArQutPmE", "mUvzcpcIAc0", "lQHatsTEqFE", "wm6XQFw1GHI",
         "EP4ptjamPYA", "rNC90QEgp-Q", "-2-vE-EbqBI"]


def main(handle="@MatFinOg"):
    os.makedirs(OUT, exist_ok=True)
    s = session()
    if os.environ.get("YT_USE_KNOWN"):
        vids = [{"id": v, "length": ""} for v in KNOWN]
    else:
        vids = channel_videos(s, handle, "videos")
    print(f"{len(vids)} long-form videos\n", flush=True)
    index = []
    for v in vids:
        vid = v["id"]
        path = os.path.join(OUT, f"{vid}.json")
        if os.path.exists(path):
            with open(path) as fh:
                rec = json.load(fh)
            index.append({k: rec.get(k) for k in ("id", "title", "length_s", "chars")})
            print(f"  {vid}  cached  {rec.get('title', '')[:70]}", flush=True)
            continue
        meta, text = transcript(s, vid)
        if meta is None:
            print(f"  {vid}  FAILED", flush=True)
            continue
        rec = {
            "id": vid,
            "title": meta.get("title", ""),
            "length_s": meta.get("length_s"),
            "listed_length": v.get("length"),
            "description": meta.get("description", ""),
            "transcript": text or "",
            "chars": len(text or ""),
        }
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        index.append({k: rec[k] for k in ("id", "title", "length_s", "chars")})
        print(f"  {vid}  {rec['length_s'] or 0:>5}s  {rec['chars']:>6} chars  "
              f"{rec['title'][:70]}", flush=True)
        time.sleep(2.0)

    with open(os.path.join(OUT, "_index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    got = sum(1 for i in index if (i.get("chars") or 0) > 500)
    print(f"\ntranscripts recovered: {got}/{len(index)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "@MatFinOg")
