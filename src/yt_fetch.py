"""Fetch YouTube channel listings and transcripts from this environment.

Three things get in the way here and each needs handling:
  * the transcript API is blocked for datacenter IPs, so captions are pulled
    from the watch page's own caption track instead;
  * YouTube 302s to a consent wall without a consent cookie;
  * the page is client-rendered, so the video list has to come out of the
    embedded ytInitialData blob rather than the DOM.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
COOKIES = {"CONSENT": "YES+cb.20210328-17-p0.en+FX+100", "SOCS": "CAI"}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    s.cookies.update(COOKIES)
    return s


# YouTube throttles this datacenter IP hard, so requests are paced globally and
# a 429 is waited out rather than retried tightly.
_last_request = [0.0]
MIN_GAP = float(os.environ.get("YT_MIN_GAP", "6.0"))


def get(s, url, tries=6, verbose=False):
    delay = 8.0
    for i in range(tries):
        gap = MIN_GAP - (time.time() - _last_request[0])
        if gap > 0:
            time.sleep(gap)
        try:
            r = s.get(url, timeout=90, allow_redirects=True)
            _last_request[0] = time.time()
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                if verbose:
                    print(f"    throttled ({r.status_code}), waiting {delay:.0f}s",
                          flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            return None
        except Exception:
            _last_request[0] = time.time()
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    return None


def initial_data(page: str):
    m = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", page, re.S)
    if not m:
        m = re.search(r'ytInitialData"\]\s*=\s*(\{.*?\});', page, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def walk(node, key):
    """Yield every value stored under `key` anywhere in a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from walk(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v, key)


def channel_videos(s, handle: str, tab: str = "videos"):
    page = get(s, f"https://www.youtube.com/{handle}/{tab}")
    if not page:
        return []
    data = initial_data(page)
    if not data:
        return []
    out, seen = [], set()
    # Current YouTube ships lockupViewModel; older layouts used videoRenderer.
    for lv in walk(data, "lockupViewModel"):
        vid = lv.get("contentId")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        title = ""
        for t in walk(lv, "metadataViewModel"):
            for txt in walk(t, "content"):
                if isinstance(txt, str) and len(txt) > len(title):
                    title = txt
            break
        if not title:
            for txt in walk(lv, "accessibilityText"):
                if isinstance(txt, str):
                    title = txt
                    break
        length = ""
        for b in walk(lv, "thumbnailBadgeViewModel"):
            t = b.get("text", "")
            if re.match(r"^\d+:\d+", t or ""):
                length = t
                break
        rows = []
        for r in walk(lv, "metadataRowViewModel"):
            for txt in walk(r, "content"):
                if isinstance(txt, str):
                    rows.append(txt)
        views = next((x for x in rows if "view" in x.lower()), "")
        published = next((x for x in rows if "ago" in x.lower()), "")
        out.append({"id": vid, "title": title, "length": length,
                    "views": views, "published": published})
    if out:
        return out

    for r in walk(data, "richItemRenderer"):
        for v in walk(r, "videoRenderer"):
            vid = v.get("videoId")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            title = ""
            t = v.get("title", {})
            if "runs" in t:
                title = "".join(x.get("text", "") for x in t["runs"])
            elif "simpleText" in t:
                title = t["simpleText"]
            length = ""
            lt = v.get("lengthText", {})
            if isinstance(lt, dict):
                length = lt.get("simpleText", "")
            views = ""
            vc = v.get("viewCountText", {})
            if isinstance(vc, dict):
                views = vc.get("simpleText", "")
            published = ""
            pt = v.get("publishedTimeText", {})
            if isinstance(pt, dict):
                published = pt.get("simpleText", "")
            out.append({"id": vid, "title": title, "length": length,
                        "views": views, "published": published})
    return out


def video_meta(s, vid: str):
    page = get(s, f"https://www.youtube.com/watch?v={vid}", verbose=True)
    if not page:
        return None
    meta = {"id": vid}
    m = re.search(r'"shortDescription":"(.*?)","', page, re.S)
    if m:
        d = m.group(1)
        try:
            d = d.encode().decode("unicode_escape")
        except Exception:
            pass
        meta["description"] = d
    m = re.search(r'<meta name="title" content="([^"]*)"', page)
    if m:
        meta["title"] = html.unescape(m.group(1))
    m = re.search(r'"lengthSeconds":"(\d+)"', page)
    if m:
        meta["length_s"] = int(m.group(1))
    tracks = re.search(r'"captionTracks":(\[.*?\])', page, re.S)
    if tracks:
        try:
            meta["captions"] = json.loads(tracks.group(1).replace("\\u0026", "&"))
        except Exception:
            meta["captions"] = None
    return meta, page


def transcript(s, vid: str):
    got = video_meta(s, vid)
    if not got:
        return None, None
    meta, _ = got
    tracks = meta.get("captions") or []
    if not tracks:
        return meta, None
    # prefer a manually written English track, else anything English
    def score(t):
        lang = (t.get("languageCode") or "")
        kind = t.get("kind", "")
        return (0 if lang.startswith("en") else 1, 0 if kind != "asr" else 1)

    track = sorted(tracks, key=score)[0]
    url = track.get("baseUrl")
    if not url:
        return meta, None
    xml = get(s, url + "&fmt=srv3", verbose=True)
    if not xml or "<" not in xml:
        xml = get(s, url)
    if not xml:
        return meta, None
    parts = re.findall(r"<text[^>]*>(.*?)</text>", xml, re.S)
    if not parts:
        parts = re.findall(r"<p[^>]*>(.*?)</p>", xml, re.S)
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", "", p)).replace("\n", " ")
                    for p in parts)
    return meta, re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    s = session()
    cmd = sys.argv[1]
    if cmd == "list":
        vids = channel_videos(s, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "videos")
        print(f"{len(vids)} videos")
        for v in vids:
            print(f"  {v['id']}  {v['length']:>8s}  {v['views']:>14s}  "
                  f"{v['published']:>16s}  {v['title'][:90]}")
    elif cmd == "transcript":
        meta, text = transcript(s, sys.argv[2])
        print(json.dumps({k: v for k, v in (meta or {}).items()
                          if k != "captions"}, indent=2)[:1500])
        print("\n--- transcript ---\n")
        print(text[:6000] if text else "NO TRANSCRIPT")
