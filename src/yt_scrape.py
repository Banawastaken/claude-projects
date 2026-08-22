"""Read a YouTube channel's video list with a real browser.

YouTube renders everything client-side, so a plain HTTP fetch returns only the
footer. Chromium is available here, so the page is loaded and scrolled until the
video grid stops growing.

Usage: yt_scrape.py <channel_url> [max_scrolls]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

CHROMIUM = "/opt/pw-browsers/chromium"


async def scrape(url: str, max_scrolls: int = 12):
    from playwright.async_api import async_playwright

    # Outbound traffic here goes through the session's agent proxy; Chromium
    # does not pick that up from the environment on its own.
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    proxy = {"server": proxy_url} if proxy_url else None

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=CHROMIUM if os.path.exists(CHROMIUM) else None,
            proxy=proxy,
            args=["--ignore-certificate-errors"])
        page = await browser.new_page(
            viewport={"width": 1400, "height": 1000},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"))
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_timeout(4000)

        # dismiss a consent wall if one appears
        for label in ("Accept all", "Reject all", "I agree"):
            try:
                btn = page.get_by_role("button", name=label)
                if await btn.count():
                    await btn.first.click(timeout=3000)
                    await page.wait_for_timeout(2500)
                    break
            except Exception:
                pass

        title = await page.title()
        meta = {}
        for name in ("description", "og:description", "og:title"):
            try:
                el = await page.query_selector(
                    f'meta[name="{name}"], meta[property="{name}"]')
                if el:
                    meta[name] = await el.get_attribute("content")
            except Exception:
                pass

        seen = 0
        for _ in range(max_scrolls):
            await page.mouse.wheel(0, 4000)
            await page.wait_for_timeout(1500)
            n = await page.evaluate(
                "document.querySelectorAll('a#video-title-link, a#video-title').length")
            if n == seen:
                break
            seen = n

        videos = await page.evaluate("""
            () => Array.from(document.querySelectorAll(
                    'a#video-title-link, a#video-title'))
                .map(a => ({
                    title: (a.getAttribute('title') || a.textContent || '').trim(),
                    href: a.href,
                }))
                .filter(v => v.title)
        """)
        body = await page.evaluate("() => document.body.innerText.slice(0, 4000)")
        await browser.close()
        return {"page_title": title, "meta": meta, "videos": videos, "body": body}


if __name__ == "__main__":
    url = sys.argv[1]
    scrolls = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    data = asyncio.run(scrape(url, scrolls))
    print(json.dumps({k: v for k, v in data.items() if k != "body"},
                     indent=2)[:12000])
    print("\n--- page text ---\n")
    print(data["body"][:2500])
