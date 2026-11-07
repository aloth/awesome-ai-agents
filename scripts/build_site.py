#!/usr/bin/env python3
"""Render the landing page from agents.json into _site/.

The entries are written into the HTML at build time, so the page works without
JavaScript and search engines see every entry. JavaScript only adds search and
category filters on top. Standard library only.

Usage:
    python3 scripts/build_site.py [output_dir]   # default: _site
"""

import datetime
import html
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = ROOT / "agents.json"


GITHUB_REPO = re.compile(r"^https://github\.com/([^/]+)/([^/#?]+)/?$")


def fetch_stars(entries):
    """Star counts for GitHub entries, keyed by URL. Empty without GITHUB_TOKEN.

    Uses the GraphQL API in batches of 100 repositories; the whole list costs
    a few points of the 5,000-point hourly budget. Failures are not fatal: the
    page is then built without star counts.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set: building without star counts")
        return {}
    repos = [(e["url"], m.group(1), m.group(2)) for e in entries if (m := GITHUB_REPO.match(e["url"]))]
    stars = {}
    for start in range(0, len(repos), 100):
        chunk = repos[start:start + 100]
        fields = " ".join(
            f"r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{ stargazerCount }}"
            for i, (_, owner, name) in enumerate(chunk)
        )
        request = urllib.request.Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": "query { " + fields + " }"}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response).get("data") or {}
        except Exception as error:  # network or API error: build without stars
            print(f"star lookup failed: {error}")
            return {}
        for i, (url, _, _) in enumerate(chunk):
            repo = data.get(f"r{i}")
            if repo:
                stars[url] = repo["stargazerCount"]
    print(f"star counts for {len(stars)} of {len(repos)} GitHub entries")
    return stars


def short_count(n):
    if n >= 1000:
        value = f"{n / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{value}k"
    return str(n)


def esc(text):
    return html.escape(str(text), quote=True)


def host(url):
    netloc = urlparse(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc


def render_entry(entry, stars):
    year = f' <span class="year">({entry["year"]})</span>' if entry.get("year") else ""
    count = stars.get(entry["url"])
    star = f'<span class="stars" title="{count:,} GitHub stars">&#9733; {short_count(count)}</span>' if count is not None else ""
    added = entry.get("added", "")
    sub = entry.get("subcategory") or ""
    search = " ".join([entry["name"], entry["description"], entry["category"], sub]).lower()
    return (
        f'<li class="entry" data-search="{esc(search)}" data-stars="{count or 0}" data-added="{esc(added)}" data-name="{esc(entry["name"].lower())}">'
        f'<a class="name" href="{esc(entry["url"])}" rel="noopener">{esc(entry["name"])}</a>{year}'
        f'<p>{esc(entry["description"])}</p>'
        f'<span class="meta"><span class="host">{esc(host(entry["url"]))}</span>{star}</span>'
        f"</li>"
    )


def render_list(data, stars):
    parts = []
    for category in data["categories"]:
        entries = [e for e in data["entries"] if e["category"] == category["name"]]
        parts.append(f'<section class="category" id="{esc(category["slug"])}" data-category="{esc(category["slug"])}">')
        parts.append(f'<h2>{esc(category["name"])} <span class="count">{len(entries)}</span></h2>')
        groups = [None] + [s["name"] for s in category["subcategories"]]
        for group in groups:
            members = [e for e in entries if e.get("subcategory") == group]
            if not members:
                continue
            parts.append('<div class="group">')
            if group:
                parts.append(f"<h3>{esc(group)}</h3>")
            parts.append('<ul class="entries">' + "".join(render_entry(e, stars) for e in members) + "</ul>")
            parts.append("</div>")
        parts.append("</section>")
    return "\n".join(parts)


def render_filters(data):
    buttons = ['<button type="button" class="chip active" data-filter="all">All</button>']
    for category in data["categories"]:
        n = sum(1 for e in data["entries"] if e["category"] == category["name"])
        buttons.append(
            f'<button type="button" class="chip" data-filter="{esc(category["slug"])}">{esc(category["name"])} <span>{n}</span></button>'
        )
    return "\n".join(buttons)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
    data = json.loads(DATA.read_text(encoding="utf-8"))
    stars = fetch_stars(data["entries"])
    # "Updated" is the last change to the list itself, not the build date: the
    # daily rebuild only refreshes star counts.
    dates = [e.get("added", "") for e in data["entries"]] + [r.get("removed", "") for r in data.get("removed_entries", [])]
    today = max(d for d in dates if d) if any(dates) else datetime.date.today().isoformat()

    page = (DOCS / "index.html").read_text(encoding="utf-8")
    replacements = {
        "{{ENTRY_COUNT}}": str(data["entry_count"]),
        "{{CATEGORY_COUNT}}": str(len(data["categories"])),
        "{{UPDATED}}": today,
        "{{FILTERS}}": render_filters(data),
        "{{LIST}}": render_list(data, stars),
    }
    for key, value in replacements.items():
        if key not in page:
            raise SystemExit(f"placeholder {key} missing from docs/index.html")
        page = page.replace(key, value)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    shutil.copy(DATA, out / "agents.json")
    for name in ("social-preview.jpg", "robots.txt"):
        shutil.copy(DOCS / name, out / name)
    sitemap = (DOCS / "sitemap.xml").read_text(encoding="utf-8").replace("{{UPDATED}}", today)
    (out / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"built {out}: {data['entry_count']} entries")


if __name__ == "__main__":
    main()
