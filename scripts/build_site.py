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
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = ROOT / "agents.json"


def esc(text):
    return html.escape(str(text), quote=True)


def host(url):
    netloc = urlparse(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc


def render_entry(entry):
    year = f' <span class="year">({entry["year"]})</span>' if entry.get("year") else ""
    sub = entry.get("subcategory") or ""
    search = " ".join([entry["name"], entry["description"], entry["category"], sub]).lower()
    return (
        f'<li class="entry" data-search="{esc(search)}">'
        f'<a class="name" href="{esc(entry["url"])}" rel="noopener">{esc(entry["name"])}</a>{year}'
        f'<p>{esc(entry["description"])}</p>'
        f'<span class="host">{esc(host(entry["url"]))}</span>'
        f"</li>"
    )


def render_list(data):
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
            parts.append('<ul class="entries">' + "".join(render_entry(e) for e in members) + "</ul>")
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
    today = datetime.date.today().isoformat()

    page = (DOCS / "index.html").read_text(encoding="utf-8")
    replacements = {
        "{{ENTRY_COUNT}}": str(data["entry_count"]),
        "{{CATEGORY_COUNT}}": str(len(data["categories"])),
        "{{UPDATED}}": today,
        "{{FILTERS}}": render_filters(data),
        "{{LIST}}": render_list(data),
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
