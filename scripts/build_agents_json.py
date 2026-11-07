#!/usr/bin/env python3
"""Build agents.json from README.md, or check README.md for format errors.

README.md is the only file contributors edit. agents.json is generated from it
on every push to main and must never be edited by hand.

Usage:
    python3 scripts/build_agents_json.py           # write agents.json
    python3 scripts/build_agents_json.py --check   # validate README.md only

Standard library only.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
OUTPUT = ROOT / "agents.json"

REPOSITORY = "https://github.com/aloth/awesome-ai-agents"
HOMEPAGE = "https://aloth.github.io/awesome-ai-agents/"

# Sections that are not part of the list itself.
SKIPPED_SECTIONS = {"Contents", "License"}
# Sections whose items are plain text examples rather than linked entries.
TEXT_ONLY_SECTIONS = {"Use Cases & Case Studies"}

SEPARATOR = " \u2014 "  # the list format: "- [Name](URL) <em dash> Description"
ENTRY = re.compile(
    r"^- \[(?P<name>[^\]]+)\]\((?P<url>[^)\s]+)\)"
    r"(?: \((?P<year>\d{4})\))?"
    r"(?P<sep> \u2014 |: | - )(?P<desc>\S.*)$"
)
INLINE_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def slugify(text):
    """GitHub-style heading anchor."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def plain(markdown):
    """Description without Markdown markup."""
    text = INLINE_LINK.sub(r"\1", markdown)
    text = text.replace("**", "").replace("`", "")
    return text.strip()


def parse(readme_text):
    entries, categories, errors = [], [], []
    category = subcategory = None

    for number, line in enumerate(readme_text.splitlines(), 1):
        line = line.rstrip()
        if line.startswith("## "):
            category = line[3:].strip()
            subcategory = None
            if category not in SKIPPED_SECTIONS:
                categories.append({"name": category, "slug": slugify(category), "subcategories": []})
            continue
        if line.startswith("### "):
            subcategory = line[4:].strip()
            if categories and category not in SKIPPED_SECTIONS:
                categories[-1]["subcategories"].append({"name": subcategory, "slug": slugify(subcategory)})
            continue
        if not line.startswith("- ") or category is None or category in SKIPPED_SECTIONS:
            continue
        if category in TEXT_ONLY_SECTIONS and not line.startswith("- ["):
            continue

        match = ENTRY.match(line)
        if not match:
            errors.append(f"README.md:{number}: entry does not match '- [Name](https://...){SEPARATOR}Description': {line[:100]}")
            continue
        if match.group("sep") != SEPARATOR:
            errors.append(f"README.md:{number}: use '{SEPARATOR.strip()}' (em dash with a space on each side) between the link and the description")
        url = match.group("url")
        if not url.startswith("https://"):
            errors.append(f"README.md:{number}: the link must start with https://")

        description_md = match.group("desc").strip()
        entry = {
            "name": match.group("name").strip(),
            "url": url,
            "description": plain(description_md),
            "description_markdown": description_md,
            "category": category,
            "subcategory": subcategory,
        }
        if match.group("year"):
            entry["year"] = int(match.group("year"))
        entries.append(entry)

    seen = {}
    for entry in entries:
        key = entry["url"].rstrip("/").lower()
        if key in seen:
            errors.append(f"duplicate link: '{entry['name']}' and '{seen[key]}' both point to {entry['url']}")
        seen[key] = entry["name"]

    categories = [c for c in categories if any(e["category"] == c["name"] for e in entries)]
    return entries, categories, errors


def build(entries, categories):
    return {
        "name": "Awesome AI Agents",
        "description": "A curated list of AI agent frameworks, tools, platforms, research papers and resources.",
        "do_not_edit": True,
        "generated_from": "README.md",
        "note": "Generated from README.md by scripts/build_agents_json.py on every push to main. Edit README.md instead.",
        "license": "CC0-1.0",
        "repository": REPOSITORY,
        "homepage": HOMEPAGE,
        "entry_count": len(entries),
        "categories": categories,
        "entries": entries,
    }


def main():
    entries, categories, errors = parse(README.read_text(encoding="utf-8"))
    for error in errors:
        print(f"::error::{error}" if "--github" in sys.argv else error, file=sys.stderr)
    if errors:
        print(f"{len(errors)} problem(s) in README.md", file=sys.stderr)
        return 1
    if "--check" in sys.argv:
        print(f"README.md OK: {len(entries)} entries in {len(categories)} categories")
        return 0
    data = json.dumps(build(entries, categories), ensure_ascii=False, indent=2) + "\n"
    OUTPUT.write_text(data, encoding="utf-8")
    print(f"wrote {OUTPUT.name}: {len(entries)} entries in {len(categories)} categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
