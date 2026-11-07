#!/usr/bin/env python3
"""Build agents.json from README.md, or check README.md for format errors.

README.md is the only file contributors edit. agents.json is generated from it
on every push to main and must never be edited by hand.

Usage:
    python3 scripts/build_agents_json.py           # write agents.json
    python3 scripts/build_agents_json.py --check   # validate README.md only

Writing agents.json reads the git history of README.md to date every entry
("added") and to list entries that were removed ("removed_entries"). This
needs a full clone (actions/checkout with fetch-depth: 0). --check does not.

Standard library only.
"""

import datetime
import json
import re
import subprocess
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
# Lenient form used for old README versions, which did not all follow the format.
LOOSE_ENTRY = re.compile(r"^- \[(?P<name>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)(?P<rest>.*)$")


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


def url_key(url):
    return url.rstrip("/").lower()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def loose_entries(readme_text):
    """Entries of any README version: {url_key: (name, url, description, category, subcategory)}."""
    found, category, subcategory = {}, None, None
    for line in readme_text.splitlines():
        if line.startswith("## "):
            category, subcategory = line[3:].strip(), None
        elif line.startswith("### "):
            subcategory = line[4:].strip()
        elif category not in SKIPPED_SECTIONS:
            match = LOOSE_ENTRY.match(line.rstrip())
            if match:
                rest = re.sub(r"^(?: \(\d{4}\))?\s*(?:\u2014|-|:)\s*", "", match.group("rest")).strip()
                found[url_key(match.group("url"))] = (match.group("name").strip(), match.group("url"), rest, category, subcategory)
    return found


def same_project(old, new):
    """Whether a removed and an added line in one commit are the same project under a new link."""
    old_name, new_name = old[0].lower(), new[0].lower()
    if old_name == new_name or old_name in new_name or new_name in old_name:
        return True
    return f"formerly {old_name}" in new[2].lower()


def history():
    """Replay the git history of README.md.

    Returns (added, previous_urls, removed). A link that changes in a commit
    that also removes the old link of the same project (same name, or the new
    description says "formerly <old name>", or the only change in that section)
    counts as a move: the entry keeps its original date.
    """
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        raise SystemExit("agents.json needs the full git history; use actions/checkout with fetch-depth: 0")
    commits = git("log", "--first-parent", "--reverse", "--format=%H %ct", "--", "README.md").split()
    added, previous, removed, alive = {}, {}, {}, {}
    for sha, stamp in zip(commits[0::2], commits[1::2]):
        day = datetime.datetime.fromtimestamp(int(stamp), datetime.timezone.utc).date().isoformat()
        now = loose_entries(git("show", f"{sha}:README.md"))
        gone = [k for k in alive if k not in now]
        new = [k for k in now if k not in alive]
        pairs = {}
        for k in new:
            match = [g for g in gone if g not in pairs.values() and same_project(alive[g], now[k])]
            if len(match) == 1:
                pairs[k] = match[0]
        for k in new:
            if k in pairs:
                continue
            section = (now[k][3], now[k][4])
            left_new = [n for n in new if n not in pairs and (now[n][3], now[n][4]) == section]
            left_gone = [g for g in gone if g not in pairs.values() and (alive[g][3], alive[g][4]) == section]
            if len(left_new) == 1 and len(left_gone) == 1:
                pairs[k] = left_gone[0]
        for k in new:
            if k in pairs:
                old = pairs[k]
                added[k] = added.pop(old)
                previous[k] = previous.pop(old, []) + [alive[old][1]]
            elif k in removed:
                added[k] = removed.pop(k)["added"]
            else:
                added.setdefault(k, day)
        for g in gone:
            if g in pairs.values():
                continue
            name, url, description, category, subcategory = alive[g]
            removed[g] = {
                "name": name, "url": url, "description": plain(description),
                "category": category, "subcategory": subcategory,
                "added": added.pop(g), "removed": day,
            }
            if g in previous:
                removed[g]["previous_urls"] = previous.pop(g)
        alive = now
    first_day = None
    if commits:
        first_day = datetime.datetime.fromtimestamp(int(commits[1]), datetime.timezone.utc).date().isoformat()
    return added, previous, sorted(removed.values(), key=lambda r: (r["removed"], r["name"].lower())), first_day


def build(entries, categories):
    added, previous, removed, first_day = history()
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    for entry in entries:
        key = url_key(entry["url"])
        entry["added"] = added.get(key, today)
        if previous.get(key):
            entry["previous_urls"] = previous[key]
    return {
        "name": "Awesome AI Agents",
        "description": "A curated list of AI agent frameworks, tools, platforms, research papers and resources.",
        "do_not_edit": True,
        "generated_from": "README.md",
        "note": "Generated from README.md by scripts/build_agents_json.py on every push to main. Edit README.md instead.",
        "license": "CC0-1.0",
        "repository": REPOSITORY,
        "homepage": HOMEPAGE,
        "fields": {
            "added": f"UTC date of the commit that first put the entry on the list. {first_day} is the date the list was created; entries with that date were part of the initial list.",
            "previous_urls": "Earlier links of the same entry, for projects that moved or were renamed.",
            "removed_entries": "Entries that were on the list and were taken off, with the dates they were added and removed.",
        },
        "entry_count": len(entries),
        "removed_count": len(removed),
        "categories": categories,
        "entries": entries,
        "removed_entries": removed,
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
