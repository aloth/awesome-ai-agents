#!/usr/bin/env python3
"""Prepare a monthly release: metrics.json, release notes and a maintenance report.

Run by .github/workflows/release.yml. Needs GITHUB_TOKEN for the GraphQL API.

Usage:
    python3 scripts/build_release.py VERSION OUTDIR [--force]

Writes metrics.json in the repository root (a dated snapshot of GitHub data per
entry, archived on Zenodo with each release), OUTDIR/notes.md and
OUTDIR/maintenance.md. Exits with code 3 when the list has not changed since
the previous release and --force is not given. Standard library only.
"""

import datetime
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO = re.compile(r"^https://github\.com/([^/]+)/([^/#?]+)/?$")
INACTIVE_DAYS = 183
NO_CHANGE = 3


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def previous_release():
    tags = [t for t in git("tag", "--list", "v*", "--sort=-creatordate").split() if t]
    return tags[0] if tags else None


def key(url):
    return url.rstrip("/").lower()


def graphql(query, token):
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("errors") and not payload.get("data"):
        raise SystemExit(f"GraphQL error: {payload['errors']}")
    return payload.get("data") or {}


def fetch(entries, token):
    repos = [(e, m.group(1), m.group(2)) for e in entries if (m := GITHUB_REPO.match(e["url"]))]
    found = {}
    for start in range(0, len(repos), 100):
        chunk = repos[start:start + 100]
        fields = " ".join(
            f"r{i}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) "
            "{ url stargazerCount forkCount isArchived pushedAt "
            "licenseInfo { spdxId } primaryLanguage { name } }"
            for i, (_, o, n) in enumerate(chunk)
        )
        data = graphql("query { " + fields + " }", token)
        for i, (entry, _, _) in enumerate(chunk):
            found[entry["url"]] = data.get(f"r{i}")
    return repos, found


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    version, outdir = sys.argv[1], Path(sys.argv[2])
    force = "--force" in sys.argv
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads((ROOT / "agents.json").read_text(encoding="utf-8"))
    entries = data["entries"]
    today = datetime.datetime.now(datetime.timezone.utc).date()

    previous = previous_release()
    before = {}
    if previous:
        before = {key(e["url"]): e for e in json.loads(git("show", f"{previous}:agents.json"))["entries"]}
    now = {key(e["url"]): e for e in entries}
    moved_now = {key(u): k for k, e in now.items() for u in e.get("previous_urls", [])}
    new = [now[k] for k in now if k not in before and not any(key(u) in before for u in now[k].get("previous_urls", []))]
    gone = [before[k] for k in before if k not in now and k not in moved_now]

    if previous and not new and not gone and not force:
        print(f"no entries added or removed since {previous}; skipping the release")
        return NO_CHANGE

    repos, found = fetch(entries, token)
    snapshot = []
    for entry, owner, name in repos:
        repo = found.get(entry["url"])
        row = {"name": entry["name"], "url": entry["url"]}
        if repo is None:
            row["status"] = "not_found"
        else:
            row.update({
                "repository": repo["url"],
                "stars": repo["stargazerCount"],
                "forks": repo["forkCount"],
                "archived": repo["isArchived"],
                "last_push": repo["pushedAt"][:10],
                "license": (repo["licenseInfo"] or {}).get("spdxId"),
                "language": (repo["primaryLanguage"] or {}).get("name"),
            })
        snapshot.append(row)

    metrics = {
        "name": "Awesome AI Agents: GitHub metrics snapshot",
        "do_not_edit": True,
        "note": "Written by scripts/build_release.py for each monthly release. One snapshot per release; earlier snapshots are in earlier releases.",
        "version": version,
        "snapshot_date": today.isoformat(),
        "source": "GitHub GraphQL API",
        "fields": {
            "repository": "Current repository URL as reported by GitHub; differs from url when the repository moved.",
            "last_push": "Date of the last push to any branch.",
            "status": "not_found when the repository no longer exists or is private.",
        },
        "repository_count": len(snapshot),
        "repositories": snapshot,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def line(e):
        return f"- [{e['name']}]({e['url']}) ({e.get('subcategory') or e['category']})"

    notes = [f"Awesome AI Agents {version}: {len(entries)} entries in {len(data['categories'])} categories."]
    if previous:
        notes.append(f"\nChanges since {previous}: {len(new)} added, {len(gone)} removed.")
    else:
        notes.append(f"\nFirst archived release. The list was started on {min(e['added'] for e in entries)}; {len(data.get('removed_entries', []))} entries have been removed over its lifetime and are kept in `removed_entries`.")
    if new and previous:
        notes.append("\n## Added\n")
        notes += [line(e) for e in sorted(new, key=lambda e: e["name"].lower())]
    if gone:
        notes.append("\n## Removed\n")
        notes += [line(e) for e in sorted(gone, key=lambda e: e["name"].lower())]
    notes.append("\n## Files\n")
    notes.append("- `README.md`: the list")
    notes.append("- `agents.json`: every entry with category, description, the date it was added and earlier links; entries taken off the list are under `removed_entries`")
    notes.append(f"- `metrics.json`: stars, forks, archive status, last push, license and language of {len(snapshot)} GitHub repositories on {today.isoformat()}")
    (outdir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    cutoff = (today - datetime.timedelta(days=INACTIVE_DAYS)).isoformat()
    rows = {
        "Archived repositories": [r for r in snapshot if r.get("archived")],
        "Repositories not found": [r for r in snapshot if r.get("status") == "not_found"],
        "Moved repositories (update the link)": [r for r in snapshot if r.get("repository") and key(r["repository"]) != key(r["url"])],
        f"No push since {cutoff}": [r for r in snapshot if r.get("last_push") and r["last_push"] < cutoff and not r.get("archived")],
    }
    report = []
    for title, items in rows.items():
        if items:
            report.append(f"\n## {title} ({len(items)})\n")
            for r in items:
                extra = f" -> {r['repository']}" if "Moved" in title else (f", last push {r['last_push']}" if r.get("last_push") else "")
                report.append(f"- [ ] [{r['name']}]({r['url']}){extra}")
    if report:
        head = [f"Maintenance check for {version}, {today.isoformat()}.",
                "",
                "Per CONTRIBUTING.md, archived projects, projects replaced by a successor and dead links are removed or updated through a regular pull request. A long time without a push alone is not a reason to remove an entry; it is listed for review."]
        (outdir / "maintenance.md").write_text("\n".join(head + report) + "\n", encoding="utf-8")
    print(f"metrics for {len(snapshot)} repositories; {len(new)} added, {len(gone)} removed; maintenance items: {sum(len(v) for v in rows.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
