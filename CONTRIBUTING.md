# Contributing to Awesome AI Agents

## Guidelines

### Adding an entry

- One line per entry: `- [Name](https://...) — Description`, one sentence, ideally under 25 words
- Sort alphabetically within the section; no duplicates across sections
- No promotional content or paid placements

What qualifies depends on the kind of entry:

| Kind | Requirement |
|---|---|
| Open-source project | Commits within the last 6 months, and 100+ GitHub stars or from an established organization |
| Product or platform without a public repository | Generally available from an established vendor |
| Research paper | Peer-reviewed, or widely cited (50+ citations) |
| Course, tutorial or community | Freely accessible and maintained |

### Staying on the list

These requirements apply when an entry is added. Existing entries stay unless the project is archived, replaced by a successor, or its link no longer works. Moved repositories are updated to their new URL. A monthly maintenance check lists such cases, and removals go through a regular pull request.

## How to Contribute

**Not sure if your project fits?** [Open a suggestion issue](https://github.com/aloth/awesome-ai-agents/issues/new?template=add-project.yml) first - we'll let you know.

**Ready to submit?** Open a PR:

1. Fork this repository
2. Add your entry in the appropriate section of `README.md`, as one line in the form `- [Name](https://...) — Description`
3. Submit a PR with a brief explanation of why it belongs here

**Edit `README.md` only.** `agents.json` and the [landing page](https://aloth.github.io/awesome-ai-agents/) are generated from `README.md` after your PR is merged. A PR that changes `agents.json` fails the format check. The check also reports entries that do not follow the line format above.

## Quality Checks

Before submitting, verify:
- [ ] Link works and is not behind a paywall (for tools/frameworks)
- [ ] Entry meets the requirement for its kind (see the table above)
- [ ] Description is one sentence, ideally under 25 words
- [ ] Correct alphabetical placement
