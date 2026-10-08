# Corpus repos for PersonaLint

Picked for: pure Python, moderate size (fast to clone/process), long commit
history, multiple active contributors each with 50+ commits, and varied
coding styles (web framework, HTTP client, formatter, CLI tools) so the
"personalization matters" claim isn't tested on only one style of codebase.

| # | Repo | Why |
|---|------|-----|
| 1 | psf/requests | HTTP client, small/clean codebase, well-known style |
| 2 | pallets/flask | Web framework, mature, multiple long-term maintainers |
| 3 | encode/httpx | Modern async HTTP client, different style from requests |
| 4 | psf/black | Code formatter — ironic/fun choice, disciplined style |
| 5 | pytest-dev/pytest | Testing framework, large contributor base |
| 6 | tiangolo/fastapi | Modern, heavily typed, single dominant author + others |
| 7 | django/django (subset) | Very large — we'll cap commits per author, not clone full history depth |
| 8 | pallets/click | CLI framework, small/focused, same org as Flask (style contrast) |
| 9 | sqlalchemy/sqlalchemy | Older, more complex codebase — different smell profile |
| 10 | encode/starlette | Small async framework, sibling project to httpx/fastapi |
| 11 | psf/requests-toolbelt | Small, good for testing on a low-commit-count repo (edge case) |
| 12 | jazzband/pip-tools | Mid-size, different domain (packaging) |

For each repo we'll select the top 3–4 authors by commit count (so each has
enough history for a meaningful fingerprint) and cap at ~150 commits/author
to keep clone + processing time reasonable on a laptop.

Total expected scale: ~12 repos × ~3.5 authors × ~150 commits ≈ 6,000 commits,
likely 40,000–80,000 functions extracted. More than enough for both the
classifier training and the hand-labeled evaluation subset.
