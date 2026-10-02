# Contributing

State the research role and input/output contract before adding a workflow. Keep
source preparation, search, held-out evaluation and evidence publication separate.
Add tests that exercise behavior and failure boundaries; preserve deterministic
event order and budget accounting. Do not claim frozen evidence supports altered
algorithms or silently repair a failed construction. Run the development and
release checks in docs/DEVELOPMENT.md. Keep model weights, private manuscripts,
credentials and full experiment outputs outside Git.

## Source distribution validation

After `uv build`, run `uv run --no-sync python scripts/check_sdist.py`.
It checks that the source distribution contains every declared test, fixture,
helper and example, creates a fresh environment from its own lock, runs the full
suite, and verifies that rebuilding its wheel preserves the distribution payload.
Dependencies must already be cached by the normal locked development setup;
this check runs offline and removes its temporary environment afterward.
