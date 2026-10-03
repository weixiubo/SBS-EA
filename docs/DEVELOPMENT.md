# Development

Use the locked Python 3.11 environment:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/check_structure.py
uv run pytest -q
uv run python scripts/check_historical.py
uv run sbs-ea reproduce
uv build
uv run python scripts/check_release.py
uv run python scripts/check_sdist.py
```

Maintained modules have a 600-line limit, functions a 120-line limit and
McCabe complexity a limit of 15. Core tests use CPU execution and offline
inputs. Type checks cover the public configuration and state boundaries.

Release checks inspect license notices and resources, install the wheel in
a separate environment, import its modules and execute search and reproduction
outside the source tree. The sdist check runs the complete suite in a fresh
locked environment and compares rebuilt wheel contents. Cache dependencies
before running the offline checks.

Review and update `PUBLIC_MANIFEST.json`, then use
`scripts/export_public.py --target PATH` with a new target directory to create
a checksum-verified source export. Each new study receives its own source,
protocol and input identities.
