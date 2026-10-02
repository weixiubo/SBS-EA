# Development

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
```

Ordinary maintained modules/functions/complexity are limited to 600/120/15;
there are currently no exceptions. Exact frozen archives are separately managed.
Core tests are CPU-only and use no model service. Release checks inspect licenses
and data, install the wheel in a separate environment, import every maintained
module and execute actual search/reproduction from outside source. Missing cached
small wheels can be fetched during setup; tests and verification are offline.

Use `scripts/export_public.py --target /new/directory` after updating the reviewed
PUBLIC_MANIFEST.json. The export verifies each listed byte. Preserve original
source/protocol checksums when adding a new study; assign new source identities.
