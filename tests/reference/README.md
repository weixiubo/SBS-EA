# Regression reference inputs

The test suite compares the current implementation with these original
producer modules and fixtures. Comparisons cover numerical results,
state transitions, event order, budgets and protocol behavior.

`MANIFEST.json` lists the dependency closure and each file's SHA-256.
The test loader verifies these identities before execution. Reference
files are distributed with the tests and excluded from runtime wheels.

Run the comparisons through the project test suite:

```sh
uv run --no-sync pytest -q
```
