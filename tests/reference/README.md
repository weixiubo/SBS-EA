# Regression reference inputs

These inputs are used by the ordinary test suite. They check numerical results,
state transitions, event order, budgets or protocol counterexamples against
independent producer code. They are not an alternate installation or a backlog.

Only the required dependency closure is retained. Every input is checked against
its producer SHA-256 before use; changing expected behavior requires explicit
review. The maintained implementations live in the normal project package.
These fixtures are excluded from wheels and cannot be selected by the CLI.

Run the normal `uv run pytest -q` check. See `MANIFEST.json` for file identities.
