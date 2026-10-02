# Validation

The maintained suite passes 148 tests, including 75 exact original/current
comparisons. Before material cleanup, the complete original all-domain suite
passed 313 tests with two resource-dependent skips. That is a historical result;
its unused domain runners are no longer shipped or executed by current CI.
Retained assertions and the disposition of old scenarios are recorded in
TEST_MIGRATION.json and SCOPE_MIGRATION.json. The two counts are not added together.
All maintained code passes lint/format, eight strict type boundaries and
600/120/15 structure limits without exceptions. Distribution and fresh-export
checks are recorded in the completed local acceptance report. New deterministic checks
compare original/current priors, budget normalization, source-EA/no-operator
results, GP results, probe trajectories, atomic selections and statistics.
Required original comparison inputs retain their exact producer bytes and assertions.

Historical evidence retains 128 calls, 127 executable calls and 170 policies.
The strict endpoint remains No-Go. Verification reconstructs mean gaps and checks
recorded confidence intervals; it does not rerun the original bootstrap draws.

Not verified here: live model generation, GPU numerical equivalence, full Taillard/
VFR experiments, additional independent construction trials or private earlier
domain workflows against restored external data. CPU fixtures do not establish
those claims. No new research conclusions are inferred from refactoring tests.

The material-cleanup checks and current retained source list are documented in
MATERIALS.md and tests/reference/MANIFEST.json. Whole-source bundles are no longer
part of the current distribution. Scientific evidence identities are unchanged.

## Final distribution audit — 2026-10-03

The current suite passed 148 collected cases. The source distribution was unpacked into a new directory and installed in its own new locked environment.
All required test fixtures and helper files were present. The rebuilt wheel
payload matched the wheel built from the maintained checkout. Installed commands
ran outside the source directory with their declared resource and optional
dependency checks. Lint, formatting, types, structural limits and lock consistency
passed. CI now includes `check_sdist.py` after the ordinary build.

This verifies the maintained offline software and distribution boundaries.
The resource-dependent experiments listed above remain unverified; the test
result does not imply a new GPU experiment or a changed scientific conclusion.
