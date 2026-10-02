# Validation

The maintained suite passes 148 tests, including 75 exact original/current
comparisons. The complete original suite separately passes 313 tests with two
resource-dependent skips. The two counts describe distinct suites and are not
added together as if all historical tests exercised the new implementation.
Every original assertion is preserved, with the suite mapping in TEST_MIGRATION.
All maintained code passes lint/format, eight strict type boundaries and
600/120/15 structure limits without exceptions. Distribution and fresh-export
checks are recorded in the completed local acceptance report. New deterministic checks
compare original/current priors, budget normalization, source-EA/no-operator
results, GP results, probe trajectories, atomic selections and statistics.
The original baseline suite is run separately, without weakening old assertions.

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
