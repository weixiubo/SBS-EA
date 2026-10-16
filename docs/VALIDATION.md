# Validation

The current suite contains 158 cases, including 75 deterministic comparisons
against original producer modules. Comparisons cover priors, budget
normalization, source-EA and no-operator results, GP search, probe trajectories,
atomic selection and statistics. `TEST_MIGRATION.json` and
`SCOPE_MIGRATION.json` record scenario ownership and study scope.

Engineering checks include Ruff, eight strict type boundaries and limits of
600 lines per module, 120 lines per function and McCabe complexity 15.
[DEVELOPMENT.md](DEVELOPMENT.md) lists the complete commands.

## Scientific evidence

The verifier checks 1,802 original files, 128 construction calls, 127
executable calls and 170 policy outcomes. It reconstructs mean gaps and checks
the recorded confidence intervals against the evidence bindings. The strict
128-of-128 endpoint was not met; the failure-aware secondary analysis retains
the failed call. [Reproduction](../reproduce/README.md) describes the full
verification scope and recorded study outcomes.

## Distribution verification

Wheel checks inspect licenses and package resources, import installed modules
and execute search and reproduction outside the source tree. The sdist check
creates a fresh locked environment, runs the full suite and compares rebuilt
wheel contents. Source exports verify all `PUBLIC_MANIFEST.json` identities.

## Artifact restoration checks

Ten artifact restoration tests cover Unicode paths, missing and corrupt parts,
repeat execution, modified-file protection, upstream assets, optional
models and independent Git history with reference and link protection. The suite counts above include these cases. The restoration and
research-material checks for the distributed archive edition are recorded in the
[research archive](https://github.com/weixiubo/SBS-EA/releases/tag/research-artifacts-2026-10-16).
