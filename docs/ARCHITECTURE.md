# Architecture

Execution follows CLI configuration, preflight checks, source preparation,
search and task evaluation. Filesystem recording belongs to `execution` and
`artifacts`; search and grammar components receive explicit inputs.

| Component | Responsibility |
| --- | --- |
| `settings`, `resources`, `datasets` | Configuration, path resolution, input hashes and PFSP split membership |
| `priors/catalog_features`, `prior_catalog` | Typed feature descriptions and source preference tables |
| `priors/parsing`, `priors/budgets`, `semantic_prior` | Proposal decoding, validation, capacity restriction and expression construction |
| `probes`, `atomic_selection`, `source_preparation` | Selection trajectories, behavioral deduplication and matched source pools |
| `source_ea/models`, `variation`, `scoring` | Candidate/result contracts, seeded operators and family-balanced selection scores |
| `pfsp_source_ea` | Population state, evaluation, variation and held-out evaluation |
| `pfsp_symbolic_gp` | Symbolic GP search and its population ranking |
| `experiment_stats` | Paired estimates, signed-rank tails and bootstrap utilities |
| `reproduction` | Original evidence verification in an isolated process |

`SearchState` owns candidate scores, invalidity, initial-generation identities
and logs. Search and no-operator control share valid-evaluation accounting.
Symbolic GP counts valid and invalid candidate attempts according to its own
protocol; comparisons specify the selected budget convention explicitly.

Proposals are parsed into supported expressions. Diagnostic domain features
support shared catalog and grammar tests; the configured experiment backend
is PFSP. Smoke and validation inputs determine source pools and selection;
held-out test instances are evaluated after policy selection.

Run records bind source hashes, package versions, dataset and proposal hashes,
prepared pools, seeds, modes and budgets. Atomic writes produce complete
result records. Resume verifies a completed result; interrupted searches
start a new run.

`SCOPE_MIGRATION.json` maps module ownership. The 75 differential cases use
original inputs identified in `tests/reference/MANIFEST.json`. Scientific
evidence has a separate identity and verification path.
