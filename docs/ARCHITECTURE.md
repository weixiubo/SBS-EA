# Maintained architecture

The dependency direction is CLI → settings/preflight → source preparation →
pure experiment/search → task evaluator. Filesystem publication lives in
execution/artifacts; the search and grammar never read local configuration.

- `settings`, `resources`, `datasets`: strict configuration, resource precedence,
  hash-bound inputs and explicit PFSP split membership.
- `priors/catalog_features`, `prior_catalog`: typed feature descriptions and fixed
  source preference tables. Diagnostic domain features are retained to test the
  shared catalog/grammar; the public experiment backend is PFSP.
- `priors/parsing`, `priors/budgets`, `semantic_prior`: proposal decoding,
  validation, priority-ordered capacity restriction and expression construction.
  No model response is executed as arbitrary Python.
- `probes`, `atomic_selection`, `source_preparation`: selection-only trajectories,
  behavioral deduplication and matched source-pool preparation.
- `source_ea/models`, `variation`, `scoring`: explicit candidate/result contracts,
  seeded stochastic operators and family-balanced robust selection scores.
- `pfsp_source_ea`: SearchState owns candidate scores, invalidity, first-generation
  identities and logs; generation evaluation is separate from variation and
  final held-out evaluation. Both search/control retain their original order.
- `pfsp_symbolic_gp`: independent comparison implementation with separate
  population ranking. It counts valid and invalid candidate attempts according
  to its original protocol; do not call that identical to the source-EA valid
  evaluation budget without an explicitly matched study design.
- `experiment_stats`: paired estimates, exact signed-rank tails and bootstrap
  utilities. `reproduction` invokes the original verifier in an isolated process.

New runs record source file hashes, package versions, dataset/proposal hashes,
prepared pools, seeds, modes and budgets. Publication uses exclusive atomic
records. A completed result can be reused after checksum validation. An
interrupted population is not guessed or silently reconstructed.

The dependency-selected scope and archived source hashes are in
SCOPE_MIGRATION.json. Earlier selector/kernel/remote launch/paper-production paths
do not supply the PFSP source-EA or its retained baseline once the three actual
atomic-selection helpers are extracted. Their original tests still run against
their original implementation through scripts/check_historical.py. They are
historical research code, not alternative current CLI implementations.
