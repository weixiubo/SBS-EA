# Current implementation

- `prior_catalog.py`: typed prior records and domain feature/catalog contracts.
- `semantic_prior.py`: semantic proposal parsing, normalization and expression construction.
- `grammar.py`: restricted AST compilation, numeric evaluation and policy composition.
- `source_ea/models.py`, `variation.py`, `scoring.py`: search state, stochastic operators and family-balanced objective.
- `pfsp_source_ea.py`: exact-budget search orchestration and compatibility exports.
- `pfsp.py` / `public_pfsp.py`: task simulation and benchmark loading.
- `experiment_stats.py` and analysis scripts: paired/crossed result analysis.

The original package name and module exports remain available. Earlier selector,
kernel, JSSP and SOLE modules remain where imported by compatibility entrypoints,
comparators or frozen diagnostics. Their presence does not make them current
method candidates. Do not choose a route based on its version-number suffix.

The regression fixture captures six pre-refactor runs, including each generation,
both evolutionary/no-operator searches, exact evaluation counts and test scores.
