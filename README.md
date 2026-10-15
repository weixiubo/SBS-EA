# SBS-EA

## Project handoff

Start with the [handoff guide](docs/HANDOFF.md) to recover the research records,
paper sources, datasets and historical development versions. The
[material release](https://github.com/weixiubo/SBS-EA/releases/tag/handoff-2026-10-15) is public;
the restoration script downloads and verifies its contents without a GitHub login.

SBS-EA studies one-call semantic guidance for evolutionary heuristic design.
The `sbs_ea` package implements PFSP source-conditioned population search,
semantic source construction, an exact-budget no-operator control and a
symbolic genetic-programming comparison.

## Installation and examples

The locked environment uses Python 3.11 and NumPy 2.0.0:

```sh
uv sync --locked
uv run sbs-ea doctor
uv run sbs-ea demo
uv run sbs-ea run --config configs/demo.json
uv run sbs-ea run --config configs/demo.json --resume
uv run sbs-ea reproduce
```

The demo searches generated seven-job PFSP instances using an NEH upper-bound
reference. Configured runs write to `~/research/runs/SBS-EA/<run_id>/`.
Assign a distinct run ID to each experiment; resume verifies completed
results against their source, input identities and checksums.

## Research workloads

Copy `configs/experiment.json`, supply a dataset and cached model proposal,
and record each input's SHA-256:

```sh
uv run sbs-ea doctor --config /path/to/study.json
uv run sbs-ea run --config /path/to/study.json
```

`sources` controls prior construction; `modes` selects `population_ea` and
`no_operator`. Both modes use the same source pool and valid-evaluation
budget. With `match_pool_size`, the first source determines realized pool
capacity and subsequent sources must match it.

Smoke and validation splits supply probes and policy selection. The test
split is evaluated after selection. The symbolic GP comparison is available
through `sbs_ea.pfsp_symbolic_gp.run_pfsp_symbolic_gp_search`, with its own
candidate-attempt accounting described in [architecture](docs/ARCHITECTURE.md).

A cached proposal is reused across configured EA seeds. Independent model
construction trials require distinct proposal records and run identities.
[Resources](docs/RESOURCES.md) specifies dataset schemas, proposal inputs
and model provenance.

## Evidence reproduction

`reproduce` verifies 1,802 original evidence files and reconstructs VFR220
mean gaps from 170 stored policy outcomes. The construction panel contains
128 calls, of which 127 were executable; its supported analysis is the
failure-aware secondary analysis. See the [reproduction guide](reproduce/README.md)
for the strict endpoint, recorded confidence intervals and study identities.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [repository layout](docs/MATERIALS.md)
- [Development](docs/DEVELOPMENT.md) and [contributing](CONTRIBUTING.md)
- [Migration](docs/MIGRATION.md), [validation](docs/VALIDATION.md) and [third-party notices](THIRD_PARTY.md)

Project-owned code is licensed under Apache-2.0. ReEvo attribution and its
MIT notice are included in the third-party documentation.
