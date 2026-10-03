# Repository contents and resource locations

| Location | Contents |
| --- | --- |
| `src/` | Installable implementation and package resources |
| `tests/` | Unit, integration-boundary and regression tests |
| `tests/reference/` | Original implementation inputs for deterministic comparisons, with per-file SHA-256 identities |
| `configs/` and `examples/` | Experiment configurations and small runnable examples |
| `reproduce/` | Scientific evidence verification and associated provenance |
| `docs/` | Architecture, configuration, migration and validation documentation |

Each checkout uses its own `.venv`, reconstructed from `uv.lock`.
External resources use the following configurable defaults:

```text
~/research/data/SBS-EA/           datasets and task resources
~/research/data/SBS-EA/evidence/  external evidence packages
~/research/models/                  model checkpoints
~/research/runs/SBS-EA/           experiment outputs
```

`PUBLIC_MANIFEST.json` defines the source export by path, size and SHA-256.
Package manifests define the wheel and source-distribution contents.
Scientific evidence and regression references retain their producer
identities; each new experiment records the source and inputs it uses.
