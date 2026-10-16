# Resources and configuration

Data defaults to `~/research/data/SBS-EA`, models to `~/research/models`,
outputs to `~/research/runs/SBS-EA/<run_id>` and external evidence to
`~/research/data/SBS-EA/evidence`. The checkout contains source and `.venv`.

Path precedence is CLI options, run-config `paths`, `SBS_EA_*_DIR` environment
variables, user JSON, then defaults. User configuration is
`$XDG_CONFIG_HOME/research/SBS-EA.json`, normally
`~/.config/research/SBS-EA.json`; `SBS_EA_LOCAL_CONFIG` selects another file.

```json
{"paths": {"data": "/path/to/data", "runs": "/path/to/runs"}}
```

Relative files resolve beside the configuration. Supported placeholders are
`{data}`, `{models}`, `{runs}` and `{evidence}`. Catalog locations also accept
a user `resources` map or `SBS_EA_RESOURCE_<ID>` override. Catalog entries
record sources, versions, license references and acquisition steps. Each
configured dataset and proposal requires an exact SHA-256 binding.

## Dataset schema

Schema 1 provides nonempty `smoke`, `val` and `test` split lists. Each instance
contains `family`, `instance_id`, `processing_times`, `best_known_makespan`
and `reference_source`. Processing times form a positive-integer job-by-machine
matrix. The reference makespan is positive and finite; its source identifies
the bound or schedule used. Instance IDs are unique across splits.

```sh
uv run python examples/make_inputs.py --directory /path/to/inputs
```

Raw Taillard/VFR adapters read
`<data>/benchmarks/pfsp/taillard_zenodo_4542886`. Their sources include
[Zenodo 4542886](https://zenodo.org/records/4542886) and the schedules identified
by the original manifests. Obtain these inputs under their source terms and
preserve explicit split and reference identities when preparing dataset JSON.

## Proposal inputs

Supply the JSON or literal response accepted by `parse_prior_proposal` in a
hash-bound file. Record the model revision, decoding parameters, generation
seed and license in the experiment provenance. A cached proposal supplies all
configured EA seeds; separate construction trials use distinct proposals.

`uv sync --locked --extra inference` provides optional local-inference
dependencies, and `--extra service` provides HTTP integration dependencies.
The search command consumes supplied proposals. CPU examples, tests and
evidence reproduction use the base environment.

## Archived research resources

The [artifact guide](ARTIFACTS.md) provides recorded research inputs,
resource configuration, file checksums and archive limitations. Use the
archive configuration for these inputs and the environment specifications
above for additional models or services.
