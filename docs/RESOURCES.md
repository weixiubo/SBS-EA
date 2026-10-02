# Resource and environment recovery

Code and its .venv live in `~/project/SBS-EA`. Data defaults to
`~/research/data/SBS-EA`, models to `~/research/models`, new runs to
`~/research/runs/SBS-EA/<run_id>`, and historical backups to `~/research/data/<project>/evidence`.
Importing the package does not create any of these directories.

Precedence: CLI `--data-dir/--models-dir/--runs-dir/--evidence-dir`, run JSON paths,
SBS_EA_DATA_DIR/MODELS_DIR/RUNS_DIR/EVIDENCE_DIR, per-user JSON paths, defaults.
The user file is `$XDG_CONFIG_HOME/research/SBS-EA.json` (normally
`~/.config/research/SBS-EA.json`); SBS_EA_LOCAL_CONFIG selects another file.
Example: `{"paths":{"data":"/path/to/data","runs":"/path/to/runs"}}`.
Run-relative files resolve beside the configuration; placeholders {data},
{models}, {runs}, {evidence} are supported. Individual doctor catalog locations
can use the user JSON resources object or SBS_EA_RESOURCE_<ID>.

The machine-readable resource_catalog.json records purpose, source, version,
license status, target and recovery instructions. An unknown original digest is
null; presence is reported as unverified until a run binds a concrete hash. The
configuration requires an exact SHA256 for every external dataset or proposal.
Never substitute a newly downloaded benchmark/reference for an old sealed study
without assigning a new experiment identity.

## Dataset schema

A schema-1 JSON file contains `splits` with nonempty smoke, val and test lists.
Each instance has exactly family, instance_id, processing_times (positive integer
job × machine matrix), best_known_makespan (positive finite reference), and
reference_source (citation or reference manifest identity). IDs must be unique
across all splits. The reference is not assumed to be an optimum.
`examples/make_inputs.py --directory /chosen/path` writes a small complete example.

Original raw adapters read Taillard/VFR files under
`<data>/benchmarks/pfsp/taillard_zenodo_4542886`. Their sources are
[Zenodo 4542886](https://zenodo.org/records/4542886) and the reference schedules
identified in the original manifests. Read the source terms before downloading
or redistributing. The adapters now diagnose missing files instead of downloading
them. The new dataset JSON interface preserves explicit splits and references.

## Proposals and model dependencies

Supply the raw JSON/literal response expected by parse_prior_proposal in a
hash-bound proposal file. Record model identity, revision, decoding parameters,
license and generation seed in your research provenance. A cached response is
reused for the configured EA seeds; it is not counted as a new API call.
`uv sync --extra inference` installs the separately locked local inference
dependencies; `--extra service` is available for HTTP integration development.
They are not needed for CPU tests, demo, cached-proposal studies or reproduction.
Live generation is not performed by the maintained search command. GPU numerical
equivalence and a new independent construction study remain unverified until
the matching model/resources are restored.

Old environments are investigation material only. Restore needed external
resources from the archive inventory; do not copy an old .venv or handwritten
site-packages patch over the locked environment.
