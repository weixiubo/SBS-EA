# Migration

| Previous entry | Current entry |
| --- | --- |
| `pairwise_bpp_poc` PFSP modules | `sbs_ea` PFSP, `source_ea` and `priors` modules |
| `运行.py` and the multi-domain CLI | `sbs-ea doctor`, `demo`, `run --config`, `reproduce` |
| `run_pfsp_source_ea_development.py` | Configuration-driven execution |
| Confirmation and analysis wrappers | `reproduce` for recorded evidence; configured runs for new experiments |
| `sole_prior_attribution` atomic helpers | `atomic_selection` and PFSP probes |
| Automatic benchmark acquisition | Explicit datasets with SHA-256 bindings |
| Source-local run outputs | Configurable external run root |

The installed package supports PFSP source-EA and its controls. Earlier
BPP/JSSP selectors and manuscript-generation workflows are recorded in
`SCOPE_MIGRATION.json`; current execution uses the `sbs-ea` entrypoint.

`TEST_MIGRATION.json` maps test ownership. The current 148-case suite includes
75 deterministic comparisons and assertions from nine original test files
relevant to the supported scope. Original evidence is verified separately
by `sbs-ea reproduce`; new runs record distinct source and input identities.
