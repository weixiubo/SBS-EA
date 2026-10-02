# Migration to 0.2

| Old entry | Maintained entry |
| --- | --- |
| pairwise_bpp_poc PFSP modules | sbs_ea PFSP/source_ea/priors modules |
| 运行.py / old broad CLI | sbs-ea doctor, demo, run --config, reproduce |
| run_pfsp_source_ea_development.py | config-driven run; source pools and evaluation in dedicated components |
| old sealed confirmation/analysis scripts | exact baseline archive and sbs-ea reproduce |
| sole_prior_attribution atomic helpers | atomic_selection and PFSP-only probes |
| source-tree benchmark auto-download | explicit local resources and hash-bound dataset JSON |
| source-tree runs/artifacts | configurable external run root |

No old namespace alias is installed. Retained public functions have their original
meaning. New interfaces deliberately fail on invalid input before experiments.
Earlier selector/kernel, BPP/JSSP multi-round judging, remote launch and manuscript
scripts are frozen because they do not supply the current PFSP search/control
dependency graph. This decision is about research usage, not version suffixes or
function length. Original modules, tests and scripts remain byte-identical in
the complete baseline archive; SCOPE_MIGRATION.json maps each old module/test.

Every original test remains executable with `uv run python scripts/check_historical.py`.
The maintained suite also migrates nine directly relevant test files without
weakening their assertions and compares current behavior against the original
implementation. Historical-only tests prove the preserved workflows still have
their baseline behavior; they are not counted as tests of the new package.

Original reproduce/ and protocol files retain their original bytes and references.
Use `sbs-ea reproduce` for the supported installed verifier. A direct invocation
of an old script belongs in the extracted original snapshot. New runs do not
reuse old sealed protocol identities or modify old evidence checksums.
