# Migration

| Old entry | Maintained entry |
| --- | --- |
| pairwise_bpp_poc PFSP modules | sbs_ea PFSP/source_ea/priors modules |
| 运行.py / old broad CLI | sbs-ea doctor, demo, run --config, reproduce |
| run_pfsp_source_ea_development.py | config-driven run; source pools and evaluation in dedicated components |
| old sealed confirmation/analysis scripts | sbs-ea reproduce for evidence; configured new runs for current code |
| sole_prior_attribution atomic helpers | atomic_selection and PFSP-only probes |
| source-tree benchmark auto-download | explicit local resources and hash-bound dataset JSON |
| source-tree runs/artifacts | configurable external run root |

The maintained scope is PFSP source-EA and its matched controls. Earlier BPP/JSSP
selector workflows, remote launchers and unused manuscript generators are removed
from the working tree. `SCOPE_MIGRATION.json` records each module decision. No old
namespace alias or alternate installation is offered.

The current suite has 148 cases, including 75 deterministic comparisons using
the required original PFSP/statistical dependency closure in `tests/reference`.
Nine directly relevant original test files retain their assertions. The previous
313-case all-domain run is a pre-cleanup validation record, not a current CI target
or evidence that unsupported old domains remain maintained.

`sbs-ea reproduce` checks the unchanged scientific evidence. New runs have new
identities and never rewrite old protocol signatures or evidence checksums.

See [MATERIALS.md](MATERIALS.md) for material ownership and deletion rules.
