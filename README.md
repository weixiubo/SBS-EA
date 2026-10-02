# SBS-EA

One-call semantic guidance for evolutionary heuristic design. The maintained
implementation is PFSP source-conditioned population EA, its exact-budget
no-operator control, semantic source construction and the symbolic GP comparison.
The installed package is `sbs_ea`; the command is `sbs-ea`.

## Install and try

Python 3.11 and NumPy 2.0.0 preserve the verified numerical environment.

```sh
uv sync --locked
uv run sbs-ea doctor
uv run sbs-ea demo
uv run sbs-ea run --config configs/demo.json
uv run sbs-ea run --config configs/demo.json --resume
uv run sbs-ea reproduce
```

The demo executes actual search on generated seven-job problems. Its NEH reference
is a fixture upper bound, not a claim of optimality or a replacement for benchmark
results. The configured run writes under `~/research/runs/SBS-EA/<run_id>/`.
Choose a new run_id for another experiment. Resume accepts a complete result only
after verifying its source/input identity and result checksum.

## Run a study

Copy `configs/experiment.json`, restore your dataset and cached model proposal,
and fill in their SHA256 values. `sbs-ea doctor --config your-study.json` validates
the resources and prepares source pools before any run directory is created.
`sbs-ea run --config your-study.json` then executes the explicitly selected searches.
No command silently downloads benchmarks or model weights, calls a paid model,
starts a server, or discovers another checkout.

`sources` control prior construction; `modes` select population_ea/no_operator.
The same source pool and exact valid-evaluation budget are used for both modes.
The first source sets realized pool capacity when match_pool_size is true; other
sources must match it. The smoke and validation splits alone build probes and
select policies. The held-out test split is evaluated only after selection.
The symbolic GP baseline is exposed separately through
`sbs_ea.pfsp_symbolic_gp.run_pfsp_symbolic_gp_search`; its evaluation semantics are
documented in ARCHITECTURE and are not relabelled as the same source-EA budget.

Cached model proposals are explicit inputs. Repeating EA seeds with one proposal
does not constitute independent model calls. Use distinct hash-bound proposals
and run IDs for independent construction trials. Live model generation and full
GPU experiments remain resource-dependent; this release supports supplied model
outputs, source-EA execution and the original evidence verifier.

## Evidence and scope

The frozen VFR220 confirmation supports a **secondary**, failure-aware claim:
127 of 128 calls were executable. The strict 128-of-128 endpoint remains No-Go;
the failed call was not repaired or replaced. Qwen transfer, macro-free v3 and
selected JSSP diagnostics retain their original negative labels. New experiments
always have a new source identity and historical_evidence=false.

`reproduce` verifies 1,802 original evidence files, call records, policy results,
and reconstructs mean gaps. It verifies the recorded confidence intervals; it
does not recompute their bootstrap draws. Historical source, tests and negative
studies are removed from the current working tree; the supported scope is PFSP source-EA.

See [architecture](docs/ARCHITECTURE.md), [resources](docs/RESOURCES.md),
[migration](docs/MIGRATION.md), [development](docs/DEVELOPMENT.md),
[validation](docs/VALIDATION.md), and [third-party notices](THIRD_PARTY.md).
No repository or research material is published automatically.

Material ownership and deletion rules: [docs/MATERIALS.md](docs/MATERIALS.md).
