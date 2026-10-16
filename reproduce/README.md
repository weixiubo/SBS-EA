# PFSP evidence reproduction

```sh
uv run sbs-ea reproduce
```

The repository command above runs the packaged verifier. The offline
verifier checks 1,802 published evidence files, construction accounting and
stored policy outcomes, then reconstructs condition mean gaps for VFR220.
Bootstrap intervals are checked against their recorded evidence bindings.

## Recorded study outcomes

The confirmation panel contains 128 construction calls, of which 127 were
executable, and 170 policy outcomes. The strict 128-of-128 endpoint was not
met. The secondary failure-aware analysis includes the failed call without
repair or replacement. Qwen transfer, macro-free v3 and selected JSSP
diagnostics retain their recorded negative outcomes and study labels.

## Evidence and execution identities

`frozen_manifest.json` identifies the original supplement and included files.
`frozen/` contains PFSP evidence, call receipts, normalization records,
search/control results and labeled post-hoc analyses. A complete original
supplement can be verified with its own
`scripts/verify_anonymous_aaai_artifact.py --root-dir .` under Python 3.11.

Re-execution of an original study requires its recorded model revision,
VFR/Taillard inputs, source, seeds and budgets. Proposal-generation hardware
follows the selected model's requirements. Current PFSP experiments use
`sbs-ea run --config` with supplied dataset and proposal records; see
[resources](../docs/RESOURCES.md). The generated-data example exercises search
and control at fixed seeds with an NEH reference.

The current public edition omits recorded model-message bodies. Its integrity
indexes bind the published records; recorded statistical outcomes are retained.
