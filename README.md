# SBS-EA

One-call semantic guidance for evolutionary heuristic design. The maintained
research line is pre-v3 candidate B, with PFSP source-conditioned search and its
matched-budget controls. Existing `pairwise_bpp_poc` imports are retained.

## Install and run

```sh
uv venv --python 3.11
uv pip install -r requirements-dev.lock
uv pip install --no-deps -e .
.venv/bin/python examples/search_smoke.py
.venv/bin/python -m pytest -q
.venv/bin/python reproduce/verify.py
```

The local example uses generated seven-job instances. It is a behavior check,
not a substitute for the frozen benchmark results. Development targets Python
3.11, matching the original version bounds and NumPy 2.0.0.

## Evidence and limits

The frozen VFR220 confirmation supports a **secondary**, failure-aware claim:
127 of 128 calls were executable. The strict 128-of-128 construction endpoint
remains No-Go; the failed call was not repaired or replaced. Qwen transfer,
macro-free v3 and selected JSSP diagnostics remain negative. Post-hoc controls
must retain their labels rather than changing the original gate.

See `reproduce/README.md` for verified evidence, external model/benchmark needs,
and commands. See `docs/ARCHITECTURE.md` and `docs/MIGRATION.md` for code layout.
Private manuscript/review tools and machine-specific launch queues are archived.
