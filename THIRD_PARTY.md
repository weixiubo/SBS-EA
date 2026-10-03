# Third-party notices

Project-owned implementation is licensed under Apache-2.0. NumPy and optional
dependencies retain their upstream terms. Benchmark inputs and model weights
are separate resources documented in [RESOURCES.md](docs/RESOURCES.md).

## ReEvo

The repository's earlier `scripts/run_reevo_bpplib_baseline.py` included an
adaptation of the [ReEvo bin-packing seed heuristic](https://github.com/ai4co/reevo/blob/main/prompts/bpp_online/seed_func.txt).
Attribution: Copyright (c) 2023 Haoran Ye, licensed under MIT.
The full notice is retained in [LICENSES/ReEvo-MIT.txt](LICENSES/ReEvo-MIT.txt)
and included in distributions. The current package implements PFSP source-EA;
the earlier bin-packing runner remains part of repository history.

Regression inputs are identified in
[tests/reference/MANIFEST.json](tests/reference/MANIFEST.json). Scientific
evidence retains its original source and file identities.
