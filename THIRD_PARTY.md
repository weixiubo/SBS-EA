# Third-party materials

Project-owned implementation is Apache-2.0. NumPy and optional installed packages
retain their upstream terms. Benchmark files and model weights are not bundled;
the resource documentation records their separate sources and license status.

The former `scripts/run_reevo_bpplib_baseline.py` embedded an adapted ReEvo
bin-packing seed heuristic from
[ReEvo seed_func.txt](https://github.com/ai4co/reevo/blob/main/prompts/bpp_online/seed_func.txt).
That unsupported runner was removed from the current working tree and is not one
of the retained regression reference files. It remains in earlier Git history.
Its attribution is retained here: MIT licensed, Copyright (c) 2023 Haoran Ye.
The complete notice is included in `LICENSES/ReEvo-MIT.txt` and distributions.
The earlier blanket statement that all baseline code was only an adapter was
incorrect. This notice covers the former embedded seed function; any additional
ReEvo code or assets require their own source and license review.

Required reference inputs and scientific evidence keep their original bytes.
`tests/reference/MANIFEST.json` lists the exact retained comparison files; it is
a file identity index, not a container for the removed runner or complete sources.
