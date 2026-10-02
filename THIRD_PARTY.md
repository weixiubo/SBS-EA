# Third-party materials

Project-owned implementation is Apache-2.0. NumPy and optional installed packages
retain their upstream terms. Benchmark files and model weights are not bundled;
RESOURCE documentation records their separate sources and license status.

The original `scripts/run_reevo_bpplib_baseline.py`, now preserved byte-for-byte
inside `tests/reference/MANIFEST.json`, embeds an adapted
ReEvo bin-packing seed heuristic. Source:
[ReEvo seed_func.txt](https://github.com/ai4co/reevo/blob/main/prompts/bpp_online/seed_func.txt).
It is MIT licensed, Copyright (c) 2023 Haoran Ye. The complete notice is included
in `LICENSES/ReEvo-MIT.txt` and distributions. The earlier blanket statement that
all baseline code was only an adapter was incorrect. ReEvo's other code/assets
must be reviewed under their own notices when restored; this attribution covers
the embedded seed function and the corresponding upstream root MIT license.

Original source and evidence archives keep their original bytes. These accompanying
notices apply to those archived copies as well as any future extraction.
