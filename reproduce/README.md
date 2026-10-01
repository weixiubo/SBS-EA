# Reproduction

`python reproduce/verify.py` works offline. It checks the original SHA256 values
from the existing anonymized supplement, preserves the failed call, validates
the 128/127 construction accounting, and reconstructs condition mean gaps from
170 stored policy outcomes on the VFR220 panel. Bootstrap intervals are checked
against recorded values, not recomputed by this lightweight command.

`frozen_manifest.json` records the original supplement digest and each included
file's original hash. `frozen/` contains the decisive PFSP evidence, raw call
receipts, normalization, search/control records and labeled post-hoc analyses.
The original supplement (including frozen source and earlier negative panels)
is retained in the private research archive under its recorded filename. To
verify that complete bundle, extract it and run its own
`scripts/verify_anonymous_aaai_artifact.py --root-dir .` with Python 3.11.
Do not use the refactored source to impersonate its historical source identity.

Full reruns require the exact recorded model/revision, the VFR/Taillard benchmark
inputs and adequate accelerator memory for proposal generation. These are not
bundled. `src/pairwise_bpp_poc/public_pfsp.py` records the benchmark sources and
loaders. Supply data/output/model paths explicitly; historical source hashes,
seeds, budgets, parser failures and no-replacement rules remain binding. The
recorded model environment belongs to the frozen experiment; current package
installation alone does not recreate a GPU model environment.

The development regression example uses generated data and runs both search
modes at three fixed seeds. It verifies state and budget preservation, not the
paper's numerical claim. External JSPLIB benchmark files are retained privately
pending redistribution-license verification.
