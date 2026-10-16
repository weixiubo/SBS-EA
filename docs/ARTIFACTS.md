# Research artifacts

This repository contains the maintained SBS-EA implementation. Its public
[research archive](https://github.com/weixiubo/SBS-EA/releases/tag/handoff-2026-10-15) contains the
research materials and their file-level checksums. The archive includes
recorded experiments, supporting inputs and historical source versions.

## Restore and use

Use Linux or WSL, with the checkout and material directory on the Linux
filesystem. Install Git, Python 3.11 and uv, then run:

```sh
git clone https://github.com/weixiubo/SBS-EA.git
cd SBS-EA
python3 scripts/handoff.py restore --root "$PWD/../SBS-EA-materials"
python3 scripts/handoff.py verify --root "$PWD/../SBS-EA-materials"
source "../SBS-EA-materials/activate.sh"
uv sync --locked
uv run --no-sync sbs-ea doctor
uv run --no-sync sbs-ea demo
uv run --no-sync sbs-ea reproduce
uv run --no-sync pytest -q
```

No GitHub account is required. Every release part is required; the script
checks downloads before extraction and verifies restored paths, SHA-256,
permissions and symbolic links. Repeating restoration reuses verified files
and refuses to overwrite local modifications. `--offline` uses existing
verified downloads. Keep new work outside the immutable `snapshot/` tree.

`resources.json` and `activate.sh` bind the restored inputs and a fresh run
output directory. Resource prerequisites for new experiments remain listed
in [RESOURCES.md](RESOURCES.md). Frozen evidence verification reconstructs
recorded results; it does not rerun GPU inference or live platform studies.

## Material organization

- Git: maintained source, tests, dependency lock and development branch.
- `snapshot/`: retained datasets, original experiment records, databases,
  checkpoints, papers and historical source snapshots.
- `snapshot/historical-git/`, when present: standalone historical Git stores,
  original-to-public commit maps, branch identities and otherwise unreferenced
  source blobs. `handoff/history-index.json` identifies each store.
- `handoff/material-disposition.json`: resource sources and explicit content
  transformations. Separately collected third-party literature PDFs are excluded;
  project papers, references and bibliography metadata remain.

Historical stores that contained excluded full texts have new commit IDs.
Their commit maps preserve the correspondence to original scientific source
identities. Retained source blobs are unchanged. Clone the selected
`repository.git` into a separate directory to work on a historical version;
keep the snapshot itself unchanged. Original branch, index and tag metadata
are retained for provenance. Existing partial upstream vendor repositories
remain identified as such and can use their recorded public origins.

```sh
git clone /path/to/materials/snapshot/historical-git/git-NN/repository.git historical-work
```

Caches, virtual environments and rebuildable distributions are omitted.
Identical payloads are stored once in the release; historical paths that are
needed to interpret records remain represented in the file manifest. The
manifest contains 42,324 retained file entries.

## Historical benchmark downloads

Three third-party ZIP downloads are not usable as standard ZIP archives
and are excluded. The usable baseline source copies and PFSP schedule CSVs
remain. In particular, the original Zenodo 4542886 TestSolutions.zip has the
same bytes and MD5 as the publisher copy but fails standard ZIP validation.
Use the recorded benchmark sources when setting up a new full study; the
packaged 1,802-file evidence verifier and generated-data examples are separate.
Hard-coded API credentials in five archived third-party EoH examples are removed;
the change records identify the original and sanitized file hashes.

## Verification records and archive scope

The release includes `verification.json`, `material-notes.json` and
`history-equivalence.json`. These record the checks performed on 2026-10-15,
source-version mappings and limitations of the historical inputs. Historical
shallow boundaries, normalized remote-reference names and reconstruction
commands are recorded in `history-equivalence.json`.
Directory resources may still be labelled `present_unverified` by `doctor`;
use the artifact verification command and the project's original evidence verifier
for their complete file-level checks.
