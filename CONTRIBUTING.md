# Contributing

Define the study role and input/output contract of each workflow. Keep source
preparation, search, held-out evaluation and evidence recording separate.
Tests should cover deterministic event order, budget accounting, selection
boundaries and failure handling.

Changes to algorithms or experiment designs receive distinct source and run
identities. Preserve construction outcomes, original evidence and protocol
checksums. Keep model weights, credentials and complete experiment outputs
in the configured resource locations.

Run the development, reproduction and distribution checks in
[DEVELOPMENT.md](docs/DEVELOPMENT.md). Review the public file set before updating
`PUBLIC_MANIFEST.json`. The source-distribution check uses cached dependencies
to create a fresh locked environment, run the full suite and compare rebuilt
wheel contents.
