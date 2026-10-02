# Material ownership and deletion policy

There is no general-purpose archive in the maintained project layout.

- Supported algorithms and workflows belong to the installed package, with a
  documented configuration and ordinary tests.
- `tests/reference/` contains only the original implementation dependencies used
  by regression tests. Its manifest checks every file before use. The original
  whole-repository source bundles were removed; their unused files are not a
  second installation or a future-work queue.
- `reproduce/` contains immutable scientific evidence and its verification tools.
  It proves recorded results; it is not a place to put unreviewed code. Signed
  evidence keeps its original bytes and source identities.
- External evidence belongs to `~/research/data/<project>/evidence`, and actual
  run records to `~/research/runs/<project>`. Both are explicit resource inputs.
- Duplicate source, obsolete wrappers and reconstructible environment residue
  are deleted after their replacements or lack of consumers are verified.
  Normal Git history records code changes; new source backup bundles are not made.

Unique private research results, manuscripts and uncommitted experimental code
are not declared useless from a directory name. Their research value must be
decided separately from whether the current package imports them.
