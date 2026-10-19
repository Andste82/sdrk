# Open items

Items that remain open, are not covered by a later milestone, and need a decision or input that cannot be taken autonomously. Format: milestone, item, why it is open, what decision or input is needed.

| Milestone | Item | Why it is open | Needed decision / input |
| --- | --- | --- | --- |
| M04 | The `golden-check` CI job cannot be confirmed green locally | the workflow only runs on GitHub; locally `python tests/golden/generate.py --check` passes with Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1 | Confirm the first CI run is green after the branch is pushed. |
| M08 | `python tests/golden/generate.py --check` could not be re-run in the final session (no NumPy in this environment) | the generator was verified green earlier in the milestone; the Rust golden tests over the committed vectors pass | Confirm the `golden-check` CI job is green on the M08 branch. |
| M09 | Windows x86_64 and Linux arm64 runs of the new tests and the CI jobs `golden-check`, `bare-metal` and `test` for M09 could not be confirmed locally | only Linux x86_64 is available in this environment; locally `cargo fmt`, `clippy -D warnings`, `cargo test --workspace --all-features`, `cargo xtask check`, `cargo xtask bare-metal`, `cargo deny check` and `python tests/golden/generate.py --check` (Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, 433 files, including the M08 vectors) pass | Confirm that the first CI run on the M09 branch is green on all three hosts; if the golden floating-point cases differ on arm64 or Windows, the tolerances of the case files (not the Rust code) are the place to look. |
