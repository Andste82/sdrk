# Open items

Items that remain open, are not covered by a later milestone, and need a decision or input that cannot be taken autonomously. Format: milestone, item, why it is open, what decision or input is needed.

| Milestone | Item | Why it is open | Needed decision / input |
| --- | --- | --- | --- |
| M04 | The `golden-check` CI job cannot be confirmed green locally | the workflow only runs on GitHub; locally `python tests/golden/generate.py --check` passes with Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1 | Confirm the first CI run is green after the branch is pushed. |
