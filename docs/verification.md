# Verification

The standalone publication directory was checked locally on 2026-10-27.

| Check | Outcome |
| --- | --- |
| `test_response_experiment.py`, `test_belief_experiment.py`, `test_closed_loop_correctness.py` | **110 passed**, CPU, Python 3.12 |
| Causal-response quickstart training | Passed; one epoch, validation metrics and checkpoint saved |
| Wheel build and separate-target installation | Passed; installed response and belief modules import successfully |
| Retained source identity and numerical CSV preservation | Passed for 611 copied source/asset files |
| Credential-pattern, internal-material and file-size scan | Passed for the publication candidate files |

The default `pytest` target is the three-file core suite above. The full extended test collection was not run; invoke extended test files explicitly when their external simulators and archived inputs are available. Core regression and quickstart success do not reproduce the complete formal multi-seed or external navigation experiments; required assets are listed in [assets](assets.md).

The GitHub workflow runs the asset check and these three core test files on push and pull request. A live GitHub workflow result is separate from the local results above.
