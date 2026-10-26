# External projects

Full external source trees are kept out of this repository. Install each project in the environment required by its own documentation and preserve its license.

| Snapshot label | Upstream project | Use |
| --- | --- | --- |
| `socialgym2_official` | [ut-amrl/social_gym](https://github.com/ut-amrl/social_gym) | Multi-robot navigation simulator |
| `sicnav_official_filtered` | [sepsamavi/safe-interactive-crowdnav](https://github.com/sepsamavi/safe-interactive-crowdnav) | SICNav and CrowdSimPlus |
| `crowdnav_official` | [vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav) | Crowd navigation baselines |
| `python_rvo2_official` | [sybrenstuvel/Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) | ORCA/RVO2 bindings |
| `sacson` | [NHirose/SACSoN](https://github.com/NHirose/SACSoN) | Social navigation baseline/data |
| `Bi3` | [Bi3 dataset project](https://fluentrobotics.com/bi3dataset/) | Real interaction trajectories |

The original recovered snapshots are fingerprinted in [external_snapshots.csv](../provenance/external_snapshots.csv). A snapshot fingerprint verifies local contents; it does not establish a fresh upstream clone's equivalence. The core test suite does not require installing these projects.

Attribution for compatibility code included in this repository is in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
