# TRACER assets

## Available in the repository

Source code, indexed configurations, selected numerical results, method graphics and a runnable CPU demonstration.

## Model downloads

Small generated-data demonstration checkpoint; formal navigation checkpoints are not included.

Use [the model guide](models.md) to download and verify the release archive. File identities are recorded in [model_release.json](../provenance/model_release.json).

## Dataset and simulator inputs

External navigation uses SocialGym2, SICNav/CrowdSimPlus and their solver/runtime dependencies. Real-trajectory studies use the corresponding dataset inputs. Install these from the [upstream source index](../third_party/README.md).

[Snapshot hashes](../provenance/external_snapshots.csv) identify the retained local source snapshots. A fresh installation should record the actual upstream commit alongside its runtime environment. Formal protocols also require matching response checkpoints, calibration files and per-episode records.

Dataset files, trained outputs and full simulator trees stay outside source control. Copied source identities are recorded in [provenance](../provenance/README.md).
