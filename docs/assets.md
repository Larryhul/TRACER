# External assets and reproducibility

## Included

Research code, retained tests, configurations, 14 selected CSVs, distinct submission/preprint source packages and a small causal-response training example.

## External dependencies

| Asset | Role | Availability |
| --- | --- | --- |
| SocialGym2 | Multi-robot social-navigation evaluation | Install from its upstream project; requires simulator/runtime setup |
| SICNav / CrowdSimPlus | External navigation and held-out scenes | Separate solver and simulator environment |
| CrowdNav / Python-RVO2 | Navigation baselines | Separate upstream dependencies |
| SACSoN / Bi3 | Real-trajectory and external response evaluation | Dataset and model terms apply |
| Response checkpoints and original per-episode runs | Frozen formal estimates and reanalysis | Retained outside this Git source release; no hosted download is advertised |

[External-source inventory](../third_party/README.md) identifies the upstream projects. [Snapshot hashes](../provenance/external_snapshots.csv) fingerprint the local source snapshots used during recovery; they are not upstream Git commit identifiers. Pin a verified upstream revision or restore the matching original snapshot before claiming exact simulator reproduction.

The original archive retained additional raw data and training weights. This repository does not imply that the entire historical asset set has been unpacked or made downloadable. CPU core checks and the quickstart use generated data and work without those assets.

## Result identity

The exact paper tables remain in the manuscript source packages. Selected CSVs also contain formal reanalysis and later external checks, with their status labels preserved. Large raw `runs.json` files, training weights and full third-party snapshots are outside this source repository.

Machine-specific source paths have portable aliases in executable defaults and CSV metadata. Every changed source is identified by original and distributed hash in [source_files.csv](../provenance/source_files.csv); manuscript packages remain byte-identical.
