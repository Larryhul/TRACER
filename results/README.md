# Selected results and evidence

The 14 CSVs in [tables](tables) preserve important numerical results, paired contrasts and per-seed records. They include formal reanalysis and external extensions in addition to the manuscript's frozen tables.

| Group | Files | Evidence |
| --- | --- | --- |
| H1 | `H1_formal_method_means`, `H1_formal_paired_contrasts`, `H1_formal_per_seed` | Factorial response tests and paired scene statistics |
| Real H1 | `Real_H1_complete_refinement_all`, `Real_H1_refinement_paired_test_contrasts` | Real-trajectory response refinement |
| H2 | `H2_frozen_identity_continuity` | Offline identity/memory comparison; preserve its timing label |
| H3 | `H3_repaired_matched_gate`, `H3_attribution_per_seed` | Repaired conditional gate attribution |
| CrowdSimPlus | `CrowdSimPlus_unseen_holdout_7method`, `CrowdSimPlus_unseen_holdout_paired_contrasts` | Held-out extension; distinct from manuscript development scenes |
| SocialGym2 | `SocialGym2_8policy_external_comparison`, `SocialGym2_external_paired_contrasts`, `SocialGym2_frozen_10seed_policy_means`, `SocialGym2_frozen_paired_contrasts` | External comparisons and frozen statistics under their recorded protocols |

All file names above have the `.csv` extension. Original `use_status`, source identifiers and numerical fields are preserved. Known machine-specific storage prefixes use portable aliases recorded in [provenance](../provenance/source_files.csv).

The exact manuscript tables are versioned in [paper/arxiv/figures](../paper/arxiv/figures), including `TRACER_table_forward.tex`, `TRACER_table_identity.tex`, `TRACER_table_recorded.tex` and `TRACER_external_v19.tex`. The anonymous package is separately preserved in [paper/submission](../paper/submission).

The SocialGym2 v19 table reports 320 paired episodes and collision-free completion of 22.5% for GoAlone and 31.6% for TRACER. Component ablations in that table have comparable completion; use the table's actual values when discussing component attribution.

Full per-episode runs, training checkpoints, development sweeps and simulator snapshots are outside this source release. See [assets](../docs/assets.md).
