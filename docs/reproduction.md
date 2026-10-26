# Reproducing TRACER

## 1. Core regression

```bash
python -m pytest tests/test_response_experiment.py tests/test_belief_experiment.py tests/test_closed_loop_correctness.py -q
```

The core checks cover response-model behavior, posterior updates, candidate/state consistency, executed-prefix handling and chance-constrained selection. Broader tests are retained for research workflows; some require original experiment artifacts or external simulators.

## 2. Data-free training example

```bash
python -m sage.experiments.train_response_baselines --config configs/quickstart.yaml --device cpu --epochs 1 --limit-train-batches 1 --limit-val-batches 1 --variants joint --output-dir experiments/quickstart
```

`quickstart.yaml` uses the causal-response generator, with a small corpus and fixed data seed. Outputs are written to the ignored `experiments/` directory. This intentionally small run checks training and serialization; it does not reproduce formal multi-seed estimates.

## 3. Formal protocols

| Experiment | Configuration / implementation | Required context |
| --- | --- | --- |
| H1 factorial response prediction | `configs/h1_paired_enhanced_formal_train.yaml`, `configs/h1_paired_enhanced_formal_eval.yaml` | Fixed corpus seeds, separate model seeds, matched additive control and paired base scenes |
| H2 identity continuity | `configs/belief_h2.yaml`, persistent trajectory modules | Matched model, identity/memory ablations and update timing |
| H3 gate attribution | `configs/h3_belief_attribution_repaired_20260906.yaml` | Conditional attribution protocol; distinguish from ungated replanning |
| External navigation | `sage/experiments/` external adapters and SocialGym2 scripts | Original simulator versions, response checkpoints, maps and execution adapter |

Inspect each entry point's `--help` before launching a run. Formal configurations reference earlier gates or training artifacts that must be restored or regenerated; the repository does not claim a one-command reconstruction of those external assets.

## 4. Interpret the evidence

The manuscript's frozen tables live in `paper/arxiv/figures/*.tex` and the corresponding anonymous source package. The selected CSV index also includes important reanalysis and external extensions; use the correct protocol for each comparison.

H1 compares response predictions under controlled joint-action interventions. H2 isolates identity and persistent memory within the response model. Controlled ungated replanning and repaired matched-gate attribution answer different questions. SocialGym2 uses a binary GoAlone/Halt adapter; its results are simulation evidence alongside real trajectory data, rather than autonomous physical-robot validation.

For every new run, retain config, seeds, split, candidate-pool identity, checkpoint-selection rule, independent unit, paired contrasts and artifact checksums. Keep smoke/pilot/sweep results labeled as development.

## Asset availability

The complete historical runtime depended on external datasets, simulators and trained weights. Their availability and source snapshot identities are documented in [assets](assets.md). [Verification](verification.md) states exactly which standalone checks were run.
