# Reproducing TRACER

## 1. Public regressions

```bash
python -m pytest -q
```

The suite covers response models, posterior updates, candidate/state consistency, executed-prefix handling and chance-constrained selection. Checks requiring CUDA or archived formal summaries report explicit skips when their inputs are absent. Run `python -m tracer.demo --output outputs/demo` for an animated CPU example.

## 2. Data-free training example

```bash
python -m tracer.experiments.train_response_baselines --config configs/quickstart.yaml --device cpu --epochs 1 --limit-train-batches 1 --limit-val-batches 1 --variants joint --output-dir experiments/quickstart
```

`quickstart.yaml` uses the causal-response generator, with a small corpus and fixed data seed. Outputs are written to the ignored `experiments/` directory. This intentionally small run checks training and serialization; it does not reproduce formal multi-seed estimates.

## 3. Formal protocols

| Experiment | Configuration / implementation | Required context |
| --- | --- | --- |
| H1 factorial response prediction | `configs/h1_paired_enhanced_formal_train.yaml`, `configs/h1_paired_enhanced_formal_eval.yaml` | Fixed corpus seeds, separate model seeds, matched additive control and paired base scenes |
| H2 identity continuity | `configs/belief_h2.yaml`, persistent trajectory modules | Matched model, identity/memory ablations and update timing |
| H3 gate attribution | `configs/h3_belief_attribution_repaired_20260906.yaml` | Conditional attribution protocol; distinguish from ungated replanning |
| External navigation | `tracer/experiments/` external adapters and SocialGym2 scripts | Original simulator versions, response checkpoints, maps and execution adapter |

The [configuration index](../configs/README.md) groups formal protocols, ablations and development runs. Each entry point's `--help` lists its overrides. Formal protocols require the matching gates, checkpoints, datasets and simulator versions listed in the asset guide.

## 4. Interpret the evidence

The paper and its reported tables are available in the [arXiv preprint](https://arxiv.org/abs/2609.18776). The selected CSV index also includes important reanalysis and external extensions; use the correct protocol for each comparison.

H1 compares response predictions under controlled joint-action interventions. H2 isolates identity and persistent memory within the response model. Controlled ungated replanning and repaired matched-gate attribution answer different questions. SocialGym2 uses a binary GoAlone/Halt adapter; its results are simulation evidence alongside real trajectory data, rather than autonomous physical-robot validation.

For every new run, retain config, seeds, split, candidate-pool identity, checkpoint-selection rule, independent unit, paired contrasts and artifact checksums. Keep smoke/pilot/sweep results labeled as development.

## Asset availability

The complete historical runtime depended on external datasets, simulators and trained weights. Their availability and source snapshot identities are documented in [assets](assets.md). [Verification](verification.md) states exactly which standalone checks were run.
