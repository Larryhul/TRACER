<p align="center"><img src="docs/assets/hero.png" alt="TRACER" width="100%"></p>

<h3 align="center">Adaptive Multi-Robot Social Navigation via Joint Human-Response Prediction and Interaction-Aware Replanning</h3>

<p align="center">
  <a href="https://github.com/Larryhul/TRACER/actions/workflows/cpu-checks.yml"><img alt="CPU checks" src="https://github.com/Larryhul/TRACER/actions/workflows/cpu-checks.yml/badge.svg"></a>
  <a href="https://arxiv.org/abs/2609.18776"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.18776-b31b1b.svg"></a>
  <a href="LICENSE"><img alt="Code license: MIT" src="https://img.shields.io/badge/code-MIT-0d9488.svg"></a>
</p>

<p align="center"><a href="#quick-start">Quick start</a> · <a href="docs/reproduction.md">Experiments</a> · <a href="docs/models.md">Models</a> · <a href="results/README.md">Results</a> · <a href="https://arxiv.org/abs/2609.18776">Paper</a> · <a href="README.zh-CN.md">中文</a></p>

**TRACER predicts how people respond to joint robot actions, executes a short plan, and updates persistent beliefs from the observed response before replanning.**

- **Predict joint-action responses:** individual-robot effects and non-additive pairwise interactions.
- **Remember interactions:** identity-bound beliefs updated from executed-prefix observations.
- **Replan online:** response-aware candidate evaluation and synchronized execution.

## Quick start

Use Python 3.12 and follow the [installation guide](docs/installation.md) to create an environment. From this repository:

```bash
python -m pip install -e ".[dev]"
python -m tracer.demo --output outputs/demo
```

The CPU example trains a small response model on generated data, then runs seven decision windows with persistent belief updates. It writes an animation, a checkpoint and the decision trace.

![TRACER CPU demonstration](docs/assets/demo.gif)

See [the demo guide](docs/demo.md) for inputs, outputs and interpretation.

## Method

<p align="center"><img src="docs/assets/method.png" alt="TRACER method overview" width="100%"></p>

## Experiments and results

| Experiment | Reported result |
| --- | --- |
| Controlled joint-action response | Pair-effect RMSE decreases 23.94% relative to the capacity-matched additive predictor |
| Identity and memory | Persistent beliefs evaluated under ID/OOD identity and memory ablations |
| SocialGym2 navigation | Collision-free completion rises from 22.5% to 31.6% against GoAlone over 320 paired episodes |

The [result index](results/README.md) identifies the protocols and status of 14 retained CSV files. Commands and required inputs are in the [experiment guide](docs/reproduction.md).

## Models

A lightweight synthetic demonstration checkpoint is available. The [model guide](docs/models.md) describes its scope and the additional assets required for formal experiments.

```bash
python tools/download_models.py
```

Downloads are verified against the release checksum. See [model details](docs/models.md) for checkpoint identity and use.

## Development

```bash
python tools/check_repository.py
python -m pytest -q
```

CPU checks run on Linux and Windows. Tests requiring archived experiment outputs or CUDA report their prerequisites through explicit skips. See [verification](docs/verification.md).

## Repository layout

```text
tracer/        Response models, belief updates and planning
tracer/legacy/ Archived-data compatibility modules
configs/       Indexed formal, ablation and development protocols
tests/         CPU regressions and asset-dependent checks
results/       Selected tables and paired statistics
provenance/    File origins and checksums
```

## Citation

Please cite the [paper](https://arxiv.org/abs/2609.18776) when using TRACER in research. [BibTeX](docs/citation.bib) · [CITATION.cff](CITATION.cff).
