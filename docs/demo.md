# TRACER CPU demonstration

```bash
python -m tracer.demo --output outputs/demo
```

The CPU example trains a small response model on generated data, then runs seven decision windows with persistent belief updates. It writes an animation, a checkpoint and the decision trace.

| Output | Contents |
| --- | --- |
| `tracer_demo.gif` | Animated execution or refinement trace |
| `tracer_demo.png` | Final overview |
| `summary.json` | Inputs, seed, scope and recorded trajectories |
| `tracer_demo.pt` | Small synthetic response checkpoint |

TRACER uses the repository response predictor, candidate-cost evaluation, synthetic simulator and persistent belief store. Only the executed prefix is used to update beliefs. The generated model is identified as a synthetic demonstration checkpoint.

This is a controlled illustration. Published benchmark results and their protocols are indexed in [results](../results/README.md).

Use `--checkpoint data/external/checkpoints/tracer-synthetic-demo.pt` after downloading the demo model to skip its small training step. Set `--seed` to select the generated episode.
