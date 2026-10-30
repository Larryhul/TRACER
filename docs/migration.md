# TRACER 0.2 migration

Python imports and commands now use `tracer`:

```python
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel
```

```bash
python -m tracer.experiments.train_response_baselines --config configs/quickstart.yaml --device cpu
```

Compatibility modules are under `tracer.legacy`. The distribution no longer installs top-level `sage`, `models`, `dataset`, `environment`, `evaluation` or `utils` packages.

State-dict checkpoint parameter keys remain unchanged. Use the matching saved configuration when loading a checkpoint. The dataset loader maps historical class paths through `tracer.compat.load_legacy_pickle`, without adding global module aliases. Load archived pickle files only from trusted sources.

The `candidate_source.type: sage` configuration value identifies a candidate-generator algorithm and remains unchanged. Numerical result files and historical source paths in provenance remain unchanged.
