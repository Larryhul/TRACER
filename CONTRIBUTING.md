# Contributing to TRACER

Create a dedicated environment using the [installation guide](docs/installation.md). Describe the intended behavior and include the configuration and input needed to reproduce a change.

```bash
python tools/check_repository.py
python -m pytest -q
```

For numerical changes, record seeds, data split, checkpoint-selection rule and the independent statistical unit. Keep development runs labeled and retain source notices. Dataset files, model outputs and private documents belong outside commits.

Use the issue forms for bugs or reproduction questions. Include the command, environment and relevant logs with credentials removed.
