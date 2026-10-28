# Installation

## Environment

Use Python 3.10 or newer; the core regression suite has been validated locally with Python 3.12. The environment snapshot documents local verification, not the original training environment.

Create a dedicated environment in this repository:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

For GPU training, install a PyTorch build compatible with the machine's CUDA runtime before installing this project. Legacy MID/pickle pipelines additionally use `python -m pip install -e ".[legacy]"`.

Run experiment commands from the repository root because configurations and artifacts use relative paths. An editable install makes Python imports available; it does not download data or weights.

## Check the installation

```bash
python tools/check_repository.py
python -m pytest tests/test_response_experiment.py tests/test_belief_experiment.py tests/test_closed_loop_correctness.py -q
```

The CPU check exercises response prediction, belief filtering and closed-loop correctness. It does not retrain the paper's models or replace external simulator evaluation.

`python -m pytest -q` selects the same three-file core suite by default. Run additional research test files explicitly after preparing their required assets.
