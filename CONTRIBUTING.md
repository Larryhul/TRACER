# Contributing

Use a separate environment for TRACER. Both SAGE and TRACER retain the Python package name `sage`; install only one in each environment.

1. Install the development dependencies in [the installation guide](docs/installation.md).
2. Make changes on a focused branch and explain the expected behavior.
3. Run the documented CPU check and any affected tests.
4. For numerical changes, record the config, seeds, checkpoint-selection rule, data split and independent statistical unit.

Keep generated weights, datasets, caches and private documents out of commits. Preserve upstream notices when changing compatibility modules. Mark exploratory experiments as development results and keep them separate from frozen manuscript evidence.

Use the issue templates for bug reports and reproduction questions. Include the Python/PyTorch versions and the exact command; remove credentials and personal data from logs.
