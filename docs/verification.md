# Verification

The release checks cover installation, model serialization, the public CPU regression suite and the [demonstration](demo.md). CI runs on Linux and Windows.

The local v0.2.0 run on Windows/Python 3.12 collected 889 tests: 887 passed and two were skipped, for CUDA availability and an archived formal-run summary. Both navigation packages were installed together with zero shared Python file paths. The installed TRACER demo ran outside the source directory using the downloaded synthetic checkpoint.

```bash
python tools/check_repository.py
python -m pytest -q
python -m tracer.demo --output outputs/demo
```

The asset check verifies retained file hashes and local documentation links. Test output reports its passed and skipped cases. Tests requiring CUDA or archived formal-run assets carry explicit prerequisites.

The release process also builds a wheel, installs it in a separate directory and checks imports and the demo outside the source tree. Compatibility checks cover old class paths and coexistence with other Python packages without registering global aliases.

These are CPU software checks. Full benchmark reruns and physical-robot validation are separate research workflows.
