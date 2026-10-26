#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.conda/envs/hulan/bin/python}

exec "$PYTHON" "$ROOT/scripts/run_socialgym2_external_baselines.py" "$@"
