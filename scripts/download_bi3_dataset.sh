#!/usr/bin/env bash
set -euo pipefail

DESTINATION=${1:-raw_data/external/bi3}
ARCHIVE=${BI3_ARCHIVE:-}
URL='https://www.dropbox.com/scl/fo/ix27cqjcknemh01u9i2f3/ANbzit4ZiK9FsnAVKGiWai0?rlkey=b4pg0n0m1wjf3j5d6g8z3eqkb&st=ol52lejh&dl=1'

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"$REPO_ROOT/.conda/envs/hulan/bin/python"}

ARGS=(--url "$URL" --destination "$DESTINATION")
if [[ -n "$ARCHIVE" ]]; then
  ARGS+=(--archive "$ARCHIVE")
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m sage.data.real_response_download_bi3 "${ARGS[@]}"
