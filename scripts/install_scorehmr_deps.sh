#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOREHMR_DIR="${REPO_ROOT}/third_party/ScoreHMR"

if [ ! -f "${SCOREHMR_DIR}/requirements.txt" ]; then
  echo "Missing ScoreHMR requirements.txt under ${SCOREHMR_DIR}" >&2
  echo "Run scripts/setup_scorehmr.sh first." >&2
  exit 1
fi

python -m pip install -r "${SCOREHMR_DIR}/requirements.txt"
python -m pip install -e "${SCOREHMR_DIR}"
