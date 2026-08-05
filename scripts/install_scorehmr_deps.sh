#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOREHMR_DIR="${REPO_ROOT}/third_party/ScoreHMR"

if [ ! -d "${SCOREHMR_DIR}" ]; then
  echo "Missing ScoreHMR checkout under ${SCOREHMR_DIR}" >&2
  echo "Run scripts/setup_scorehmr.sh first." >&2
  exit 1
fi

python -m pip install -r "${REPO_ROOT}/requirements-scorehmr.txt"
python -m pip install -e "${SCOREHMR_DIR}"
